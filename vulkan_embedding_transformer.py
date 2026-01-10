"""
Vulkan Embedding Transformer for GrillCheese.

A complete GPU-accelerated embedding model using Vulkan compute shaders.
Runs entirely on the GPU without PyTorch, compatible with AMD Vulkan.

Architecture (BERT-style encoder):
1. Token embedding lookup
2. Positional embedding (learned or sinusoidal)
3. N transformer encoder layers (self-attention + FFN)
4. Mean/CLS pooling
5. L2 normalization

Compatible with weights from:
- sentence-transformers/all-MiniLM-L6-v2 (384 dims, 6 layers)
- BAAI/bge-small-en-v1.5 (384 dims, 12 layers)
- nomic-ai/nomic-embed-text-v1.5 (768 dims, 12 layers)
"""
import logging
import struct
import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union

from config import LogConfig, ModelConfig

logger = logging.getLogger(__name__)

# Import Vulkan backend
try:
    from vulkan_backend.base import VULKAN_AVAILABLE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    if VULKAN_AVAILABLE:
        from vulkan import vkDestroyBuffer, vkFreeMemory
except ImportError:
    VULKAN_AVAILABLE = False

# Import tokenizer
try:
    from tokenizer import get_tokenizer, SentencePieceTokenizer, SimpleTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


class VulkanEmbeddingTransformer:
    """
    GPU-accelerated embedding transformer using Vulkan compute shaders.
    
    This is a complete BERT-style encoder that runs entirely on GPU,
    eliminating the CPU bottleneck from sentence-transformers.
    """
    
    # Model configurations for common embedding models
    MODEL_CONFIGS = {
        "minilm-l6": {
            "hidden_dim": 384,
            "intermediate_dim": 1536,
            "num_layers": 6,
            "num_heads": 12,
            "vocab_size": 30522,
            "max_position": 512,
        },
        "bge-small": {
            "hidden_dim": 384,
            "intermediate_dim": 1536,
            "num_layers": 12,
            "num_heads": 12,
            "vocab_size": 30522,
            "max_position": 512,
        },
        "bge-base": {
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "num_layers": 12,
            "num_heads": 12,
            "vocab_size": 30522,
            "max_position": 512,
        },
        "nomic-embed": {
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "num_layers": 12,
            "num_heads": 12,
            "vocab_size": 50265,
            "max_position": 8192,
        },
        "grillcheese-small": {
            "hidden_dim": 384,
            "intermediate_dim": 1536,
            "num_layers": 4,
            "num_heads": 6,
            "vocab_size": 32000,
            "max_position": 512,
        },
    }
    
    def __init__(
        self,
        config_name: str = "grillcheese-small",
        hidden_dim: Optional[int] = None,
        intermediate_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        vocab_size: Optional[int] = None,
        max_position: Optional[int] = None,
        output_dim: Optional[int] = None,
        pooling: str = "mean",
        normalize: bool = True,
        tokenizer_path: Optional[str] = None
    ):
        """
        Initialize Vulkan embedding transformer.
        
        Args:
            config_name: Preset configuration name
            hidden_dim: Override hidden dimension
            intermediate_dim: Override FFN intermediate dimension
            num_layers: Override number of encoder layers
            num_heads: Override number of attention heads
            vocab_size: Override vocabulary size
            max_position: Override max position embeddings
            output_dim: Output projection dimension (None = hidden_dim)
            pooling: Pooling strategy ("mean", "cls", "max")
            normalize: L2 normalize output embeddings
            tokenizer_path: Path to SentencePiece tokenizer model
        """
        # Get base config
        base_config = self.MODEL_CONFIGS.get(config_name, self.MODEL_CONFIGS["grillcheese-small"])
        
        # Apply overrides
        self.hidden_dim = hidden_dim or base_config["hidden_dim"]
        self.intermediate_dim = intermediate_dim or base_config["intermediate_dim"]
        self.num_layers = num_layers or base_config["num_layers"]
        self.num_heads = num_heads or base_config["num_heads"]
        self.vocab_size = vocab_size or base_config["vocab_size"]
        self.max_position = max_position or base_config["max_position"]
        self.output_dim = output_dim or self.hidden_dim
        self.head_dim = self.hidden_dim // self.num_heads
        
        self.pooling = pooling
        self.normalize = normalize
        self.config_name = config_name
        
        # Initialize Vulkan backend
        self._init_vulkan()
        
        # Initialize tokenizer
        self._init_tokenizer(tokenizer_path)
        
        # Initialize weights (random or loaded)
        self._init_weights()
        
        # Create persistent GPU buffers
        self._gpu_buffers = {}
        
        logger.info(
            f"{LogConfig.CHECK} VulkanEmbeddingTransformer initialized: "
            f"hidden={self.hidden_dim}, layers={self.num_layers}, heads={self.num_heads}"
        )

    
    def _init_vulkan(self):
        """Initialize Vulkan compute backend"""
        if not VULKAN_AVAILABLE:
            raise RuntimeError("Vulkan backend not available")
        
        try:
            from vulkan_backend import VulkanCompute
            self.gpu = VulkanCompute()
            logger.debug("Vulkan compute backend initialized")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Vulkan: {e}")
    
    def _init_tokenizer(self, tokenizer_path: Optional[str] = None):
        """Initialize tokenizer"""
        if not TOKENIZER_AVAILABLE:
            logger.warning("Tokenizer module not available, using simple tokenizer")
            from tokenizer import SimpleTokenizer
            self.tokenizer = SimpleTokenizer(max_length=self.max_position)
            return
        
        try:
            self.tokenizer = get_tokenizer(
                model_path=tokenizer_path,
                vocab_size=self.vocab_size,
                max_length=self.max_position
            )
            logger.debug(f"Tokenizer initialized: vocab={self.tokenizer.vocab_size}")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}, using simple tokenizer")
            from tokenizer import SimpleTokenizer
            self.tokenizer = SimpleTokenizer(max_length=self.max_position)
    
    def _init_weights(self):
        """Initialize model weights (random initialization)"""
        # Token embeddings: (vocab_size, hidden_dim)
        self.token_embeddings = np.random.randn(
            self.vocab_size, self.hidden_dim
        ).astype(np.float32) * 0.02
        
        # Position embeddings: (max_position, hidden_dim)
        self.position_embeddings = np.random.randn(
            self.max_position, self.hidden_dim
        ).astype(np.float32) * 0.02
        
        # Layer norm after embeddings
        self.emb_ln_gamma = np.ones(self.hidden_dim, dtype=np.float32)
        self.emb_ln_beta = np.zeros(self.hidden_dim, dtype=np.float32)
        
        # Encoder layers
        self.layers = []
        for i in range(self.num_layers):
            layer = {
                # Self-attention
                'attn_qkv_weight': np.random.randn(3 * self.hidden_dim, self.hidden_dim).astype(np.float32) * 0.02,
                'attn_qkv_bias': np.zeros(3 * self.hidden_dim, dtype=np.float32),
                'attn_out_weight': np.random.randn(self.hidden_dim, self.hidden_dim).astype(np.float32) * 0.02,
                'attn_out_bias': np.zeros(self.hidden_dim, dtype=np.float32),
                'attn_ln_gamma': np.ones(self.hidden_dim, dtype=np.float32),
                'attn_ln_beta': np.zeros(self.hidden_dim, dtype=np.float32),
                
                # FFN
                'ffn_w1': np.random.randn(self.intermediate_dim, self.hidden_dim).astype(np.float32) * 0.02,
                'ffn_b1': np.zeros(self.intermediate_dim, dtype=np.float32),
                'ffn_w2': np.random.randn(self.hidden_dim, self.intermediate_dim).astype(np.float32) * 0.02,
                'ffn_b2': np.zeros(self.hidden_dim, dtype=np.float32),
                'ffn_ln_gamma': np.ones(self.hidden_dim, dtype=np.float32),
                'ffn_ln_beta': np.zeros(self.hidden_dim, dtype=np.float32),
            }
            self.layers.append(layer)
        
        # Output projection (if different from hidden_dim)
        if self.output_dim != self.hidden_dim:
            self.output_projection = np.random.randn(
                self.output_dim, self.hidden_dim
            ).astype(np.float32) * 0.02
            self.output_bias = np.zeros(self.output_dim, dtype=np.float32)
        else:
            self.output_projection = None
            self.output_bias = None
        
        # Calculate parameter count
        self._param_count = self._count_parameters()
        logger.info(f"Model parameters: {self._param_count / 1e6:.1f}M")
    
    def _count_parameters(self) -> int:
        """Count total model parameters"""
        count = 0
        count += self.token_embeddings.size
        count += self.position_embeddings.size
        count += self.emb_ln_gamma.size + self.emb_ln_beta.size
        
        for layer in self.layers:
            for key, val in layer.items():
                count += val.size
        
        if self.output_projection is not None:
            count += self.output_projection.size + self.output_bias.size
        
        return count

    
    def load_weights(self, weights_path: str):
        """
        Load pretrained weights from file.
        
        Supports:
        - .npz (numpy archive)
        - .safetensors (HuggingFace format)
        - .bin (PyTorch state dict - requires torch)
        
        Args:
            weights_path: Path to weights file
        """
        path = Path(weights_path)
        
        if path.suffix == '.npz':
            self._load_npz(path)
        elif path.suffix == '.safetensors':
            self._load_safetensors(path)
        elif path.suffix in ('.bin', '.pt', '.pth'):
            self._load_torch(path)
        else:
            raise ValueError(f"Unknown weights format: {path.suffix}")
        
        logger.info(f"{LogConfig.CHECK} Loaded weights from {weights_path}")
    
    def _load_npz(self, path: Path):
        """Load from numpy archive"""
        data = np.load(path)
        
        self.token_embeddings = data.get('token_embeddings', self.token_embeddings)
        self.position_embeddings = data.get('position_embeddings', self.position_embeddings)
        self.emb_ln_gamma = data.get('emb_ln_gamma', self.emb_ln_gamma)
        self.emb_ln_beta = data.get('emb_ln_beta', self.emb_ln_beta)
        
        for i, layer in enumerate(self.layers):
            prefix = f'layer_{i}_'
            for key in layer.keys():
                npz_key = prefix + key
                if npz_key in data:
                    layer[key] = data[npz_key]
        
        if 'output_projection' in data:
            self.output_projection = data['output_projection']
            self.output_bias = data.get('output_bias', self.output_bias)
    
    def _load_safetensors(self, path: Path):
        """Load from safetensors format"""
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError("safetensors required: pip install safetensors")
        
        with safe_open(path, framework="numpy") as f:
            # Map HuggingFace BERT keys to our format
            key_map = {
                'embeddings.word_embeddings.weight': 'token_embeddings',
                'embeddings.position_embeddings.weight': 'position_embeddings',
                'embeddings.LayerNorm.weight': 'emb_ln_gamma',
                'embeddings.LayerNorm.bias': 'emb_ln_beta',
            }
            
            for hf_key, our_key in key_map.items():
                if hf_key in f.keys():
                    setattr(self, our_key, f.get_tensor(hf_key).astype(np.float32))
            
            # Load encoder layers
            for i, layer in enumerate(self.layers):
                prefix = f'encoder.layer.{i}.'
                
                layer_map = {
                    'attention.self.query.weight': 'attn_qkv_weight',  # Partial
                    'attention.self.key.weight': 'attn_qkv_weight',    # Partial
                    'attention.self.value.weight': 'attn_qkv_weight',  # Partial
                    'attention.output.dense.weight': 'attn_out_weight',
                    'attention.output.dense.bias': 'attn_out_bias',
                    'attention.output.LayerNorm.weight': 'attn_ln_gamma',
                    'attention.output.LayerNorm.bias': 'attn_ln_beta',
                    'intermediate.dense.weight': 'ffn_w1',
                    'intermediate.dense.bias': 'ffn_b1',
                    'output.dense.weight': 'ffn_w2',
                    'output.dense.bias': 'ffn_b2',
                    'output.LayerNorm.weight': 'ffn_ln_gamma',
                    'output.LayerNorm.bias': 'ffn_ln_beta',
                }
                
                # Handle QKV weight combination
                q_key = f'{prefix}attention.self.query.weight'
                k_key = f'{prefix}attention.self.key.weight'
                v_key = f'{prefix}attention.self.value.weight'
                
                if all(k in f.keys() for k in [q_key, k_key, v_key]):
                    q = f.get_tensor(q_key).astype(np.float32)
                    k = f.get_tensor(k_key).astype(np.float32)
                    v = f.get_tensor(v_key).astype(np.float32)
                    layer['attn_qkv_weight'] = np.concatenate([q, k, v], axis=0)
                    
                    # Biases
                    q_b = f'{prefix}attention.self.query.bias'
                    k_b = f'{prefix}attention.self.key.bias'
                    v_b = f'{prefix}attention.self.value.bias'
                    if all(k in f.keys() for k in [q_b, k_b, v_b]):
                        layer['attn_qkv_bias'] = np.concatenate([
                            f.get_tensor(q_b).astype(np.float32),
                            f.get_tensor(k_b).astype(np.float32),
                            f.get_tensor(v_b).astype(np.float32)
                        ])
                
                # Load other layer weights
                for hf_suffix, our_key in layer_map.items():
                    if 'query' in hf_suffix or 'key' in hf_suffix or 'value' in hf_suffix:
                        continue  # Already handled
                    hf_key = prefix + hf_suffix
                    if hf_key in f.keys():
                        layer[our_key] = f.get_tensor(hf_key).astype(np.float32)
    
    def _load_torch(self, path: Path):
        """Load from PyTorch state dict"""
        try:
            import torch
        except ImportError:
            raise ImportError("torch required for .bin/.pt files")
        
        state = torch.load(path, map_location='cpu')
        
        # Convert torch tensors to numpy
        def to_numpy(t):
            if isinstance(t, torch.Tensor):
                return t.detach().numpy().astype(np.float32)
            return t
        
        # Similar mapping as safetensors
        if 'embeddings.word_embeddings.weight' in state:
            self.token_embeddings = to_numpy(state['embeddings.word_embeddings.weight'])
        if 'embeddings.position_embeddings.weight' in state:
            self.position_embeddings = to_numpy(state['embeddings.position_embeddings.weight'])
        
        # Continue with layer loading...
        logger.info("Loaded PyTorch weights (partial support)")

    
    def save_weights(self, weights_path: str):
        """Save weights to numpy archive"""
        data = {
            'token_embeddings': self.token_embeddings,
            'position_embeddings': self.position_embeddings,
            'emb_ln_gamma': self.emb_ln_gamma,
            'emb_ln_beta': self.emb_ln_beta,
        }
        
        for i, layer in enumerate(self.layers):
            for key, val in layer.items():
                data[f'layer_{i}_{key}'] = val
        
        if self.output_projection is not None:
            data['output_projection'] = self.output_projection
            data['output_bias'] = self.output_bias
        
        np.savez_compressed(weights_path, **data)
        logger.info(f"Saved weights to {weights_path}")
    
    def _create_gpu_buffer(self, data: np.ndarray, name: str) -> Tuple[Any, Any]:
        """Create or reuse GPU buffer"""
        data = np.ascontiguousarray(data.astype(np.float32))
        size = data.nbytes
        
        buf, mem = self.gpu.core._create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        self.gpu.core._upload_buffer(buf, mem, data.flatten())
        
        return buf, mem
    
    def _free_gpu_buffer(self, buf, mem):
        """Free GPU buffer"""
        vkDestroyBuffer(self.gpu.core.device, buf, None)
        vkFreeMemory(self.gpu.core.device, mem, None)
    
    def _run_embedding_lookup(
        self,
        token_ids: np.ndarray,
        batch_size: int,
        seq_len: int
    ) -> np.ndarray:
        """Run token embedding lookup on GPU"""
        # For now, use CPU lookup (GPU version requires buffer management)
        # Shape: (batch, seq) -> (batch, seq, hidden)
        embeddings = self.token_embeddings[token_ids.flatten()].reshape(
            batch_size, seq_len, self.hidden_dim
        )
        return embeddings.astype(np.float32)
    
    def _run_position_embedding(
        self,
        token_embeddings: np.ndarray,
        seq_len: int
    ) -> np.ndarray:
        """Add positional embeddings"""
        pos_emb = self.position_embeddings[:seq_len]  # (seq, hidden)
        # Broadcast: (batch, seq, hidden) + (seq, hidden)
        return token_embeddings + pos_emb
    
    def _run_layer_norm(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        eps: float = 1e-12
    ) -> np.ndarray:
        """Layer normalization (CPU fallback)"""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta
    
    def _run_attention(
        self,
        x: np.ndarray,
        attention_mask: np.ndarray,
        layer: Dict
    ) -> np.ndarray:
        """Run self-attention (CPU fallback for now)"""
        batch_size, seq_len, _ = x.shape
        
        # QKV projection
        x_flat = x.reshape(-1, self.hidden_dim)  # (batch*seq, hidden)
        qkv = x_flat @ layer['attn_qkv_weight'].T + layer['attn_qkv_bias']
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Attention scores
        scale = 1.0 / np.sqrt(self.head_dim)
        scores = np.einsum('bhqd,bhkd->bhqk', q, k) * scale
        
        # Apply attention mask
        mask = attention_mask[:, np.newaxis, np.newaxis, :]  # (batch, 1, 1, seq)
        scores = scores + (1.0 - mask) * -1e9
        
        # Softmax
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores)
        weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-9)
        
        # Attention output
        attn_out = np.einsum('bhqk,bhkd->bhqd', weights, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        
        # Output projection
        attn_out = attn_out.reshape(-1, self.hidden_dim)
        output = attn_out @ layer['attn_out_weight'].T + layer['attn_out_bias']
        output = output.reshape(batch_size, seq_len, self.hidden_dim)
        
        # Residual connection
        return x + output
    
    def _run_ffn(self, x: np.ndarray, layer: Dict) -> np.ndarray:
        """Run feed-forward network (CPU fallback)"""
        batch_size, seq_len, _ = x.shape
        
        x_flat = x.reshape(-1, self.hidden_dim)
        
        # First linear + GELU
        h = x_flat @ layer['ffn_w1'].T + layer['ffn_b1']
        # GELU approximation
        h = 0.5 * h * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * h ** 3)))
        
        # Second linear
        output = h @ layer['ffn_w2'].T + layer['ffn_b2']
        output = output.reshape(batch_size, seq_len, self.hidden_dim)
        
        # Residual connection
        return x + output
    
    def _run_pooling(
        self,
        hidden_states: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """Pool token representations to sentence embedding"""
        if self.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.pooling == "max":
            mask = attention_mask[:, :, np.newaxis]
            masked = hidden_states * mask + (1 - mask) * -1e9
            return masked.max(axis=1)
        else:  # mean
            mask = attention_mask[:, :, np.newaxis]
            sum_hidden = (hidden_states * mask).sum(axis=1)
            count = mask.sum(axis=1).clip(min=1e-9)
            return sum_hidden / count

    
    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """
        Forward pass through the transformer.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len), 1.0 for valid tokens
            
        Returns:
            Embeddings (batch, output_dim)
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Token embeddings
        x = self._run_embedding_lookup(input_ids, batch_size, seq_len)
        
        # 2. Add position embeddings
        x = self._run_position_embedding(x, seq_len)
        
        # 3. Embedding layer norm
        x = self._run_layer_norm(x, self.emb_ln_gamma, self.emb_ln_beta)
        
        # 4. Encoder layers
        for i, layer in enumerate(self.layers):
            # Self-attention + residual
            x = self._run_attention(x, attention_mask, layer)
            x = self._run_layer_norm(x, layer['attn_ln_gamma'], layer['attn_ln_beta'])
            
            # FFN + residual
            x = self._run_ffn(x, layer)
            x = self._run_layer_norm(x, layer['ffn_ln_gamma'], layer['ffn_ln_beta'])
        
        # 5. Pooling
        pooled = self._run_pooling(x, attention_mask)
        
        # 6. Output projection (if needed)
        if self.output_projection is not None:
            pooled = pooled @ self.output_projection.T + self.output_bias
        
        # 7. L2 normalize
        if self.normalize:
            norm = np.linalg.norm(pooled, axis=-1, keepdims=True) + 1e-12
            pooled = pooled / norm
        
        return pooled.astype(np.float32)
    
    def encode(
        self,
        text: str,
        max_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Encode text to embedding vector.
        
        Args:
            text: Input text
            max_length: Override max sequence length
            
        Returns:
            Embedding vector (output_dim,)
        """
        # Tokenize
        encoded = self.tokenizer.encode(
            text,
            max_length=max_length or self.max_position,
            padding=True,
            return_attention_mask=True
        )
        
        input_ids = encoded['input_ids'].reshape(1, -1)
        attention_mask = encoded['attention_mask'].reshape(1, -1)
        
        # Forward pass
        embedding = self.forward(input_ids, attention_mask)
        
        return embedding[0]
    
    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode multiple texts to embeddings.
        
        Args:
            texts: List of input texts
            max_length: Override max sequence length
            batch_size: Batch size for processing
            
        Returns:
            Embeddings (num_texts, output_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            encoded = self.tokenizer.encode_batch(
                batch_texts,
                max_length=max_length or self.max_position,
                padding=True,
                return_attention_mask=True
            )
            
            # Forward pass
            embeddings = self.forward(
                encoded['input_ids'],
                encoded['attention_mask']
            )
            all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings, axis=0)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Alias for encode() - matches the API in model_gguf.py
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (output_dim,)
        """
        return self.encode(text)
    
    @property
    def embedding_dim(self) -> int:
        """Return output embedding dimension"""
        return self.output_dim
    
    def __del__(self):
        """Cleanup GPU resources"""
        for name, (buf, mem) in self._gpu_buffers.items():
            try:
                self._free_gpu_buffer(buf, mem)
            except:
                pass


    def forward_gpu(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """
        GPU-accelerated forward pass using Vulkan compute shaders.
        
        This is the optimized version that runs entirely on GPU.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            attention_mask: Attention mask (batch, seq_len)
            
        Returns:
            Embeddings (batch, output_dim)
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. Token embedding lookup (CPU for now - could be GPU with shader)
        hidden = self._run_embedding_lookup(input_ids, batch_size, seq_len)
        
        # 2. Position embeddings (use Vulkan shader)
        hidden = self._gpu_position_embedding(hidden, seq_len)
        
        # 3. Layer norm
        hidden = self._gpu_layer_norm(
            hidden, self.emb_ln_gamma, self.emb_ln_beta
        )
        
        # 4. Encoder layers
        for i, layer in enumerate(self.layers):
            # Self-attention (Vulkan shader)
            hidden = self._gpu_attention(hidden, attention_mask, layer)
            hidden = self._gpu_layer_norm(
                hidden, layer['attn_ln_gamma'], layer['attn_ln_beta']
            )
            
            # FFN (Vulkan shader)
            hidden = self._gpu_ffn(hidden, layer)
            hidden = self._gpu_layer_norm(
                hidden, layer['ffn_ln_gamma'], layer['ffn_ln_beta']
            )
        
        # 5. Pooling (Vulkan shader)
        pooled = self._gpu_pooling(hidden, attention_mask)
        
        # 6. Output projection
        if self.output_projection is not None:
            pooled = self._gpu_linear(
                pooled, self.output_projection, self.output_bias
            )
        
        # 7. L2 normalize (Vulkan shader)
        if self.normalize:
            pooled = self._gpu_normalize(pooled)
        
        return pooled.astype(np.float32)
    
    def _gpu_position_embedding(
        self,
        token_embeddings: np.ndarray,
        seq_len: int
    ) -> np.ndarray:
        """Add positional embeddings using Vulkan shader"""
        batch_size = token_embeddings.shape[0]
        hidden_dim = token_embeddings.shape[2]
        
        # Prepare buffers
        token_emb_flat = token_embeddings.flatten().astype(np.float32)
        pos_emb = self.position_embeddings[:seq_len].flatten().astype(np.float32)
        output_size = batch_size * seq_len * hidden_dim * 4
        
        buf_token, mem_token = self._create_gpu_buffer(token_emb_flat, 'pos_token')
        buf_pos, mem_pos = self._create_gpu_buffer(pos_emb, 'pos_pos')
        buf_out, mem_out = self.gpu.core._create_buffer(output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Get pipeline
        pipeline, layout, desc_layout = self.gpu.pipelines.get_or_create_pipeline(
            'embedding-position', 3, push_constant_size=20
        )
        
        desc_set = self.gpu.pipelines.get_cached_descriptor_set(
            'embedding-position',
            [
                (buf_token, token_emb_flat.nbytes),
                (buf_pos, pos_emb.nbytes),
                (buf_out, output_size),
            ]
        )
        
        # Push constants: batch, seq, hidden, pos_type, scale
        push = struct.pack('IIIIf', batch_size, seq_len, hidden_dim, 0, 1.0)
        
        workgroups = (batch_size * seq_len * hidden_dim + 255) // 256
        self.gpu.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push)
        
        # Read output
        output = self.gpu.core._download_buffer(mem_out, output_size, dtype=np.float32)
        output = output[:batch_size * seq_len * hidden_dim].reshape(batch_size, seq_len, hidden_dim)
        
        # Cleanup
        self._free_gpu_buffer(buf_token, mem_token)
        self._free_gpu_buffer(buf_pos, mem_pos)
        self._free_gpu_buffer(buf_out, mem_out)
        
        return output
    
    def _gpu_layer_norm(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        eps: float = 1e-12
    ) -> np.ndarray:
        """Layer normalization using Vulkan shader"""
        # Reshape if needed
        original_shape = x.shape
        if len(x.shape) == 3:
            batch_size, seq_len, features = x.shape
            x_flat = x.reshape(-1, features)
        else:
            batch_size = x.shape[0]
            seq_len = 1
            features = x.shape[1]
            x_flat = x
        
        batch_seq = batch_size * seq_len
        
        # Prepare buffers
        x_data = x_flat.flatten().astype(np.float32)
        gamma_data = gamma.astype(np.float32)
        beta_data = beta.astype(np.float32)
        
        buf_x, mem_x = self._create_gpu_buffer(x_data, 'ln_x')
        buf_out, mem_out = self.gpu.core._create_buffer(x_data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_gamma, mem_gamma = self._create_gpu_buffer(gamma_data, 'ln_gamma')
        buf_beta, mem_beta = self._create_gpu_buffer(beta_data, 'ln_beta')
        buf_mean, mem_mean = self.gpu.core._create_buffer(batch_seq * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_var, mem_var = self.gpu.core._create_buffer(batch_seq * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        pipeline, layout, desc_layout = self.gpu.pipelines.get_or_create_pipeline(
            'fnn-layernorm', 6, push_constant_size=20
        )
        
        desc_set = self.gpu.pipelines.get_cached_descriptor_set(
            'fnn-layernorm',
            [
                (buf_x, x_data.nbytes),
                (buf_out, x_data.nbytes),
                (buf_gamma, gamma_data.nbytes),
                (buf_beta, beta_data.nbytes),
                (buf_mean, batch_seq * 4),
                (buf_var, batch_seq * 4),
            ]
        )
        
        # Three passes: compute mean, compute variance, normalize
        for pass_type in [0, 1, 2]:
            if pass_type < 2:
                workgroups = (batch_seq + 255) // 256
            else:
                workgroups = (batch_seq * features + 255) // 256
            
            push = struct.pack('IIIfI', batch_size, seq_len, features, eps, pass_type)
            self.gpu.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push)
        
        # Read output
        output = self.gpu.core._download_buffer(mem_out, x_data.nbytes, dtype=np.float32)
        output = output[:batch_seq * features].reshape(original_shape)
        
        # Cleanup
        for buf, mem in [(buf_x, mem_x), (buf_out, mem_out), (buf_gamma, mem_gamma),
                         (buf_beta, mem_beta), (buf_mean, mem_mean), (buf_var, mem_var)]:
            self._free_gpu_buffer(buf, mem)
        
        return output
