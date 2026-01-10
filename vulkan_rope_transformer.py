"""
Vulkan Embedding Transformer with RoPE for GrillCheese.

A complete GPU-accelerated embedding model using Vulkan compute shaders
with Rotary Position Embeddings (RoPE) for better position encoding.

Key Features:
- RoPE instead of learned positional embeddings (better extrapolation)
- Flash Attention 2 with fused RoPE (memory efficient)
- Runs entirely on GPU via Vulkan (AMD compatible)
- SentencePiece tokenization

Architecture:
1. Token embedding lookup
2. N transformer layers with:
   - RoPE-fused self-attention (flash attention 2)
   - Pre-LayerNorm
   - Feed-forward network with GELU
3. Mean/CLS pooling
4. L2 normalization
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


class VulkanEmbeddingConfig:
    """Configuration for Vulkan Embedding Transformer"""
    
    # Preset configurations
    PRESETS = {
        "grillcheese-tiny": {
            "hidden_dim": 256,
            "intermediate_dim": 1024,
            "num_layers": 3,
            "num_heads": 4,
            "vocab_size": 32000,
            "max_seq_len": 512,
            "rope_base": 10000.0,
        },
        "grillcheese-small": {
            "hidden_dim": 384,
            "intermediate_dim": 1536,
            "num_layers": 4,
            "num_heads": 6,
            "vocab_size": 32000,
            "max_seq_len": 512,
            "rope_base": 10000.0,
        },
        "grillcheese-base": {
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "num_layers": 6,
            "num_heads": 12,
            "vocab_size": 32000,
            "max_seq_len": 512,
            "rope_base": 10000.0,
        },
        "bge-small-v1.5": {
            "hidden_dim": 384,
            "intermediate_dim": 1536,
            "num_layers": 12,
            "num_heads": 12,
            "vocab_size": 30522,
            "max_seq_len": 512,
            "rope_base": 10000.0,
        },
        "nomic-embed-v1.5": {
            "hidden_dim": 768,
            "intermediate_dim": 3072,
            "num_layers": 12,
            "num_heads": 12,
            "vocab_size": 50265,
            "max_seq_len": 8192,
            "rope_base": 10000.0,
        },
    }
    
    def __init__(
        self,
        preset: str = "grillcheese-small",
        hidden_dim: Optional[int] = None,
        intermediate_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        vocab_size: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        rope_base: float = 10000.0,
        rope_scaling: float = 1.0,
        output_dim: Optional[int] = None,
        pooling: str = "mean",
        normalize: bool = True,
    ):
        # Load preset defaults
        defaults = self.PRESETS.get(preset, self.PRESETS["grillcheese-small"])
        
        # Apply overrides
        self.hidden_dim = hidden_dim or defaults["hidden_dim"]
        self.intermediate_dim = intermediate_dim or defaults["intermediate_dim"]
        self.num_layers = num_layers or defaults["num_layers"]
        self.num_heads = num_heads or defaults["num_heads"]
        self.vocab_size = vocab_size or defaults["vocab_size"]
        self.max_seq_len = max_seq_len or defaults["max_seq_len"]
        self.rope_base = rope_base
        self.rope_scaling = rope_scaling
        
        # Derived
        self.head_dim = self.hidden_dim // self.num_heads
        self.output_dim = output_dim or self.hidden_dim
        self.pooling = pooling
        self.normalize = normalize
        
        # Validation
        assert self.hidden_dim % self.num_heads == 0, \
            f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
        assert self.head_dim % 2 == 0, \
            f"head_dim ({self.head_dim}) must be even for RoPE"


class VulkanEmbeddingTransformerRoPE:
    """
    GPU-accelerated embedding transformer with RoPE.
    
    Uses Vulkan compute shaders for all operations, compatible with AMD GPUs.
    RoPE (Rotary Position Embeddings) provides better position encoding than
    learned positional embeddings.
    """
    
    def __init__(
        self,
        config: Optional[VulkanEmbeddingConfig] = None,
        tokenizer_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize Vulkan Embedding Transformer with RoPE.
        
        Args:
            config: Model configuration (uses default if None)
            tokenizer_path: Path to SentencePiece model
            weights_path: Path to pretrained weights
            use_gpu: Use GPU acceleration (falls back to CPU if False or unavailable)
        """
        self.config = config or VulkanEmbeddingConfig()
        self.use_gpu = use_gpu and VULKAN_AVAILABLE
        
        # Initialize components
        self._init_vulkan()
        self._init_tokenizer(tokenizer_path)
        self._init_weights()
        self._precompute_rope_cache()
        
        # Load weights if provided
        if weights_path:
            self.load_weights(weights_path)
        
        # GPU buffer cache
        self._gpu_buffers: Dict[str, Tuple[Any, Any]] = {}
        
        logger.info(
            f"{LogConfig.CHECK} VulkanEmbeddingTransformerRoPE initialized: "
            f"hidden={self.config.hidden_dim}, layers={self.config.num_layers}, "
            f"heads={self.config.num_heads}, RoPE base={self.config.rope_base}"
        )
    
    def _init_vulkan(self):
        """Initialize Vulkan compute backend"""
        if not self.use_gpu:
            self.gpu = None
            logger.info("Running in CPU mode")
            return
            
        try:
            from vulkan_backend import VulkanCompute
            self.gpu = VulkanCompute()
            logger.debug("Vulkan compute backend initialized")
        except Exception as e:
            logger.warning(f"Vulkan init failed: {e}, falling back to CPU")
            self.gpu = None
            self.use_gpu = False
    
    def _init_tokenizer(self, tokenizer_path: Optional[str]):
        """Initialize tokenizer"""
        if not TOKENIZER_AVAILABLE:
            from tokenizer import SimpleTokenizer
            self.tokenizer = SimpleTokenizer(max_length=self.config.max_seq_len)
            return
            
        try:
            self.tokenizer = get_tokenizer(
                model_path=tokenizer_path,
                vocab_size=self.config.vocab_size,
                max_length=self.config.max_seq_len
            )
        except Exception as e:
            logger.warning(f"Tokenizer load failed: {e}")
            from tokenizer import SimpleTokenizer
            self.tokenizer = SimpleTokenizer(max_length=self.config.max_seq_len)
    
    def _init_weights(self):
        """Initialize model weights with Xavier initialization"""
        cfg = self.config
        
        # Token embeddings (no position embeddings - we use RoPE)
        scale = np.sqrt(2.0 / (cfg.vocab_size + cfg.hidden_dim))
        self.token_embeddings = np.random.randn(
            cfg.vocab_size, cfg.hidden_dim
        ).astype(np.float32) * scale
        
        # Embedding LayerNorm
        self.emb_ln_gamma = np.ones(cfg.hidden_dim, dtype=np.float32)
        self.emb_ln_beta = np.zeros(cfg.hidden_dim, dtype=np.float32)
        
        # Encoder layers
        self.layers = []
        for i in range(cfg.num_layers):
            scale_attn = np.sqrt(2.0 / (cfg.hidden_dim * 2))
            scale_ffn = np.sqrt(2.0 / (cfg.hidden_dim + cfg.intermediate_dim))
            
            layer = {
                # Attention: separate Q, K, V projections (needed for RoPE)
                'q_weight': np.random.randn(cfg.hidden_dim, cfg.hidden_dim).astype(np.float32) * scale_attn,
                'q_bias': np.zeros(cfg.hidden_dim, dtype=np.float32),
                'k_weight': np.random.randn(cfg.hidden_dim, cfg.hidden_dim).astype(np.float32) * scale_attn,
                'k_bias': np.zeros(cfg.hidden_dim, dtype=np.float32),
                'v_weight': np.random.randn(cfg.hidden_dim, cfg.hidden_dim).astype(np.float32) * scale_attn,
                'v_bias': np.zeros(cfg.hidden_dim, dtype=np.float32),
                'out_weight': np.random.randn(cfg.hidden_dim, cfg.hidden_dim).astype(np.float32) * scale_attn,
                'out_bias': np.zeros(cfg.hidden_dim, dtype=np.float32),
                'attn_ln_gamma': np.ones(cfg.hidden_dim, dtype=np.float32),
                'attn_ln_beta': np.zeros(cfg.hidden_dim, dtype=np.float32),
                
                # FFN
                'ffn_up_weight': np.random.randn(cfg.intermediate_dim, cfg.hidden_dim).astype(np.float32) * scale_ffn,
                'ffn_up_bias': np.zeros(cfg.intermediate_dim, dtype=np.float32),
                'ffn_down_weight': np.random.randn(cfg.hidden_dim, cfg.intermediate_dim).astype(np.float32) * scale_ffn,
                'ffn_down_bias': np.zeros(cfg.hidden_dim, dtype=np.float32),
                'ffn_ln_gamma': np.ones(cfg.hidden_dim, dtype=np.float32),
                'ffn_ln_beta': np.zeros(cfg.hidden_dim, dtype=np.float32),
            }
            self.layers.append(layer)
        
        # Output projection (if different from hidden_dim)
        if cfg.output_dim != cfg.hidden_dim:
            scale_out = np.sqrt(2.0 / (cfg.hidden_dim + cfg.output_dim))
            self.output_proj_weight = np.random.randn(
                cfg.output_dim, cfg.hidden_dim
            ).astype(np.float32) * scale_out
            self.output_proj_bias = np.zeros(cfg.output_dim, dtype=np.float32)
        else:
            self.output_proj_weight = None
            self.output_proj_bias = None
        
        # Calculate parameter count
        self._param_count = self._count_parameters()
        logger.info(f"Model parameters: {self._param_count / 1e6:.2f}M")
    
    def _count_parameters(self) -> int:
        """Count total parameters"""
        count = self.token_embeddings.size
        count += self.emb_ln_gamma.size + self.emb_ln_beta.size
        
        for layer in self.layers:
            for val in layer.values():
                count += val.size
        
        if self.output_proj_weight is not None:
            count += self.output_proj_weight.size + self.output_proj_bias.size
        
        return count
    
    def _precompute_rope_cache(self):
        """Precompute RoPE cos/sin tables for efficiency"""
        cfg = self.config
        
        # Frequencies: theta_i = base^(-2i/head_dim) for i in [0, head_dim/2)
        dim_pairs = cfg.head_dim // 2
        freq_exp = -2.0 * np.arange(dim_pairs) / cfg.head_dim
        freqs = np.power(cfg.rope_base, freq_exp).astype(np.float32)  # (head_dim/2,)
        
        # Position indices: [0, 1, 2, ..., max_seq_len-1]
        positions = np.arange(cfg.max_seq_len).astype(np.float32)  # (max_seq_len,)
        
        # Theta matrix: positions @ freqs = (max_seq_len, head_dim/2)
        theta = np.outer(positions / cfg.rope_scaling, freqs)
        
        # Cos and Sin caches
        self.rope_cos_cache = np.cos(theta).astype(np.float32)  # (max_seq_len, head_dim/2)
        self.rope_sin_cache = np.sin(theta).astype(np.float32)  # (max_seq_len, head_dim/2)
        
        logger.debug(f"RoPE cache precomputed for seq_len={cfg.max_seq_len}")

    
    def _apply_rope(
        self,
        x: np.ndarray,
        seq_len: int,
        offset: int = 0
    ) -> np.ndarray:
        """
        Apply RoPE to Q or K tensor.
        
        Args:
            x: Input tensor (batch, seq, heads, head_dim)
            seq_len: Actual sequence length
            offset: Position offset (for KV cache)
            
        Returns:
            Rotated tensor (same shape)
        """
        batch, seq, heads, head_dim = x.shape
        
        # Get cos/sin for this sequence length
        cos = self.rope_cos_cache[offset:offset + seq_len]  # (seq, head_dim/2)
        sin = self.rope_sin_cache[offset:offset + seq_len]  # (seq, head_dim/2)
        
        # Reshape for broadcasting: (1, seq, 1, head_dim/2)
        cos = cos[np.newaxis, :, np.newaxis, :]
        sin = sin[np.newaxis, :, np.newaxis, :]
        
        # Split into pairs
        x_even = x[..., 0::2]  # (batch, seq, heads, head_dim/2)
        x_odd = x[..., 1::2]   # (batch, seq, heads, head_dim/2)
        
        # Apply rotation
        # x'[2i]   = x[2i] * cos - x[2i+1] * sin
        # x'[2i+1] = x[2i] * sin + x[2i+1] * cos
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        
        # Interleave back
        x_rot = np.empty_like(x)
        x_rot[..., 0::2] = x_rot_even
        x_rot[..., 1::2] = x_rot_odd
        
        return x_rot
    
    def _layer_norm(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        eps: float = 1e-6
    ) -> np.ndarray:
        """Layer normalization"""
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + eps)
        return gamma * x_norm + beta
    
    def _attention_with_rope(
        self,
        x: np.ndarray,
        attention_mask: np.ndarray,
        layer: Dict,
        seq_len: int
    ) -> np.ndarray:
        """
        Self-attention with RoPE applied to Q and K.
        
        Args:
            x: Input (batch, seq, hidden)
            attention_mask: Mask (batch, seq), 1.0 for valid
            layer: Layer weights dict
            seq_len: Actual sequence length
            
        Returns:
            Attention output (batch, seq, hidden)
        """
        cfg = self.config
        batch_size = x.shape[0]
        
        # Project to Q, K, V
        q = x @ layer['q_weight'].T + layer['q_bias']  # (batch, seq, hidden)
        k = x @ layer['k_weight'].T + layer['k_bias']
        v = x @ layer['v_weight'].T + layer['v_bias']
        
        # Reshape for multi-head attention: (batch, seq, heads, head_dim)
        q = q.reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)
        k = k.reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)
        v = v.reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)
        
        # Apply RoPE to Q and K (not V!)
        q = self._apply_rope(q, seq_len)
        k = self._apply_rope(k, seq_len)
        
        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(cfg.head_dim)
        scores = np.einsum('bhqd,bhkd->bhqk', q, k) * scale  # (batch, heads, seq, seq)
        
        # Apply mask: (batch, 1, 1, seq)
        mask = attention_mask[:, np.newaxis, np.newaxis, :]
        scores = scores + (1.0 - mask) * -1e9
        
        # Softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-9)
        
        # Attention output
        attn_out = np.einsum('bhqk,bhkd->bhqd', weights, v)  # (batch, heads, seq, head_dim)
        
        # Reshape back: (batch, seq, hidden)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, cfg.hidden_dim)
        
        # Output projection
        output = attn_out @ layer['out_weight'].T + layer['out_bias']
        
        return output
    
    def _ffn(self, x: np.ndarray, layer: Dict) -> np.ndarray:
        """Feed-forward network with GELU"""
        # Up projection
        h = x @ layer['ffn_up_weight'].T + layer['ffn_up_bias']
        
        # GELU activation (tanh approximation)
        h = 0.5 * h * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * h ** 3)))
        
        # Down projection
        output = h @ layer['ffn_down_weight'].T + layer['ffn_down_bias']
        
        return output
    
    def _pool(
        self,
        hidden_states: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """Pool token representations to sentence embedding"""
        if self.config.pooling == "cls":
            return hidden_states[:, 0, :]
        elif self.config.pooling == "max":
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
            input_ids: Token IDs (batch, seq)
            attention_mask: Attention mask (batch, seq)
            
        Returns:
            Embeddings (batch, output_dim)
        """
        batch_size, seq_len = input_ids.shape
        cfg = self.config
        
        # 1. Token embeddings (no position embeddings - RoPE handles position)
        x = self.token_embeddings[input_ids.flatten()].reshape(batch_size, seq_len, cfg.hidden_dim)
        
        # 2. Embedding LayerNorm
        x = self._layer_norm(x, self.emb_ln_gamma, self.emb_ln_beta)
        
        # 3. Transformer layers
        for layer in self.layers:
            # Pre-LN Attention
            ln_x = self._layer_norm(x, layer['attn_ln_gamma'], layer['attn_ln_beta'])
            attn_out = self._attention_with_rope(ln_x, attention_mask, layer, seq_len)
            x = x + attn_out  # Residual
            
            # Pre-LN FFN
            ln_x = self._layer_norm(x, layer['ffn_ln_gamma'], layer['ffn_ln_beta'])
            ffn_out = self._ffn(ln_x, layer)
            x = x + ffn_out  # Residual
        
        # 4. Pooling
        pooled = self._pool(x, attention_mask)
        
        # 5. Output projection
        if self.output_proj_weight is not None:
            pooled = pooled @ self.output_proj_weight.T + self.output_proj_bias
        
        # 6. L2 normalize
        if self.config.normalize:
            norm = np.linalg.norm(pooled, axis=-1, keepdims=True) + 1e-12
            pooled = pooled / norm
        
        return pooled.astype(np.float32)

    
    def encode(self, text: str, max_length: Optional[int] = None) -> np.ndarray:
        """
        Encode text to embedding vector.
        
        Args:
            text: Input text
            max_length: Override max sequence length
            
        Returns:
            Embedding vector (output_dim,)
        """
        encoded = self.tokenizer.encode(
            text,
            max_length=max_length or self.config.max_seq_len,
            padding=True,
            return_attention_mask=True
        )
        
        input_ids = encoded['input_ids'].reshape(1, -1)
        attention_mask = encoded['attention_mask'].reshape(1, -1)
        
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
            batch_size: Processing batch size
            
        Returns:
            Embeddings (num_texts, output_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            encoded = self.tokenizer.encode_batch(
                batch_texts,
                max_length=max_length or self.config.max_seq_len,
                padding=True,
                return_attention_mask=True
            )
            
            embeddings = self.forward(encoded['input_ids'], encoded['attention_mask'])
            all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings, axis=0)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Alias for encode() - matches model_gguf.py API"""
        return self.encode(text)
    
    @property
    def embedding_dim(self) -> int:
        """Return output embedding dimension"""
        return self.config.output_dim
    
    def save_weights(self, path: str):
        """Save weights to numpy archive"""
        data = {
            'token_embeddings': self.token_embeddings,
            'emb_ln_gamma': self.emb_ln_gamma,
            'emb_ln_beta': self.emb_ln_beta,
            'config_hidden_dim': self.config.hidden_dim,
            'config_num_layers': self.config.num_layers,
            'config_num_heads': self.config.num_heads,
            'config_rope_base': self.config.rope_base,
        }
        
        for i, layer in enumerate(self.layers):
            for key, val in layer.items():
                data[f'layer_{i}_{key}'] = val
        
        if self.output_proj_weight is not None:
            data['output_proj_weight'] = self.output_proj_weight
            data['output_proj_bias'] = self.output_proj_bias
        
        np.savez_compressed(path, **data)
        logger.info(f"Saved weights to {path}")
    
    def load_weights(self, path: str):
        """Load weights from numpy archive or safetensors"""
        path = Path(path)
        
        if path.suffix == '.npz':
            self._load_npz(path)
        elif path.suffix == '.safetensors':
            self._load_safetensors(path)
        else:
            raise ValueError(f"Unknown format: {path.suffix}")
        
        logger.info(f"{LogConfig.CHECK} Loaded weights from {path}")
    
    def _load_npz(self, path: Path):
        """Load from numpy archive"""
        data = np.load(path)
        
        if 'token_embeddings' in data:
            self.token_embeddings = data['token_embeddings']
        if 'emb_ln_gamma' in data:
            self.emb_ln_gamma = data['emb_ln_gamma']
        if 'emb_ln_beta' in data:
            self.emb_ln_beta = data['emb_ln_beta']
        
        for i, layer in enumerate(self.layers):
            for key in layer.keys():
                npz_key = f'layer_{i}_{key}'
                if npz_key in data:
                    layer[key] = data[npz_key]
        
        if 'output_proj_weight' in data:
            self.output_proj_weight = data['output_proj_weight']
            self.output_proj_bias = data['output_proj_bias']
    
    def _load_safetensors(self, path: Path):
        """Load from safetensors format (HuggingFace models)"""
        try:
            from safetensors import safe_open
        except ImportError:
            raise ImportError("safetensors required: pip install safetensors")
        
        with safe_open(path, framework="numpy") as f:
            keys = f.keys()
            
            # BERT-style key mapping
            if 'embeddings.word_embeddings.weight' in keys:
                self.token_embeddings = f.get_tensor('embeddings.word_embeddings.weight').astype(np.float32)
            
            if 'embeddings.LayerNorm.weight' in keys:
                self.emb_ln_gamma = f.get_tensor('embeddings.LayerNorm.weight').astype(np.float32)
            if 'embeddings.LayerNorm.bias' in keys:
                self.emb_ln_beta = f.get_tensor('embeddings.LayerNorm.bias').astype(np.float32)
            
            # Load encoder layers
            for i, layer in enumerate(self.layers):
                prefix = f'encoder.layer.{i}.'
                
                # Q, K, V weights (separate in our format)
                if f'{prefix}attention.self.query.weight' in keys:
                    layer['q_weight'] = f.get_tensor(f'{prefix}attention.self.query.weight').astype(np.float32)
                if f'{prefix}attention.self.query.bias' in keys:
                    layer['q_bias'] = f.get_tensor(f'{prefix}attention.self.query.bias').astype(np.float32)
                if f'{prefix}attention.self.key.weight' in keys:
                    layer['k_weight'] = f.get_tensor(f'{prefix}attention.self.key.weight').astype(np.float32)
                if f'{prefix}attention.self.key.bias' in keys:
                    layer['k_bias'] = f.get_tensor(f'{prefix}attention.self.key.bias').astype(np.float32)
                if f'{prefix}attention.self.value.weight' in keys:
                    layer['v_weight'] = f.get_tensor(f'{prefix}attention.self.value.weight').astype(np.float32)
                if f'{prefix}attention.self.value.bias' in keys:
                    layer['v_bias'] = f.get_tensor(f'{prefix}attention.self.value.bias').astype(np.float32)
                
                # Output projection
                if f'{prefix}attention.output.dense.weight' in keys:
                    layer['out_weight'] = f.get_tensor(f'{prefix}attention.output.dense.weight').astype(np.float32)
                if f'{prefix}attention.output.dense.bias' in keys:
                    layer['out_bias'] = f.get_tensor(f'{prefix}attention.output.dense.bias').astype(np.float32)
                
                # Attention LayerNorm
                if f'{prefix}attention.output.LayerNorm.weight' in keys:
                    layer['attn_ln_gamma'] = f.get_tensor(f'{prefix}attention.output.LayerNorm.weight').astype(np.float32)
                if f'{prefix}attention.output.LayerNorm.bias' in keys:
                    layer['attn_ln_beta'] = f.get_tensor(f'{prefix}attention.output.LayerNorm.bias').astype(np.float32)
                
                # FFN
                if f'{prefix}intermediate.dense.weight' in keys:
                    layer['ffn_up_weight'] = f.get_tensor(f'{prefix}intermediate.dense.weight').astype(np.float32)
                if f'{prefix}intermediate.dense.bias' in keys:
                    layer['ffn_up_bias'] = f.get_tensor(f'{prefix}intermediate.dense.bias').astype(np.float32)
                if f'{prefix}output.dense.weight' in keys:
                    layer['ffn_down_weight'] = f.get_tensor(f'{prefix}output.dense.weight').astype(np.float32)
                if f'{prefix}output.dense.bias' in keys:
                    layer['ffn_down_bias'] = f.get_tensor(f'{prefix}output.dense.bias').astype(np.float32)
                
                # FFN LayerNorm
                if f'{prefix}output.LayerNorm.weight' in keys:
                    layer['ffn_ln_gamma'] = f.get_tensor(f'{prefix}output.LayerNorm.weight').astype(np.float32)
                if f'{prefix}output.LayerNorm.bias' in keys:
                    layer['ffn_ln_beta'] = f.get_tensor(f'{prefix}output.LayerNorm.bias').astype(np.float32)

    
    # =========================================================================
    # GPU-Accelerated Methods (Vulkan)
    # =========================================================================
    
    def forward_gpu(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray
    ) -> np.ndarray:
        """
        GPU-accelerated forward pass using Vulkan compute shaders.
        
        Uses flash-attention2-rope for fused RoPE + attention.
        
        Args:
            input_ids: Token IDs (batch, seq)
            attention_mask: Attention mask (batch, seq)
            
        Returns:
            Embeddings (batch, output_dim)
        """
        if not self.use_gpu or self.gpu is None:
            return self.forward(input_ids, attention_mask)
        
        batch_size, seq_len = input_ids.shape
        cfg = self.config
        
        # 1. Token embeddings (CPU lookup, then upload)
        x = self.token_embeddings[input_ids.flatten()].reshape(batch_size, seq_len, cfg.hidden_dim)
        
        # 2. Embedding LayerNorm (GPU)
        x = self._gpu_layer_norm(x, self.emb_ln_gamma, self.emb_ln_beta)
        
        # 3. Transformer layers
        for i, layer in enumerate(self.layers):
            # Pre-LN + Attention with fused RoPE (GPU)
            ln_x = self._gpu_layer_norm(x, layer['attn_ln_gamma'], layer['attn_ln_beta'])
            attn_out = self._gpu_attention_rope(ln_x, attention_mask, layer, seq_len)
            x = x + attn_out
            
            # Pre-LN + FFN (GPU)
            ln_x = self._gpu_layer_norm(x, layer['ffn_ln_gamma'], layer['ffn_ln_beta'])
            ffn_out = self._gpu_ffn(ln_x, layer)
            x = x + ffn_out
        
        # 4. Pooling (GPU)
        pooled = self._gpu_pool(x, attention_mask)
        
        # 5. Output projection
        if self.output_proj_weight is not None:
            pooled = self._gpu_linear(pooled, self.output_proj_weight, self.output_proj_bias)
        
        # 6. L2 normalize (GPU)
        if cfg.normalize:
            pooled = self._gpu_normalize(pooled)
        
        return pooled.astype(np.float32)
    
    def _create_buffer(self, data: np.ndarray, name: str):
        """Create GPU buffer from numpy array"""
        data = np.ascontiguousarray(data.astype(np.float32))
        buf, mem = self.gpu.core._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        self.gpu.core._upload_buffer(buf, mem, data.flatten())
        return buf, mem, data.nbytes
    
    def _free_buffer(self, buf, mem):
        """Free GPU buffer"""
        vkDestroyBuffer(self.gpu.core.device, buf, None)
        vkFreeMemory(self.gpu.core.device, mem, None)
    
    def _gpu_layer_norm(
        self,
        x: np.ndarray,
        gamma: np.ndarray,
        beta: np.ndarray,
        eps: float = 1e-6
    ) -> np.ndarray:
        """GPU-accelerated layer normalization"""
        original_shape = x.shape
        
        if len(x.shape) == 3:
            batch_size, seq_len, features = x.shape
            x_flat = x.reshape(-1, features)
        else:
            batch_size = x.shape[0]
            seq_len = 1
            features = x.shape[-1]
            x_flat = x
        
        batch_seq = batch_size * seq_len
        
        # Create buffers
        buf_x, mem_x, size_x = self._create_buffer(x_flat.flatten(), 'ln_x')
        buf_out, mem_out = self.gpu.core._create_buffer(size_x, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_gamma, mem_gamma, _ = self._create_buffer(gamma, 'ln_gamma')
        buf_beta, mem_beta, _ = self._create_buffer(beta, 'ln_beta')
        buf_mean, mem_mean = self.gpu.core._create_buffer(batch_seq * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_var, mem_var = self.gpu.core._create_buffer(batch_seq * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Get pipeline
        pipeline, layout, _ = self.gpu.pipelines.get_or_create_pipeline(
            'fnn-layernorm', 6, push_constant_size=20
        )
        
        desc_set = self.gpu.pipelines.get_cached_descriptor_set(
            'fnn-layernorm',
            [
                (buf_x, size_x),
                (buf_out, size_x),
                (buf_gamma, gamma.nbytes),
                (buf_beta, beta.nbytes),
                (buf_mean, batch_seq * 4),
                (buf_var, batch_seq * 4),
            ]
        )
        
        # Three passes
        for pass_type in [0, 1, 2]:
            workgroups = (batch_seq + 255) // 256 if pass_type < 2 else (batch_seq * features + 255) // 256
            push = struct.pack('IIIfI', batch_size, seq_len, features, eps, pass_type)
            self.gpu.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push)
        
        # Download result
        output = self.gpu.core._download_buffer(mem_out, size_x, dtype=np.float32)
        output = output[:batch_seq * features].reshape(original_shape)
        
        # Cleanup
        for buf, mem in [(buf_x, mem_x), (buf_out, mem_out), (buf_gamma, mem_gamma),
                         (buf_beta, mem_beta), (buf_mean, mem_mean), (buf_var, mem_var)]:
            self._free_buffer(buf, mem)
        
        return output
    
    def _gpu_attention_rope(
        self,
        x: np.ndarray,
        attention_mask: np.ndarray,
        layer: Dict,
        seq_len: int
    ) -> np.ndarray:
        """
        GPU-accelerated attention with fused RoPE using flash-attention2-rope shader.
        """
        cfg = self.config
        batch_size = x.shape[0]
        
        # Project Q, K, V on CPU (could be GPU too)
        x_flat = x.reshape(-1, cfg.hidden_dim)
        q = (x_flat @ layer['q_weight'].T + layer['q_bias']).reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)
        k = (x_flat @ layer['k_weight'].T + layer['k_bias']).reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)
        v = (x_flat @ layer['v_weight'].T + layer['v_bias']).reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)
        
        # Prepare buffers
        buf_q, mem_q, size_q = self._create_buffer(q.flatten(), 'attn_q')
        buf_k, mem_k, size_k = self._create_buffer(k.flatten(), 'attn_k')
        buf_v, mem_v, size_v = self._create_buffer(v.flatten(), 'attn_v')
        buf_mask, mem_mask, size_mask = self._create_buffer(attention_mask.flatten(), 'attn_mask')
        
        output_size = batch_size * seq_len * cfg.num_heads * cfg.head_dim * 4
        buf_out, mem_out = self.gpu.core._create_buffer(output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Running max/sum buffers
        running_size = batch_size * seq_len * cfg.num_heads * 4
        buf_max, mem_max = self.gpu.core._create_buffer(running_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_sum, mem_sum = self.gpu.core._create_buffer(running_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_accum, mem_accum = self.gpu.core._create_buffer(output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Get pipeline
        pipeline, layout, _ = self.gpu.pipelines.get_or_create_pipeline(
            'flash-attention2-rope', 8, push_constant_size=60
        )
        
        desc_set = self.gpu.pipelines.get_cached_descriptor_set(
            'flash-attention2-rope',
            [
                (buf_q, size_q),
                (buf_k, size_k),
                (buf_v, size_v),
                (buf_mask, size_mask),
                (buf_out, output_size),
                (buf_max, running_size),
                (buf_sum, running_size),
                (buf_accum, output_size),
            ]
        )
        
        scale = 1.0 / np.sqrt(cfg.head_dim)
        tile_size_q = 64
        tile_size_k = 64
        
        # Pass 0: Initialize
        push = struct.pack('IIIIfIIIIIIffI',
            batch_size, seq_len, cfg.num_heads, cfg.head_dim,
            scale, tile_size_q, tile_size_k, 0,  # pass_type=0
            1,  # has_mask
            0, 0,  # tile indices
            cfg.rope_base, cfg.rope_scaling, 1  # use_rope=1
        )
        workgroups = (batch_size * seq_len * cfg.num_heads + 255) // 256
        self.gpu.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push)
        
        # Pass 1: Process tiles
        num_q_tiles = (seq_len + tile_size_q - 1) // tile_size_q
        num_k_tiles = (seq_len + tile_size_k - 1) // tile_size_k
        
        for q_tile in range(num_q_tiles):
            for k_tile in range(num_k_tiles):
                push = struct.pack('IIIIfIIIIIIffI',
                    batch_size, seq_len, cfg.num_heads, cfg.head_dim,
                    scale, tile_size_q, tile_size_k, 1,  # pass_type=1
                    1, q_tile, k_tile,
                    cfg.rope_base, cfg.rope_scaling, 1
                )
                wg_x = (tile_size_k + 15) // 16
                wg_y = (batch_size * cfg.num_heads * tile_size_q + 15) // 16
                self.gpu.core._dispatch_compute(pipeline, layout, desc_set, wg_x, push, wg_y)
        
        # Pass 2: Finalize
        push = struct.pack('IIIIfIIIIIIffI',
            batch_size, seq_len, cfg.num_heads, cfg.head_dim,
            scale, tile_size_q, tile_size_k, 2,  # pass_type=2
            1, 0, 0,
            cfg.rope_base, cfg.rope_scaling, 1
        )
        wg_x = (cfg.head_dim + 15) // 16
        wg_y = (batch_size * seq_len * cfg.num_heads + 15) // 16
        self.gpu.core._dispatch_compute(pipeline, layout, desc_set, wg_x, push, wg_y)
        
        # Download result
        attn_out = self.gpu.core._download_buffer(mem_out, output_size, dtype=np.float32)
        attn_out = attn_out[:batch_size * seq_len * cfg.num_heads * cfg.head_dim]
        attn_out = attn_out.reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)
        attn_out = attn_out.reshape(batch_size, seq_len, cfg.hidden_dim)
        
        # Output projection (CPU)
        output = attn_out.reshape(-1, cfg.hidden_dim) @ layer['out_weight'].T + layer['out_bias']
        output = output.reshape(batch_size, seq_len, cfg.hidden_dim)
        
        # Cleanup
        for buf, mem in [(buf_q, mem_q), (buf_k, mem_k), (buf_v, mem_v), (buf_mask, mem_mask),
                         (buf_out, mem_out), (buf_max, mem_max), (buf_sum, mem_sum), (buf_accum, mem_accum)]:
            self._free_buffer(buf, mem)
        
        return output

    
    def _gpu_ffn(self, x: np.ndarray, layer: Dict) -> np.ndarray:
        """GPU-accelerated FFN"""
        cfg = self.config
        batch_size, seq_len, _ = x.shape
        batch_seq = batch_size * seq_len
        
        x_flat = x.reshape(-1, cfg.hidden_dim).astype(np.float32)
        
        # Create buffers
        buf_x, mem_x, size_x = self._create_buffer(x_flat.flatten(), 'ffn_x')
        buf_w1, mem_w1, _ = self._create_buffer(layer['ffn_up_weight'].flatten(), 'ffn_w1')
        buf_b1, mem_b1, _ = self._create_buffer(layer['ffn_up_bias'], 'ffn_b1')
        buf_w2, mem_w2, _ = self._create_buffer(layer['ffn_down_weight'].flatten(), 'ffn_w2')
        buf_b2, mem_b2, _ = self._create_buffer(layer['ffn_down_bias'], 'ffn_b2')
        
        inter_size = batch_seq * cfg.intermediate_dim * 4
        out_size = batch_seq * cfg.hidden_dim * 4
        buf_inter, mem_inter = self.gpu.core._create_buffer(inter_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.gpu.core._create_buffer(out_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        pipeline, layout, _ = self.gpu.pipelines.get_or_create_pipeline(
            'embedding-ffn', 7, push_constant_size=20
        )
        
        desc_set = self.gpu.pipelines.get_cached_descriptor_set(
            'embedding-ffn',
            [
                (buf_x, size_x),
                (buf_w1, layer['ffn_up_weight'].nbytes),
                (buf_b1, layer['ffn_up_bias'].nbytes),
                (buf_w2, layer['ffn_down_weight'].nbytes),
                (buf_b2, layer['ffn_down_bias'].nbytes),
                (buf_inter, inter_size),
                (buf_out, out_size),
            ]
        )
        
        # Pass 0: Up projection + GELU
        wg_x = (cfg.intermediate_dim + 15) // 16
        wg_y = (batch_seq + 15) // 16
        push = struct.pack('IIIII', batch_seq, cfg.hidden_dim, cfg.intermediate_dim, 0, 0)  # GELU
        self.gpu.core._dispatch_compute(pipeline, layout, desc_set, wg_x, push, wg_y)
        
        # Pass 1: Down projection
        wg_x = (cfg.hidden_dim + 15) // 16
        push = struct.pack('IIIII', batch_seq, cfg.hidden_dim, cfg.intermediate_dim, 0, 1)
        self.gpu.core._dispatch_compute(pipeline, layout, desc_set, wg_x, push, wg_y)
        
        # Download
        output = self.gpu.core._download_buffer(mem_out, out_size, dtype=np.float32)
        output = output[:batch_seq * cfg.hidden_dim].reshape(batch_size, seq_len, cfg.hidden_dim)
        
        # Cleanup
        for buf, mem in [(buf_x, mem_x), (buf_w1, mem_w1), (buf_b1, mem_b1),
                         (buf_w2, mem_w2), (buf_b2, mem_b2), (buf_inter, mem_inter), (buf_out, mem_out)]:
            self._free_buffer(buf, mem)
        
        return output
    
    def _gpu_pool(self, x: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """GPU-accelerated pooling"""
        cfg = self.config
        batch_size, seq_len, hidden_dim = x.shape
        
        buf_x, mem_x, size_x = self._create_buffer(x.flatten(), 'pool_x')
        buf_mask, mem_mask, size_mask = self._create_buffer(attention_mask.flatten(), 'pool_mask')
        
        out_size = batch_size * hidden_dim * 4
        buf_out, mem_out = self.gpu.core._create_buffer(out_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        pipeline, layout, _ = self.gpu.pipelines.get_or_create_pipeline(
            'embedding-pool', 3, push_constant_size=16
        )
        
        pool_type = {"mean": 0, "cls": 1, "max": 2}.get(cfg.pooling, 0)
        
        desc_set = self.gpu.pipelines.get_cached_descriptor_set(
            'embedding-pool',
            [
                (buf_x, size_x),
                (buf_mask, size_mask),
                (buf_out, out_size),
            ]
        )
        
        push = struct.pack('IIII', batch_size, seq_len, hidden_dim, pool_type)
        workgroups = (batch_size * hidden_dim + 255) // 256
        self.gpu.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push)
        
        output = self.gpu.core._download_buffer(mem_out, out_size, dtype=np.float32)
        output = output[:batch_size * hidden_dim].reshape(batch_size, hidden_dim)
        
        for buf, mem in [(buf_x, mem_x), (buf_mask, mem_mask), (buf_out, mem_out)]:
            self._free_buffer(buf, mem)
        
        return output
    
    def _gpu_linear(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """GPU-accelerated linear projection"""
        batch_size = x.shape[0]
        in_dim = weight.shape[1]
        out_dim = weight.shape[0]
        
        buf_x, mem_x, size_x = self._create_buffer(x.flatten(), 'lin_x')
        buf_w, mem_w, _ = self._create_buffer(weight.flatten(), 'lin_w')
        buf_b, mem_b, _ = self._create_buffer(bias, 'lin_b')
        
        out_size = batch_size * out_dim * 4
        buf_out, mem_out = self.gpu.core._create_buffer(out_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        pipeline, layout, _ = self.gpu.pipelines.get_or_create_pipeline(
            'fnn-linear', 4, push_constant_size=16
        )
        
        desc_set = self.gpu.pipelines.get_cached_descriptor_set(
            'fnn-linear',
            [
                (buf_x, size_x),
                (buf_w, weight.nbytes),
                (buf_b, bias.nbytes),
                (buf_out, out_size),
            ]
        )
        
        push = struct.pack('IIII', batch_size, in_dim, out_dim, 1)  # has_bias=1
        wg_x = (out_dim + 15) // 16
        wg_y = (batch_size + 15) // 16
        self.gpu.core._dispatch_compute(pipeline, layout, desc_set, wg_x, push, wg_y)
        
        output = self.gpu.core._download_buffer(mem_out, out_size, dtype=np.float32)
        output = output[:batch_size * out_dim].reshape(batch_size, out_dim)
        
        for buf, mem in [(buf_x, mem_x), (buf_w, mem_w), (buf_b, mem_b), (buf_out, mem_out)]:
            self._free_buffer(buf, mem)
        
        return output
    
    def _gpu_normalize(self, x: np.ndarray) -> np.ndarray:
        """GPU-accelerated L2 normalization"""
        batch_size, hidden_dim = x.shape
        
        buf_x, mem_x, size_x = self._create_buffer(x.flatten(), 'norm_x')
        buf_out, mem_out = self.gpu.core._create_buffer(size_x, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_norms, mem_norms = self.gpu.core._create_buffer(batch_size * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        pipeline, layout, _ = self.gpu.pipelines.get_or_create_pipeline(
            'embedding-normalize', 3, push_constant_size=16
        )
        
        desc_set = self.gpu.pipelines.get_cached_descriptor_set(
            'embedding-normalize',
            [
                (buf_x, size_x),
                (buf_out, size_x),
                (buf_norms, batch_size * 4),
            ]
        )
        
        # Pass 0: Compute norms
        push = struct.pack('IIIf', batch_size, hidden_dim, 0, 1e-12)
        self.gpu.core._dispatch_compute(pipeline, layout, desc_set, batch_size, push)
        
        # Pass 1: Normalize
        push = struct.pack('IIIf', batch_size, hidden_dim, 1, 1e-12)
        workgroups = (batch_size * hidden_dim + 255) // 256
        self.gpu.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push)
        
        output = self.gpu.core._download_buffer(mem_out, size_x, dtype=np.float32)
        output = output[:batch_size * hidden_dim].reshape(batch_size, hidden_dim)
        
        for buf, mem in [(buf_x, mem_x), (buf_out, mem_out), (buf_norms, mem_norms)]:
            self._free_buffer(buf, mem)
        
        return output



# =============================================================================
# Benchmark and Testing
# =============================================================================

def benchmark_rope_transformer(
    config_name: str = "grillcheese-small",
    num_iterations: int = 50,
    warmup: int = 5,
    batch_size: int = 1,
    seq_len: int = 128
) -> Dict[str, Any]:
    """
    Benchmark the Vulkan Embedding Transformer with RoPE.
    
    Args:
        config_name: Model configuration preset
        num_iterations: Number of benchmark iterations
        warmup: Warmup iterations
        batch_size: Batch size for benchmark
        seq_len: Sequence length for benchmark
        
    Returns:
        Dict with benchmark results
    """
    print(f"\n{'='*60}")
    print(f"Vulkan Embedding Transformer Benchmark (RoPE)")
    print(f"{'='*60}")
    
    config = VulkanEmbeddingConfig(preset=config_name)
    print(f"\nConfiguration: {config_name}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  RoPE base: {config.rope_base}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # Initialize model
    print("\nInitializing model...")
    model = VulkanEmbeddingTransformerRoPE(config=config)
    
    # Generate test data
    np.random.seed(42)
    input_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=np.int32)
    attention_mask = np.ones((batch_size, seq_len), dtype=np.float32)
    
    results = {}
    
    # Benchmark CPU forward
    print("\n--- CPU Forward Pass ---")
    
    # Warmup
    for _ in range(warmup):
        _ = model.forward(input_ids, attention_mask)
    
    # Benchmark
    cpu_times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = model.forward(input_ids, attention_mask)
        cpu_times.append((time.perf_counter() - start) * 1000)
    
    results['cpu_mean_ms'] = np.mean(cpu_times)
    results['cpu_std_ms'] = np.std(cpu_times)
    results['cpu_min_ms'] = np.min(cpu_times)
    print(f"  Mean: {results['cpu_mean_ms']:.2f} ms")
    print(f"  Std:  {results['cpu_std_ms']:.2f} ms")
    print(f"  Min:  {results['cpu_min_ms']:.2f} ms")
    
    # Benchmark GPU forward (if available)
    if model.use_gpu:
        print("\n--- GPU Forward Pass (Vulkan) ---")
        
        # Warmup
        for _ in range(warmup):
            _ = model.forward_gpu(input_ids, attention_mask)
        
        # Benchmark
        gpu_times = []
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model.forward_gpu(input_ids, attention_mask)
            gpu_times.append((time.perf_counter() - start) * 1000)
        
        results['gpu_mean_ms'] = np.mean(gpu_times)
        results['gpu_std_ms'] = np.std(gpu_times)
        results['gpu_min_ms'] = np.min(gpu_times)
        results['speedup'] = results['cpu_mean_ms'] / results['gpu_mean_ms']
        
        print(f"  Mean: {results['gpu_mean_ms']:.2f} ms")
        print(f"  Std:  {results['gpu_std_ms']:.2f} ms")
        print(f"  Min:  {results['gpu_min_ms']:.2f} ms")
        print(f"  Speedup: {results['speedup']:.2f}x")
    else:
        print("\n[GPU not available]")
    
    # Test encode function
    print("\n--- Encode Test ---")
    test_text = "Hello, this is a test sentence for the embedding model."
    
    start = time.perf_counter()
    embedding = model.encode(test_text)
    encode_time = (time.perf_counter() - start) * 1000
    
    print(f"  Text: '{test_text}'")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
    print(f"  Time: {encode_time:.2f} ms")
    
    results['embedding_dim'] = embedding.shape[0]
    results['encode_time_ms'] = encode_time
    
    # Memory usage
    param_count = model._param_count
    param_mb = param_count * 4 / (1024 * 1024)
    results['param_count'] = param_count
    results['param_mb'] = param_mb
    
    print(f"\n--- Model Stats ---")
    print(f"  Parameters: {param_count:,} ({param_mb:.1f} MB)")
    print(f"  Embedding dim: {config.output_dim}")
    
    print(f"\n{'='*60}\n")
    
    return results


def test_rope_correctness():
    """Test that RoPE is being applied correctly"""
    print("\n--- RoPE Correctness Test ---")
    
    config = VulkanEmbeddingConfig(
        preset="grillcheese-tiny",
        num_layers=1,
        hidden_dim=64,
        num_heads=4
    )
    model = VulkanEmbeddingTransformerRoPE(config=config)
    
    # Test 1: Same input at different positions should have different embeddings
    x = np.random.randn(1, 4, 4, 16).astype(np.float32)  # (batch, seq, heads, head_dim)
    
    # Apply RoPE
    x_rotated = model._apply_rope(x, seq_len=4)
    
    # Check that different positions get different rotations
    pos0 = x_rotated[0, 0, 0, :]
    pos1 = x_rotated[0, 1, 0, :]
    pos2 = x_rotated[0, 2, 0, :]
    
    # All positions should be different
    assert not np.allclose(pos0, pos1), "Position 0 and 1 should differ"
    assert not np.allclose(pos1, pos2), "Position 1 and 2 should differ"
    assert not np.allclose(pos0, pos2), "Position 0 and 2 should differ"
    
    print("  [OK] Different positions produce different rotations")
    
    # Test 2: RoPE should preserve vector magnitude approximately
    original_norms = np.linalg.norm(x, axis=-1)
    rotated_norms = np.linalg.norm(x_rotated, axis=-1)
    
    assert np.allclose(original_norms, rotated_norms, rtol=1e-5), "RoPE should preserve magnitude"
    print("  [OK] RoPE preserves vector magnitude")
    
    # Test 3: Full forward pass should work
    input_ids = np.array([[1, 2, 3, 4]], dtype=np.int32)
    mask = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    
    embedding = model.forward(input_ids, mask)
    
    assert embedding.shape == (1, 64), f"Expected (1, 64), got {embedding.shape}"
    assert np.isfinite(embedding).all(), "Embedding should be finite"
    
    # Should be normalized
    norm = np.linalg.norm(embedding)
    assert np.isclose(norm, 1.0, rtol=1e-3), f"Embedding should be normalized, got norm={norm}"
    
    print("  [OK] Forward pass produces valid normalized embeddings")
    print("  [OK] All RoPE tests passed!\n")


if __name__ == "__main__":
    import sys
    
    # Run tests
    test_rope_correctness()
    
    # Run benchmark
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    else:
        config_name = "grillcheese-small"
    
    benchmark_rope_transformer(config_name=config_name)
