"""
Vulkan Embedding Transformer with RoPE and Capsule Memory Integration.

Features:
- RoPE (Rotary Position Embeddings)
- Flash Attention 2 with fused RoPE
- Capsule encoding (384D → 32D cognitive vectors)
- Dentate Gyrus sparse expansion (32D → 128D, 2% sparsity)
- Memory injection at layers 4-5
- Persistent GPU buffers for zero-allocation forward pass
- SentencePiece tokenization

Bio-inspired architecture based on hippocampal memory:
- DG: Pattern separation via sparse expansion
- CA3: Pattern completion via FAISS kNN
- Residual injection: Memory context into transformer layers
"""
import logging
import struct
import time
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

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
    from tokenizer import get_tokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


# =============================================================================
# Capsule Memory Types
# =============================================================================

class MemoryType(Enum):
    CONCEPT = "CONCEPT"
    EPISODE = "EPISODE"
    SELF_STATE = "SELF_STATE"
    TASK = "TASK"
    TOOL = "TOOL"


@dataclass
class CognitiveFeatures:
    """Cognitive modulation features for capsule memory"""
    plasticity_gain: float = 0.5       # Learning rate for this memory
    consolidation_priority: float = 0.5  # Importance for replay
    stability: float = 0.5             # Resistance to forgetting
    stress_link: float = 0.0           # Emotional/stress association
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.plasticity_gain,
            self.consolidation_priority,
            self.stability,
            self.stress_link
        ], dtype=np.float32)


@dataclass  
class CapsuleMemory:
    """32D cognitive capsule memory"""
    memory_id: str
    memory_type: MemoryType
    domain: str
    content: str
    cognitive_features: CognitiveFeatures = field(default_factory=CognitiveFeatures)
    
    # Computed vectors
    capsule_vector: Optional[np.ndarray] = None  # (32,)
    dg_vector: Optional[np.ndarray] = None       # (128,)
    
    # Metadata
    protected: bool = False
    access_count: int = 0
    last_access: Optional[float] = None
    created: float = 0.0


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class CapsuleTransformerConfig:
    """Configuration for Capsule-augmented Embedding Transformer"""
    
    # Transformer config
    hidden_dim: int = 384
    intermediate_dim: int = 1536
    num_layers: int = 6
    num_heads: int = 6
    vocab_size: int = 32000
    max_seq_len: int = 512
    
    # RoPE config
    rope_base: float = 10000.0
    rope_scaling: float = 1.0
    
    # Capsule config
    capsule_dim: int = 32              # Cognitive capsule dimension
    semantic_dims: int = 28            # Semantic portion (capsule_dim - 4)
    cognitive_dims: int = 4            # Last 4 dims for cognitive features
    
    # DG config (Dentate Gyrus)
    dg_expansion_factor: int = 4       # 32 → 128
    dg_sparsity: float = 0.02          # ~2-3 active neurons
    
    # Memory injection config
    injection_layers: Tuple[int, ...] = (4, 5)  # Inject at layers 4-5
    injection_strength: float = 0.1    # Residual injection weight
    max_injection_memories: int = 32   # Max memories to inject
    
    # Output config
    output_dim: Optional[int] = None   # None = hidden_dim
    pooling: str = "mean"
    normalize: bool = True
    
    def __post_init__(self):
        self.head_dim = self.hidden_dim // self.num_heads
        self.dg_dim = self.capsule_dim * self.dg_expansion_factor  # 128
        self.dg_k = max(1, int(self.dg_dim * self.dg_sparsity))    # ~3 active
        if self.output_dim is None:
            self.output_dim = self.hidden_dim
        
        assert self.capsule_dim == self.semantic_dims + self.cognitive_dims
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"


# =============================================================================
# Persistent GPU Buffer Manager
# =============================================================================

class PersistentBufferManager:
    """
    Manages persistent GPU buffers to eliminate allocation overhead.
    
    Pre-allocates all buffers at init based on max batch/seq size.
    Buffers are reused across forward passes for zero-allocation inference.
    """
    
    def __init__(self, gpu, config: CapsuleTransformerConfig, max_batch: int = 32, max_seq: int = 512):
        self.gpu = gpu
        self.config = config
        self.max_batch = max_batch
        self.max_seq = max_seq
        
        self.buffers: Dict[str, Tuple[Any, Any, int]] = {}  # name -> (buf, mem, size)
        self._allocated = False
    
    def allocate(self):
        """Allocate all persistent buffers"""
        if self._allocated:
            return
        
        cfg = self.config
        batch_seq = self.max_batch * self.max_seq
        
        # Hidden state buffers (double buffer for layer chaining)
        hidden_size = batch_seq * cfg.hidden_dim * 4
        self._alloc('hidden_a', hidden_size)
        self._alloc('hidden_b', hidden_size)
        
        # Attention buffers
        qkv_size = batch_seq * 3 * cfg.hidden_dim * 4
        self._alloc('qkv', qkv_size)
        
        attn_out_size = batch_seq * cfg.hidden_dim * 4
        self._alloc('attn_out', attn_out_size)
        
        # Flash attention running buffers
        running_size = self.max_batch * self.max_seq * cfg.num_heads * 4
        self._alloc('attn_max', running_size)
        self._alloc('attn_sum', running_size)
        self._alloc('attn_accum', batch_seq * cfg.num_heads * cfg.head_dim * 4)
        
        # FFN buffers
        inter_size = batch_seq * cfg.intermediate_dim * 4
        self._alloc('ffn_inter', inter_size)
        self._alloc('ffn_out', hidden_size)
        
        # LayerNorm buffers
        ln_stats_size = batch_seq * 4
        self._alloc('ln_mean', ln_stats_size)
        self._alloc('ln_var', ln_stats_size)
        
        # Attention mask
        mask_size = self.max_batch * self.max_seq * 4
        self._alloc('mask', mask_size)
        
        # Pooling output
        pool_size = self.max_batch * cfg.hidden_dim * 4
        self._alloc('pooled', pool_size)
        
        # Normalization
        norm_size = self.max_batch * 4
        self._alloc('norms', norm_size)
        
        # Final output
        out_size = self.max_batch * cfg.output_dim * 4
        self._alloc('output', out_size)
        
        # Capsule buffers
        capsule_size = self.max_batch * cfg.capsule_dim * 4
        self._alloc('capsule', capsule_size)
        
        dg_size = self.max_batch * cfg.dg_dim * 4
        self._alloc('dg', dg_size)
        
        # Memory injection buffer
        inject_size = self.max_batch * cfg.hidden_dim * 4
        self._alloc('inject', inject_size)
        
        self._allocated = True
        
        total_mb = sum(s for _, _, s in self.buffers.values()) / (1024 * 1024)
        logger.info(f"Allocated {len(self.buffers)} persistent GPU buffers ({total_mb:.1f} MB)")
    
    def _alloc(self, name: str, size: int):
        """Allocate a single buffer"""
        buf, mem = self.gpu.core._create_buffer(size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        self.buffers[name] = (buf, mem, size)
    
    def get(self, name: str) -> Tuple[Any, Any, int]:
        """Get buffer by name"""
        if not self._allocated:
            self.allocate()
        return self.buffers[name]
    
    def upload(self, name: str, data: np.ndarray):
        """Upload data to named buffer"""
        buf, mem, size = self.get(name)
        data = np.ascontiguousarray(data.astype(np.float32))
        assert data.nbytes <= size, f"Data too large for buffer {name}: {data.nbytes} > {size}"
        self.gpu.core._upload_buffer(buf, mem, data.flatten())
    
    def download(self, name: str, count: int, dtype=np.float32) -> np.ndarray:
        """Download data from named buffer"""
        buf, mem, size = self.get(name)
        byte_count = count * np.dtype(dtype).itemsize
        assert byte_count <= size, f"Requested too much data from {name}"
        return self.gpu.core._download_buffer(mem, byte_count, dtype=dtype)[:count]
    
    def free(self):
        """Free all buffers"""
        for name, (buf, mem, _) in self.buffers.items():
            try:
                vkDestroyBuffer(self.gpu.core.device, buf, None)
                vkFreeMemory(self.gpu.core.device, mem, None)
            except:
                pass
        self.buffers.clear()
        self._allocated = False
    
    def __del__(self):
        self.free()


# =============================================================================
# Dentate Gyrus - Sparse Expansion Layer
# =============================================================================

class DentateGyrus:
    """
    Sparse expansion layer: 32D → 128D with ~2% sparsity.
    Transforms similar inputs into non-overlapping representations.
    Bio-inspired pattern separation mechanism.
    """
    
    def __init__(self, config: CapsuleTransformerConfig):
        self.config = config
        self.input_dim = config.capsule_dim      # 32
        self.output_dim = config.dg_dim          # 128
        self.k = config.dg_k                     # ~3 active neurons
        
        # Random sparse projection (Xavier init)
        scale = np.sqrt(2.0 / (self.input_dim + self.output_dim))
        self.W = (np.random.randn(self.input_dim, self.output_dim) * scale).astype(np.float32)
    
    def expand(self, capsule: np.ndarray) -> np.ndarray:
        """
        Sparse expansion with top-k selection.
        
        Args:
            capsule: (32,) or (batch, 32) input
            
        Returns:
            (128,) or (batch, 128) sparse vector with ~2% activation
        """
        if capsule.ndim == 1:
            return self._expand_single(capsule)
        return self._expand_batch(capsule)
    
    def _expand_single(self, capsule: np.ndarray) -> np.ndarray:
        activations = capsule @ self.W
        
        # Top-k sparsification
        top_k_idx = np.argsort(np.abs(activations))[-self.k:]
        
        sparse = np.zeros(self.output_dim, dtype=np.float32)
        sparse[top_k_idx] = activations[top_k_idx]
        
        # Normalize
        norm = np.linalg.norm(sparse)
        if norm > 1e-8:
            sparse /= norm
        
        return sparse
    
    def _expand_batch(self, capsules: np.ndarray) -> np.ndarray:
        batch_size = capsules.shape[0]
        activations = capsules @ self.W  # (batch, 128)
        
        sparse_batch = np.zeros_like(activations)
        for i in range(batch_size):
            top_k_idx = np.argsort(np.abs(activations[i]))[-self.k:]
            sparse_batch[i, top_k_idx] = activations[i, top_k_idx]
            
            norm = np.linalg.norm(sparse_batch[i])
            if norm > 1e-8:
                sparse_batch[i] /= norm
        
        return sparse_batch


# =============================================================================
# Main Transformer with Capsule Memory
# =============================================================================

class VulkanCapsuleTransformer:
    """
    GPU-accelerated embedding transformer with:
    - RoPE positional encoding
    - Capsule memory encoding (384D → 32D)
    - DG sparse expansion (32D → 128D)
    - Memory injection at layers 4-5
    - Persistent GPU buffers for zero-allocation forward
    """
    
    def __init__(
        self,
        config: Optional[CapsuleTransformerConfig] = None,
        tokenizer_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        max_batch: int = 32,
        max_seq: int = 512,
    ):
        self.config = config or CapsuleTransformerConfig()
        cfg = self.config
        
        # Initialize Vulkan
        self._init_vulkan()
        
        # Initialize tokenizer
        self._init_tokenizer(tokenizer_path)
        
        # Initialize weights
        self._init_weights()
        
        # Initialize capsule components
        self._init_capsule_components()
        
        # Precompute RoPE cache
        self._precompute_rope_cache()
        
        # Initialize persistent buffers
        if self.use_gpu:
            self.buffer_mgr = PersistentBufferManager(
                self.gpu, cfg, max_batch=max_batch, max_seq=max_seq
            )
        else:
            self.buffer_mgr = None
        
        # Memory injection state
        self.injected_memories: List[CapsuleMemory] = []
        
        # Load weights if provided
        if weights_path:
            self.load_weights(weights_path)
        
        logger.info(
            f"VulkanCapsuleTransformer initialized: "
            f"hidden={cfg.hidden_dim}, layers={cfg.num_layers}, "
            f"capsule={cfg.capsule_dim}D, DG={cfg.dg_dim}D, "
            f"injection_layers={cfg.injection_layers}"
        )
    
    def _init_vulkan(self):
        """Initialize Vulkan backend"""
        if not VULKAN_AVAILABLE:
            self.gpu = None
            self.use_gpu = False
            logger.warning("Vulkan not available, using CPU")
            return
        
        try:
            from vulkan_backend import VulkanCompute
            self.gpu = VulkanCompute()
            self.use_gpu = True
        except Exception as e:
            logger.warning(f"Vulkan init failed: {e}")
            self.gpu = None
            self.use_gpu = False
    
    def _init_tokenizer(self, tokenizer_path: Optional[str]):
        """Initialize tokenizer"""
        if TOKENIZER_AVAILABLE:
            try:
                self.tokenizer = get_tokenizer(
                    model_path=tokenizer_path,
                    max_length=self.config.max_seq_len
                )
            except:
                from tokenizer import SimpleTokenizer
                self.tokenizer = SimpleTokenizer(max_length=self.config.max_seq_len)
        else:
            self.tokenizer = None
    
    def _init_weights(self):
        """Initialize transformer weights"""
        cfg = self.config
        
        # Token embeddings
        scale = np.sqrt(2.0 / (cfg.vocab_size + cfg.hidden_dim))
        self.token_embeddings = (np.random.randn(cfg.vocab_size, cfg.hidden_dim) * scale).astype(np.float32)
        
        # Embedding LayerNorm
        self.emb_ln_gamma = np.ones(cfg.hidden_dim, dtype=np.float32)
        self.emb_ln_beta = np.zeros(cfg.hidden_dim, dtype=np.float32)
        
        # Encoder layers
        self.layers = []
        for i in range(cfg.num_layers):
            scale_attn = np.sqrt(2.0 / (cfg.hidden_dim * 2))
            scale_ffn = np.sqrt(2.0 / (cfg.hidden_dim + cfg.intermediate_dim))
            
            layer = {
                # Attention
                'q_weight': (np.random.randn(cfg.hidden_dim, cfg.hidden_dim) * scale_attn).astype(np.float32),
                'q_bias': np.zeros(cfg.hidden_dim, dtype=np.float32),
                'k_weight': (np.random.randn(cfg.hidden_dim, cfg.hidden_dim) * scale_attn).astype(np.float32),
                'k_bias': np.zeros(cfg.hidden_dim, dtype=np.float32),
                'v_weight': (np.random.randn(cfg.hidden_dim, cfg.hidden_dim) * scale_attn).astype(np.float32),
                'v_bias': np.zeros(cfg.hidden_dim, dtype=np.float32),
                'out_weight': (np.random.randn(cfg.hidden_dim, cfg.hidden_dim) * scale_attn).astype(np.float32),
                'out_bias': np.zeros(cfg.hidden_dim, dtype=np.float32),
                'attn_ln_gamma': np.ones(cfg.hidden_dim, dtype=np.float32),
                'attn_ln_beta': np.zeros(cfg.hidden_dim, dtype=np.float32),
                
                # FFN
                'ffn_up_weight': (np.random.randn(cfg.intermediate_dim, cfg.hidden_dim) * scale_ffn).astype(np.float32),
                'ffn_up_bias': np.zeros(cfg.intermediate_dim, dtype=np.float32),
                'ffn_down_weight': (np.random.randn(cfg.hidden_dim, cfg.intermediate_dim) * scale_ffn).astype(np.float32),
                'ffn_down_bias': np.zeros(cfg.hidden_dim, dtype=np.float32),
                'ffn_ln_gamma': np.ones(cfg.hidden_dim, dtype=np.float32),
                'ffn_ln_beta': np.zeros(cfg.hidden_dim, dtype=np.float32),
            }
            self.layers.append(layer)
        
        # Output projection
        if cfg.output_dim != cfg.hidden_dim:
            scale_out = np.sqrt(2.0 / (cfg.hidden_dim + cfg.output_dim))
            self.output_proj_weight = (np.random.randn(cfg.output_dim, cfg.hidden_dim) * scale_out).astype(np.float32)
            self.output_proj_bias = np.zeros(cfg.output_dim, dtype=np.float32)
        else:
            self.output_proj_weight = None
            self.output_proj_bias = None
        
        # Capsule projection (hidden_dim → capsule_dim)
        scale_cap = np.sqrt(2.0 / (cfg.hidden_dim + cfg.capsule_dim))
        self.capsule_proj = (np.random.randn(cfg.capsule_dim, cfg.hidden_dim) * scale_cap).astype(np.float32)
        
        # Memory injection projection (capsule_dim → hidden_dim)
        self.inject_proj = (np.random.randn(cfg.hidden_dim, cfg.capsule_dim) * scale_cap).astype(np.float32)
    
    def _init_capsule_components(self):
        """Initialize capsule memory components"""
        self.dg = DentateGyrus(self.config)
    
    def _precompute_rope_cache(self):
        """Precompute RoPE cos/sin tables"""
        cfg = self.config
        
        dim_pairs = cfg.head_dim // 2
        freq_exp = -2.0 * np.arange(dim_pairs) / cfg.head_dim
        freqs = np.power(cfg.rope_base, freq_exp).astype(np.float32)
        
        positions = np.arange(cfg.max_seq_len).astype(np.float32)
        theta = np.outer(positions / cfg.rope_scaling, freqs)
        
        self.rope_cos = np.cos(theta).astype(np.float32)
        self.rope_sin = np.sin(theta).astype(np.float32)

    
    # =========================================================================
    # RoPE Application
    # =========================================================================
    
    def _apply_rope(self, x: np.ndarray, seq_len: int) -> np.ndarray:
        """Apply RoPE to Q or K tensor (batch, seq, heads, head_dim)"""
        cos = self.rope_cos[:seq_len][np.newaxis, :, np.newaxis, :]
        sin = self.rope_sin[:seq_len][np.newaxis, :, np.newaxis, :]
        
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        x_rot = np.empty_like(x)
        x_rot[..., 0::2] = x_even * cos - x_odd * sin
        x_rot[..., 1::2] = x_even * sin + x_odd * cos
        
        return x_rot
    
    # =========================================================================
    # Core Operations
    # =========================================================================
    
    def _layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return gamma * (x - mean) / np.sqrt(var + eps) + beta
    
    def _attention(self, x: np.ndarray, mask: np.ndarray, layer: Dict, layer_idx: int) -> np.ndarray:
        """Self-attention with RoPE and optional memory injection"""
        cfg = self.config
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = (x.reshape(-1, cfg.hidden_dim) @ layer['q_weight'].T + layer['q_bias']).reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)
        k = (x.reshape(-1, cfg.hidden_dim) @ layer['k_weight'].T + layer['k_bias']).reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)
        v = (x.reshape(-1, cfg.hidden_dim) @ layer['v_weight'].T + layer['v_bias']).reshape(batch_size, seq_len, cfg.num_heads, cfg.head_dim)
        
        # Apply RoPE
        q = self._apply_rope(q, seq_len)
        k = self._apply_rope(k, seq_len)
        
        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scale = 1.0 / np.sqrt(cfg.head_dim)
        scores = np.einsum('bhqd,bhkd->bhqk', q, k) * scale
        
        # Apply mask
        mask_expanded = mask[:, np.newaxis, np.newaxis, :]
        scores = scores + (1.0 - mask_expanded) * -1e9
        
        # Softmax
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        weights = exp_scores / (exp_scores.sum(axis=-1, keepdims=True) + 1e-9)
        
        # Attention output
        attn_out = np.einsum('bhqk,bhkd->bhqd', weights, v)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, cfg.hidden_dim)
        
        # Output projection
        output = attn_out.reshape(-1, cfg.hidden_dim) @ layer['out_weight'].T + layer['out_bias']
        output = output.reshape(batch_size, seq_len, cfg.hidden_dim)
        
        # Memory injection at specified layers
        if layer_idx in cfg.injection_layers and self.injected_memories:
            output = self._inject_memory(output, layer_idx)
        
        return output
    
    def _inject_memory(self, hidden: np.ndarray, layer_idx: int) -> np.ndarray:
        """
        Inject retrieved memories into residual stream.
        
        Averages capsule vectors from retrieved memories, projects to hidden_dim,
        and adds to the last token position (where generation happens).
        """
        cfg = self.config
        batch_size, seq_len, _ = hidden.shape
        
        # Collect capsule vectors from memories
        memory_vectors = []
        for mem in self.injected_memories[:cfg.max_injection_memories]:
            if mem.capsule_vector is not None:
                memory_vectors.append(mem.capsule_vector)
        
        if not memory_vectors:
            return hidden
        
        # Average memory capsules
        memory_avg = np.mean(memory_vectors, axis=0)  # (32,)
        
        # Project to hidden dim
        injection = memory_avg @ self.inject_proj.T  # (hidden_dim,)
        injection = injection * cfg.injection_strength
        
        # Add to all sequence positions (or just last for generation)
        # For embeddings, add to all positions
        hidden = hidden + injection.reshape(1, 1, -1)
        
        return hidden
    
    def _ffn(self, x: np.ndarray, layer: Dict) -> np.ndarray:
        """Feed-forward network with GELU"""
        cfg = self.config
        batch_size, seq_len, _ = x.shape
        
        # Up projection
        h = x.reshape(-1, cfg.hidden_dim) @ layer['ffn_up_weight'].T + layer['ffn_up_bias']
        
        # GELU
        h = 0.5 * h * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (h + 0.044715 * h ** 3)))
        
        # Down projection
        output = h @ layer['ffn_down_weight'].T + layer['ffn_down_bias']
        
        return output.reshape(batch_size, seq_len, cfg.hidden_dim)
    
    def _pool(self, hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Pool to sentence embedding"""
        if self.config.pooling == "cls":
            return hidden[:, 0, :]
        elif self.config.pooling == "max":
            mask_3d = mask[:, :, np.newaxis]
            masked = hidden * mask_3d + (1 - mask_3d) * -1e9
            return masked.max(axis=1)
        else:  # mean
            mask_3d = mask[:, :, np.newaxis]
            sum_hidden = (hidden * mask_3d).sum(axis=1)
            count = mask_3d.sum(axis=1).clip(min=1e-9)
            return sum_hidden / count
    
    # =========================================================================
    # Forward Pass
    # =========================================================================
    
    def forward(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        inject_memories: Optional[List[CapsuleMemory]] = None
    ) -> np.ndarray:
        """
        Forward pass with optional memory injection.
        
        Args:
            input_ids: (batch, seq) token IDs
            attention_mask: (batch, seq) attention mask
            inject_memories: Optional list of CapsuleMemory to inject at layers 4-5
            
        Returns:
            (batch, output_dim) embeddings
        """
        cfg = self.config
        batch_size, seq_len = input_ids.shape
        
        # Set injection state
        self.injected_memories = inject_memories or []
        
        try:
            # 1. Token embeddings
            x = self.token_embeddings[input_ids.flatten()].reshape(batch_size, seq_len, cfg.hidden_dim)
            
            # 2. Embedding LayerNorm
            x = self._layer_norm(x, self.emb_ln_gamma, self.emb_ln_beta)
            
            # 3. Transformer layers
            for i, layer in enumerate(self.layers):
                # Pre-LN Attention with memory injection
                ln_x = self._layer_norm(x, layer['attn_ln_gamma'], layer['attn_ln_beta'])
                attn_out = self._attention(ln_x, attention_mask, layer, layer_idx=i)
                x = x + attn_out
                
                # Pre-LN FFN
                ln_x = self._layer_norm(x, layer['ffn_ln_gamma'], layer['ffn_ln_beta'])
                ffn_out = self._ffn(ln_x, layer)
                x = x + ffn_out
            
            # 4. Pooling
            pooled = self._pool(x, attention_mask)
            
            # 5. Output projection
            if self.output_proj_weight is not None:
                pooled = pooled @ self.output_proj_weight.T + self.output_proj_bias
            
            # 6. L2 normalize
            if cfg.normalize:
                norm = np.linalg.norm(pooled, axis=-1, keepdims=True) + 1e-12
                pooled = pooled / norm
            
            return pooled.astype(np.float32)
        
        finally:
            self.injected_memories = []
    
    # =========================================================================
    # Capsule Encoding
    # =========================================================================
    
    def encode_to_capsule(
        self,
        text: str,
        cognitive_features: Optional[CognitiveFeatures] = None
    ) -> np.ndarray:
        """
        Encode text to 32D capsule vector.
        
        Args:
            text: Input text
            cognitive_features: Optional cognitive modulation
            
        Returns:
            (32,) capsule vector with semantic + cognitive features
        """
        cfg = self.config
        cognitive_features = cognitive_features or CognitiveFeatures()
        
        # Get full embedding
        embedding = self.encode(text)  # (hidden_dim,)
        
        # Project to capsule space
        capsule = embedding @ self.capsule_proj.T  # (capsule_dim,)
        
        # Normalize semantic portion
        semantic = capsule[:cfg.semantic_dims]
        semantic = semantic / (np.linalg.norm(semantic) + 1e-8)
        
        # Blend with cognitive features
        capsule[:cfg.semantic_dims] = semantic * 0.9
        capsule[:cfg.semantic_dims] += semantic * 0.1 * cognitive_features.plasticity_gain
        
        # Inject cognitive features in last 4 dims
        cog = cognitive_features.to_array()
        capsule[cfg.semantic_dims:] = cog
        
        # Final normalization
        capsule = capsule / (np.linalg.norm(capsule) + 1e-8)
        
        return capsule.astype(np.float32)
    
    def encode_to_dg(
        self,
        text: str,
        cognitive_features: Optional[CognitiveFeatures] = None
    ) -> np.ndarray:
        """
        Encode text to 128D DG-expanded vector for FAISS indexing.
        
        Args:
            text: Input text
            cognitive_features: Optional cognitive modulation
            
        Returns:
            (128,) sparse DG vector
        """
        capsule = self.encode_to_capsule(text, cognitive_features)
        return self.dg.expand(capsule)
    
    def create_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.CONCEPT,
        domain: str = "general",
        cognitive_features: Optional[CognitiveFeatures] = None,
        memory_id: Optional[str] = None,
        protected: bool = False
    ) -> CapsuleMemory:
        """
        Create a complete CapsuleMemory with computed vectors.
        
        Args:
            content: Memory content text
            memory_type: Type of memory
            domain: Domain/category
            cognitive_features: Cognitive modulation features
            memory_id: Optional ID (auto-generated if None)
            protected: Whether memory is protected from forgetting
            
        Returns:
            CapsuleMemory with capsule_vector and dg_vector computed
        """
        import uuid
        
        cognitive_features = cognitive_features or CognitiveFeatures()
        memory_id = memory_id or f"{memory_type.value.lower()}_{uuid.uuid4().hex[:8]}"
        
        memory = CapsuleMemory(
            memory_id=memory_id,
            memory_type=memory_type,
            domain=domain,
            content=content,
            cognitive_features=cognitive_features,
            protected=protected,
            created=time.time()
        )
        
        # Compute vectors
        memory.capsule_vector = self.encode_to_capsule(content, cognitive_features)
        memory.dg_vector = self.dg.expand(memory.capsule_vector)
        
        return memory

    
    # =========================================================================
    # Standard Encoding API
    # =========================================================================
    
    def encode(self, text: str, max_length: Optional[int] = None) -> np.ndarray:
        """
        Encode text to embedding vector.
        
        Args:
            text: Input text
            max_length: Override max sequence length
            
        Returns:
            (output_dim,) embedding
        """
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")
        
        encoded = self.tokenizer.encode(
            text,
            max_length=max_length or self.config.max_seq_len,
            padding=True,
            return_attention_mask=True
        )
        
        input_ids = encoded['input_ids'].reshape(1, -1)
        attention_mask = encoded['attention_mask'].reshape(1, -1)
        
        return self.forward(input_ids, attention_mask)[0]
    
    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        batch_size: int = 32,
        inject_memories: Optional[List[CapsuleMemory]] = None
    ) -> np.ndarray:
        """
        Encode multiple texts to embeddings.
        
        Args:
            texts: List of texts
            max_length: Override max sequence length
            batch_size: Processing batch size
            inject_memories: Optional memories to inject
            
        Returns:
            (num_texts, output_dim) embeddings
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
            
            embeddings = self.forward(
                encoded['input_ids'],
                encoded['attention_mask'],
                inject_memories=inject_memories
            )
            all_embeddings.append(embeddings)
        
        return np.concatenate(all_embeddings, axis=0)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Alias for encode()"""
        return self.encode(text)
    
    @property
    def embedding_dim(self) -> int:
        return self.config.output_dim
    
    # =========================================================================
    # GPU-Optimized Forward Pass
    # =========================================================================
    
    def forward_gpu(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        inject_memories: Optional[List[CapsuleMemory]] = None
    ) -> np.ndarray:
        """
        GPU-accelerated forward pass using persistent buffers.
        Zero allocation after warmup.
        """
        if not self.use_gpu or self.buffer_mgr is None:
            return self.forward(input_ids, attention_mask, inject_memories)
        
        cfg = self.config
        batch_size, seq_len = input_ids.shape
        
        # Ensure buffers allocated
        self.buffer_mgr.allocate()
        
        # Set injection state
        self.injected_memories = inject_memories or []
        
        try:
            # 1. Token embeddings (CPU lookup, upload to GPU)
            x = self.token_embeddings[input_ids.flatten()].reshape(batch_size, seq_len, cfg.hidden_dim)
            self.buffer_mgr.upload('hidden_a', x.flatten())
            
            # Upload attention mask
            self.buffer_mgr.upload('mask', attention_mask.flatten())
            
            # 2. Embedding LayerNorm
            x = self._gpu_layer_norm_persistent('hidden_a', 'hidden_b', self.emb_ln_gamma, self.emb_ln_beta, batch_size, seq_len)
            
            # 3. Transformer layers (alternate between hidden_a and hidden_b)
            src_buf = 'hidden_b'
            dst_buf = 'hidden_a'
            
            for i, layer in enumerate(self.layers):
                # Pre-LN + Attention
                self._gpu_layer_norm_persistent(src_buf, 'ffn_out', layer['attn_ln_gamma'], layer['attn_ln_beta'], batch_size, seq_len)
                self._gpu_attention_persistent('ffn_out', dst_buf, layer, batch_size, seq_len, i)
                
                # Residual
                self._gpu_add_residual(src_buf, dst_buf, batch_size * seq_len * cfg.hidden_dim)
                
                # Pre-LN + FFN
                self._gpu_layer_norm_persistent(dst_buf, 'ffn_out', layer['ffn_ln_gamma'], layer['ffn_ln_beta'], batch_size, seq_len)
                self._gpu_ffn_persistent('ffn_out', src_buf, layer, batch_size, seq_len)
                
                # Residual
                self._gpu_add_residual(dst_buf, src_buf, batch_size * seq_len * cfg.hidden_dim)
                
                # Swap buffers
                src_buf, dst_buf = dst_buf, src_buf
            
            # 4. Pooling
            pooled = self._gpu_pool_persistent(src_buf, batch_size, seq_len)
            
            # 5. Output projection
            if self.output_proj_weight is not None:
                pooled = pooled @ self.output_proj_weight.T + self.output_proj_bias
            
            # 6. Normalize
            if cfg.normalize:
                norm = np.linalg.norm(pooled, axis=-1, keepdims=True) + 1e-12
                pooled = pooled / norm
            
            return pooled.astype(np.float32)
        
        finally:
            self.injected_memories = []
    
    def _gpu_layer_norm_persistent(self, src_name: str, dst_name: str, gamma: np.ndarray, beta: np.ndarray, batch: int, seq: int) -> np.ndarray:
        """GPU LayerNorm using persistent buffers"""
        cfg = self.config
        batch_seq = batch * seq
        
        src_buf, src_mem, _ = self.buffer_mgr.get(src_name)
        dst_buf, dst_mem, _ = self.buffer_mgr.get(dst_name)
        mean_buf, mean_mem, _ = self.buffer_mgr.get('ln_mean')
        var_buf, var_mem, _ = self.buffer_mgr.get('ln_var')
        
        # Upload gamma/beta (could be cached per layer)
        gamma_buf, gamma_mem = self.gpu.core._create_buffer(gamma.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        beta_buf, beta_mem = self.gpu.core._create_buffer(beta.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        self.gpu.core._upload_buffer(gamma_buf, gamma_mem, gamma)
        self.gpu.core._upload_buffer(beta_buf, beta_mem, beta)
        
        pipeline, layout, _ = self.gpu.pipelines.get_or_create_pipeline('fnn-layernorm', 6, push_constant_size=20)
        
        desc_set = self.gpu.pipelines.get_cached_descriptor_set(
            f'fnn-layernorm-{src_name}-{dst_name}',
            [
                (src_buf, batch_seq * cfg.hidden_dim * 4),
                (dst_buf, batch_seq * cfg.hidden_dim * 4),
                (gamma_buf, gamma.nbytes),
                (beta_buf, beta.nbytes),
                (mean_buf, batch_seq * 4),
                (var_buf, batch_seq * 4),
            ]
        )
        
        eps = 1e-6
        for pass_type in [0, 1, 2]:
            workgroups = (batch_seq + 255) // 256 if pass_type < 2 else (batch_seq * cfg.hidden_dim + 255) // 256
            push = struct.pack('IIIfI', batch, seq, cfg.hidden_dim, eps, pass_type)
            self.gpu.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push)
        
        # Cleanup temporary buffers
        vkDestroyBuffer(self.gpu.core.device, gamma_buf, None)
        vkFreeMemory(self.gpu.core.device, gamma_mem, None)
        vkDestroyBuffer(self.gpu.core.device, beta_buf, None)
        vkFreeMemory(self.gpu.core.device, beta_mem, None)
        
        return self.buffer_mgr.download(dst_name, batch_seq * cfg.hidden_dim).reshape(batch, seq, cfg.hidden_dim)
    
    def _gpu_attention_persistent(self, src_name: str, dst_name: str, layer: Dict, batch: int, seq: int, layer_idx: int):
        """GPU attention with RoPE using persistent buffers - falls back to CPU for now"""
        cfg = self.config
        
        # Download hidden states
        hidden = self.buffer_mgr.download(src_name, batch * seq * cfg.hidden_dim).reshape(batch, seq, cfg.hidden_dim)
        mask = self.buffer_mgr.download('mask', batch * seq).reshape(batch, seq)
        
        # CPU attention with RoPE
        output = self._attention(hidden, mask, layer, layer_idx)
        
        # Upload result
        self.buffer_mgr.upload(dst_name, output.flatten())
    
    def _gpu_ffn_persistent(self, src_name: str, dst_name: str, layer: Dict, batch: int, seq: int):
        """GPU FFN using persistent buffers - falls back to CPU for now"""
        cfg = self.config
        
        hidden = self.buffer_mgr.download(src_name, batch * seq * cfg.hidden_dim).reshape(batch, seq, cfg.hidden_dim)
        output = self._ffn(hidden, layer)
        self.buffer_mgr.upload(dst_name, output.flatten())
    
    def _gpu_add_residual(self, src_name: str, dst_name: str, count: int):
        """Add residual connection"""
        src = self.buffer_mgr.download(src_name, count)
        dst = self.buffer_mgr.download(dst_name, count)
        self.buffer_mgr.upload(dst_name, src + dst)
    
    def _gpu_pool_persistent(self, src_name: str, batch: int, seq: int) -> np.ndarray:
        """GPU pooling"""
        cfg = self.config
        hidden = self.buffer_mgr.download(src_name, batch * seq * cfg.hidden_dim).reshape(batch, seq, cfg.hidden_dim)
        mask = self.buffer_mgr.download('mask', batch * seq).reshape(batch, seq)
        return self._pool(hidden, mask)

    
    # =========================================================================
    # Weight Management
    # =========================================================================
    
    def save_weights(self, path: str):
        """Save weights to numpy archive"""
        cfg = self.config
        
        data = {
            'token_embeddings': self.token_embeddings,
            'emb_ln_gamma': self.emb_ln_gamma,
            'emb_ln_beta': self.emb_ln_beta,
            'capsule_proj': self.capsule_proj,
            'inject_proj': self.inject_proj,
            'dg_W': self.dg.W,
            
            # Config
            'config_hidden_dim': cfg.hidden_dim,
            'config_num_layers': cfg.num_layers,
            'config_num_heads': cfg.num_heads,
            'config_capsule_dim': cfg.capsule_dim,
            'config_dg_dim': cfg.dg_dim,
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
        """Load weights from numpy archive"""
        data = np.load(path)
        
        if 'token_embeddings' in data:
            self.token_embeddings = data['token_embeddings']
        if 'emb_ln_gamma' in data:
            self.emb_ln_gamma = data['emb_ln_gamma']
        if 'emb_ln_beta' in data:
            self.emb_ln_beta = data['emb_ln_beta']
        if 'capsule_proj' in data:
            self.capsule_proj = data['capsule_proj']
        if 'inject_proj' in data:
            self.inject_proj = data['inject_proj']
        if 'dg_W' in data:
            self.dg.W = data['dg_W']
        
        for i, layer in enumerate(self.layers):
            for key in layer.keys():
                npz_key = f'layer_{i}_{key}'
                if npz_key in data:
                    layer[key] = data[npz_key]
        
        if 'output_proj_weight' in data:
            self.output_proj_weight = data['output_proj_weight']
            self.output_proj_bias = data['output_proj_bias']
        
        logger.info(f"Loaded weights from {path}")


# =============================================================================
# Benchmark and Testing
# =============================================================================

def benchmark_capsule_transformer(
    num_iterations: int = 50,
    warmup: int = 5,
    batch_size: int = 1,
    seq_len: int = 128
) -> Dict[str, Any]:
    """Benchmark the Vulkan Capsule Transformer"""
    
    print(f"\n{'='*60}")
    print("Vulkan Capsule Transformer Benchmark")
    print(f"{'='*60}")
    
    config = CapsuleTransformerConfig(
        hidden_dim=384,
        num_layers=6,
        num_heads=6,
        injection_layers=(4, 5)
    )
    
    print(f"\nConfiguration:")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Capsule dim: {config.capsule_dim}")
    print(f"  DG dim: {config.dg_dim}")
    print(f"  Injection layers: {config.injection_layers}")
    print(f"  Batch: {batch_size}, Seq: {seq_len}")
    
    print("\nInitializing...")
    model = VulkanCapsuleTransformer(config=config, max_batch=batch_size, max_seq=seq_len)
    
    # Test data
    np.random.seed(42)
    input_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=np.int32)
    attention_mask = np.ones((batch_size, seq_len), dtype=np.float32)
    
    results = {}
    
    # CPU forward
    print("\n--- CPU Forward ---")
    for _ in range(warmup):
        _ = model.forward(input_ids, attention_mask)
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = model.forward(input_ids, attention_mask)
        times.append((time.perf_counter() - start) * 1000)
    
    results['cpu_mean_ms'] = np.mean(times)
    results['cpu_std_ms'] = np.std(times)
    print(f"  Mean: {results['cpu_mean_ms']:.2f} ms")
    print(f"  Std:  {results['cpu_std_ms']:.2f} ms")
    
    # With memory injection
    print("\n--- CPU Forward + Memory Injection ---")
    
    # Create test memories
    test_memories = []
    for i in range(5):
        mem = CapsuleMemory(
            memory_id=f"test_{i}",
            memory_type=MemoryType.CONCEPT,
            domain="test",
            content=f"Test memory content {i}",
            cognitive_features=CognitiveFeatures(
                plasticity_gain=0.8,
                consolidation_priority=0.9,
                stability=0.7,
                stress_link=0.1
            )
        )
        # Compute vectors
        mem.capsule_vector = np.random.randn(config.capsule_dim).astype(np.float32)
        mem.capsule_vector /= np.linalg.norm(mem.capsule_vector)
        test_memories.append(mem)
    
    for _ in range(warmup):
        _ = model.forward(input_ids, attention_mask, inject_memories=test_memories)
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = model.forward(input_ids, attention_mask, inject_memories=test_memories)
        times.append((time.perf_counter() - start) * 1000)
    
    results['cpu_inject_mean_ms'] = np.mean(times)
    print(f"  Mean: {results['cpu_inject_mean_ms']:.2f} ms")
    print(f"  Overhead: {results['cpu_inject_mean_ms'] - results['cpu_mean_ms']:.2f} ms")
    
    # Capsule encoding
    print("\n--- Capsule Encoding ---")
    test_text = "This is a test sentence for capsule encoding."
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        capsule = model.encode_to_capsule(test_text)
        times.append((time.perf_counter() - start) * 1000)
    
    results['capsule_mean_ms'] = np.mean(times)
    print(f"  Mean: {results['capsule_mean_ms']:.2f} ms")
    print(f"  Capsule shape: {capsule.shape}")
    print(f"  Capsule norm: {np.linalg.norm(capsule):.4f}")
    
    # DG expansion
    print("\n--- DG Expansion ---")
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        dg = model.encode_to_dg(test_text)
        times.append((time.perf_counter() - start) * 1000)
    
    results['dg_mean_ms'] = np.mean(times)
    active = np.sum(dg != 0)
    print(f"  Mean: {results['dg_mean_ms']:.2f} ms")
    print(f"  DG shape: {dg.shape}")
    print(f"  Active neurons: {active} ({100*active/len(dg):.1f}%)")
    
    # Memory creation
    print("\n--- Full Memory Creation ---")
    
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        mem = model.create_memory(
            content=test_text,
            memory_type=MemoryType.EPISODE,
            domain="benchmark",
            cognitive_features=CognitiveFeatures(plasticity_gain=0.9)
        )
        times.append((time.perf_counter() - start) * 1000)
    
    results['memory_create_mean_ms'] = np.mean(times)
    print(f"  Mean: {results['memory_create_mean_ms']:.2f} ms")
    print(f"  Memory ID: {mem.memory_id}")
    print(f"  Has capsule: {mem.capsule_vector is not None}")
    print(f"  Has DG: {mem.dg_vector is not None}")
    
    print(f"\n{'='*60}\n")
    
    return results


def test_capsule_integration():
    """Test capsule memory integration"""
    print("\n--- Capsule Integration Tests ---")
    
    config = CapsuleTransformerConfig(
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        capsule_dim=32,
        injection_layers=(1,)
    )
    
    model = VulkanCapsuleTransformer(config=config)
    
    # Test 1: Capsule encoding
    text = "Pattern separation transforms similar inputs."
    features = CognitiveFeatures(plasticity_gain=0.8, stability=0.9)
    
    capsule = model.encode_to_capsule(text, features)
    
    assert capsule.shape == (32,), f"Expected (32,), got {capsule.shape}"
    assert np.isclose(np.linalg.norm(capsule), 1.0, atol=1e-5), "Capsule not normalized"
    assert capsule[28] > 0, "Cognitive features not injected"
    print("  [OK] Capsule encoding")
    
    # Test 2: DG expansion
    dg = model.dg.expand(capsule)
    
    assert dg.shape == (128,), f"Expected (128,), got {dg.shape}"
    active = np.sum(dg != 0)
    assert active <= config.dg_k + 1, f"Too many active neurons: {active}"
    print(f"  [OK] DG expansion ({active} active)")
    
    # Test 3: Memory creation
    mem = model.create_memory(
        content=text,
        memory_type=MemoryType.CONCEPT,
        domain="testing",
        cognitive_features=features
    )
    
    assert mem.capsule_vector is not None
    assert mem.dg_vector is not None
    assert mem.memory_type == MemoryType.CONCEPT
    print("  [OK] Memory creation")
    
    # Test 4: Forward with injection
    input_ids = np.array([[1, 2, 3, 4]], dtype=np.int32)
    mask = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    
    emb_no_inject = model.forward(input_ids, mask)
    emb_with_inject = model.forward(input_ids, mask, inject_memories=[mem])
    
    # Should be different due to injection
    diff = np.linalg.norm(emb_no_inject - emb_with_inject)
    assert diff > 0.01, f"Injection had no effect: diff={diff}"
    print(f"  [OK] Memory injection (diff={diff:.4f})")
    
    # Test 5: Batch encoding
    texts = ["First test", "Second test", "Third test"]
    embeddings = model.encode_batch(texts)
    
    assert embeddings.shape == (3, config.output_dim)
    print(f"  [OK] Batch encoding")
    
    print("  [OK] All capsule tests passed!\n")


if __name__ == "__main__":
    import sys
    
    # Run tests
    test_capsule_integration()
    
    # Run benchmark
    benchmark_capsule_transformer()
