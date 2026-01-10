"""
Vulkan-accelerated embeddings using llama.cpp or custom compute shaders.

This module provides GPU-accelerated embedding generation that uses the same
Vulkan backend as the rest of GrillCheese, eliminating the CPU bottleneck
from sentence-transformers.

Two strategies:
1. LlamaCppEmbedder: Uses llama.cpp's native embedding mode (Vulkan accelerated)
2. VulkanEmbedder: Custom compute shader implementation (coming soon)
"""
import logging
import numpy as np
import time
from typing import Optional, List, Union
from pathlib import Path

from config import ModelConfig, LogConfig

logger = logging.getLogger(__name__)

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False


class LlamaCppEmbedder:
    """
    Vulkan-accelerated embeddings using llama.cpp.
    
    Uses a dedicated embedding model (or the main model in embedding mode)
    with GPU acceleration through the same Vulkan backend used for inference.
    
    Recommended models:
    - nomic-embed-text-v1.5 (768 dims, best quality for Vulkan)
    - bge-small-en-v1.5 (384 dims, compatible with existing memory)
    - all-MiniLM-L6-v2-gguf (384 dims, lightweight)
    """
    
    # Known embedding models with their dimensions
    KNOWN_MODELS = {
        "nomic-embed-text-v1.5": 768,
        "nomic-embed-text-v1": 768,
        "bge-small-en-v1.5": 384,
        "bge-base-en-v1.5": 768,
        "all-MiniLM-L6-v2": 384,
        "e5-small-v2": 384,
        "e5-base-v2": 768,
        "gte-small": 384,
        "gte-base": 768,
    }
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_dim: int = 384,
        n_gpu_layers: int = -1,
        n_ctx: int = 512,
        pooling: str = "mean",
        normalize: bool = True
    ):
        """
        Initialize Vulkan-accelerated embedder.
        
        Args:
            model_path: Path to GGUF embedding model. If None, will search for common models.
            embedding_dim: Expected embedding dimension (for validation)
            n_gpu_layers: GPU layers (-1 = all on GPU)
            n_ctx: Context length for embedding model
            pooling: Pooling strategy ("mean", "cls", "last")
            normalize: Whether to L2-normalize embeddings
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError("llama-cpp-python required for Vulkan embeddings")
        
        self.embedding_dim = embedding_dim
        self.pooling = pooling
        self.normalize = normalize
        self._model_path = model_path
        
        # Find embedding model
        if model_path is None:
            model_path = self._find_embedding_model()
        
        if model_path is None:
            raise FileNotFoundError(
                "No embedding model found. Download a GGUF embedding model:\n"
                "  nomic-embed-text-v1.5.Q4_K_M.gguf (recommended)\n"
                "  bge-small-en-v1.5-q4_k_m.gguf (384 dims, compatible)\n"
                "Place in the 'models/' directory."
            )
        
        logger.info(f"Loading Vulkan embedding model: {model_path}")
        
        # Initialize with embedding=True for models that support it
        # Otherwise use logits_all=True to get hidden states
        try:
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                embedding=True,  # Enable embedding mode
                n_threads=1,  # Let GPU do the work
            )
            self._use_native_embedding = True
            logger.info(f"{LogConfig.CHECK} Vulkan embedding model loaded (native embedding mode)")
        except Exception as e:
            logger.warning(f"Native embedding mode failed: {e}, using hidden state extraction")
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                logits_all=True,
            )
            self._use_native_embedding = False
            logger.info(f"{LogConfig.CHECK} Vulkan embedding model loaded (hidden state mode)")
        
        # Detect actual embedding dimension
        self._detect_embedding_dim()
    
    def _find_embedding_model(self) -> Optional[Path]:
        """Find an embedding model in the models directory"""
        from config import BASE_DIR
        models_dir = BASE_DIR / "models"
        
        # Search patterns for common embedding models
        patterns = [
            "*embed*.gguf",
            "*bge*.gguf",
            "*e5*.gguf",
            "*gte*.gguf",
            "*minilm*.gguf",
            "*nomic*.gguf",
        ]
        
        for pattern in patterns:
            matches = list(models_dir.glob(pattern))
            if matches:
                # Prefer Q4_K_M quantization
                for m in matches:
                    if "q4_k_m" in m.name.lower() or "Q4_K_M" in m.name:
                        return m
                return matches[0]
        
        return None
    
    def _detect_embedding_dim(self):
        """Detect embedding dimension from model"""
        try:
            test_emb = self._get_embedding_internal("test")
            actual_dim = len(test_emb)
            if actual_dim != self.embedding_dim:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self.embedding_dim}, "
                    f"got {actual_dim}. Updating config."
                )
                self.embedding_dim = actual_dim
        except Exception as e:
            logger.warning(f"Could not detect embedding dim: {e}")
    
    def _get_embedding_internal(self, text: str) -> np.ndarray:
        """Internal embedding computation"""
        if self._use_native_embedding:
            # Use llama.cpp's native embedding function
            embedding = self.llm.embed(text)
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            return embedding
        else:
            # Extract from hidden states
            tokens = self.llm.tokenize(text.encode())
            output = self.llm.eval(tokens)
            # Get last hidden state and pool
            # This is approximate - proper implementation needs model architecture knowledge
            return self._pool_hidden_states(output, len(tokens))
    
    def _pool_hidden_states(self, hidden_states, seq_len: int) -> np.ndarray:
        """Pool hidden states to get embedding"""
        if self.pooling == "last":
            embedding = hidden_states[-1]
        elif self.pooling == "cls":
            embedding = hidden_states[0]
        else:  # mean
            embedding = np.mean(hidden_states[:seq_len], axis=0)
        
        return embedding.astype(np.float32)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text using Vulkan-accelerated model.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (embedding_dim,)
        """
        embedding = self._get_embedding_internal(text)
        
        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Embedding matrix (num_texts, embedding_dim)
        """
        embeddings = []
        for text in texts:
            emb = self.get_embedding(text)
            embeddings.append(emb)
        return np.array(embeddings, dtype=np.float32)


class VulkanEmbedder:
    """
    Custom Vulkan compute shader embeddings.
    
    Uses GrillCheese's Vulkan backend to implement embedding computation
    entirely in GLSL compute shaders for maximum performance.
    
    Architecture:
    1. Token embedding lookup (embedding-lookup.glsl)
    2. Positional encoding
    3. Transformer layers (attention + FFN)
    4. Mean pooling
    5. Projection to target dimension
    """
    
    def __init__(
        self,
        vocab_size: int = 32000,
        embedding_dim: int = 384,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_seq_len: int = 512
    ):
        """
        Initialize custom Vulkan embedder.
        
        Args:
            vocab_size: Vocabulary size for token embeddings
            embedding_dim: Output embedding dimension
            hidden_dim: Transformer hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Initialize Vulkan backend
        try:
            from vulkan_backend import VulkanCompute
            self.gpu = VulkanCompute()
            logger.info(f"{LogConfig.CHECK} VulkanEmbedder initialized")
        except Exception as e:
            raise RuntimeError(f"Vulkan backend required: {e}")
        
        # Initialize weights (would be loaded from trained model)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Token embeddings
        self.token_embeddings = np.random.randn(
            self.vocab_size, self.hidden_dim
        ).astype(np.float32) * 0.02
        
        # Positional embeddings
        self.position_embeddings = np.random.randn(
            self.max_seq_len, self.hidden_dim
        ).astype(np.float32) * 0.02
        
        # Output projection
        self.output_projection = np.random.randn(
            self.hidden_dim, self.embedding_dim
        ).astype(np.float32) * 0.02
        
        # Transformer layer weights would go here
        # For now, this is a placeholder
        self.layers = []
    
    def tokenize(self, text: str) -> np.ndarray:
        """Simple tokenization (placeholder - use proper tokenizer)"""
        # This is a placeholder - in production use a proper tokenizer
        # that matches the embedding model's vocabulary
        tokens = []
        for char in text.lower()[:self.max_seq_len]:
            token_id = ord(char) % self.vocab_size
            tokens.append(token_id)
        return np.array(tokens, dtype=np.uint32)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Compute embedding using Vulkan compute shaders.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (embedding_dim,)
        """
        # Tokenize
        tokens = self.tokenize(text)
        seq_len = len(tokens)
        
        if seq_len == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Token embedding lookup (GPU)
        # For now using CPU - would use embedding-lookup.glsl shader
        token_embs = self.token_embeddings[tokens]  # (seq_len, hidden_dim)
        
        # Add positional embeddings
        pos_embs = self.position_embeddings[:seq_len]
        hidden = token_embs + pos_embs  # (seq_len, hidden_dim)
        
        # Mean pooling
        pooled = np.mean(hidden, axis=0)  # (hidden_dim,)
        
        # Project to embedding dim
        embedding = pooled @ self.output_projection  # (embedding_dim,)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def load_weights(self, weights_path: str):
        """Load pretrained weights from file"""
        import pickle
        with open(weights_path, 'rb') as f:
            weights = pickle.load(f)
        
        self.token_embeddings = weights['token_embeddings']
        self.position_embeddings = weights['position_embeddings']
        self.output_projection = weights['output_projection']
        if 'layers' in weights:
            self.layers = weights['layers']
        
        logger.info(f"Loaded weights from {weights_path}")


class HybridEmbedder:
    """
    Hybrid embedder that tries multiple backends in order.
    
    Priority:
    1. LlamaCppEmbedder (Vulkan accelerated GGUF model)
    2. VulkanEmbedder (custom shaders - if weights available)
    3. SentenceTransformer (CPU fallback)
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        prefer_vulkan: bool = True,
        fallback_to_cpu: bool = True
    ):
        """
        Initialize hybrid embedder.
        
        Args:
            embedding_dim: Target embedding dimension
            prefer_vulkan: Try Vulkan backends first
            fallback_to_cpu: Allow CPU fallback
        """
        self.embedding_dim = embedding_dim
        self._embedder = None
        self._backend_name = "none"
        
        if prefer_vulkan:
            # Try llama.cpp Vulkan embedder
            try:
                self._embedder = LlamaCppEmbedder(embedding_dim=embedding_dim)
                self._backend_name = "vulkan_llama"
                self.embedding_dim = self._embedder.embedding_dim
                logger.info(f"{LogConfig.CHECK} Using Vulkan embedder (llama.cpp)")
                return
            except Exception as e:
                logger.debug(f"LlamaCppEmbedder not available: {e}")
        
        if fallback_to_cpu:
            # Fallback to sentence-transformers
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(
                    ModelConfig.EMBEDDING_MODEL,
                    device='cpu'
                )
                self._backend_name = "sentence_transformer"
                logger.info(f"{LogConfig.WARNING} Using CPU embedder (sentence-transformers)")
            except ImportError:
                logger.error("No embedding backend available!")
                raise RuntimeError("No embedding backend available")
    
    @property
    def backend(self) -> str:
        """Return the active backend name"""
        return self._backend_name
    
    @property
    def is_gpu_accelerated(self) -> bool:
        """Check if using GPU acceleration"""
        return self._backend_name in ("vulkan_llama", "vulkan_custom")
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text"""
        if self._embedder is None:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        if self._backend_name == "sentence_transformer":
            return self._embedder.encode(text, convert_to_numpy=True).astype(np.float32)
        else:
            return self._embedder.get_embedding(text)
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts"""
        if self._embedder is None:
            return np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        
        if self._backend_name == "sentence_transformer":
            return self._embedder.encode(texts, convert_to_numpy=True).astype(np.float32)
        elif hasattr(self._embedder, 'get_embeddings_batch'):
            return self._embedder.get_embeddings_batch(texts)
        else:
            # Fallback to sequential
            return np.array([self.get_embedding(t) for t in texts], dtype=np.float32)


# Benchmark utility
def benchmark_embedders(texts: Optional[List[str]] = None, iterations: int = 10):
    """
    Benchmark available embedding backends.
    
    Args:
        texts: Test texts (default: predefined set)
        iterations: Number of iterations per text
    """
    if texts is None:
        texts = [
            "Hello world",
            "What is machine learning?",
            "Tell me a story about artificial intelligence.",
            "The quick brown fox jumps over the lazy dog.",
        ]
    
    print("=" * 60)
    print("Embedding Backend Benchmark")
    print("=" * 60)
    
    results = {}
    
    # Test LlamaCpp Vulkan
    try:
        embedder = LlamaCppEmbedder()
        
        # Warmup
        embedder.get_embedding("warmup")
        
        times = []
        for _ in range(iterations):
            for text in texts:
                start = time.time()
                emb = embedder.get_embedding(text)
                times.append(time.time() - start)
        
        avg_ms = np.mean(times) * 1000
        results["vulkan_llama"] = avg_ms
        print(f"Vulkan (llama.cpp): {avg_ms:.2f} ms/embedding (dim={len(emb)})")
    except Exception as e:
        print(f"Vulkan (llama.cpp): Not available - {e}")
    
    # Test sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(ModelConfig.EMBEDDING_MODEL, device='cpu')
        
        # Warmup
        embedder.encode("warmup")
        
        times = []
        for _ in range(iterations):
            for text in texts:
                start = time.time()
                emb = embedder.encode(text, convert_to_numpy=True)
                times.append(time.time() - start)
        
        avg_ms = np.mean(times) * 1000
        results["sentence_transformer"] = avg_ms
        print(f"SentenceTransformer (CPU): {avg_ms:.2f} ms/embedding (dim={len(emb)})")
    except Exception as e:
        print(f"SentenceTransformer: Not available - {e}")
    
    print("=" * 60)
    
    if len(results) >= 2:
        best = min(results, key=results.get)
        speedup = max(results.values()) / min(results.values())
        print(f"Best: {best} ({speedup:.1f}x faster)")
    
    return results


if __name__ == "__main__":
    benchmark_embedders()
