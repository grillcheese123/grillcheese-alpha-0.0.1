"""
Phi-3 Model using GGUF format with llama-cpp-python (Vulkan/CUDA support)
Uses sentence-transformers for proper semantic embeddings
"""
import logging
import numpy as np
import sys
import os
from typing import List, Optional
from contextlib import contextmanager

from config import ModelConfig, LogConfig, find_gguf_model

# Configure logging
logging.basicConfig(level=LogConfig.LEVEL, format=LogConfig.FORMAT)
logger = logging.getLogger(__name__)

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False

# Try to import sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


@contextmanager
def suppress_llama_output():
    """Suppress llama.cpp debug output to stderr"""
    stderr_fileno = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fileno)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fileno)
    os.close(devnull)
    try:
        yield
    finally:
        os.dup2(old_stderr, stderr_fileno)
        os.close(old_stderr)


class Phi3GGUF:
    """
    Phi-3 model using GGUF format with llama-cpp-python
    Uses sentence-transformers for semantic embeddings (required for memory retrieval)
    """
    
    def __init__(self, model_path: Optional[str] = r"models\Phi-3-mini-4k-instruct-q4.gguf", n_gpu_layers: int = -1):
        """
        Initialize Phi-3 GGUF model
        
        Args:
            model_path: Path to GGUF model file (if None, will search default paths)
            n_gpu_layers: Number of layers to offload to GPU (-1 = all, 0 = CPU only)
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python not installed. Install with: "
                "pip install llama-cpp-python\n"
                "Note: On Windows, this may require Visual Studio Build Tools."
            )
        
        # Find model path
        if model_path is None:
            model_path = find_gguf_model()
            if model_path is None:
                raise FileNotFoundError(
                    "GGUF model not found. Please download Phi-3-mini GGUF model from:\n"
                    "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf\n"
                    "Or specify model_path parameter"
                )
        
        logger.info(f"Loading GGUF model from: {model_path}")
        
        # Initialize llama-cpp with GPU support
        try:
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,  # 4k context window
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                n_threads=4 if n_gpu_layers == 0 else None,
                logits_all=False,  # Suppress debug output
            )
            gpu_info = "all" if n_gpu_layers < 0 else str(n_gpu_layers)
            logger.info(f"{LogConfig.CHECK} GGUF model loaded (GPU layers: {gpu_info})")
            self.device = "gpu" if n_gpu_layers != 0 else "cpu"
        except Exception as e:
            raise RuntimeError(f"Failed to load GGUF model: {e}")
        
        # Initialize embedding model (sentence-transformers)
        self._init_embedding_model()
    
    def _init_embedding_model(self):
        """Initialize sentence-transformers for proper semantic embeddings"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning(
                f"{LogConfig.WARNING} sentence-transformers not available. "
                "Install with: pip install sentence-transformers\n"
                "Falling back to hash-based embeddings (not recommended for production)"
            )
            self.embedder = None
            self.embedding_dim = ModelConfig.PHI3_EMBEDDING_DIM  # Fallback dimension
            return
        
        try:
            logger.info(f"Loading embedding model: {ModelConfig.SENTENCE_TRANSFORMER_MODEL}")
            self.embedder = SentenceTransformer(ModelConfig.SENTENCE_TRANSFORMER_MODEL)
            self.embedding_dim = ModelConfig.SENTENCE_TRANSFORMER_DIM
            logger.info(f"{LogConfig.CHECK} Embedding model loaded (dim={self.embedding_dim})")
        except Exception as e:
            logger.warning(f"{LogConfig.WARNING} Failed to load embedding model: {e}")
            self.embedder = None
            self.embedding_dim = ModelConfig.PHI3_EMBEDDING_DIM
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Extract semantic embedding from text
        
        Uses sentence-transformers for high-quality embeddings suitable
        for semantic similarity search in the memory system.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        if self.embedder is not None:
            # Use sentence-transformers (recommended)
            embedding = self.embedder.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        else:
            # Fallback to hash-based embedding (not recommended)
            return self._fallback_embedding(text)
    
    def _fallback_embedding(self, text: str) -> np.ndarray:
        """
        Fallback hash-based embedding when sentence-transformers unavailable
        
        WARNING: This produces deterministic but non-semantic embeddings.
        Memory retrieval will not work properly with this method.
        """
        logger.warning("Using fallback hash-based embedding - semantic search will not work properly")
        
        try:
            tokens = self.llm.tokenize(text.encode('utf-8'))
        except:
            tokens = []
        
        embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        
        if len(tokens) > 0:
            token_ids = np.array(tokens[:min(512, len(tokens))], dtype=np.int64)
            for i, token_id in enumerate(token_ids):
                idx = i % self.embedding_dim
                val = (hash(str(token_id)) % 10000) / 10000.0
                embedding[idx] = (embedding[idx] + val) % 1.0
            
            norm = np.linalg.norm(embedding)
            if norm > 1e-8:
                embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def generate(self, prompt: str, context: List[str]) -> str:
        """
        Generate response with memory context
        
        Args:
            prompt: User prompt
            context: List of retrieved memory texts
        
        Returns:
            Generated response text
        """
        # Build prompt with context
        context_items = context[:ModelConfig.MAX_CONTEXT_ITEMS] if context else []
        context_text = "\n".join([f"Context: {c}" for c in context_items])
        
        if context_text:
            full_prompt = f"{context_text}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = f"User: {prompt}\nAssistant:"
        
        try:
            # Suppress llama-cpp debug output
            with suppress_llama_output():
                output = self.llm(
                    full_prompt,
                    max_tokens=ModelConfig.MAX_NEW_TOKENS_GPU,
                    temperature=ModelConfig.TEMPERATURE,
                    top_p=ModelConfig.TOP_P,
                    stop=["User:", "\nUser:", "\n\nUser:", "Instruction>", "[MY_STATE]", "<|end|>", "<|endoftext|>", "</s>"],
                    echo=False,
                )
            
            # Extract text from response
            if isinstance(output, dict):
                choices = output.get('choices', [])
                response = choices[0].get('text', '') if choices else ''
            elif isinstance(output, str):
                response = output
            else:
                response = str(output)
            
            # Clean up response - remove system prompt echoes
            response = response.strip()
            
            # Stop at common separators
            for separator in ["\nUser:", "\n\nUser:", "User:", "Instruction>", "[MY_STATE]", "\nYou are", "\n\nYou are"]:
                if separator in response:
                    response = response.split(separator)[0].strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"[Error during generation: {str(e)}]"
