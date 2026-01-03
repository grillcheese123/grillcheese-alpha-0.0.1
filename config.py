"""
GrillCheese Configuration
Centralized configuration for all components
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
SHADERS_DIR = BASE_DIR / "shaders"
DATA_DIR = BASE_DIR / "data"

# Model configuration
class ModelConfig:
    # Phi-3 embedding dimension (from hidden states)
    PHI3_EMBEDDING_DIM = 3072
    
    # Sentence-transformers embedding dimension (for GGUF fallback)
    # all-MiniLM-L6-v2: 384 dims, fast, good quality
    # all-mpnet-base-v2: 768 dims, better quality, slower
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
    SENTENCE_TRANSFORMER_DIM = 384
    
    # Which embedding dimension to use (defaults to sentence-transformer for compatibility)
    # Set to PHI3_EMBEDDING_DIM if using PyTorch Phi-3 exclusively
    EMBEDDING_DIM = SENTENCE_TRANSFORMER_DIM
    
    @staticmethod
    def detect_embedding_dim(model):
        """
        Detect embedding dimension from model.
        
        Args:
            model: Model instance (Phi3GGUF, Phi3Model, or similar)
            
        Returns:
            Embedding dimension, or default if detection fails
        """
        if hasattr(model, 'embedding_dim'):
            return model.embedding_dim
        elif hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            return model.config.hidden_size
        elif hasattr(model, 'hidden_size'):
            return model.hidden_size
        else:
            # Fallback to default
            return ModelConfig.SENTENCE_TRANSFORMER_DIM
    
    # GGUF model paths (searched in order)
    GGUF_MODEL_PATHS = [
        "models/Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        "models/Phi-3-mini-4k-instruct-q4_K_M.gguf",
        "models/phi-3-mini-4k-instruct.gguf",
    ]
    
    # Generation settings
    MAX_NEW_TOKENS_GPU = 100
    MAX_NEW_TOKENS_CPU = 50
    TEMPERATURE = 0.7
    TOP_P = 0.9
    MAX_CONTEXT_ITEMS = 3


# Memory configuration
class MemoryConfig:
    DB_PATH = "memories.db"
    MAX_MEMORIES = 100000
    EMBEDDING_DIM = ModelConfig.EMBEDDING_DIM
    
    # Top-K retrieval default
    DEFAULT_K = 3
    
    # GPU memory buffer size (number of embeddings to keep in GPU memory)
    GPU_BUFFER_SIZE = 10000


# SNN configuration
class SNNConfig:
    N_NEURONS = 1000
    
    # LIF neuron parameters (tuned for visualization)
    DT = 0.01           # 10ms timestep
    TAU_MEM = 5.0       # 5ms membrane time constant (fast response)
    V_THRESH = 0.5      # Lower threshold for easier spiking
    
    # Simulation settings
    TIMESTEPS = 50      # Number of timesteps per forward pass
    INPUT_SCALE = 20.0  # Scaling factor for input embeddings


# Server configuration
class ServerConfig:
    HOST = "127.0.0.1"
    PORT = 8080
    WS_PATH = "/ws"


# Logging configuration
class LogConfig:
    LEVEL = "INFO"
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Use ASCII-safe characters for Windows compatibility
    CHECK = "[OK]"
    CROSS = "[X]"
    WARNING = "[!]"


# Helper function to find GGUF model
def find_gguf_model() -> str | None:
    """Find first available GGUF model file"""
    for path in ModelConfig.GGUF_MODEL_PATHS:
        full_path = BASE_DIR / path
        if full_path.exists():
            return str(full_path)
    return None

