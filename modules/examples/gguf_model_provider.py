"""
GGUF Model Provider Plugin

Wraps the existing Phi3GGUF model as a plugin-compatible provider.
"""
import numpy as np
from typing import List, Optional

from modules.base import BaseModelProvider
from model_gguf import Phi3GGUF
from config import ModelConfig, find_gguf_model


class GGUFModelProvider(BaseModelProvider):
    """
    GGUF model provider plugin.
    
    Wraps the existing Phi3GGUF implementation.
    """
    
    def __init__(self, model_path: Optional[str] = None, n_gpu_layers: int = -1):
        """
        Initialize GGUF model provider.
        
        Args:
            model_path: Path to GGUF model file (None = auto-detect)
            n_gpu_layers: Number of GPU layers (-1 = all)
        """
        if model_path is None:
            model_path = find_gguf_model()
        
        self._model = Phi3GGUF(model_path=model_path, n_gpu_layers=n_gpu_layers)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Extract semantic embedding from text."""
        return self._model.get_embedding(text)
    
    def generate(self, prompt: str, context: List[str]) -> str:
        """Generate response with memory context."""
        return self._model.generate(prompt, context)
    
    @property
    def embedding_dim(self) -> int:
        """Embedding dimension."""
        return self._model.embedding_dim

