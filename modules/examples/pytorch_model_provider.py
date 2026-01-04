"""
PyTorch Model Provider Plugin

Wraps the existing Phi3Model as a plugin-compatible provider.
"""
import numpy as np
from typing import List

from modules.base import BaseModelProvider
from model import Phi3Model


class PyTorchModelProvider(BaseModelProvider):
    """
    PyTorch model provider plugin.
    
    Wraps the existing Phi3Model implementation.
    """
    
    def __init__(self):
        """Initialize PyTorch model provider."""
        self._model = Phi3Model()
    
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

