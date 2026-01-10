"""
PyTorch Model Provider Plugin
Wraps existing Phi3Model as a plugin
"""
import numpy as np
from typing import List, Optional, Dict, Any

from modules.base import BaseModelProvider
from model import Phi3Model


class PyTorchModelProvider(BaseModelProvider):
    """
    Wrapper around existing Phi3Model for plugin compatibility
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PyTorch model provider
        
        Args:
            config: Optional configuration dict
        """
        self._model = Phi3Model()
    
    @property
    def embedding_dim(self) -> int:
        """Dimension of embeddings"""
        return self._model.embedding_dim
    
    @property
    def device(self) -> str:
        """Device the model is running on"""
        return getattr(self._model, 'device', 'cpu')
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Extract semantic embedding from text"""
        return self._model.get_embedding(text)
    
    def generate(self, prompt: str, context: List[str]) -> str:
        """Generate text response"""
        return self._model.generate(prompt, context)
