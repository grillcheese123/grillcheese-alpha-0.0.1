"""
GGUF Model Provider Plugin
Wraps existing Phi3GGUF as a plugin
"""
import numpy as np
from typing import List, Optional, Dict, Any

from modules.base import BaseModelProvider
from model_gguf import Phi3GGUF
from config import ModelConfig, find_gguf_model


class GGUFModelProvider(BaseModelProvider):
    """
    Wrapper around existing Phi3GGUF for plugin compatibility
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize GGUF model provider
        
        Args:
            config: Optional configuration dict
        """
        config = config or {}
        model_path = config.get("model_path")
        if model_path is None:
            model_path = find_gguf_model()
        
        n_gpu_layers = config.get("n_gpu_layers", -1)
        
        self._model = Phi3GGUF(model_path=model_path, n_gpu_layers=n_gpu_layers)
    
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
    
    def generate_with_tools(
        self,
        prompt: str,
        context: List[str],
        tools: List[Any],
        tool_executor: Any,
        max_iterations: int = 5
    ) -> str:
        """Generate response with tool calling support"""
        # Delegate to underlying Phi3GGUF model
        return self._model.generate_with_tools(
            prompt=prompt,
            context=context,
            tools=tools,
            tool_executor=tool_executor,
            max_iterations=max_iterations
        )
