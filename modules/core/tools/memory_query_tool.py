"""
Memory Query Tool
Queries the memory store for information
"""
from typing import Dict, Any, Optional
import numpy as np

from modules.base import BaseTool


class MemoryQueryTool(BaseTool):
    """
    Tool for querying the memory store
    """
    
    def __init__(self, memory_backend=None, registry=None):
        """
        Initialize memory query tool
        
        Args:
            memory_backend: Optional memory backend instance (deprecated, use registry)
            registry: Optional ModuleRegistry instance for lazy loading
        """
        self.memory_backend = memory_backend
        self.registry = registry
    
    def _get_memory_backend(self):
        """Get memory backend from registry if not already set"""
        if self.memory_backend:
            return self.memory_backend
        
        if self.registry:
            self.memory_backend = self.registry.get_active_memory_backend()
            return self.memory_backend
        
        return None
    
    @property
    def name(self) -> str:
        return "memory_query"
    
    @property
    def description(self) -> str:
        return "Queries the memory store for similar memories. Provide a query text to search for related memories."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Query text to search for in memories"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of memories to retrieve (default: 3)",
                    "default": 3,
                    "minimum": 1,
                    "maximum": 10
                }
            },
            "required": ["query"]
        }
    
    def execute(self, **kwargs) -> Any:
        """
        Query memory store
        
        Args:
            query: Query text
            k: Number of results (default: 3)
            
        Returns:
            List of retrieved memories
        """
        # Get memory backend (lazy initialization from registry)
        memory_backend = self._get_memory_backend()
        if not memory_backend:
            return {"error": "Memory backend not available"}
        
        query = kwargs.get("query", "")
        k = kwargs.get("k", 3)
        
        if not query:
            return {"error": "No query provided"}
        
        try:
            # Get model provider for embeddings
            model_provider = None
            if self.registry:
                model_provider = self.registry.get_active_model_provider()
            
            if not model_provider:
                # Try to get from memory backend if it has one
                model_provider = getattr(memory_backend, '_model', None)
            
            if not model_provider:
                return {"error": "Model provider not available for embeddings"}
            
            query_embedding = model_provider.get_embedding(query)
            
            # Retrieve memories - use query_text if available, otherwise embedding
            if hasattr(memory_backend, 'retrieve'):
                # Check if retrieve accepts query_text parameter
                import inspect
                sig = inspect.signature(memory_backend.retrieve)
                if 'query_text' in sig.parameters:
                    results = memory_backend.retrieve(
                        query_embedding,
                        k=k,
                        query_text=query,
                        include_identity=False
                    )
                else:
                    results = memory_backend.retrieve(query_embedding, k=k)
            
            return {
                "query": query,
                "results": [
                    {"text": text, "similarity": float(score)}
                    for text, score in results
                ],
                "count": len(results)
            }
        except Exception as e:
            return {
                "error": str(e),
                "query": query
            }
