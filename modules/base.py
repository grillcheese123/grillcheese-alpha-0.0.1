"""
Base interfaces for GrillCheese plugin system
Defines abstract base classes for all plugin types
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from fastapi import FastAPI


class BaseMemoryBackend(ABC):
    """
    Base interface for memory storage backends
    
    Must match MemoryStore API for compatibility:
    - store(embedding, text, metadata=None) -> str (memory_id)
    - retrieve(query_embedding, k=3) -> List[Tuple[str, float]]
    - clear() -> None
    - get_stats() -> Dict[str, Any]
    - get_identity() -> Optional[str]
    """
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of embeddings used by this backend"""
        pass
    
    @abstractmethod
    def store(
        self,
        embedding: np.ndarray,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a memory
        
        Args:
            embedding: Embedding vector (embedding_dim,)
            text: Memory text content
            metadata: Optional metadata dictionary
            
        Returns:
            memory_id: Unique identifier for the stored memory
        """
        pass
    
    @abstractmethod
    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        include_identity: bool = True,
        reranker: Optional[Any] = None,
        query_text: Optional[str] = None,
        emotion_bias: Optional[Dict[str, float]] = None,
        temporal_bias: Optional[Dict[int, float]] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve similar memories
        
        Args:
            query_embedding: Query embedding vector (embedding_dim,)
            k: Number of memories to retrieve
            include_identity: Whether to include identity memory (optional, default: True)
            reranker: Optional reranking function (optional)
            query_text: Optional query text for reranking (optional)
            emotion_bias: Optional emotion-based bias (optional)
            temporal_bias: Optional temporal recency bias (optional)
            
        Returns:
            List of (text, similarity_score) tuples
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all memories"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory statistics
        
        Returns:
            Dictionary with stats (e.g., num_memories, total_size, etc.)
        """
        pass
    
    @abstractmethod
    def get_identity(self) -> Optional[str]:
        """
        Get system identity text
        
        Returns:
            Identity text or None if not set
        """
        pass
    
    def store_identity(
        self,
        embedding: np.ndarray,
        identity_text: str
    ) -> None:
        """
        Store system identity (optional, defaults to store)
        
        Args:
            embedding: Identity embedding vector
            identity_text: Identity description text
        """
        self.store(embedding, identity_text, metadata={"is_identity": True})


class BaseModelProvider(ABC):
    """
    Base interface for LLM model providers
    
    Must match Phi3GGUF/Phi3Model API for compatibility:
    - get_embedding(text) -> np.ndarray
    - generate(prompt, context) -> str
    - embedding_dim property
    """
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimension of embeddings produced by this model"""
        pass
    
    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Extract semantic embedding from text
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: List[str]
    ) -> str:
        """
        Generate text response
        
        Args:
            prompt: User prompt
            context: List of context strings (e.g., retrieved memories)
            
        Returns:
            Generated text response
        """
        pass
    
    def generate_with_tools(
        self,
        prompt: str,
        context: List[str],
        tools: List['BaseTool'],
        tool_executor: 'ToolExecutor',
        max_iterations: int = 5
    ) -> str:
        """
        Generate response with tool calling support (optional)
        
        Default implementation falls back to regular generate()
        Subclasses can override to support tool calling
        
        Args:
            prompt: User prompt
            context: List of context strings
            tools: List of available tools
            tool_executor: Tool executor instance
            max_iterations: Maximum tool calling iterations
            
        Returns:
            Generated text response (may include tool results)
        """
        return self.generate(prompt, context)


class BaseProcessingHook(ABC):
    """
    Base interface for processing middleware hooks
    
    Chainable hooks that modify request/response
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Hook name for identification"""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """
        Hook priority (lower = earlier in chain)
        
        Hooks are executed in priority order:
        - Pre-process: lowest priority first
        - Post-process: lowest priority first
        """
        pass
    
    async def pre_process(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Pre-process prompt before generation
        
        Args:
            prompt: Original prompt
            context: Request context (e.g., user_id, session_id, etc.)
            
        Returns:
            Modified prompt
        """
        return prompt
    
    async def post_process(
        self,
        response_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Post-process response after generation
        
        Args:
            response_data: Response dictionary (e.g., {"text": "...", "memories": [...]})
            context: Request context
            
        Returns:
            Modified response dictionary
        """
        return response_data
    
    async def on_error(
        self,
        error: Exception,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Handle errors during processing
        
        Args:
            error: Exception that occurred
            context: Request context
            
        Returns:
            Error response dictionary or None to propagate error
        """
        return None


class BaseAPIExtension(ABC):
    """
    Base interface for API extensions
    
    Allows plugins to add custom FastAPI routes and WebSocket handlers
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Extension name for identification"""
        pass
    
    @abstractmethod
    def register_routes(self, app: FastAPI) -> None:
        """
        Register FastAPI routes
        
        Args:
            app: FastAPI application instance
        """
        pass
    
    def register_websockets(self, app: FastAPI) -> None:
        """
        Register WebSocket handlers (optional)
        
        Args:
            app: FastAPI application instance
        """
        pass


class BaseTool(ABC):
    """
    Base interface for AI tools (MCP-style function calling)
    
    Tools can be called by the model during generation
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name (must be unique)"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the model"""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        JSON schema for tool parameters
        
        Example:
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                }
            },
            "required": ["query"]
        }
        """
        pass
    
    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate tool parameters (optional override)
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            True if valid, False otherwise
        """
        # Basic validation: check required parameters
        required = self.parameters.get("required", [])
        for param in required:
            if param not in kwargs:
                return False
        return True
    
    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """
        Execute the tool
        
        Args:
            **kwargs: Tool parameters (validated)
            
        Returns:
            Tool result (will be serialized to JSON for model)
        """
        pass
    
    async def execute_async(self, **kwargs) -> Any:
        """
        Async execution (optional override)
        
        Default implementation calls execute() in executor
        
        Args:
            **kwargs: Tool parameters
            
        Returns:
            Tool result
        """
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.execute(**kwargs))
