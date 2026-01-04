"""
Base interfaces for GrillCheese plugin system

Defines abstract base classes that all plugins must implement.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
from fastapi import FastAPI


class BaseMemoryBackend(ABC):
    """
    Base interface for memory storage backends.
    
    All memory backends must implement this interface to be compatible
    with the GrillCheese memory system.
    """
    
    @abstractmethod
    def store(self, embedding: np.ndarray, text: str) -> None:
        """
        Store a memory with its embedding.
        
        Args:
            embedding: Embedding vector (numpy array)
            text: Text content of the memory
        """
        pass
    
    @abstractmethod
    def retrieve(self, embedding: np.ndarray, k: int = 3) -> List[str]:
        """
        Retrieve k most similar memories.
        
        Args:
            embedding: Query embedding vector
            k: Number of memories to retrieve
            
        Returns:
            List of retrieved memory texts
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """
        Clear all memories (protected memories may be preserved).
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get memory store statistics.
        
        Returns:
            Dictionary with stats (e.g., total_memories, etc.)
        """
        pass
    
    @abstractmethod
    def get_identity(self) -> Optional[str]:
        """
        Get the stored identity text.
        
        Returns:
            Identity text or None if not set
        """
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Embedding dimension for this memory backend."""
        pass


class BaseModelProvider(ABC):
    """
    Base interface for LLM model providers.
    
    All model providers must implement this interface to be compatible
    with the GrillCheese generation system.
    """
    
    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Extract semantic embedding from text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector (numpy array)
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str, context: List[str]) -> str:
        """
        Generate response with memory context.
        
        Args:
            prompt: User prompt
            context: List of retrieved memory texts
            
        Returns:
            Generated response text
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
        Generate response with tool calling support.
        
        This method can be overridden by providers that support native tool calling.
        Default implementation falls back to regular generate().
        
        Args:
            prompt: User prompt
            context: List of retrieved memory texts
            tools: List of available tools
            tool_executor: Tool executor instance
            max_iterations: Maximum number of tool calling iterations
            
        Returns:
            Generated response text
        """
        # Default implementation: no tool calling
        return self.generate(prompt, context)
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Embedding dimension for this model."""
        pass


class BaseProcessingHook(ABC):
    """
    Base interface for processing hooks (middleware).
    
    Hooks can modify prompts and responses during processing.
    They are executed in registration order.
    """
    
    @abstractmethod
    async def pre_process(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Pre-process the user prompt before generation.
        
        Args:
            prompt: Original user prompt
            context: Additional context dictionary
            
        Returns:
            Modified prompt
        """
        pass
    
    @abstractmethod
    async def post_process(self, response_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process the response data after generation.
        
        Args:
            response_data: Response data dictionary
            context: Additional context dictionary
            
        Returns:
            Modified response data
        """
        pass
    
    async def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """
        Handle errors during processing.
        
        Args:
            error: Exception that occurred
            context: Additional context dictionary
        """
        pass


class BaseAPIExtension(ABC):
    """
    Base interface for API extensions.
    
    Extensions can add custom FastAPI routes and WebSocket handlers.
    """
    
    @abstractmethod
    def register_routes(self, app: FastAPI) -> None:
        """
        Register custom FastAPI routes.
        
        Args:
            app: FastAPI application instance
        """
        pass
    
    @abstractmethod
    def register_websockets(self, app: FastAPI) -> None:
        """
        Register custom WebSocket handlers.
        
        Args:
            app: FastAPI application instance
        """
        pass


class BaseTool(ABC):
    """
    Base interface for AI tools (MCP-style function calling).
    
    Tools can be called by the model during generation to perform
    external actions like web search, file operations, etc.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Tool name (must be unique).
        
        Returns:
            Tool name string
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Tool description (used by model to understand when to call it).
        
        Returns:
            Tool description string
        """
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """
        JSON schema for tool parameters.
        
        Returns:
            JSON schema dictionary (OpenAPI format)
        """
        pass
    
    @abstractmethod
    async def execute(self, **kwargs: Any) -> Any:
        """
        Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool parameters (validated against schema)
            
        Returns:
            Tool execution result (must be JSON-serializable)
        """
        pass
    
    def validate_parameters(self, **kwargs: Any) -> bool:
        """
        Validate tool parameters against schema.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            True if valid, False otherwise
        """
        schema = self.parameters
        required = schema.get('required', [])
        
        # Check required parameters
        for param in required:
            if param not in kwargs:
                return False
        
        # Check parameter types (basic validation)
        properties = schema.get('properties', {})
        for param_name, param_value in kwargs.items():
            if param_name not in properties:
                continue  # Allow extra parameters
            
            param_schema = properties[param_name]
            param_type = param_schema.get('type')
            
            if param_type == 'string' and not isinstance(param_value, str):
                return False
            elif param_type == 'integer' and not isinstance(param_value, int):
                return False
            elif param_type == 'number' and not isinstance(param_value, (int, float)):
                return False
            elif param_type == 'boolean' and not isinstance(param_value, bool):
                return False
            elif param_type == 'array' and not isinstance(param_value, list):
                return False
            elif param_type == 'object' and not isinstance(param_value, dict):
                return False
        
        return True

