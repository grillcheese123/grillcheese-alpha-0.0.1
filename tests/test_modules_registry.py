"""
Tests for modules.registry - Plugin registry
"""
import pytest
import sys
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.registry import ModuleRegistry
from modules.base import BaseMemoryBackend, BaseModelProvider, BaseTool


class MockMemoryBackend(BaseMemoryBackend):
    """Mock memory backend for testing"""
    def __init__(self):
        self._dim = 384
    
    def store(self, embedding, text):
        pass
    
    def retrieve(self, embedding, k=3):
        return []
    
    def clear(self):
        pass
    
    def get_stats(self):
        return {"total": 0}
    
    def get_identity(self):
        return None
    
    @property
    def embedding_dim(self):
        return self._dim


class MockModelProvider(BaseModelProvider):
    """Mock model provider for testing"""
    def __init__(self):
        self._dim = 384
    
    def get_embedding(self, text):
        return np.array([1.0] * self._dim)
    
    def generate(self, prompt, context):
        return "Test response"
    
    @property
    def embedding_dim(self):
        return self._dim


class MockTool(BaseTool):
    """Mock tool for testing"""
    @property
    def name(self):
        return "mock_tool"
    
    @property
    def description(self):
        return "Mock tool for testing"
    
    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "param1": {"type": "string"}
            },
            "required": ["param1"]
        }
    
    async def execute(self, **kwargs):
        return {"result": "success"}


class TestModuleRegistry:
    """Tests for ModuleRegistry"""
    
    def setup_method(self):
        """Reset registry state before each test"""
        # Clear the singleton instance
        ModuleRegistry._instance = None
    
    def test_registry_singleton(self):
        """Test that ModuleRegistry is a singleton"""
        registry1 = ModuleRegistry()
        registry2 = ModuleRegistry()
        
        assert registry1 is registry2
    
    def test_register_memory_backend(self):
        """Test registering a memory backend"""
        registry = ModuleRegistry()
        backend = MockMemoryBackend()
        
        registry.register_memory_backend("test_backend", backend)
        
        assert "test_backend" in registry.memory_backends
        assert registry.memory_backends["test_backend"] is backend
    
    def test_register_model_provider(self):
        """Test registering a model provider"""
        registry = ModuleRegistry()
        provider = MockModelProvider()
        
        registry.register_model_provider("test_provider", provider)
        
        assert "test_provider" in registry.model_providers
        assert registry.model_providers["test_provider"] is provider
    
    def test_register_tool(self):
        """Test registering a tool"""
        registry = ModuleRegistry()
        tool = MockTool()
        
        registry.register_tool(tool)
        
        assert tool.name in registry.tools
        assert registry.tools[tool.name] is tool
    
    def test_get_tools(self):
        """Test getting all tools"""
        registry = ModuleRegistry()
        tool1 = MockTool()
        
        # Create another tool with different name
        class MockTool2(MockTool):
            @property
            def name(self):
                return "mock_tool2"
        
        tool2 = MockTool2()
        
        registry.register_tool(tool1)
        registry.register_tool(tool2)
        
        tools = registry.get_tools()
        assert len(tools) == 2
        assert tool1 in tools
        assert tool2 in tools
    
    def test_get_tool(self):
        """Test getting a specific tool"""
        registry = ModuleRegistry()
        tool = MockTool()
        
        registry.register_tool(tool)
        
        retrieved = registry.get_tool("mock_tool")
        assert retrieved is tool
        
        # Non-existent tool
        assert registry.get_tool("nonexistent") is None
    
    def test_set_active_memory_backend(self):
        """Test setting active memory backend"""
        registry = ModuleRegistry()
        backend = MockMemoryBackend()
        
        registry.register_memory_backend("test_backend", backend)
        
        result = registry.set_active_memory_backend("test_backend")
        assert result == True
        assert registry._active_memory_backend == "test_backend"
        assert registry.get_active_memory_backend() is backend
    
    def test_set_active_memory_backend_nonexistent(self):
        """Test setting active memory backend that doesn't exist"""
        registry = ModuleRegistry()
        # Clear any previous state
        registry._active_memory_backend = None
        
        result = registry.set_active_memory_backend("nonexistent")
        assert result == False
        assert registry._active_memory_backend is None
    
    def test_set_active_model_provider(self):
        """Test setting active model provider"""
        registry = ModuleRegistry()
        provider = MockModelProvider()
        
        registry.register_model_provider("test_provider", provider)
        
        result = registry.set_active_model_provider("test_provider")
        assert result == True
        assert registry._active_model_provider == "test_provider"
        assert registry.get_active_model_provider() is provider
    
    def test_register_processing_hook(self):
        """Test registering a processing hook"""
        registry = ModuleRegistry()
        
        class MockHook:
            pass
        
        hook = MockHook()
        registry.register_processing_hook(hook)
        
        assert hook in registry.processing_hooks
    
    def test_register_api_extension(self):
        """Test registering an API extension"""
        registry = ModuleRegistry()
        
        class MockExtension:
            pass
        
        extension = MockExtension()
        registry.register_api_extension(extension)
        
        assert extension in registry.api_extensions
    
    def test_get_active_memory_backend_none_set(self):
        """Test getting active memory backend when none is set"""
        registry = ModuleRegistry()
        # Clear any previous state
        registry._active_memory_backend = None
        
        assert registry.get_active_memory_backend() is None
    
    def test_get_active_model_provider_none_set(self):
        """Test getting active model provider when none is set"""
        registry = ModuleRegistry()
        # Clear any previous state
        registry._active_model_provider = None
        
        assert registry.get_active_model_provider() is None

