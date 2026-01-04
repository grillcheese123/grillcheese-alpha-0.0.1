"""
Tests for modules.base - Base interfaces
"""
import pytest
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.base import (
    BaseMemoryBackend,
    BaseModelProvider,
    BaseProcessingHook,
    BaseAPIExtension,
    BaseTool
)


class TestBaseMemoryBackend:
    """Tests for BaseMemoryBackend interface"""
    
    def test_base_memory_backend_is_abstract(self):
        """BaseMemoryBackend should be abstract and cannot be instantiated"""
        with pytest.raises(TypeError):
            BaseMemoryBackend()
    
    def test_memory_backend_interface_methods(self):
        """Test that a concrete implementation has all required methods"""
        class TestBackend(BaseMemoryBackend):
            def store(self, embedding, text):
                pass
            
            def retrieve(self, embedding, k=3):
                return []
            
            def clear(self):
                pass
            
            def get_stats(self):
                return {}
            
            def get_identity(self):
                return None
            
            @property
            def embedding_dim(self):
                return 384
        
        backend = TestBackend()
        assert backend.embedding_dim == 384
        assert backend.get_stats() == {}
        assert backend.retrieve(np.array([1.0] * 384), k=1) == []


class TestBaseModelProvider:
    """Tests for BaseModelProvider interface"""
    
    def test_base_model_provider_is_abstract(self):
        """BaseModelProvider should be abstract and cannot be instantiated"""
        with pytest.raises(TypeError):
            BaseModelProvider()
    
    def test_model_provider_interface_methods(self):
        """Test that a concrete implementation has all required methods"""
        class TestProvider(BaseModelProvider):
            def get_embedding(self, text):
                return np.array([1.0] * 384)
            
            def generate(self, prompt, context):
                return "Test response"
            
            @property
            def embedding_dim(self):
                return 384
        
        provider = TestProvider()
        assert provider.embedding_dim == 384
        assert len(provider.get_embedding("test")) == 384
        assert provider.generate("test", []) == "Test response"
    
    def test_generate_with_tools_default(self):
        """Test default generate_with_tools implementation"""
        class TestProvider(BaseModelProvider):
            def get_embedding(self, text):
                return np.array([1.0] * 384)
            
            def generate(self, prompt, context):
                return "Test response"
            
            @property
            def embedding_dim(self):
                return 384
        
        provider = TestProvider()
        # Default implementation should fall back to generate()
        result = provider.generate_with_tools("test", [], [], None)
        assert result == "Test response"


class TestBaseProcessingHook:
    """Tests for BaseProcessingHook interface"""
    
    def test_base_processing_hook_is_abstract(self):
        """BaseProcessingHook should be abstract and cannot be instantiated"""
        with pytest.raises(TypeError):
            BaseProcessingHook()
    
    def test_processing_hook_interface_methods(self):
        """Test that a concrete implementation has all required methods"""
        class TestHook(BaseProcessingHook):
            async def pre_process(self, prompt, context):
                return prompt
            
            async def post_process(self, response_data, context):
                return response_data
            
            async def on_error(self, error, context):
                pass
        
        hook = TestHook()
        import asyncio
        assert asyncio.run(hook.pre_process("test", {})) == "test"
        assert asyncio.run(hook.post_process({}, {})) == {}


class TestBaseAPIExtension:
    """Tests for BaseAPIExtension interface"""
    
    def test_base_api_extension_is_abstract(self):
        """BaseAPIExtension should be abstract and cannot be instantiated"""
        with pytest.raises(TypeError):
            BaseAPIExtension()
    
    def test_api_extension_interface_methods(self):
        """Test that a concrete implementation has all required methods"""
        class TestExtension(BaseAPIExtension):
            def register_routes(self, app):
                pass
            
            def register_websockets(self, app):
                pass
        
        extension = TestExtension()
        # Just verify it can be instantiated
        assert extension is not None


class TestBaseTool:
    """Tests for BaseTool interface"""
    
    def test_base_tool_is_abstract(self):
        """BaseTool should be abstract and cannot be instantiated"""
        with pytest.raises(TypeError):
            BaseTool()
    
    def test_tool_interface_properties(self):
        """Test that a concrete implementation has all required properties"""
        class TestTool(BaseTool):
            @property
            def name(self):
                return "test_tool"
            
            @property
            def description(self):
                return "Test tool"
            
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
        
        tool = TestTool()
        assert tool.name == "test_tool"
        assert tool.description == "Test tool"
        assert "param1" in tool.parameters["properties"]
    
    def test_tool_parameter_validation(self):
        """Test parameter validation"""
        class TestTool(BaseTool):
            @property
            def name(self):
                return "test_tool"
            
            @property
            def description(self):
                return "Test tool"
            
            @property
            def parameters(self):
                return {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string"},
                        "param2": {"type": "integer"}
                    },
                    "required": ["param1"]
                }
            
            async def execute(self, **kwargs):
                return {"result": "success"}
        
        tool = TestTool()
        
        # Valid parameters
        assert tool.validate_parameters(param1="test", param2=5) == True
        
        # Missing required parameter
        assert tool.validate_parameters(param2=5) == False
        
        # Wrong type
        assert tool.validate_parameters(param1="test", param2="not_int") == False
        
        # Extra parameters (should be allowed)
        assert tool.validate_parameters(param1="test", param2=5, extra="ok") == True

