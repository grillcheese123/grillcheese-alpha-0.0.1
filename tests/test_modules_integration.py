"""
Integration tests for the module system
"""
import pytest
import sys
import asyncio
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.registry import ModuleRegistry
from modules.tools import ToolExecutor
from modules.base import BaseMemoryBackend, BaseModelProvider, BaseTool


class IntegrationTestBackend(BaseMemoryBackend):
    """Integration test memory backend"""
    def __init__(self):
        self.stored = []
    
    def store(self, embedding, text):
        self.stored.append((embedding, text))
    
    def retrieve(self, embedding, k=3):
        return [text for _, text in self.stored[:k]]
    
    def clear(self):
        self.stored.clear()
    
    def get_stats(self):
        return {"total": len(self.stored)}
    
    def get_identity(self):
        return "test_identity"
    
    @property
    def embedding_dim(self):
        return 384


class IntegrationTestProvider(BaseModelProvider):
    """Integration test model provider"""
    def get_embedding(self, text):
        return np.array([1.0] * 384)
    
    def generate(self, prompt, context):
        return f"Response to: {prompt}"
    
    @property
    def embedding_dim(self):
        return 384


class IntegrationTestTool(BaseTool):
    """Integration test tool"""
    @property
    def name(self):
        return "integration_tool"
    
    @property
    def description(self):
        return "Integration test tool"
    
    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string"}
            },
            "required": ["input"]
        }
    
    async def execute(self, **kwargs):
        return {"output": f"Processed: {kwargs.get('input')}"}


class TestModuleIntegration:
    """Integration tests for the module system"""
    
    def test_full_registry_workflow(self):
        """Test complete registry workflow"""
        registry = ModuleRegistry()
        
        # Register components
        backend = IntegrationTestBackend()
        provider = IntegrationTestProvider()
        tool = IntegrationTestTool()
        
        registry.register_memory_backend("test_backend", backend)
        registry.register_model_provider("test_provider", provider)
        registry.register_tool(tool)
        
        # Set active components
        registry.set_active_memory_backend("test_backend")
        registry.set_active_model_provider("test_provider")
        
        # Retrieve active components
        active_backend = registry.get_active_memory_backend()
        active_provider = registry.get_active_model_provider()
        tools = registry.get_tools()
        
        assert active_backend is backend
        assert active_provider is provider
        assert tool in tools
    
    def test_tool_executor_with_registry(self):
        """Test ToolExecutor integration with registry"""
        registry = ModuleRegistry()
        tool = IntegrationTestTool()
        registry.register_tool(tool)
        
        executor = ToolExecutor(registry)
        
        result = asyncio.run(executor.execute_tool("integration_tool", input="test"))
        
        assert result.error is None
        assert result.result["output"] == "Processed: test"
    
    def test_memory_backend_operations(self):
        """Test memory backend operations"""
        backend = IntegrationTestBackend()
        
        embedding1 = np.array([1.0] * 384)
        embedding2 = np.array([2.0] * 384)
        
        backend.store(embedding1, "Memory 1")
        backend.store(embedding2, "Memory 2")
        
        assert backend.get_stats()["total"] == 2
        
        retrieved = backend.retrieve(embedding1, k=1)
        assert len(retrieved) == 1
        assert "Memory" in retrieved[0]
        
        backend.clear()
        assert backend.get_stats()["total"] == 0
    
    def test_model_provider_operations(self):
        """Test model provider operations"""
        provider = IntegrationTestProvider()
        
        embedding = provider.get_embedding("test text")
        assert len(embedding) == 384
        
        response = provider.generate("test prompt", ["context1", "context2"])
        assert "test prompt" in response
    
    def test_tool_validation_and_execution(self):
        """Test tool parameter validation and execution"""
        tool = IntegrationTestTool()
        
        # Valid parameters
        assert tool.validate_parameters(input="test") == True
        
        # Invalid parameters (missing required)
        assert tool.validate_parameters() == False
        
        # Execute tool
        result = asyncio.run(tool.execute(input="test_value"))
        assert result["output"] == "Processed: test_value"
    
    def test_registry_singleton_persistence(self):
        """Test that registry singleton persists across instances"""
        registry1 = ModuleRegistry()
        registry1.register_tool(IntegrationTestTool())
        
        registry2 = ModuleRegistry()
        tools = registry2.get_tools()
        
        assert len(tools) > 0
    
    @pytest.mark.asyncio
    async def test_async_tool_execution(self):
        """Test async tool execution"""
        registry = ModuleRegistry()
        tool = IntegrationTestTool()
        registry.register_tool(tool)
        
        executor = ToolExecutor(registry)
        
        result = await executor.execute_tool("integration_tool", input="async_test")
        
        assert result.error is None
        assert "async_test" in result.result["output"]
    
    def test_multiple_tools_same_registry(self):
        """Test registering and using multiple tools"""
        registry = ModuleRegistry()
        
        class Tool1(IntegrationTestTool):
            @property
            def name(self):
                return "tool1"
        
        class Tool2(IntegrationTestTool):
            @property
            def name(self):
                return "tool2"
        
        tool1 = Tool1()
        tool2 = Tool2()
        
        registry.register_tool(tool1)
        registry.register_tool(tool2)
        
        executor = ToolExecutor(registry)
        
        result1 = asyncio.run(executor.execute_tool("tool1", input="test1"))
        result2 = asyncio.run(executor.execute_tool("tool2", input="test2"))
        
        assert result1.result["output"] == "Processed: test1"
        assert result2.result["output"] == "Processed: test2"

