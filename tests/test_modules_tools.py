"""
Tests for modules.tools - Tool execution framework
"""
import pytest
import sys
import json
import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.tools import ToolCall, ToolResult, ToolExecutor
from modules.base import BaseTool


class MockTool(BaseTool):
    """Mock tool for testing"""
    def __init__(self, name="test_tool", should_fail=False):
        self._name = name
        self._should_fail = should_fail
    
    @property
    def name(self):
        return self._name
    
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
        if self._should_fail:
            raise Exception("Tool execution failed")
        return {"result": kwargs.get("param1", "default")}


class MockRegistry:
    """Mock registry for testing"""
    def __init__(self):
        self.tools = {}
    
    def get_tool(self, name):
        return self.tools.get(name)


class TestToolCall:
    """Tests for ToolCall dataclass"""
    
    def test_tool_call_creation(self):
        """Test creating a ToolCall"""
        call = ToolCall(
            tool_name="test_tool",
            parameters={"param1": "value1"}
        )
        
        assert call.tool_name == "test_tool"
        assert call.parameters == {"param1": "value1"}
        assert call.call_id is not None
    
    def test_tool_call_custom_id(self):
        """Test ToolCall with custom call_id"""
        call = ToolCall(
            tool_name="test_tool",
            parameters={},
            call_id="custom_id"
        )
        
        assert call.call_id == "custom_id"


class TestToolResult:
    """Tests for ToolResult dataclass"""
    
    def test_tool_result_success(self):
        """Test creating a successful ToolResult"""
        result = ToolResult(
            call_id="test_id",
            result={"data": "success"},
            execution_time=0.5
        )
        
        assert result.call_id == "test_id"
        assert result.result == {"data": "success"}
        assert result.error is None
        assert result.execution_time == 0.5
    
    def test_tool_result_error(self):
        """Test creating a ToolResult with error"""
        result = ToolResult(
            call_id="test_id",
            result=None,
            error="Tool not found",
            execution_time=0.1
        )
        
        assert result.error == "Tool not found"
        assert result.result is None


class TestToolExecutor:
    """Tests for ToolExecutor"""
    
    def test_execute_tool_success(self):
        """Test executing a tool successfully"""
        registry = MockRegistry()
        tool = MockTool()
        registry.tools["test_tool"] = tool
        
        executor = ToolExecutor(registry)
        
        result = asyncio.run(executor.execute_tool("test_tool", param1="test_value"))
        
        assert result.error is None
        assert result.result == {"result": "test_value"}
        assert result.execution_time >= 0  # Allow 0.0 for very fast executions
    
    def test_execute_tool_not_found(self):
        """Test executing a tool that doesn't exist"""
        registry = MockRegistry()
        executor = ToolExecutor(registry)
        
        result = asyncio.run(executor.execute_tool("nonexistent", param1="test"))
        
        assert result.error is not None
        assert "not found" in result.error.lower()
        assert result.result is None
    
    def test_execute_tool_invalid_parameters(self):
        """Test executing a tool with invalid parameters"""
        registry = MockRegistry()
        tool = MockTool()
        registry.tools["test_tool"] = tool
        
        executor = ToolExecutor(registry)
        
        # Missing required parameter
        result = asyncio.run(executor.execute_tool("test_tool"))
        
        assert result.error is not None
        assert "invalid" in result.error.lower() or "parameter" in result.error.lower()
    
    def test_execute_tool_execution_error(self):
        """Test tool execution that raises an exception"""
        registry = MockRegistry()
        tool = MockTool(should_fail=True)
        registry.tools["test_tool"] = tool
        
        executor = ToolExecutor(registry)
        
        result = asyncio.run(executor.execute_tool("test_tool", param1="test"))
        
        assert result.error is not None
        assert "failed" in result.error.lower()
    
    def test_execute_tool_chain(self):
        """Test executing multiple tools in sequence"""
        registry = MockRegistry()
        tool1 = MockTool("tool1")
        tool2 = MockTool("tool2")
        registry.tools["tool1"] = tool1
        registry.tools["tool2"] = tool2
        
        executor = ToolExecutor(registry)
        
        calls = [
            ToolCall("tool1", {"param1": "value1"}),
            ToolCall("tool2", {"param1": "value2"})
        ]
        
        results = asyncio.run(executor.execute_tool_chain(calls))
        
        assert len(results) == 2
        assert results[0].result == {"result": "value1"}
        assert results[1].result == {"result": "value2"}
    
    def test_format_tool_results(self):
        """Test formatting tool results for prompt injection"""
        executor = ToolExecutor(MockRegistry())
        
        results = [
            ToolResult("id1", {"data": "result1"}, execution_time=0.1),
            ToolResult("id2", {"data": "result2"}, error="Error message", execution_time=0.2)
        ]
        
        formatted = executor.format_tool_results(results)
        
        assert "Tool Result: id1" in formatted
        assert "result1" in formatted
        assert "Tool Result: id2" in formatted
        assert "Error" in formatted
    
    def test_parse_tool_calls_single(self):
        """Test parsing a single tool call"""
        executor = ToolExecutor(MockRegistry())
        
        text = '{"tool": "test_tool", "parameters": {"param1": "value1"}}'
        calls = executor.parse_tool_calls(text)
        
        assert len(calls) == 1
        assert calls[0].tool_name == "test_tool"
        assert calls[0].parameters == {"param1": "value1"}
    
    def test_parse_tool_calls_array(self):
        """Test parsing multiple tool calls"""
        executor = ToolExecutor(MockRegistry())
        
        text = '[{"tool": "tool1", "parameters": {"param1": "v1"}}, {"tool": "tool2", "parameters": {"param1": "v2"}}]'
        calls = executor.parse_tool_calls(text)
        
        assert len(calls) == 2
        assert calls[0].tool_name == "tool1"
        assert calls[1].tool_name == "tool2"
    
    def test_parse_tool_calls_no_matches(self):
        """Test parsing text with no tool calls"""
        executor = ToolExecutor(MockRegistry())
        
        text = "This is just regular text with no tool calls."
        calls = executor.parse_tool_calls(text)
        
        assert len(calls) == 0
    
    def test_parse_tool_calls_malformed_json(self):
        """Test parsing malformed JSON"""
        executor = ToolExecutor(MockRegistry())
        
        text = '{"tool": "test", "parameters": {invalid json}'
        calls = executor.parse_tool_calls(text)
        
        # Should handle gracefully and return empty or partial results
        assert isinstance(calls, list)

