"""
Tool execution framework for MCP-style function calling
Supports tool calling during model generation
"""
import json
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.base import BaseTool
    from modules.registry import ModuleRegistry

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call request"""
    tool_name: str
    parameters: Dict[str, Any]
    call_id: str = ""
    
    def __post_init__(self):
        if not self.call_id:
            import uuid
            self.call_id = str(uuid.uuid4())


@dataclass
class ToolResult:
    """Represents a tool execution result"""
    call_id: str
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "call_id": self.call_id,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time
        }


class ToolExecutor:
    """
    Executes tools and manages tool calls
    
    Supports both sync and async tool execution
    """
    
    def __init__(self, registry: 'ModuleRegistry'):
        """
        Initialize tool executor
        
        Args:
            registry: ModuleRegistry instance for accessing tools
        """
        self.registry = registry
    
    def execute_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> ToolResult:
        """
        Execute a single tool
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            ToolResult with execution result
        """
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return ToolResult(
                call_id="",
                result=None,
                error=f"Tool '{tool_name}' not found"
            )
        
        # Ensure tool has registry if it needs it (e.g., MemoryQueryTool)
        if hasattr(tool, '__class__') and tool.__class__.__name__ == 'MemoryQueryTool':
            if not hasattr(tool, 'registry') or tool.registry is None:
                tool.registry = self.registry
        
        # Validate parameters
        if not tool.validate_parameters(**kwargs):
            return ToolResult(
                call_id="",
                result=None,
                error=f"Invalid parameters for tool '{tool_name}'"
            )
        
        # Execute tool
        call_id = ToolCall(tool_name=tool_name, parameters=kwargs).call_id
        start_time = time.time()
        
        try:
            result = tool.execute(**kwargs)
            execution_time = time.time() - start_time
            
            return ToolResult(
                call_id=call_id,
                result=result,
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool execution error: {e}", exc_info=True)
            
            return ToolResult(
                call_id=call_id,
                result=None,
                error=str(e),
                execution_time=execution_time
            )
    
    async def execute_tool_async(
        self,
        tool_name: str,
        **kwargs
    ) -> ToolResult:
        """
        Execute a tool asynchronously
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Tool parameters
            
        Returns:
            ToolResult with execution result
        """
        tool = self.registry.get_tool(tool_name)
        if not tool:
            return ToolResult(
                call_id="",
                result=None,
                error=f"Tool '{tool_name}' not found"
            )
        
        # Validate parameters
        if not tool.validate_parameters(**kwargs):
            return ToolResult(
                call_id="",
                result=None,
                error=f"Invalid parameters for tool '{tool_name}'"
            )
        
        # Execute tool (async if available)
        call_id = ToolCall(tool_name=tool_name, parameters=kwargs).call_id
        start_time = time.time()
        
        try:
            if hasattr(tool, 'execute_async'):
                result = await tool.execute_async(**kwargs)
            else:
                import asyncio
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: tool.execute(**kwargs))
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                call_id=call_id,
                result=result,
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Tool execution error: {e}", exc_info=True)
            
            return ToolResult(
                call_id=call_id,
                result=None,
                error=str(e),
                execution_time=execution_time
            )
    
    def execute_tool_chain(
        self,
        calls: List[ToolCall]
    ) -> List[ToolResult]:
        """
        Execute multiple tools sequentially
        
        Args:
            calls: List of ToolCall instances
            
        Returns:
            List of ToolResult instances
        """
        results = []
        for call in calls:
            result = self.execute_tool(call.tool_name, **call.parameters)
            results.append(result)
        return results
    
    async def execute_tool_chain_async(
        self,
        calls: List[ToolCall]
    ) -> List[ToolResult]:
        """
        Execute multiple tools concurrently (async)
        
        Args:
            calls: List of ToolCall instances
            
        Returns:
            List of ToolResult instances
        """
        import asyncio
        
        tasks = [
            self.execute_tool_async(call.tool_name, **call.parameters)
            for call in calls
        ]
        
        results = await asyncio.gather(*tasks)
        return list(results)
    
    @staticmethod
    def parse_tool_calls(text: str) -> List[ToolCall]:
        """
        Parse tool calls from model output
        
        Supports JSON format:
        - Single call: {"tool": "name", "parameters": {...}}
        - Multiple calls: [{"tool": "name1", ...}, {"tool": "name2", ...}]
        
        Args:
            text: Model output text (may contain JSON tool calls)
            
        Returns:
            List of ToolCall instances
        """
        calls = []
        
        # Try to find JSON in text
        # Look for JSON objects/arrays - improved to handle nested braces
        import re
        
        # More robust pattern: find JSON objects that contain "tool" key
        # This handles cases where JSON is mixed with other text
        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*"tool"(?:[^{}]|(?:\{[^{}]*\}))*\}'
        json_array_pattern = r'\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\{(?:[^{}]|(?:\{[^{}]*\}))*"tool"(?:[^{}]|(?:\{[^{}]*\}))*\}(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]'
        
        # Try array first (multiple calls)
        matches = re.findall(json_array_pattern, text, re.DOTALL)
        if matches:
            for match in matches:
                try:
                    data = json.loads(match)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and "tool" in item:
                                calls.append(ToolCall(
                                    tool_name=item["tool"],
                                    parameters=item.get("parameters", {})
                                ))
                        if calls:
                            return calls
                except json.JSONDecodeError:
                    continue
        
        # Try single object - use a more greedy approach to find complete JSON
        # First try to find JSON objects with balanced braces
        brace_count = 0
        start_idx = -1
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    # Found a complete JSON object
                    json_str = text[start_idx:i+1]
                    if '"tool"' in json_str:
                        try:
                            data = json.loads(json_str)
                            if isinstance(data, dict) and "tool" in data:
                                calls.append(ToolCall(
                                    tool_name=data["tool"],
                                    parameters=data.get("parameters", {})
                                ))
                                # Found one, return it (or continue to find multiple)
                                if calls:
                                    return calls
                        except json.JSONDecodeError:
                            pass
                    start_idx = -1
        
        # Handle incomplete JSON (missing closing brace) - try to fix it
        # Look for JSON that starts with {"tool" but doesn't have balanced braces
        if not calls and '"tool"' in text:
            # Find the start of a JSON object with "tool"
            tool_start = text.find('{"tool"')
            if tool_start != -1:
                # Extract from start to end of text (or reasonable limit)
                potential_json = text[tool_start:]
                # Try to find where parameters end
                params_start = potential_json.find('"parameters"')
                if params_start != -1:
                    # Find the opening brace of parameters
                    params_brace = potential_json.find('{', params_start)
                    if params_brace != -1:
                        # Count braces to find where parameters should end
                        brace_count = 0
                        for i, char in enumerate(potential_json[params_brace:], start=params_brace):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    # Found end of parameters, add closing brace for main object
                                    fixed_json = potential_json[:i+1] + '}'
                                    try:
                                        data = json.loads(fixed_json)
                                        if isinstance(data, dict) and "tool" in data:
                                            calls.append(ToolCall(
                                                tool_name=data["tool"],
                                                parameters=data.get("parameters", {})
                                            ))
                                            if calls:
                                                return calls
                                    except json.JSONDecodeError:
                                        pass
                                    break
        
        # Fallback: try regex pattern (less reliable but catches edge cases)
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                if isinstance(data, dict) and "tool" in data:
                    # Check if we already have this call
                    existing = any(c.tool_name == data["tool"] and c.parameters == data.get("parameters", {}) for c in calls)
                    if not existing:
                        calls.append(ToolCall(
                            tool_name=data["tool"],
                            parameters=data.get("parameters", {})
                        ))
            except json.JSONDecodeError:
                continue
        
        return calls
    
    @staticmethod
    def format_tool_result(tool_name: str, result: ToolResult) -> str:
        """
        Format tool result for injection into model prompt
        
        Args:
            tool_name: Name of the tool
            result: ToolResult instance
            
        Returns:
            Formatted string for model context
        """
        if result.error:
            return f"Tool Result: {tool_name}\nError: {result.error}\n\n"
        
        try:
            result_json = json.dumps(result.result, indent=2)
            return f"Tool Result: {tool_name}\n{result_json}\n\n"
        except (TypeError, ValueError):
            # Fallback to string representation
            return f"Tool Result: {tool_name}\n{str(result.result)}\n\n"
