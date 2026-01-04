"""
Tool execution framework for GrillCheese plugin system

Provides MCP-style tool calling capabilities for AI models.
"""
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseTool

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Represents a tool call request."""
    tool_name: str
    parameters: Dict[str, Any]
    call_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ToolResult:
    """Represents the result of a tool execution."""
    call_id: str
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0


class ToolExecutor:
    """
    Executes tools and manages tool calls.
    
    Provides MCP-style tool execution with validation and error handling.
    """
    
    def __init__(self, registry):
        """
        Initialize tool executor.
        
        Args:
            registry: ModuleRegistry instance for accessing tools
        """
        self.registry = registry
    
    async def execute_tool(self, name: str, **kwargs: Any) -> ToolResult:
        """
        Execute a single tool.
        
        Args:
            name: Tool name
            **kwargs: Tool parameters
            
        Returns:
            ToolResult with execution result or error
        """
        start_time = time.time()
        call_id = str(uuid.uuid4())
        
        # Get tool from registry
        tool = self.registry.get_tool(name)
        if tool is None:
            return ToolResult(
                call_id=call_id,
                result=None,
                error=f"Tool '{name}' not found",
                execution_time=time.time() - start_time
            )
        
        # Validate parameters
        if not tool.validate_parameters(**kwargs):
            return ToolResult(
                call_id=call_id,
                result=None,
                error=f"Invalid parameters for tool '{name}'",
                execution_time=time.time() - start_time
            )
        
        # Execute tool
        try:
            result = await tool.execute(**kwargs)
            execution_time = time.time() - start_time
            
            logger.debug(f"Tool '{name}' executed successfully in {execution_time:.3f}s")
            
            return ToolResult(
                call_id=call_id,
                result=result,
                error=None,
                execution_time=execution_time
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(f"Tool '{name}' execution error: {e}")
            
            return ToolResult(
                call_id=call_id,
                result=None,
                error=error_msg,
                execution_time=execution_time
            )
    
    async def execute_tool_chain(self, calls: List[ToolCall]) -> List[ToolResult]:
        """
        Execute multiple tools in sequence.
        
        Args:
            calls: List of ToolCall objects
            
        Returns:
            List of ToolResult objects in same order as calls
        """
        results = []
        
        for call in calls:
            result = await self.execute_tool(call.tool_name, **call.parameters)
            result.call_id = call.call_id  # Preserve original call ID
            results.append(result)
        
        return results
    
    def format_tool_results(self, results: List[ToolResult]) -> str:
        """
        Format tool results for injection into model prompt.
        
        Args:
            results: List of ToolResult objects
            
        Returns:
            Formatted string for model prompt
        """
        formatted = []
        
        for result in results:
            if result.error:
                formatted.append(
                    f"Tool Result: {result.call_id}\n"
                    f"Error: {result.error}\n"
                )
            else:
                # Convert result to JSON string
                try:
                    result_json = json.dumps(result.result, indent=2)
                except (TypeError, ValueError):
                    result_json = str(result.result)
                
                formatted.append(
                    f"Tool Result: {result.call_id}\n"
                    f"{result_json}\n"
                )
        
        return "\n".join(formatted)
    
    def parse_tool_calls(self, text: str) -> List[ToolCall]:
        """
        Parse tool calls from model output.
        
        Looks for JSON objects with 'tool' and 'parameters' keys.
        Supports both single objects and arrays.
        
        Args:
            text: Model output text
            
        Returns:
            List of ToolCall objects
        """
        tool_calls = []
        
        # Try to find JSON objects in the text
        # Look for patterns like {"tool": "...", "parameters": {...}}
        import re
        
        # Pattern for JSON objects
        json_pattern = r'\{[^{}]*"tool"[^{}]*\{[^{}]*\}[^{}]*\}|\[[^\[\]]*\{[^{}]*"tool"[^{}]*\{[^{}]*\}[^{}]*\}[^\[\]]*\]'
        
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                data = json.loads(match)
                
                # Handle array of tool calls
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and 'tool' in item:
                            tool_calls.append(ToolCall(
                                tool_name=item['tool'],
                                parameters=item.get('parameters', {})
                            ))
                
                # Handle single tool call
                elif isinstance(data, dict) and 'tool' in data:
                    tool_calls.append(ToolCall(
                        tool_name=data['tool'],
                        parameters=data.get('parameters', {})
                    ))
            
            except json.JSONDecodeError:
                # Try to extract tool name and parameters more flexibly
                # This is a fallback for non-standard formats
                continue
        
        # Also try to find tool calls in a more structured format
        # Look for lines like: Tool: name, Parameters: {...}
        lines = text.split('\n')
        current_tool = None
        current_params = {}
        
        for line in lines:
            if 'tool:' in line.lower() or '"tool"' in line.lower():
                # Extract tool name
                try:
                    if current_tool:
                        tool_calls.append(ToolCall(
                            tool_name=current_tool,
                            parameters=current_params
                        ))
                    current_tool = None
                    current_params = {}
                    
                    # Try to parse JSON from this line
                    json_match = re.search(r'\{[^{}]*\}', line)
                    if json_match:
                        data = json.loads(json_match.group())
                        if 'tool' in data:
                            current_tool = data['tool']
                            current_params = data.get('parameters', {})
                except:
                    pass
        
        if current_tool:
            tool_calls.append(ToolCall(
                tool_name=current_tool,
                parameters=current_params
            ))
        
        return tool_calls

