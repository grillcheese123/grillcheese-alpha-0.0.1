"""
File Operations Tool
Reads and writes files
"""
from typing import Dict, Any
from pathlib import Path

from modules.base import BaseTool


class FileOperationsTool(BaseTool):
    """
    Tool for reading and writing files
    """
    
    @property
    def name(self) -> str:
        return "file_operations"
    
    @property
    def description(self) -> str:
        return "Reads and writes files. Use 'read' operation to read a file, 'write' operation to write a file."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write"],
                    "description": "Operation to perform: 'read' or 'write'"
                },
                "path": {
                    "type": "string",
                    "description": "File path (relative to current directory)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (required for 'write' operation)"
                }
            },
            "required": ["operation", "path"]
        }
    
    def validate_parameters(self, **kwargs) -> bool:
        """Validate parameters"""
        if not super().validate_parameters(**kwargs):
            return False
        
        operation = kwargs.get("operation")
        if operation == "write" and "content" not in kwargs:
            return False
        
        return True
    
    def execute(self, **kwargs) -> Any:
        """
        Execute file operation
        
        Args:
            operation: 'read' or 'write'
            path: File path
            content: Content to write (for 'write' operation)
            
        Returns:
            Operation result
        """
        operation = kwargs.get("operation")
        path_str = kwargs.get("path", "")
        
        if not path_str:
            return {"error": "No path provided"}
        
        try:
            path = Path(path_str)
            
            if operation == "read":
                if not path.exists():
                    return {"error": f"File not found: {path_str}"}
                
                content = path.read_text(encoding="utf-8")
                return {
                    "operation": "read",
                    "path": path_str,
                    "content": content,
                    "size": len(content)
                }
            
            elif operation == "write":
                content = kwargs.get("content", "")
                path.write_text(content, encoding="utf-8")
                return {
                    "operation": "write",
                    "path": path_str,
                    "size": len(content),
                    "success": True
                }
            
            else:
                return {"error": f"Unknown operation: {operation}"}
        
        except Exception as e:
            return {
                "error": str(e),
                "operation": operation,
                "path": path_str
            }
