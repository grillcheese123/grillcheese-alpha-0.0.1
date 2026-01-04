"""
File Operations Tool

Reads and writes files (with safety restrictions).
"""
from typing import Any, Dict
from pathlib import Path
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.base import BaseTool
from config import BASE_DIR


class FileOperationsTool(BaseTool):
    """
    File operations tool for reading and writing files.
    
    Restricted to the project directory for safety.
    """
    
    # Allowed directories (relative to BASE_DIR)
    ALLOWED_DIRS = [
        "data",
        "brain_state",
        "learning_state"
    ]
    
    @property
    def name(self) -> str:
        """Tool name."""
        return "file_operations"
    
    @property
    def description(self) -> str:
        """Tool description."""
        return "Reads and writes files within allowed project directories. Use 'read' or 'write' operation."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """JSON schema for parameters."""
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
                    "description": "File path relative to allowed directories"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (required for 'write' operation)"
                }
            },
            "required": ["operation", "path"]
        }
    
    def _is_path_allowed(self, file_path: str) -> bool:
        """Check if file path is within allowed directories."""
        try:
            full_path = Path(BASE_DIR) / file_path
            resolved = full_path.resolve()
            base_resolved = BASE_DIR.resolve()
            
            # Check if path is within base directory
            if not str(resolved).startswith(str(base_resolved)):
                return False
            
            # Check if path is within an allowed subdirectory
            relative = resolved.relative_to(base_resolved)
            parts = relative.parts
            
            if len(parts) == 0:
                return False
            
            return parts[0] in self.ALLOWED_DIRS
        
        except Exception:
            return False
    
    async def execute(self, **kwargs: Any) -> Any:
        """
        Execute file operation.
        
        Args:
            operation: 'read' or 'write'
            path: File path
            content: Content to write (for 'write' operation)
            
        Returns:
            Operation result
        """
        operation = kwargs.get('operation')
        path = kwargs.get('path')
        content = kwargs.get('content')
        
        if not self._is_path_allowed(path):
            return {
                "error": f"Path '{path}' is not within allowed directories: {', '.join(self.ALLOWED_DIRS)}"
            }
        
        full_path = BASE_DIR / path
        
        if operation == "read":
            try:
                if not full_path.exists():
                    return {"error": f"File not found: {path}"}
                
                with open(full_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                
                return {
                    "operation": "read",
                    "path": path,
                    "content": file_content,
                    "size": len(file_content)
                }
            except Exception as e:
                return {
                    "operation": "read",
                    "path": path,
                    "error": str(e)
                }
        
        elif operation == "write":
            if content is None:
                return {"error": "Content is required for write operation"}
            
            try:
                # Create parent directories if needed
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                return {
                    "operation": "write",
                    "path": path,
                    "size": len(content),
                    "success": True
                }
            except Exception as e:
                return {
                    "operation": "write",
                    "path": path,
                    "error": str(e)
                }
        
        else:
            return {"error": f"Unknown operation: {operation}"}

