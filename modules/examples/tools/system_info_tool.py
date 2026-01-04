"""
System Info Tool

Provides system information.
"""
from typing import Any, Dict
import platform
import sys
import os
from pathlib import Path

from modules.base import BaseTool
from config import BASE_DIR


class SystemInfoTool(BaseTool):
    """
    System information tool.
    """
    
    @property
    def name(self) -> str:
        """Tool name."""
        return "system_info"
    
    @property
    def description(self) -> str:
        """Tool description."""
        return "Provides system information including OS, Python version, and project directory status."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """JSON schema for parameters."""
        return {
            "type": "object",
            "properties": {
                "info_type": {
                    "type": "string",
                    "enum": ["all", "os", "python", "project"],
                    "description": "Type of information to retrieve"
                }
            },
            "required": []
        }
    
    async def execute(self, **kwargs: Any) -> Any:
        """
        Execute system info tool.
        
        Args:
            info_type: Type of information ('all', 'os', 'python', 'project')
            
        Returns:
            System information dictionary
        """
        info_type = kwargs.get('info_type', 'all')
        result = {}
        
        if info_type in ('all', 'os'):
            result['os'] = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            }
        
        if info_type in ('all', 'python'):
            result['python'] = {
                "version": sys.version,
                "version_info": {
                    "major": sys.version_info.major,
                    "minor": sys.version_info.minor,
                    "micro": sys.version_info.micro
                },
                "executable": sys.executable,
                "platform": sys.platform
            }
        
        if info_type in ('all', 'project'):
            try:
                project_info = {
                    "base_dir": str(BASE_DIR),
                    "exists": BASE_DIR.exists()
                }
                
                # Check for key directories
                key_dirs = ['models', 'shaders', 'data', 'brain_state', 'modules']
                project_info['directories'] = {}
                for dir_name in key_dirs:
                    dir_path = BASE_DIR / dir_name
                    project_info['directories'][dir_name] = {
                        "exists": dir_path.exists(),
                        "is_dir": dir_path.is_dir() if dir_path.exists() else False
                    }
                
                result['project'] = project_info
            except Exception as e:
                result['project'] = {"error": str(e)}
        
        return result

