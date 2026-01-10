"""
System Info Tool
Provides system information
"""
from typing import Dict, Any
import platform
import sys
import os

from modules.base import BaseTool


class SystemInfoTool(BaseTool):
    """
    Tool for getting system information
    """
    
    @property
    def name(self) -> str:
        return "system_info"
    
    @property
    def description(self) -> str:
        return "Provides information about the system, including OS, Python version, and environment details."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "info_type": {
                    "type": "string",
                    "enum": ["all", "os", "python", "environment"],
                    "description": "Type of information to retrieve"
                }
            },
            "required": []
        }
    
    def execute(self, **kwargs) -> Any:
        """
        Get system information
        
        Args:
            info_type: Type of information ('all', 'os', 'python', 'environment')
            
        Returns:
            System information dictionary
        """
        info_type = kwargs.get("info_type", "all")
        
        result = {}
        
        if info_type in ["all", "os"]:
            result["os"] = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor()
            }
        
        if info_type in ["all", "python"]:
            result["python"] = {
                "version": sys.version,
                "version_info": list(sys.version_info),
                "executable": sys.executable
            }
        
        if info_type in ["all", "environment"]:
            result["environment"] = {
                "cwd": os.getcwd(),
                "env_vars": {k: v for k, v in os.environ.items() if not k.startswith("_")}
            }
        
        return result
