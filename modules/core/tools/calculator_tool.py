"""
Calculator Tool

Performs mathematical calculations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from typing import Any, Dict
import math

from modules.base import BaseTool


class CalculatorTool(BaseTool):
    """
    Calculator tool for performing mathematical operations.
    """
    
    @property
    def name(self) -> str:
        """Tool name."""
        return "calculator"
    
    @property
    def description(self) -> str:
        """Tool description."""
        return "Performs mathematical calculations. Supports basic arithmetic and common math functions."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        """JSON schema for parameters."""
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'sin(pi/2)')"
                }
            },
            "required": ["expression"]
        }
    
    async def execute(self, **kwargs: Any) -> Any:
        """
        Execute calculator tool.
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Calculation result
        """
        expression = kwargs.get('expression', '')
        
        # Safe evaluation with limited functions
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({
            'abs': abs,
            'round': round,
            'min': min,
            'max': max,
            'sum': sum,
            'pow': pow
        })
        
        try:
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return {
                "expression": expression,
                "result": result,
                "type": type(result).__name__
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e)
            }

