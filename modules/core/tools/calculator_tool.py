"""
Calculator Tool
Performs mathematical calculations
"""
from typing import Dict, Any
import math

from modules.base import BaseTool


class CalculatorTool(BaseTool):
    """
    Tool for performing mathematical calculations
    """
    
    @property
    def name(self) -> str:
        return "calculator"
    
    @property
    def description(self) -> str:
        return "Performs mathematical calculations. Supports basic arithmetic, trigonometry, logarithms, and more."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sin(pi/2)', 'log(10)')"
                }
            },
            "required": ["expression"]
        }
    
    def execute(self, **kwargs) -> Any:
        """
        Execute calculation
        
        Args:
            expression: Mathematical expression to evaluate
            
        Returns:
            Calculation result
        """
        expression = kwargs.get("expression", "")
        
        if not expression:
            return {"error": "No expression provided"}
        
        try:
            # Safe evaluation with limited functions
            allowed_names = {
                k: v for k, v in math.__dict__.items() if not k.startswith("__")
            }
            # Add common constants
            allowed_names["pi"] = math.pi
            allowed_names["e"] = math.e
            
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return {
                "expression": expression,
                "result": result
            }
        except Exception as e:
            return {
                "expression": expression,
                "error": str(e)
            }
