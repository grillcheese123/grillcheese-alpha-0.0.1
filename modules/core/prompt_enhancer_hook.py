"""
Prompt Enhancement Processing Hook
Example hook that enhances prompts before generation
"""
from typing import Dict, Any

from modules.base import BaseProcessingHook


class PromptEnhancerHook(BaseProcessingHook):
    """
    Example processing hook that enhances prompts
    """
    
    @property
    def name(self) -> str:
        return "prompt_enhancer"
    
    @property
    def priority(self) -> int:
        return 10
    
    async def pre_process(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Enhance prompt with additional context
        
        Args:
            prompt: Original prompt
            context: Request context
            
        Returns:
            Enhanced prompt
        """
        # Simple enhancement: add helpful prefix
        enhanced = f"Please provide a helpful and accurate response to the following:\n\n{prompt}"
        return enhanced
    
    async def post_process(
        self,
        response_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Post-process response (optional)
        
        Args:
            response_data: Response dictionary
            context: Request context
            
        Returns:
            Modified response dictionary
        """
        # No modification by default
        return response_data
