"""
Prompt Enhancer Processing Hook

Example processing hook that enhances prompts before generation.
"""
from typing import Dict, Any

from modules.base import BaseProcessingHook


class PromptEnhancerHook(BaseProcessingHook):
    """
    Example processing hook that enhances user prompts.
    
    Adds helpful context and formatting to prompts.
    """
    
    async def pre_process(self, prompt: str, context: Dict[str, Any]) -> str:
        """
        Enhance the prompt before generation.
        
        Args:
            prompt: Original user prompt
            context: Additional context
            
        Returns:
            Enhanced prompt
        """
        # Simple enhancement: ensure prompt ends properly
        if not prompt.strip().endswith(('.', '!', '?', ':')):
            prompt = prompt.strip() + '.'
        
        return prompt
    
    async def post_process(self, response_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process the response data.
        
        Args:
            response_data: Response data dictionary
            context: Additional context
            
        Returns:
            Modified response data
        """
        # No modification needed for this example
        return response_data
    
    async def on_error(self, error: Exception, context: Dict[str, Any]) -> None:
        """
        Handle errors during processing.
        
        Args:
            error: Exception that occurred
            context: Additional context
        """
        # Log error (handled by main system)
        pass

