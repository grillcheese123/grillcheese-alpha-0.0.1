"""
GrillCheese Plugin/Module System

Provides a flexible plugin architecture for extending GrillCheese with:
- Memory backends
- Model providers
- Processing hooks
- API extensions
- AI tools (MCP-style)
"""
from .base import (
    BaseMemoryBackend,
    BaseModelProvider,
    BaseProcessingHook,
    BaseAPIExtension,
    BaseTool
)

__all__ = [
    'BaseMemoryBackend',
    'BaseModelProvider',
    'BaseProcessingHook',
    'BaseAPIExtension',
    'BaseTool'
]

