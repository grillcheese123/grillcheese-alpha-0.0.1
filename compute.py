"""
Compute module - exports SNNCompute for SNN operations
The HybridCompute class has been removed as it was just a passthrough wrapper.
"""
from vulkan_backend import SNNCompute

# Re-export SNNCompute for backwards compatibility
__all__ = ['SNNCompute']
