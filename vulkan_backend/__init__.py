"""
Vulkan compute backend module for GPU-accelerated neural network operations.

This module provides GPU acceleration for:
- Spiking Neural Networks (SNN)
- Feedforward Neural Networks (FNN)
- Attention mechanisms
- Memory operations
- FAISS similarity search
- Place and time cells
"""

from vulkan_backend.base import VULKAN_AVAILABLE
from vulkan_backend.vulkan_compute import VulkanCompute
from vulkan_backend.snn_compute import SNNCompute

__all__ = ['VULKAN_AVAILABLE', 'VulkanCompute', 'SNNCompute']

