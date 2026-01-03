"""
Feedforward Neural Network operations for Vulkan backend.
GPU-accelerated FNN operations: activations, layer normalization, linear layers, dropout.
"""

import numpy as np
import struct
from .base import VULKAN_AVAILABLE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanFNN:
    """FNN operations: activations, layer normalization, linear layers, dropout"""
    
    def __init__(self, core, pipelines, shaders):
        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders
    
    def activation_relu(self, input_data):
        """Apply ReLU activation: max(0, x)"""
        data = input_data.astype(np.float32).flatten()
        total_elements = len(data)
        
        # Create buffers
        buf_in, mem_in = self.core._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_in, mem_in, data)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-relu', 2, push_constant_size=4
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-relu',
            [(buf_in, data.nbytes), (buf_out, data.nbytes)]
        )
        
        # Pack push constants
        push_constants = struct.pack('I', total_elements)
        
        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )
        
        # Download results
        result = self.core._download_buffer(mem_out, data.nbytes, dtype=np.float32)
        result = result[:total_elements]
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_in, None)
        vkDestroyBuffer(self.core.device, buf_out, None)
        vkFreeMemory(self.core.device, mem_in, None)
        vkFreeMemory(self.core.device, mem_out, None)
        
        return result.reshape(input_data.shape) if input_data.ndim > 1 else result
    
    def activation_gelu(self, input_data):
        """Apply GELU activation"""
        data = input_data.astype(np.float32).flatten()
        total_elements = len(data)
        
        # Create buffers
        buf_in, mem_in = self.core._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_in, mem_in, data)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-gelu', 2, push_constant_size=4
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-gelu',
            [(buf_in, data.nbytes), (buf_out, data.nbytes)]
        )
        
        # Pack push constants
        push_constants = struct.pack('I', total_elements)
        
        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )
        
        # Download results
        result = self.core._download_buffer(mem_out, data.nbytes, dtype=np.float32)
        result = result[:total_elements]
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_in, None)
        vkDestroyBuffer(self.core.device, buf_out, None)
        vkFreeMemory(self.core.device, mem_in, None)
        vkFreeMemory(self.core.device, mem_out, None)
        
        return result.reshape(input_data.shape) if input_data.ndim > 1 else result
    
    def activation_silu(self, input_data):
        """Apply SiLU (Swish) activation: x * sigmoid(x)"""
        data = input_data.astype(np.float32).flatten()
        total_elements = len(data)
        
        # Create buffers
        buf_in, mem_in = self.core._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(data.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_in, mem_in, data)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-silu', 2, push_constant_size=4
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-silu',
            [(buf_in, data.nbytes), (buf_out, data.nbytes)]
        )
        
        # Pack push constants
        push_constants = struct.pack('I', total_elements)
        
        # Dispatch
        workgroups = (total_elements + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )
        
        # Download results
        result = self.core._download_buffer(mem_out, data.nbytes, dtype=np.float32)
        result = result[:total_elements]
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_in, None)
        vkDestroyBuffer(self.core.device, buf_out, None)
        vkFreeMemory(self.core.device, mem_in, None)
        vkFreeMemory(self.core.device, mem_out, None)
        
        return result.reshape(input_data.shape) if input_data.ndim > 1 else result
    
    def activation_softmax(self, input_data, axis=-1):
        """
        Apply softmax activation: exp(x) / sum(exp(x))
        
        Args:
            input_data: Input array
            axis: Axis along which to compute softmax (default: -1)
        
        Returns:
            Softmax probabilities
        """
        data = input_data.astype(np.float32)
        original_shape = data.shape
        
        # Handle different input shapes - shader expects (batch, seq_len, features)
        if data.ndim == 1:
            batch_size, seq_len, features = 1, 1, len(data)
            data = data.reshape(1, 1, -1)
        elif data.ndim == 2:
            batch_size, seq_len, features = data.shape[0], 1, data.shape[1]
            data = data.reshape(data.shape[0], 1, -1)
        else:
            batch_size, seq_len, features = data.shape
        
        data_flat = data.flatten()
        
        # Create buffers - shader needs 4 buffers: input, output, max_vals, sum_exp
        buf_in, mem_in = self.core._create_buffer(data_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(data_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_max, mem_max = self.core._create_buffer(batch_size * seq_len * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_sum, mem_sum = self.core._create_buffer(batch_size * seq_len * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_in, mem_in, data_flat)
        
        # Get or create pipeline - 4 buffers, 24 bytes push constants (5 uints + padding)
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-softmax', 4, push_constant_size=24
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-softmax',
            [
                (buf_in, data_flat.nbytes),
                (buf_out, data_flat.nbytes),
                (buf_max, batch_size * seq_len * 4),
                (buf_sum, batch_size * seq_len * 4)
            ]
        )
        
        # Pass 1: Compute max for numerical stability
        push_constants = struct.pack('IIIII', batch_size, seq_len, features, 0, features)
        workgroups = ((batch_size * seq_len) + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )
        
        # Pass 2: Compute sum of exponentials
        push_constants = struct.pack('IIIII', batch_size, seq_len, features, 1, features)
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )
        
        # Pass 3: Normalize
        push_constants = struct.pack('IIIII', batch_size, seq_len, features, 2, features)
        workgroups = (len(data_flat) + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )
        
        # Download results
        result = self.core._download_buffer(mem_out, data_flat.nbytes, dtype=np.float32)
        result = result[:len(data_flat)].reshape(original_shape)
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_in, None)
        vkDestroyBuffer(self.core.device, buf_out, None)
        vkDestroyBuffer(self.core.device, buf_max, None)
        vkDestroyBuffer(self.core.device, buf_sum, None)
        vkFreeMemory(self.core.device, mem_in, None)
        vkFreeMemory(self.core.device, mem_out, None)
        vkFreeMemory(self.core.device, mem_max, None)
        vkFreeMemory(self.core.device, mem_sum, None)
        
        return result

