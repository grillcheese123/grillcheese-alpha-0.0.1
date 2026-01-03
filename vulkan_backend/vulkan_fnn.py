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
        """Apply softmax activation"""
        data = input_data.astype(np.float32)
        original_shape = data.shape
        
        if axis == -1:
            axis = len(data.shape) - 1
        
        # Flatten for processing
        if data.ndim > 1:
            # Reshape to (batch, features) for 2D, or flatten for 1D
            if data.ndim == 2:
                batch_size, feature_dim = data.shape
                data_flat = data.flatten()
            else:
                # For higher dimensions, flatten last axis
                batch_size = int(np.prod(data.shape[:-1]))
                feature_dim = data.shape[-1]
                data_flat = data.reshape(-1, feature_dim).flatten()
        else:
            batch_size = 1
            feature_dim = len(data)
            data_flat = data.flatten()
        
        total_elements = len(data_flat)
        
        # Create buffers
        buf_in, mem_in = self.core._create_buffer(data_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(data_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_in, mem_in, data_flat)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-softmax', 2, push_constant_size=8
        )
        
        # Pack push constants: batch_size, feature_dim
        push_constants = struct.pack('II', batch_size, feature_dim)
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-softmax',
            [(buf_in, data_flat.nbytes), (buf_out, data_flat.nbytes)]
        )
        
        # Dispatch
        workgroups = (batch_size + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )
        
        # Download results
        result = self.core._download_buffer(mem_out, data_flat.nbytes, dtype=np.float32)
        result = result[:total_elements]
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_in, None)
        vkDestroyBuffer(self.core.device, buf_out, None)
        vkFreeMemory(self.core.device, mem_in, None)
        vkFreeMemory(self.core.device, mem_out, None)
        
        # Reshape to original shape
        return result.reshape(original_shape)

