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
    
    def xavier_init(self, input_dim: int, output_dim: int, seed: int = 42) -> np.ndarray:
        """
        GPU-accelerated Xavier initialization
        
        Generates weights from normal distribution scaled by sqrt(2.0 / input_dim)
        Uses shader: fnn-xavier-init.glsl
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            seed: Random seed for reproducibility
            
        Returns:
            Weight matrix (output_dim, input_dim) with Xavier initialization
        """
        scale = np.sqrt(2.0 / input_dim)
        weights_flat = np.zeros(input_dim * output_dim, dtype=np.float32)
        
        # Create output buffer
        buf_weights, mem_weights = self.core._create_buffer(weights_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Check if shader is available
        if 'fnn-xavier-init' not in self.core.shaders:
            raise RuntimeError("fnn-xavier-init shader not compiled. Run: glslc shaders/fnn-xavier-init.glsl -o shaders/spv/fnn-xavier-init.spv")
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'fnn-xavier-init', 1, push_constant_size=16
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'fnn-xavier-init',
            [(buf_weights, weights_flat.nbytes)]
        )
        
        # Pack push constants: input_dim, output_dim, scale, seed
        push_constants = struct.pack('IIfI', input_dim, output_dim, scale, seed)
        
        # Dispatch: 2D workgroups (one thread per weight)
        workgroups_x = (input_dim + 15) // 16
        workgroups_y = (output_dim + 15) // 16
        
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y, 1
        )
        
        # Download results
        result = self.core._download_buffer(mem_weights, weights_flat.nbytes, dtype=np.float32)
        result = result[:input_dim * output_dim]
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_weights, None)
        vkFreeMemory(self.core.device, mem_weights, None)
        
        return result.reshape(output_dim, input_dim)
    
    def activation_gelu_backward(self, grad_output, input_data):
        """
        GPU-accelerated GELU backward pass
        
        Args:
            grad_output: Gradient from next layer (same shape as input_data)
            input_data: Input to GELU (for computing derivative)
        
        Returns:
            Gradient w.r.t. input
        """
        grad_out = grad_output.astype(np.float32).flatten()
        input_flat = input_data.astype(np.float32).flatten()
        total_elements = len(input_flat)
        
        if len(grad_out) != total_elements:
            raise ValueError(f"grad_output size {len(grad_out)} != input_data size {total_elements}")
        
        # Create buffers
        buf_grad_out, mem_grad_out = self.core._create_buffer(grad_out.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_input, mem_input = self.core._create_buffer(input_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_grad_in, mem_grad_in = self.core._create_buffer(input_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_grad_out, mem_grad_out, grad_out)
        self.core._upload_buffer(buf_input, mem_input, input_flat)
        
        # Check if shader is available
        if 'activation-gelu-backward' not in self.shaders:
            # CPU fallback
            sqrt_2_over_pi = 0.7978845608028654
            coeff = 0.044715
            grad_in = np.zeros_like(input_flat)
            for i in range(total_elements):
                x = input_flat[i]
                x_cubed = x * x * x
                z = sqrt_2_over_pi * (x + coeff * x_cubed)
                tanh_z = np.tanh(z)
                sech_z = 1.0 / np.cosh(z)
                sech_sq = sech_z * sech_z
                dz_dx = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x)
                gelu_grad = 0.5 * (1.0 + tanh_z + x * sech_sq * dz_dx)
                grad_in[i] = grad_out[i] * gelu_grad
            return grad_in.reshape(input_data.shape)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'activation-gelu-backward', 3, push_constant_size=4
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'activation-gelu-backward',
            [
                (buf_grad_out, grad_out.nbytes),
                (buf_input, input_flat.nbytes),
                (buf_grad_in, input_flat.nbytes)
            ]
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
        result = self.core._download_buffer(mem_grad_in, input_flat.nbytes, dtype=np.float32)
        result = result[:total_elements].reshape(input_data.shape)
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_grad_out, None)
        vkDestroyBuffer(self.core.device, buf_input, None)
        vkDestroyBuffer(self.core.device, buf_grad_in, None)
        vkFreeMemory(self.core.device, mem_grad_out, None)
        vkFreeMemory(self.core.device, mem_input, None)
        vkFreeMemory(self.core.device, mem_grad_in, None)
        
        return result

