"""
Spiking Neural Network (SNN) operations for Vulkan backend.
"""

import numpy as np
import struct
from .base import VULKAN_AVAILABLE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanSNN:
    """SNN operations: LIF neurons and learning rules"""
    
    def __init__(self, core, pipelines):
        """Initialize with VulkanCore and VulkanPipelines instances"""
        self.core = core
        self.pipelines = pipelines
    
    def lif_step(self, input_current, membrane, refractory, 
                 dt=0.001, tau_mem=20.0, v_thresh=1.0):
        """Run LIF shader on GPU"""
        n = len(input_current)
        
        # Create buffers
        buf_in, mem_in = self.core._create_buffer(
            input_current.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_mem, mem_mem = self.core._create_buffer(
            membrane.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_ref, mem_ref = self.core._create_buffer(
            refractory.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        buf_out, mem_out = self.core._create_buffer(
            n * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        
        # Upload data
        self.core._upload_buffer(buf_in, mem_in, input_current.astype(np.float32))
        self.core._upload_buffer(buf_mem, mem_mem, membrane.astype(np.float32))
        self.core._upload_buffer(buf_ref, mem_ref, refractory.astype(np.float32))
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'lif-neuron', 4, push_constant_size=32
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [
                (buf_in, input_current.nbytes),
                (buf_mem, membrane.nbytes),
                (buf_ref, refractory.nbytes),
                (buf_out, n * 4)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack(
            'Ifffffff',
            n, dt, tau_mem, 0.0, 0.0, v_thresh, 1.0, 2.0
        )
        
        # Dispatch
        workgroups = (n + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )
        
        # Download results
        membrane_out = self.core._download_buffer(mem_mem, membrane.nbytes)
        refractory_out = self.core._download_buffer(mem_ref, refractory.nbytes)
        spikes_out = self.core._download_buffer(mem_out, n * 4)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_in, None)
        vkDestroyBuffer(self.core.device, buf_mem, None)
        vkDestroyBuffer(self.core.device, buf_ref, None)
        vkDestroyBuffer(self.core.device, buf_out, None)
        vkFreeMemory(self.core.device, mem_in, None)
        vkFreeMemory(self.core.device, mem_mem, None)
        vkFreeMemory(self.core.device, mem_ref, None)
        vkFreeMemory(self.core.device, mem_out, None)
        
        return membrane_out, refractory_out, spikes_out
    
    def hebbian_learning(self, pre_activations, post_activations, weights,
                         learning_rate=0.01, weight_decay=0.0):
        """
        Apply Hebbian learning rule: ΔW = η * <pre * post> - λ * W
        """
        batch_size, time_steps, pre_dim = pre_activations.shape
        _, _, post_dim = post_activations.shape
        
        # Flatten activations
        pre_flat = pre_activations.astype(np.float32).flatten()
        post_flat = post_activations.astype(np.float32).flatten()
        weights_flat = weights.astype(np.float32).flatten()
        
        # Create buffers
        buf_pre, mem_pre = self.core._create_buffer(pre_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_post, mem_post = self.core._create_buffer(post_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_weights, mem_weights = self.core._create_buffer(weights_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_pre, mem_pre, pre_flat)
        self.core._upload_buffer(buf_post, mem_post, post_flat)
        self.core._upload_buffer(buf_weights, mem_weights, weights_flat)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'hebbian-learning', 3, push_constant_size=32
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [
                (buf_pre, pre_flat.nbytes),
                (buf_post, post_flat.nbytes),
                (buf_weights, weights_flat.nbytes)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack(
            'IIIIff', batch_size, time_steps, pre_dim, post_dim, learning_rate, weight_decay
        )
        
        # Dispatch
        workgroups_x = (pre_dim + 15) // 16
        workgroups_y = (post_dim + 15) // 16
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y, 1
        )
        
        # Download results
        weights_out = self.core._download_buffer(mem_weights, weights_flat.nbytes, dtype=np.float32)
        weights_out = weights_out[:post_dim * pre_dim].reshape(post_dim, pre_dim)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_pre, None)
        vkDestroyBuffer(self.core.device, buf_post, None)
        vkDestroyBuffer(self.core.device, buf_weights, None)
        vkFreeMemory(self.core.device, mem_pre, None)
        vkFreeMemory(self.core.device, mem_post, None)
        vkFreeMemory(self.core.device, mem_weights, None)
        
        return weights_out
    
    def stdp_learning(self, pre_activations, post_activations, weights,
                      pre_trace, post_trace,
                      lr_potentiation=0.01, lr_depression=0.01, trace_decay=0.9):
        """
        Apply STDP learning rule with eligibility traces
        """
        batch_size, time_steps, pre_dim = pre_activations.shape
        _, _, post_dim = post_activations.shape
        
        # Flatten activations
        pre_flat = pre_activations.astype(np.float32).flatten()
        post_flat = post_activations.astype(np.float32).flatten()
        weights_flat = weights.astype(np.float32).flatten()
        pre_trace_flat = pre_trace.astype(np.float32).flatten()
        post_trace_flat = post_trace.astype(np.float32).flatten()
        
        # Create buffers
        buf_pre, mem_pre = self.core._create_buffer(pre_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_post, mem_post = self.core._create_buffer(post_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_weights, mem_weights = self.core._create_buffer(weights_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_pre_trace, mem_pre_trace = self.core._create_buffer(pre_trace_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_post_trace, mem_post_trace = self.core._create_buffer(post_trace_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_pre, mem_pre, pre_flat)
        self.core._upload_buffer(buf_post, mem_post, post_flat)
        self.core._upload_buffer(buf_weights, mem_weights, weights_flat)
        self.core._upload_buffer(buf_pre_trace, mem_pre_trace, pre_trace_flat)
        self.core._upload_buffer(buf_post_trace, mem_post_trace, post_trace_flat)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'stdp-learning', 5, push_constant_size=32
        )
        
        # Create descriptor set
        descriptor_set = self.pipelines._create_descriptor_set(
            desc_layout,
            [
                (buf_pre, pre_flat.nbytes),
                (buf_post, post_flat.nbytes),
                (buf_weights, weights_flat.nbytes),
                (buf_pre_trace, pre_trace_flat.nbytes),
                (buf_post_trace, post_trace_flat.nbytes)
            ]
        )
        
        # Pass 1: Update traces
        push_constants = struct.pack(
            'IIIIfffI', batch_size, time_steps, pre_dim, post_dim,
            lr_potentiation, lr_depression, trace_decay, 0
        )
        workgroups_x = (max(pre_dim, post_dim) + 15) // 16
        workgroups_y = (batch_size + 15) // 16
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y, 1
        )
        
        # Download updated traces
        pre_trace_out = self.core._download_buffer(mem_pre_trace, pre_trace_flat.nbytes, dtype=np.float32)
        post_trace_out = self.core._download_buffer(mem_post_trace, post_trace_flat.nbytes, dtype=np.float32)
        pre_trace_out = pre_trace_out[:batch_size * pre_dim].reshape(batch_size, pre_dim)
        post_trace_out = post_trace_out[:batch_size * post_dim].reshape(batch_size, post_dim)
        
        # Upload updated traces back
        self.core._upload_buffer(buf_pre_trace, mem_pre_trace, pre_trace_out.flatten())
        self.core._upload_buffer(buf_post_trace, mem_post_trace, post_trace_out.flatten())
        
        # Pass 2: Update weights
        push_constants = struct.pack(
            'IIIIfffI', batch_size, time_steps, pre_dim, post_dim,
            lr_potentiation, lr_depression, trace_decay, 1
        )
        workgroups_x = (pre_dim + 15) // 16
        workgroups_y = (post_dim + 15) // 16
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y, 1
        )
        
        # Download results
        weights_out = self.core._download_buffer(mem_weights, weights_flat.nbytes, dtype=np.float32)
        weights_out = weights_out[:post_dim * pre_dim].reshape(post_dim, pre_dim)
        pre_trace_out = self.core._download_buffer(mem_pre_trace, pre_trace_flat.nbytes, dtype=np.float32)
        post_trace_out = self.core._download_buffer(mem_post_trace, post_trace_flat.nbytes, dtype=np.float32)
        pre_trace_out = pre_trace_out[:batch_size * pre_dim].reshape(batch_size, pre_dim)
        post_trace_out = post_trace_out[:batch_size * post_dim].reshape(batch_size, post_dim)
        
        # Free descriptor set
        vkFreeDescriptorSets(self.core.device, self.core.descriptor_pool, 1, [descriptor_set])
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_pre, None)
        vkDestroyBuffer(self.core.device, buf_post, None)
        vkDestroyBuffer(self.core.device, buf_weights, None)
        vkDestroyBuffer(self.core.device, buf_pre_trace, None)
        vkDestroyBuffer(self.core.device, buf_post_trace, None)
        vkFreeMemory(self.core.device, mem_pre, None)
        vkFreeMemory(self.core.device, mem_post, None)
        vkFreeMemory(self.core.device, mem_weights, None)
        vkFreeMemory(self.core.device, mem_pre_trace, None)
        vkFreeMemory(self.core.device, mem_post_trace, None)
        
        return weights_out, pre_trace_out, post_trace_out

