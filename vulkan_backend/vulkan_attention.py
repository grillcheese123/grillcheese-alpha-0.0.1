"""
Attention operations for Vulkan backend.
GPU-accelerated attention mechanisms for transformers.
"""

import numpy as np
import struct
from .base import VULKAN_AVAILABLE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT

if VULKAN_AVAILABLE:
    from vulkan import *


class VulkanAttention:
    """Attention operations: scores, mask, output, concat heads"""
    
    def __init__(self, core, pipelines, shaders):
        self.core = core
        self.pipelines = pipelines
        self.shaders = shaders
    
    def attention_scores(self, queries, keys, num_heads, head_dim, scale=None):
        """
        Compute attention scores: Q @ K^T / sqrt(head_dim)
        
        Args:
            queries: Query tensor (batch, seq_len, num_heads, head_dim) or (batch, seq_len, num_heads * head_dim)
            keys: Key tensor (batch, seq_len, num_heads, head_dim) or (batch, seq_len, num_heads * head_dim)
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            scale: Optional scaling factor (default: 1/sqrt(head_dim))
        
        Returns:
            Attention scores (batch, num_heads, seq_len, seq_len)
        """
        q = queries.astype(np.float32)
        k = keys.astype(np.float32)
        
        if scale is None:
            scale = 1.0 / np.sqrt(head_dim)
        
        # Handle flattened head dimension
        if q.ndim == 3:
            batch_size, seq_len, _ = q.shape
            q = q.reshape(batch_size, seq_len, num_heads, head_dim)
            k = k.reshape(batch_size, seq_len, num_heads, head_dim)
        else:
            batch_size, seq_len, num_heads, head_dim = q.shape
        
        q_flat = q.flatten()
        k_flat = k.flatten()
        
        # Create buffers
        buf_q, mem_q = self.core._create_buffer(q_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_k, mem_k = self.core._create_buffer(k_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_scores, mem_scores = self.core._create_buffer(batch_size * num_heads * seq_len * seq_len * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_q, mem_q, q_flat)
        self.core._upload_buffer(buf_k, mem_k, k_flat)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'attention-scores', 4, push_constant_size=24
        )
        
        # Create dummy V buffer (required by shader)
        buf_v_dummy, mem_v_dummy = self.core._create_buffer(q_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        self.core._upload_buffer(buf_v_dummy, mem_v_dummy, q_flat)
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'attention-scores',
            [
                (buf_q, q_flat.nbytes),
                (buf_k, k_flat.nbytes),
                (buf_v_dummy, q_flat.nbytes),
                (buf_scores, batch_size * num_heads * seq_len * seq_len * 4)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('IIIIfI', batch_size, seq_len, num_heads, head_dim, scale, 0)
        
        # Dispatch
        workgroups_x = (seq_len + 15) // 16
        workgroups_y = ((batch_size * num_heads * seq_len) + 15) // 16
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups_x, push_constants, workgroups_y
        )
        
        # Download results
        result = self.core._download_buffer(mem_scores, batch_size * num_heads * seq_len * seq_len * 4, dtype=np.float32)
        result = result[:batch_size * num_heads * seq_len * seq_len].reshape(batch_size, num_heads, seq_len, seq_len)
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_q, None)
        vkDestroyBuffer(self.core.device, buf_k, None)
        vkDestroyBuffer(self.core.device, buf_v_dummy, None)
        vkDestroyBuffer(self.core.device, buf_scores, None)
        vkFreeMemory(self.core.device, mem_q, None)
        vkFreeMemory(self.core.device, mem_k, None)
        vkFreeMemory(self.core.device, mem_v_dummy, None)
        vkFreeMemory(self.core.device, mem_scores, None)
        
        return result
    
    def attention_mask(self, attention_scores, use_causal=True, mask_value=-1e9):
        """
        Apply causal mask to attention scores
        
        Args:
            attention_scores: Attention scores (batch, num_heads, seq_len, seq_len)
            use_causal: Whether to apply causal masking
            mask_value: Value to use for masked positions
        
        Returns:
            Masked attention scores
        """
        scores = attention_scores.astype(np.float32)
        batch_size, num_heads, seq_len, _ = scores.shape
        
        scores_flat = scores.flatten()
        
        # Create causal mask (1 = allow, 0 = mask)
        mask = np.ones((seq_len, seq_len), dtype=np.float32)
        if use_causal:
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    mask[i, j] = 0.0
        
        # Create buffers
        buf_scores, mem_scores = self.core._create_buffer(scores_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_mask, mem_mask = self.core._create_buffer(seq_len * seq_len * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_scores, mem_scores, scores_flat)
        self.core._upload_buffer(buf_mask, mem_mask, mask.flatten())
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'attention-mask', 2, push_constant_size=20
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'attention-mask',
            [
                (buf_scores, scores_flat.nbytes),
                (buf_mask, seq_len * seq_len * 4)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('IIIIf', batch_size, num_heads, seq_len, 1 if use_causal else 0, mask_value)
        
        # Dispatch
        workgroups = (len(scores_flat) + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )
        
        # Download results
        result = self.core._download_buffer(mem_scores, scores_flat.nbytes, dtype=np.float32)
        result = result[:len(scores_flat)].reshape(batch_size, num_heads, seq_len, seq_len)
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_scores, None)
        vkDestroyBuffer(self.core.device, buf_mask, None)
        vkFreeMemory(self.core.device, mem_scores, None)
        vkFreeMemory(self.core.device, mem_mask, None)
        
        return result
    
    def attention_output(self, attention_weights, values, num_heads, head_dim):
        """
        Compute attention output: weights @ values
        
        Args:
            attention_weights: Attention weights (batch, num_heads, seq_len, seq_len)
            values: Value tensor (batch, seq_len, num_heads, head_dim)
            num_heads: Number of attention heads
            head_dim: Dimension of each head
        
        Returns:
            Attention output (batch, seq_len, num_heads, head_dim)
        """
        weights = attention_weights.astype(np.float32)
        v = values.astype(np.float32)
        
        batch_size, num_heads_w, seq_len, _ = weights.shape
        
        if v.ndim == 3:
            v = v.reshape(batch_size, seq_len, num_heads, head_dim)
        
        weights_flat = weights.flatten()
        v_flat = v.flatten()
        
        output_size = batch_size * seq_len * num_heads * head_dim * 4
        
        # Create buffers
        buf_weights, mem_weights = self.core._create_buffer(weights_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_v, mem_v = self.core._create_buffer(v_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(output_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_weights, mem_weights, weights_flat)
        self.core._upload_buffer(buf_v, mem_v, v_flat)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'attention-output', 3, push_constant_size=16
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'attention-output',
            [
                (buf_weights, weights_flat.nbytes),
                (buf_v, v_flat.nbytes),
                (buf_out, output_size)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('IIII', batch_size, seq_len, num_heads, head_dim)
        
        # Dispatch
        workgroups = ((batch_size * seq_len * num_heads * head_dim) + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )
        
        # Download results
        result = self.core._download_buffer(mem_out, output_size, dtype=np.float32)
        result = result[:batch_size * seq_len * num_heads * head_dim].reshape(batch_size, seq_len, num_heads, head_dim)
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_weights, None)
        vkDestroyBuffer(self.core.device, buf_v, None)
        vkDestroyBuffer(self.core.device, buf_out, None)
        vkFreeMemory(self.core.device, mem_weights, None)
        vkFreeMemory(self.core.device, mem_v, None)
        vkFreeMemory(self.core.device, mem_out, None)
        
        return result
    
    def attention_concat_heads(self, attention_output):
        """
        Concatenate attention heads
        
        Args:
            attention_output: Attention output (batch, seq_len, num_heads, head_dim)
        
        Returns:
            Concatenated output (batch, seq_len, num_heads * head_dim)
        """
        output = attention_output.astype(np.float32)
        batch_size, seq_len, num_heads, head_dim = output.shape
        
        output_flat = output.flatten()
        concat_size = batch_size * seq_len * num_heads * head_dim * 4
        
        # Create buffers
        buf_in, mem_in = self.core._create_buffer(output_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_out, mem_out = self.core._create_buffer(concat_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload data
        self.core._upload_buffer(buf_in, mem_in, output_flat)
        
        # Get or create pipeline
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'attention-concat-heads', 2, push_constant_size=16
        )
        
        # Get cached descriptor set
        descriptor_set = self.pipelines.get_cached_descriptor_set(
            'attention-concat-heads',
            [
                (buf_in, output_flat.nbytes),
                (buf_out, concat_size)
            ]
        )
        
        # Pack push constants
        push_constants = struct.pack('IIII', batch_size, seq_len, num_heads, head_dim)
        
        # Dispatch
        workgroups = ((batch_size * seq_len * num_heads * head_dim) + 255) // 256
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set,
            workgroups, push_constants
        )
        
        # Download results
        result = self.core._download_buffer(mem_out, concat_size, dtype=np.float32)
        result = result[:batch_size * seq_len * num_heads * head_dim].reshape(batch_size, seq_len, num_heads * head_dim)
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_in, None)
        vkDestroyBuffer(self.core.device, buf_out, None)
        vkFreeMemory(self.core.device, mem_in, None)
        vkFreeMemory(self.core.device, mem_out, None)
        
        return result

