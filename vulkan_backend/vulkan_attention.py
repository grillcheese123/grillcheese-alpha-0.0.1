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
    
    def flash_attention2(
        self,
        queries,
        keys,
        values,
        num_heads,
        head_dim,
        tile_size_q=64,
        tile_size_k=64,
        scale=None,
        mask=None,
        causal=False
    ):
        """
        Flash Attention 2: Tiled attention with online softmax
        
        Processes attention in blocks to reduce memory from O(N²) to O(N).
        Uses online softmax algorithm for numerical stability.
        Optimized for GPU tiling support.
        
        Args:
            queries: Query tensor (batch, seq_len, num_heads, head_dim) or (batch, seq_len, num_heads * head_dim)
            keys: Key tensor (batch, seq_len, num_heads, head_dim) or (batch, seq_len, num_heads * head_dim)
            values: Value tensor (batch, seq_len, num_heads, head_dim) or (batch, seq_len, num_heads * head_dim)
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            tile_size_q: Tile size for query dimension (default: 64, optimal for most GPUs)
            tile_size_k: Tile size for key dimension (default: 64, optimal for most GPUs)
            scale: Optional scaling factor (default: 1/sqrt(head_dim))
            mask: Optional attention mask (batch, seq_len) - 0.0 = mask out, 1.0 = keep
            causal: Whether to apply causal masking (default: False)
        
        Returns:
            Attention output (batch, seq_len, num_heads, head_dim)
        """
        q = queries.astype(np.float32)
        k = keys.astype(np.float32)
        v = values.astype(np.float32)
        
        if scale is None:
            scale = 1.0 / np.sqrt(head_dim)
        
        # Handle flattened head dimension
        if q.ndim == 3:
            batch_size, seq_len, _ = q.shape
            q = q.reshape(batch_size, seq_len, num_heads, head_dim)
            k = k.reshape(batch_size, seq_len, num_heads, head_dim)
            v = v.reshape(batch_size, seq_len, num_heads, head_dim)
        else:
            batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Create causal mask if requested
        if causal and mask is None:
            mask = np.ones((batch_size, seq_len), dtype=np.float32)
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    mask[:, j] = 0.0
        
        # Flatten tensors
        q_flat = q.flatten()
        k_flat = k.flatten()
        v_flat = v.flatten()
        
        # Calculate number of tiles
        num_tiles_q = (seq_len + tile_size_q - 1) // tile_size_q
        num_tiles_k = (seq_len + tile_size_k - 1) // tile_size_k
        
        # Temporary buffers for online softmax
        num_q_positions = batch_size * seq_len * num_heads
        running_max_size = num_q_positions * 4  # float32
        running_sum_size = num_q_positions * 4
        output_accum_size = batch_size * seq_len * num_heads * head_dim * 4
        
        # Create buffers
        buf_q, mem_q = self.core._create_buffer(q_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_k, mem_k = self.core._create_buffer(k_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_v, mem_v = self.core._create_buffer(v_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_running_max, mem_running_max = self.core._create_buffer(running_max_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_running_sum, mem_running_sum = self.core._create_buffer(running_sum_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_output_accum, mem_output_accum = self.core._create_buffer(output_accum_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        buf_output, mem_output = self.core._create_buffer(output_accum_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
        
        # Upload Q, K, V
        self.core._upload_buffer(buf_q, mem_q, q_flat)
        self.core._upload_buffer(buf_k, mem_k, k_flat)
        self.core._upload_buffer(buf_v, mem_v, v_flat)
        
        # Handle mask buffer
        buf_mask = None
        mem_mask = None
        mask_flat = None
        if mask is not None:
            mask_flat = mask.astype(np.float32).flatten()
            buf_mask, mem_mask = self.core._create_buffer(mask_flat.nbytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)
            self.core._upload_buffer(buf_mask, mem_mask, mask_flat)
        
        # Check if shader is available
        if 'flash-attention2' not in self.shaders:
            raise RuntimeError(
                "flash-attention2 shader not compiled. "
                "Run: glslc -fshader-stage=compute shaders/flash-attention2.glsl -o shaders/spv/flash-attention2.spv"
            )
        
        # Get or create pipeline
        num_bindings = 8  # Q, K, V, mask, output, running_max, running_sum, output_accum
        pipeline, pipeline_layout, desc_layout = self.pipelines.get_or_create_pipeline(
            'flash-attention2', num_bindings, push_constant_size=44  # 11 uint/float values
        )
        
        # Pass 0: Initialize running max, sum, and accumulator
        descriptor_set_init = self.pipelines.get_cached_descriptor_set(
            'flash-attention2',
            [
                (buf_q, q_flat.nbytes),
                (buf_k, k_flat.nbytes),
                (buf_v, v_flat.nbytes),
                (buf_mask if mask is not None else buf_q, mask_flat.nbytes if mask is not None else q_flat.nbytes),
                (buf_output, output_accum_size),
                (buf_running_max, running_max_size),
                (buf_running_sum, running_sum_size),
                (buf_output_accum, output_accum_size)
            ]
        )
        
        push_constants_init = struct.pack(
            'IIIIfIIIII',
            batch_size, seq_len, num_heads, head_dim,
            scale,
            tile_size_q, tile_size_k,
            0,  # pass_type = 0 (initialize)
            1 if mask is not None else 0,  # has_mask
            0, 0  # q_tile_idx, k_tile_idx (not used in init)
        )
        
        # Dispatch initialization
        workgroups_init_x = 16
        workgroups_init_y = (num_q_positions + 15) // 16
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set_init,
            workgroups_init_x, push_constants_init, workgroups_init_y
        )
        
        # Pass 1: Process all tiles
        for q_tile in range(num_tiles_q):
            for k_tile in range(num_tiles_k):
                descriptor_set_tile = self.pipelines.get_cached_descriptor_set(
                    'flash-attention2',
                    [
                        (buf_q, q_flat.nbytes),
                        (buf_k, k_flat.nbytes),
                        (buf_v, v_flat.nbytes),
                        (buf_mask if mask is not None else buf_q, mask_flat.nbytes if mask is not None else q_flat.nbytes),
                        (buf_output, output_accum_size),
                        (buf_running_max, running_max_size),
                        (buf_running_sum, running_sum_size),
                        (buf_output_accum, output_accum_size)
                    ]
                )
                
                push_constants_tile = struct.pack(
                    'IIIIfIIIII',
                    batch_size, seq_len, num_heads, head_dim,
                    scale,
                    tile_size_q, tile_size_k,
                    1,  # pass_type = 1 (process tile)
                    1 if mask is not None else 0,  # has_mask
                    q_tile, k_tile
                )
                
                # Dispatch tile processing
                workgroups_tile_x = (tile_size_k + 15) // 16
                workgroups_tile_y = (batch_size * num_heads * tile_size_q + 15) // 16
                self.core._dispatch_compute(
                    pipeline, pipeline_layout, descriptor_set_tile,
                    workgroups_tile_x, push_constants_tile, workgroups_tile_y
                )
        
        # Pass 2: Finalize output
        descriptor_set_final = self.pipelines.get_cached_descriptor_set(
            'flash-attention2',
            [
                (buf_q, q_flat.nbytes),
                (buf_k, k_flat.nbytes),
                (buf_v, v_flat.nbytes),
                (buf_mask if mask is not None else buf_q, mask_flat.nbytes if mask is not None else q_flat.nbytes),
                (buf_output, output_accum_size),
                (buf_running_max, running_max_size),
                (buf_running_sum, running_sum_size),
                (buf_output_accum, output_accum_size)
            ]
        )
        
        push_constants_final = struct.pack(
            'IIIIfIIIII',
            batch_size, seq_len, num_heads, head_dim,
            scale,
            tile_size_q, tile_size_k,
            2,  # pass_type = 2 (finalize)
            1 if mask is not None else 0,  # has_mask
            0, 0  # q_tile_idx, k_tile_idx (not used in finalize)
        )
        
        # Dispatch finalization
        workgroups_final_x = (head_dim + 15) // 16
        workgroups_final_y = (num_q_positions + 15) // 16
        self.core._dispatch_compute(
            pipeline, pipeline_layout, descriptor_set_final,
            workgroups_final_x, push_constants_final, workgroups_final_y
        )
        
        # Download results
        result = self.core._download_buffer(mem_output, output_accum_size, dtype=np.float32)
        result = result[:batch_size * seq_len * num_heads * head_dim].reshape(
            batch_size, seq_len, num_heads, head_dim
        )
        
        # Cleanup
        vkDestroyBuffer(self.core.device, buf_q, None)
        vkDestroyBuffer(self.core.device, buf_k, None)
        vkDestroyBuffer(self.core.device, buf_v, None)
        vkDestroyBuffer(self.core.device, buf_running_max, None)
        vkDestroyBuffer(self.core.device, buf_running_sum, None)
        vkDestroyBuffer(self.core.device, buf_output_accum, None)
        vkDestroyBuffer(self.core.device, buf_output, None)
        vkFreeMemory(self.core.device, mem_q, None)
        vkFreeMemory(self.core.device, mem_k, None)
        vkFreeMemory(self.core.device, mem_v, None)
        vkFreeMemory(self.core.device, mem_running_max, None)
        vkFreeMemory(self.core.device, mem_running_sum, None)
        vkFreeMemory(self.core.device, mem_output_accum, None)
        vkFreeMemory(self.core.device, mem_output, None)
        
        if mask is not None:
            vkDestroyBuffer(self.core.device, buf_mask, None)
            vkFreeMemory(self.core.device, mem_mask, None)
        
        return result


