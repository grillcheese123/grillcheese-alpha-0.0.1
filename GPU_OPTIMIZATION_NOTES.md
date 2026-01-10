"""
GPU Optimization Notes for VulkanEmbeddingTransformerRoPE
=========================================================

Current Issue: GPU path is ~2.4x SLOWER than CPU due to buffer allocation overhead.

Root Causes:
1. Creating/destroying GPU buffers for EVERY operation
2. Downloading intermediate results back to CPU between layers
3. Each vkCreateBuffer/vkFreeMemory is expensive
4. CPU-GPU sync points after every shader dispatch

Optimization Strategy:
----------------------

Phase 1: Persistent Buffers
- Pre-allocate all GPU buffers at init time based on max batch/seq size
- Reuse buffers across forward passes
- Use buffer pools for temporary storage

Phase 2: Keep Data on GPU
- Chain shader dispatches without CPU readback
- Only download final embedding result
- Use double-buffering for layer outputs

Phase 3: Fuse Operations
- Combine LayerNorm + Attention into single dispatch
- Combine FFN up/down projections
- Reduce total number of shader dispatches from ~20 per layer to ~4

Expected Performance After Optimization:
- Buffer allocation: 0ms (one-time at init)
- GPU forward: ~15-25ms (vs current 278ms)
- Speedup: 5-10x faster than CPU

Implementation Steps:
1. Add _allocate_persistent_buffers() method
2. Add forward_gpu_optimized() that reuses buffers
3. Create fused shader variants
4. Add warmup pass to ensure JIT compilation

Code Pattern for Persistent Buffers:
```python
def _allocate_persistent_buffers(self, max_batch=32, max_seq=512):
    cfg = self.config
    
    # Hidden state buffers (double buffer for layer chaining)
    hidden_size = max_batch * max_seq * cfg.hidden_dim * 4
    self._buf_hidden_a = self._create_buffer_persistent(hidden_size)
    self._buf_hidden_b = self._create_buffer_persistent(hidden_size)
    
    # Attention scratch
    qkv_size = max_batch * max_seq * 3 * cfg.hidden_dim * 4
    self._buf_qkv = self._create_buffer_persistent(qkv_size)
    
    # FFN intermediate
    inter_size = max_batch * max_seq * cfg.intermediate_dim * 4
    self._buf_ffn_inter = self._create_buffer_persistent(inter_size)
    
    # Flash attention running buffers
    running_size = max_batch * max_seq * cfg.num_heads * 4
    self._buf_attn_max = self._create_buffer_persistent(running_size)
    self._buf_attn_sum = self._create_buffer_persistent(running_size)
    
    # Output buffer
    output_size = max_batch * cfg.output_dim * 4
    self._buf_output = self._create_buffer_persistent(output_size)
```

For now, the CPU path is fast enough for single-text embeddings (~117ms).
GPU optimization should be prioritized for batch embedding workloads.
"""
