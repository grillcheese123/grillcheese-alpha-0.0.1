# Vulkan Backend Module Structure

This module has been refactored from a single 2720-line file into a modular structure with files under 500 lines each.

## Module Structure

- `base.py` - Constants and Vulkan availability check
- `vulkan_core.py` - Core Vulkan initialization, buffer management, and dispatch
- `vulkan_pipelines.py` - Pipeline creation and descriptor set management
- `vulkan_snn.py` - SNN operations (LIF, Hebbian, STDP)
- `vulkan_faiss.py` - FAISS operations (distance computation, top-k selection) ✅
- `vulkan_fnn.py` - FNN operations (linear, layernorm, activations, dropout) - TODO
- `vulkan_attention.py` - Attention operations - TODO
- `vulkan_memory.py` - Memory operations - TODO
- `vulkan_cells.py` - Place/time cell operations - TODO
- `vulkan_compute.py` - Main class that composes all modules
- `snn_compute.py` - High-level SNN interface

## Usage

The API remains the same:

```python
from vulkan_backend import VulkanCompute, SNNCompute

# Create compute backend
backend = VulkanCompute()

# Use operations
membrane, refractory, spikes = backend.lif_step(...)
```

## FAISS Integration

The FAISS module (`vulkan_faiss.py`) provides GPU-accelerated vector similarity search:
- `compute_distances()` - Compute pairwise distances (L2, cosine, dot product)
- `topk()` - Select top-k smallest distances

These are exposed through `VulkanCompute` as:
- `faiss_compute_distances()` 
- `faiss_topk()`

The `MemoryStore` class automatically uses FAISS when GPU is available.

## Adding Remaining Modules

The remaining modules (FNN, Attention, Memory, Cells) need to be created following the same pattern as `vulkan_snn.py`:

1. Create a class that takes `core` and `pipelines` in `__init__`
2. Use `self.core._create_buffer()`, `self.core._upload_buffer()`, etc.
3. Use `self.pipelines.get_or_create_pipeline()` for pipeline management
4. Use `self.pipelines._create_descriptor_set()` for descriptor sets
5. Delegate methods in `vulkan_compute.py` to the module instances

