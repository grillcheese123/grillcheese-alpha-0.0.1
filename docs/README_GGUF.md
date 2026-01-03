# GGUF Model Setup for GPU Acceleration

## Quick Start

GGUF format provides **10-20x faster** inference than PyTorch on CPU by using GPU acceleration via Vulkan/CUDA.

### 1. Install llama-cpp-python

```bash
# Install with GPU support
pip install llama-cpp-python

# Or with uv:
uv add llama-cpp-python
```

### 2. Download GGUF Model

Download from HuggingFace:
```bash
# Using huggingface-cli (recommended)
pip install huggingface-hub
huggingface-cli download microsoft/Phi-3-mini-4k-instruct-gguf Phi-3-mini-4k-instruct-Q4_K_M.gguf --local-dir models/

# Or manually download from:
# https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
# Save to: grillcheese/backend/models/Phi-3-mini-4k-instruct-Q4_K_M.gguf
```

**Recommended quantization: Q4_K_M** (good quality/speed balance)

### 3. Use It!

The CLI will automatically detect and use the GGUF model:

```bash
python cli.py "Hello, how are you?"
```

You should see:
```
Using GGUF model: models/Phi-3-mini-4k-instruct-Q4_K_M.gguf
✓ GGUF model loaded (GPU accelerated)
```

## Performance

- **PyTorch (CPU)**: 2-3 minutes per generation
- **GGUF (GPU/Vulkan)**: 5-15 seconds per generation ⚡

That's **10-20x faster**!

## Model Sizes

| Quantization | Size | Quality | Speed |
|--------------|------|---------|-------|
| Q4_0 | ~2.3 GB | Good | Fastest |
| **Q4_K_M** | ~2.6 GB | **Better** | **Fast** ⭐ |
| Q5_K_M | ~3.1 GB | Great | Medium |
| Q8_0 | ~4.4 GB | Excellent | Slower |

**Recommendation: Q4_K_M** for best balance

## Troubleshooting

### Model not found
- Ensure model is in `models/` directory
- Check filename matches exactly (case-sensitive)
- Or specify path: `Phi3GGUF(model_path="path/to/model.gguf")`

### GPU not used
- llama-cpp-python should auto-detect GPU
- Check GPU drivers are up to date
- For AMD: Ensure Vulkan is installed

### Slow performance
- Make sure GPU acceleration is working
- Try a lower quantization (Q4_0) for faster inference
- Check that `n_gpu_layers=-1` is set (all layers on GPU)

