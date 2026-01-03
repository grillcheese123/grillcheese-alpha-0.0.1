# Windows Setup Guide

## Current Status: ✅ Working with PyTorch Model

The system is **fully functional** using the PyTorch model. It works immediately without any additional setup!

```bash
python cli.py "Hello, how are you?"
```

The system automatically:
1. Tries to load GGUF model (if available)
2. Falls back to PyTorch model (current setup)
3. Works on CPU (slower but functional)

## Performance

- **Current (PyTorch CPU):** 2-3 minutes per generation
- **Future (GGUF GPU):** 5-15 seconds per generation (after setup)

## Setting Up GGUF for GPU Acceleration (Optional)

GGUF requires compilation on Windows, which needs Visual Studio Build Tools.

### Quick Summary

1. **Now:** PyTorch model works immediately ✅
2. **Later:** Install build tools → Build llama-cpp-python → Download GGUF model

### Detailed Steps (For Future)

See `WINDOWS_INSTALL_NOTES.md` for complete instructions.

**TL;DR:**
- Install Visual Studio Build Tools (~6GB)
- Install CMake
- Build llama-cpp-python with Vulkan support
- Download GGUF model
- System automatically uses it!

## No Action Needed

**The system works right now** with PyTorch model. GGUF setup is optional for future GPU acceleration. The code automatically detects and uses GGUF when available - no code changes needed!

