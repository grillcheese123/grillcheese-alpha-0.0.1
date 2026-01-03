#!/usr/bin/env python3
"""Test if GGUF model can be loaded"""
import os

print("Testing GGUF model setup...")
print()

# Check if llama-cpp-python is installed
try:
    from llama_cpp import Llama
    print("[OK] llama-cpp-python is installed")
except ImportError as e:
    print(f"[ERROR] llama-cpp-python is NOT installed: {e}")
    print("\nTo install:")
    print("  1. Open 'Developer PowerShell for VS 2022'")
    print("  2. Run: $env:CMAKE_ARGS='-DLLAMA_VULKAN=on'; pip install llama-cpp-python")
    exit(1)

# Check if model file exists
model_paths = [
    "models/Phi-3-mini-4k-instruct-q4_K_M.gguf",
    "models/Phi-3-mini-4k-instruct-q4.gguf",
    "models/Phi-3-mini-4k-instruct-fp16.gguf",
]

model_path = None
for path in model_paths:
    if os.path.exists(path):
        model_path = path
        print(f"[OK] Model file found: {path}")
        size_gb = os.path.getsize(path) / (1024**3)
        print(f"     Size: {size_gb:.2f} GB")
        break

if not model_path:
    print("[ERROR] No GGUF model file found")
    print("\nTo download:")
    print("  python download_model.py")
    exit(1)

# Try to load the model
print(f"\nAttempting to load model...")
try:
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1,  # Try GPU first
        verbose=False,
    )
    print("[OK] Model loaded successfully!")
    print(f"     Device: GPU (if available, else CPU)")
except Exception as e:
    print(f"[WARNING] Failed to load with GPU, trying CPU...")
    try:
        llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_gpu_layers=0,  # CPU only
            verbose=False,
        )
        print("[OK] Model loaded on CPU")
    except Exception as e2:
        print(f"[ERROR] Failed to load model: {e2}")
        exit(1)

print("\n[SUCCESS] GGUF model is ready to use!")
print("You can now run: python cli.py 'Hello!'")

