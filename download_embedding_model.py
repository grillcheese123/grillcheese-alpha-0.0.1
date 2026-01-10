#!/usr/bin/env python3
"""
Download GGUF embedding models for Vulkan-accelerated embeddings.

This downloads lightweight embedding models that can run on the same
Vulkan backend as your main Phi-3 model, eliminating the CPU bottleneck.
"""
import os
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Available embedding models (repo_id, filename, dims, description)
EMBEDDING_MODELS = {
    "nomic-768": (
        "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "nomic-embed-text-v1.5.Q4_K_M.gguf",
        768,
        "Best quality, 768 dims. Requires reindexing existing memories."
    ),
    "nomic-384": (
        "nomic-ai/nomic-embed-text-v1.5-GGUF",
        "nomic-embed-text-v1.5.Q4_K_M.gguf",  # Same model, we project down
        768,
        "High quality, projects to 384 dims for compatibility."
    ),
    "bge-small": (
        "second-state/BGE-Small-EN-v1.5-GGUF",
        "bge-small-en-v1.5-q4_k_m.gguf",
        384,
        "Lightweight, 384 dims. Compatible with existing memories."
    ),
    "all-minilm": (
        "leliuga/all-MiniLM-L6-v2-GGUF",
        "all-MiniLM-L6-v2.Q4_K_M.gguf",
        384,
        "Ultra-light, 384 dims. Fast but lower quality."
    ),
}

DEFAULT_MODEL = "bge-small"  # 384 dims for compatibility


def download_embedding_model(model_key: str = DEFAULT_MODEL, models_dir: str = "models"):
    """
    Download a GGUF embedding model.
    
    Args:
        model_key: Key from EMBEDDING_MODELS dict
        models_dir: Directory to save model
    """
    if not HF_AVAILABLE:
        print("[X] huggingface_hub not installed.")
        print("    pip install huggingface_hub")
        return False
    
    if model_key not in EMBEDDING_MODELS:
        print(f"[X] Unknown model: {model_key}")
        print(f"    Available: {list(EMBEDDING_MODELS.keys())}")
        return False
    
    repo_id, filename, dims, description = EMBEDDING_MODELS[model_key]
    
    models_path = Path(models_dir)
    models_path.mkdir(exist_ok=True)
    
    dest_path = models_path / filename
    
    if dest_path.exists():
        print(f"[OK] Model already exists: {dest_path}")
        return True
    
    print(f"Downloading {model_key} embedding model...")
    print(f"  Repo: {repo_id}")
    print(f"  File: {filename}")
    print(f"  Dims: {dims}")
    print(f"  Note: {description}")
    print()
    
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(models_path),
            local_dir_use_symlinks=False
        )
        print(f"\n[OK] Downloaded to: {downloaded_path}")
        return True
    except Exception as e:
        print(f"\n[X] Download failed: {e}")
        return False


def list_models():
    """List available embedding models"""
    print("=" * 60)
    print("Available GGUF Embedding Models")
    print("=" * 60)
    
    for key, (repo, filename, dims, desc) in EMBEDDING_MODELS.items():
        compat = "✓ compatible" if dims == 384 else "⚠ requires reindex"
        print(f"\n  {key}:")
        print(f"    Dimensions: {dims} ({compat})")
        print(f"    Description: {desc}")
    
    print("\n" + "=" * 60)
    print(f"Default: {DEFAULT_MODEL}")
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download GGUF embedding models")
    parser.add_argument(
        "model",
        nargs="?",
        default=DEFAULT_MODEL,
        help=f"Model to download (default: {DEFAULT_MODEL})"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--dir",
        default="models",
        help="Models directory (default: models)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    success = download_embedding_model(args.model, args.dir)
    
    if success:
        print("\n" + "=" * 60)
        print("NEXT STEPS")
        print("=" * 60)
        print("1. Run benchmark to verify Vulkan embeddings:")
        print("   python benchmark.py")
        print()
        print("2. Embeddings will automatically use Vulkan backend")
        print("   when model_gguf.py initializes.")
        print()
        if EMBEDDING_MODELS[args.model][2] != 384:
            print("3. IMPORTANT: Your existing memories use 384 dims.")
            print("   You may need to reindex or project embeddings.")
        print("=" * 60)


if __name__ == "__main__":
    main()
