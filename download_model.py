#!/usr/bin/env python3
"""Download Phi-3-mini GGUF model from Hugging Face"""
import os
from huggingface_hub import hf_hub_download, list_repo_files

def main():
    repo_id = "microsoft/Phi-3-mini-4k-instruct-gguf"
    
    print(f"Checking available files in {repo_id}...")
    files = list(list_repo_files(repo_id))
    gguf_files = [f for f in files if f.endswith('.gguf')]
    
    print("\nAvailable GGUF files:")
    for f in sorted(gguf_files):
        print(f"  - {f}")
    
    # Prefer Q4_K_M, fallback to q4, then fp16
    preferred_files = [
        "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
        "Phi-3-mini-4k-instruct-q4_K_M.gguf",
        "Phi-3-mini-4k-instruct-q4.gguf",
        "Phi-3-mini-4k-instruct-fp16.gguf"
    ]
    
    filename = None
    for pref in preferred_files:
        if pref in gguf_files:
            filename = pref
            break
    
    if not filename:
        # Use first available
        filename = gguf_files[0] if gguf_files else None
    
    if not filename:
        print("\n[ERROR] No GGUF files found in repository")
        return
    
    print(f"\nDownloading: {filename}")
    print("This may take a few minutes (file is ~2-3 GB)...")
    
    os.makedirs("models", exist_ok=True)
    
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir="models",
            local_dir_use_symlinks=False
        )
        print(f"\n[OK] Downloaded successfully!")
        print(f"  Location: {file_path}")
        print(f"\nThe model is ready to use!")
        print(f"  Filename: {filename}")
        
        # Rename if needed to match expected names
        if filename == "Phi-3-mini-4k-instruct-q4.gguf":
            # Create a copy with expected name for compatibility
            import shutil
            expected_path = "models/Phi-3-mini-4k-instruct-q4_K_M.gguf"
            if not os.path.exists(expected_path):
                shutil.copy(file_path, expected_path)
                print(f"  Also created: {expected_path} (for compatibility)")
        
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        return

if __name__ == "__main__":
    main()

