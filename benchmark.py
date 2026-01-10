#!/usr/bin/env python3
"""
GrillCheese Inference Benchmark
Quick benchmark for inference speed on your hardware
"""
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import find_gguf_model


def benchmark_embeddings():
    """Benchmark embedding backends specifically"""
    print("\n" + "=" * 60)
    print("EMBEDDING BACKEND BENCHMARK")
    print("=" * 60)
    
    try:
        from vulkan_embeddings import benchmark_embedders
        results = benchmark_embedders()
        return results
    except ImportError as e:
        print(f"[!] vulkan_embeddings not available: {e}")
        return {}


def main():
    print("=" * 60)
    print("GrillCheese Inference Benchmark")
    print("=" * 60)
    
    # Check for GGUF model
    model_path = find_gguf_model()
    if not model_path:
        print("[X] No GGUF model found. Download one first:")
        print("    python download_model.py")
        sys.exit(1)
    
    print(f"\n[1/5] Loading GGUF model: {model_path}")
    start = time.time()
    
    try:
        from model_gguf import Phi3GGUF
        model = Phi3GGUF(model_path=model_path, n_gpu_layers=-1)
        load_time = time.time() - start
        print(f"      Loaded in {load_time:.1f}s")
    except Exception as e:
        print(f"[X] Failed to load model: {e}")
        sys.exit(1)
    
    # Embedding benchmark
    print("\n[2/5] Embedding Speed...")
    texts = ["Hello world", "What is machine learning?", "Tell me a story about AI."]
    
    # Warmup
    model.get_embedding("warmup")
    
    times = []
    for text in texts:
        start = time.time()
        emb = model.get_embedding(text)
        times.append(time.time() - start)
    
    avg_emb_ms = sum(times) / len(times) * 1000
    print(f"      Avg: {avg_emb_ms:.2f} ms | Dim: {len(emb)}")
    
    # Generation benchmark - short
    print("\n[3/5] Generation Speed (short)...")
    prompts = ["Hi", "What is 2+2?", "Hello"]
    
    # Warmup
    model.generate("hi", [])
    
    total_tokens = 0
    total_time = 0
    for prompt in prompts:
        start = time.time()
        response = model.generate(prompt, [])
        elapsed = time.time() - start
        tokens = len(response) / 4  # Rough estimate
        total_tokens += tokens
        total_time += elapsed
    
    short_tps = total_tokens / total_time if total_time > 0 else 0
    print(f"      {total_tokens:.0f} tokens in {total_time:.1f}s = {short_tps:.1f} tokens/sec")
    
    # Generation benchmark - with context
    print("\n[4/5] Generation Speed (with context)...")
    prompt = "What should I work on today?"
    context = [
        "You are GrillCheese, a helpful AI assistant.",
        "The user is interested in machine learning.",
        "Previously discussed Python programming.",
    ]
    
    times = []
    token_counts = []
    for _ in range(3):
        start = time.time()
        response = model.generate(prompt, context)
        elapsed = time.time() - start
        times.append(elapsed)
        token_counts.append(len(response) / 4)
    
    avg_time = sum(times) / len(times)
    avg_tokens = sum(token_counts) / len(token_counts)
    context_tps = avg_tokens / avg_time if avg_time > 0 else 0
    print(f"      {avg_tokens:.0f} tokens in {avg_time:.1f}s = {context_tps:.1f} tokens/sec")
    
    # Full pipeline benchmark
    print("\n[5/5] Full Pipeline (embed + store + retrieve + generate + SNN)...")
    
    try:
        import tempfile
        import os
        import numpy as np
        from memory_store import MemoryStore
        from vulkan_backend import SNNCompute
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "bench.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=model.embedding_dim)
            snn = SNNCompute(n_neurons=1000, use_vulkan=True)
            
            # Add some memories
            for i in range(10):
                emb = model.get_embedding(f"Memory about topic {i}")
                memory.store(emb, f"Memory {i}")
            
            # Benchmark full pipeline
            prompt = "Tell me about the project"
            
            start = time.time()
            emb = model.get_embedding(prompt)
            t_emb = time.time() - start
            
            start = time.time()
            memory.store(emb, prompt)
            t_store = time.time() - start
            
            start = time.time()
            context = memory.retrieve(emb, k=3)
            t_retrieve = time.time() - start
            
            start = time.time()
            response = model.generate(prompt, context)
            t_gen = time.time() - start
            
            start = time.time()
            spikes = snn.process(emb)
            t_snn = time.time() - start
            
            total = t_emb + t_store + t_retrieve + t_gen + t_snn
            gen_tokens = len(response) / 4
            
            print(f"      Embed: {t_emb*1000:.1f}ms | Store: {t_store*1000:.1f}ms | "
                  f"Retrieve: {t_retrieve*1000:.1f}ms")
            print(f"      Generate: {t_gen:.2f}s ({gen_tokens:.0f} tokens) | SNN: {t_snn*1000:.1f}ms")
            print(f"      Total: {total:.2f}s")
            
    except Exception as e:
        print(f"      [!] Skipped: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Embedding extraction:     {avg_emb_ms:.1f} ms")
    print(f"  Generation (short):       {short_tps:.1f} tokens/sec")
    print(f"  Generation (w/ context):  {context_tps:.1f} tokens/sec")
    print("=" * 60)
    
    # Recommendations
    if short_tps > 50:
        print("\n[OK] Great performance! GPU acceleration working well.")
    elif short_tps > 20:
        print("\n[OK] Good performance. Target of 20+ tokens/sec achieved.")
    else:
        print("\n[!] Below target. Check GPU offloading settings.")
    
    # Embedding backend comparison
    emb_results = benchmark_embeddings()
    
    if emb_results:
        print("\n" + "=" * 60)
        print("EMBEDDING OPTIMIZATION RECOMMENDATION")
        print("=" * 60)
        if "vulkan_llama" in emb_results:
            if emb_results.get("sentence_transformer", float('inf')) > emb_results.get("vulkan_llama", float('inf')):
                speedup = emb_results["sentence_transformer"] / emb_results["vulkan_llama"]
                print(f"[OK] Vulkan embeddings are {speedup:.1f}x faster than CPU!")
                print("     Your setup is optimized for AMD GPU.")
            else:
                print("[!] Vulkan embeddings slower than expected.")
                print("    Check if a GGUF embedding model is properly loaded.")
        else:
            print("[!] Vulkan embeddings not available.")
            print("    Download a GGUF embedding model to accelerate embeddings:")
            print("    - nomic-embed-text-v1.5.Q4_K_M.gguf (recommended)")
            print("    - bge-small-en-v1.5-q4_k_m.gguf (384 dims, compatible)")


if __name__ == "__main__":
    main()

