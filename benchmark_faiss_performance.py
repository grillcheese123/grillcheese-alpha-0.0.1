"""
Performance benchmark for GPU FAISS vs CPU similarity search
Run with: python benchmark_faiss_performance.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import time
import numpy as np
from vulkan_backend import VulkanCompute
from memory_store import MemoryStore


def benchmark_similarity_search(num_memories, embedding_dim, k, num_queries):
    print(f"\n{'='*70}")
    print(f"Configuration:")
    print(f"  Memories: {num_memories:,}")
    print(f"  Embedding Dim: {embedding_dim}")
    print(f"  Top-K: {k}")
    print(f"  Queries: {num_queries}")
    print(f"{'='*70}\n")
    
    np.random.seed(42)
    database = np.random.randn(num_memories, embedding_dim).astype(np.float32)
    queries = np.random.randn(num_queries, embedding_dim).astype(np.float32)
    
    database /= np.linalg.norm(database, axis=1, keepdims=True) + 1e-8
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8
    
    gpu = VulkanCompute()
    
    print("GPU FAISS benchmark...")
    gpu_times = []
    for i in range(num_queries):
        start = time.perf_counter()
        distances = gpu.faiss_compute_distances(
            queries[i:i+1], database, distance_type='cosine'
        )
        topk_indices, _ = gpu.faiss_topk(distances, k)
        elapsed = time.perf_counter() - start
        gpu_times.append(elapsed * 1000)
    
    print("CPU NumPy benchmark...")
    cpu_times = []
    for i in range(num_queries):
        start = time.perf_counter()
        similarities = database @ queries[i]
        distances = 1.0 - similarities
        topk_indices = np.argsort(distances)[:k]
        elapsed = time.perf_counter() - start
        cpu_times.append(elapsed * 1000)
    
    gpu_mean = np.mean(gpu_times)
    gpu_std = np.std(gpu_times)
    gpu_p95 = np.percentile(gpu_times, 95)
    gpu_p99 = np.percentile(gpu_times, 99)
    
    cpu_mean = np.mean(cpu_times)
    cpu_std = np.std(cpu_times)
    cpu_p95 = np.percentile(cpu_times, 95)
    cpu_p99 = np.percentile(cpu_times, 99)
    
    speedup = cpu_mean / gpu_mean
    
    print(f"\nResults:")
    print(f"  GPU FAISS:")
    print(f"    Mean: {gpu_mean:.2f} ms")
    print(f"    Std:  {gpu_std:.2f} ms")
    print(f"    P95:  {gpu_p95:.2f} ms")
    print(f"    P99:  {gpu_p99:.2f} ms")
    print(f"\n  CPU NumPy:")
    print(f"    Mean: {cpu_mean:.2f} ms")
    print(f"    Std:  {cpu_std:.2f} ms")
    print(f"    P95:  {cpu_p95:.2f} ms")
    print(f"    P99:  {cpu_p99:.2f} ms")
    print(f"\n  Speedup: {speedup:.2f}x")
    
    return {
        'num_memories': num_memories,
        'gpu_mean': gpu_mean,
        'cpu_mean': cpu_mean,
        'speedup': speedup
    }


def benchmark_memory_store():
    print(f"\n{'='*70}")
    print("MemoryStore Integration Benchmark")
    print(f"{'='*70}\n")
    
    embedding_dim = 384
    num_memories = 5000
    k = 3
    num_queries = 100
    
    memory = MemoryStore(
        db_path="benchmark_test.db",
        max_memories=10000,
        embedding_dim=embedding_dim
    )
    
    memory.clear()
    
    print("Populating memory store...")
    np.random.seed(42)
    for i in range(num_memories):
        embedding = np.random.randn(embedding_dim).astype(np.float32)
        embedding /= np.linalg.norm(embedding) + 1e-8
        memory.store(embedding, f"Memory {i}")
    
    print(f"Stored {num_memories:,} memories")
    print(f"Running {num_queries} retrieval queries...\n")
    
    retrieval_times = []
    for _ in range(num_queries):
        query = np.random.randn(embedding_dim).astype(np.float32)
        query /= np.linalg.norm(query) + 1e-8
        
        start = time.perf_counter()
        results = memory.retrieve(query, k=k, include_identity=False)
        elapsed = time.perf_counter() - start
        retrieval_times.append(elapsed * 1000)
    
    mean_time = np.mean(retrieval_times)
    std_time = np.std(retrieval_times)
    p95_time = np.percentile(retrieval_times, 95)
    p99_time = np.percentile(retrieval_times, 99)
    
    print(f"Retrieval Performance:")
    print(f"  Mean: {mean_time:.2f} ms")
    print(f"  Std:  {std_time:.2f} ms")
    print(f"  P95:  {p95_time:.2f} ms")
    print(f"  P99:  {p99_time:.2f} ms")
    
    memory.clear()
    Path("benchmark_test.db").unlink(missing_ok=True)


def main():
    configs = [
        {'num_memories': 1000, 'embedding_dim': 384, 'k': 3, 'num_queries': 50},
        {'num_memories': 5000, 'embedding_dim': 384, 'k': 5, 'num_queries': 50},
        {'num_memories': 10000, 'embedding_dim': 384, 'k': 10, 'num_queries': 50},
        {'num_memories': 20000, 'embedding_dim': 384, 'k': 10, 'num_queries': 50},
    ]
    
    results = []
    for config in configs:
        result = benchmark_similarity_search(**config)
        results.append(result)
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"{'Memories':<12} {'GPU (ms)':<12} {'CPU (ms)':<12} {'Speedup':<12}")
    print(f"{'-'*70}")
    for result in results:
        print(f"{result['num_memories']:<12,} "
              f"{result['gpu_mean']:<12.2f} "
              f"{result['cpu_mean']:<12.2f} "
              f"{result['speedup']:<12.2f}x")
    
    benchmark_memory_store()


if __name__ == '__main__':
    main()
