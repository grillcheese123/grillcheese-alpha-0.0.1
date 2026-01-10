"""
Quick benchmark to find GPU/CPU crossover point
"""
import numpy as np
import time

# Test different vector counts
vector_counts = [1000, 5000, 10000, 25000, 50000, 100000]
dim = 128
k = 32

print(f"Finding GPU/CPU crossover point (dim={dim}, k={k})")
print(f"{'Vectors':>10} | {'Numpy (ms)':>12} | {'GPU (ms)':>12} | {'Winner':>8}")
print("-" * 50)

from vulkan_faiss_optimized import VulkanFAISSOptimized

for num_vectors in vector_counts:
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    query = np.random.randn(1, dim).astype(np.float32)
    
    # Numpy
    times_np = []
    for _ in range(5):
        start = time.perf_counter()
        diff = query[:, np.newaxis, :] - vectors[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        idx = np.argsort(dists, axis=1)[:, :k]
        times_np.append((time.perf_counter() - start) * 1000)
    np_time = np.mean(times_np[1:])  # Skip warmup
    
    # GPU (reuse index for warmup)
    try:
        index = VulkanFAISSOptimized(dim=dim, max_vectors=num_vectors, k=k)
        index.build(vectors)
        
        # Warmup
        _ = index.search(query, k)
        _ = index.search(query, k)
        
        times_gpu = []
        for _ in range(5):
            start = time.perf_counter()
            _, _ = index.search(query, k)
            times_gpu.append((time.perf_counter() - start) * 1000)
        gpu_time = np.mean(times_gpu)
        
        winner = "GPU" if gpu_time < np_time else "Numpy"
        print(f"{num_vectors:>10} | {np_time:>12.2f} | {gpu_time:>12.2f} | {winner:>8}")
        
        del index
    except Exception as e:
        print(f"{num_vectors:>10} | {np_time:>12.2f} | {'ERROR':>12} | {'Numpy':>8}")
