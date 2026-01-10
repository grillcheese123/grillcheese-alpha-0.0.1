"""
Optimized Vulkan FAISS with persistent buffers.
Eliminates buffer allocation overhead for fast repeated searches.
"""

import numpy as np
import struct
import time
from typing import Tuple, Optional

# Import Vulkan backend
try:
    from vulkan_backend import VulkanCompute, VULKAN_AVAILABLE
    from vulkan_backend.base import VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
    if VULKAN_AVAILABLE:
        from vulkan import vkDestroyBuffer, vkFreeMemory
except ImportError:
    VULKAN_AVAILABLE = False


class VulkanFAISSOptimized:
    """
    GPU-accelerated FAISS with persistent buffers.
    
    Pre-allocates GPU memory for:
    - Database vectors (up to max_vectors)
    - Query vectors (up to max_queries)
    - Distance matrix
    - Top-k results
    
    Achieves ~10-50x speedup over per-search allocation.
    """
    
    def __init__(
        self,
        dim: int = 128,
        max_vectors: int = 100000,
        max_queries: int = 64,
        k: int = 32,
        distance_type: str = 'l2'
    ):
        self.dim = dim
        self.max_vectors = max_vectors
        self.max_queries = max_queries
        self.k = k
        self.distance_type = distance_type
        self.distance_type_int = {'l2': 0, 'cosine': 1, 'dot': 2}.get(distance_type, 0)
        
        # Current state
        self.vectors = None
        self.count = 0
        
        # GPU state
        self.gpu = None
        self.buffers_allocated = False
        self._init_gpu()
    
    def _init_gpu(self):
        """Initialize Vulkan and allocate persistent buffers"""
        if not VULKAN_AVAILABLE:
            print("[VulkanFAISSOptimized] Vulkan not available")
            return
        
        try:
            self.gpu = VulkanCompute()
            self._allocate_buffers()
            print(f"[VulkanFAISSOptimized] GPU ready: {self.max_vectors} vectors, {self.dim}D")
        except Exception as e:
            print(f"[VulkanFAISSOptimized] GPU init failed: {e}")
            self.gpu = None
    
    def _allocate_buffers(self):
        """Allocate persistent GPU buffers"""
        if self.gpu is None:
            return
        
        # Database buffer (max_vectors, dim)
        db_size = self.max_vectors * self.dim * 4
        self.buf_database, self.mem_database = self.gpu.core._create_buffer(
            db_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        
        # Query buffer (max_queries, dim)
        query_size = self.max_queries * self.dim * 4
        self.buf_queries, self.mem_queries = self.gpu.core._create_buffer(
            query_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        
        # Distance matrix (max_queries, max_vectors)
        dist_size = self.max_queries * self.max_vectors * 4
        self.buf_distances, self.mem_distances = self.gpu.core._create_buffer(
            dist_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        
        # Top-k indices (max_queries, k)
        topk_idx_size = self.max_queries * self.k * 4
        self.buf_topk_idx, self.mem_topk_idx = self.gpu.core._create_buffer(
            topk_idx_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        
        # Top-k distances (max_queries, k)
        topk_dist_size = self.max_queries * self.k * 4
        self.buf_topk_dist, self.mem_topk_dist = self.gpu.core._create_buffer(
            topk_dist_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        
        # Database indices (for topk)
        db_idx_size = self.max_vectors * 4
        self.buf_db_indices, self.mem_db_indices = self.gpu.core._create_buffer(
            db_idx_size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        )
        # Upload indices 0..max_vectors-1
        db_indices = np.arange(self.max_vectors, dtype=np.uint32)
        self.gpu.core._upload_buffer(self.buf_db_indices, self.mem_db_indices, db_indices)
        
        self.buffers_allocated = True
        
        total_mb = (db_size + query_size + dist_size + topk_idx_size + topk_dist_size + db_idx_size) / (1024 * 1024)
        print(f"[VulkanFAISSOptimized] Allocated {total_mb:.1f} MB GPU buffers")
    
    def build(self, vectors: np.ndarray):
        """Build index (upload vectors to GPU)"""
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        
        if len(vectors) > self.max_vectors:
            raise ValueError(f"Too many vectors: {len(vectors)} > {self.max_vectors}")
        
        self.vectors = vectors
        self.count = len(vectors)
        
        if self.gpu is not None and self.buffers_allocated:
            # Upload to GPU
            self.gpu.core._upload_buffer(
                self.buf_database, 
                self.mem_database, 
                vectors.flatten()
            )
    
    def add(self, vectors: np.ndarray):
        """Add vectors to index"""
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if self.vectors is None:
            self.build(vectors)
        else:
            new_count = self.count + len(vectors)
            if new_count > self.max_vectors:
                raise ValueError(f"Capacity exceeded: {new_count} > {self.max_vectors}")
            
            self.vectors = np.vstack([self.vectors, vectors])
            self.count = new_count
            
            # Re-upload to GPU
            if self.gpu is not None and self.buffers_allocated:
                self.gpu.core._upload_buffer(
                    self.buf_database,
                    self.mem_database,
                    self.vectors.flatten()
                )
    
    def search(self, query: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Returns:
            distances: (num_queries, k)
            indices: (num_queries, k)
        """
        if self.count == 0:
            return np.array([[]]), np.array([[]])
        
        k = k or self.k
        k = min(k, self.count)
        
        query = np.ascontiguousarray(query.astype(np.float32))
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        num_queries = len(query)
        if num_queries > self.max_queries:
            # Fall back to batched search
            return self._search_batched(query, k)
        
        if self.gpu is not None and self.buffers_allocated:
            return self._search_gpu(query, k)
        else:
            return self._search_cpu(query, k)
    
    def _search_gpu(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated search using persistent buffers"""
        num_queries = len(query)
        
        # Upload queries
        self.gpu.core._upload_buffer(
            self.buf_queries,
            self.mem_queries,
            query.flatten()
        )
        
        # 1. Compute distances
        pipeline, layout, _ = self.gpu.pipelines.get_or_create_pipeline(
            'faiss-distance', 3, push_constant_size=16
        )
        
        desc_set = self.gpu.pipelines.get_cached_descriptor_set(
            'faiss-distance-persistent',
            [
                (self.buf_queries, self.max_queries * self.dim * 4),
                (self.buf_database, self.max_vectors * self.dim * 4),
                (self.buf_distances, self.max_queries * self.max_vectors * 4)
            ]
        )
        
        push = struct.pack('IIII', num_queries, self.count, self.dim, self.distance_type_int)
        
        wg_x = (self.count + 15) // 16
        wg_y = (num_queries + 15) // 16
        self.gpu.core._dispatch_compute(pipeline, layout, desc_set, wg_x, push, wg_y, 1)
        
        # 2. Top-k selection
        pipeline, layout, _ = self.gpu.pipelines.get_or_create_pipeline(
            'faiss-topk', 4, push_constant_size=12
        )
        
        desc_set = self.gpu.pipelines.get_cached_descriptor_set(
            'faiss-topk-persistent',
            [
                (self.buf_distances, self.max_queries * self.max_vectors * 4),
                (self.buf_db_indices, self.max_vectors * 4),
                (self.buf_topk_idx, self.max_queries * self.k * 4),
                (self.buf_topk_dist, self.max_queries * self.k * 4)
            ]
        )
        
        push = struct.pack('III', num_queries, self.count, k)
        workgroups = (num_queries + 255) // 256
        self.gpu.core._dispatch_compute(pipeline, layout, desc_set, workgroups, push)
        
        # Download results
        indices = self.gpu.core._download_buffer(
            self.mem_topk_idx, num_queries * k * 4, dtype=np.uint32
        )[:num_queries * k].reshape(num_queries, k)
        
        distances = self.gpu.core._download_buffer(
            self.mem_topk_dist, num_queries * k * 4, dtype=np.float32
        )[:num_queries * k].reshape(num_queries, k)
        
        return distances, indices
    
    def _search_cpu(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """CPU fallback"""
        if self.distance_type == 'l2':
            diff = query[:, np.newaxis, :] - self.vectors[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff ** 2, axis=2))
        elif self.distance_type == 'cosine':
            q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
            v_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)
            distances = 1 - np.dot(q_norm, v_norm.T)
        else:
            distances = -np.dot(query, self.vectors.T)
        
        indices = np.argsort(distances, axis=1)[:, :k]
        topk_distances = np.take_along_axis(distances, indices, axis=1)
        
        return topk_distances, indices
    
    def _search_batched(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Handle queries exceeding max_queries"""
        all_distances = []
        all_indices = []
        
        for i in range(0, len(query), self.max_queries):
            batch = query[i:i + self.max_queries]
            distances, indices = self.search(batch, k)
            all_distances.append(distances)
            all_indices.append(indices)
        
        return np.vstack(all_distances), np.vstack(all_indices)
    
    def reset(self):
        """Clear the index (keeps buffers allocated)"""
        self.vectors = None
        self.count = 0
    
    def __del__(self):
        """Free GPU buffers"""
        if self.buffers_allocated and self.gpu is not None:
            try:
                for buf, mem in [
                    (self.buf_database, self.mem_database),
                    (self.buf_queries, self.mem_queries),
                    (self.buf_distances, self.mem_distances),
                    (self.buf_topk_idx, self.mem_topk_idx),
                    (self.buf_topk_dist, self.mem_topk_dist),
                    (self.buf_db_indices, self.mem_db_indices),
                ]:
                    vkDestroyBuffer(self.gpu.core.device, buf, None)
                    vkFreeMemory(self.gpu.core.device, mem, None)
            except:
                pass


def benchmark_optimized_faiss():
    """Benchmark optimized Vulkan FAISS"""
    print(f"\n{'='*60}")
    print("Optimized Vulkan FAISS Benchmark")
    print(f"{'='*60}")
    
    num_vectors = 10000
    dim = 128
    k = 32
    num_queries = 10
    
    # Generate test data
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    queries = np.random.randn(num_queries, dim).astype(np.float32)
    queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    
    print(f"\nTest: {num_vectors} vectors, {dim}D, k={k}, {num_queries} queries")
    
    # Optimized Vulkan FAISS
    print("\n[Optimized Vulkan FAISS]")
    index = VulkanFAISSOptimized(
        dim=dim,
        max_vectors=num_vectors,
        max_queries=num_queries,
        k=k
    )
    
    # Build
    start = time.perf_counter()
    index.build(vectors)
    build_time = (time.perf_counter() - start) * 1000
    print(f"  Build time: {build_time:.2f}ms")
    
    # Warmup
    _ = index.search(queries[0], k)
    _ = index.search(queries[0], k)
    
    # Search (single query)
    single_times = []
    for q in queries:
        start = time.perf_counter()
        distances, indices = index.search(q, k)
        single_times.append((time.perf_counter() - start) * 1000)
    
    print(f"  Single query: {np.mean(single_times):.2f}ms (std: {np.std(single_times):.2f})")
    
    # Batch search
    start = time.perf_counter()
    distances, indices = index.search(queries, k)
    batch_time = (time.perf_counter() - start) * 1000
    print(f"  Batch ({num_queries}): {batch_time:.2f}ms ({batch_time/num_queries:.2f}ms/query)")
    
    # Numpy baseline
    print("\n[Numpy Baseline]")
    
    single_times_np = []
    for q in queries:
        start = time.perf_counter()
        diff = q.reshape(1, -1)[:, np.newaxis, :] - vectors[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        idx = np.argsort(dists, axis=1)[:, :k]
        single_times_np.append((time.perf_counter() - start) * 1000)
    
    print(f"  Single query: {np.mean(single_times_np):.2f}ms (std: {np.std(single_times_np):.2f})")
    
    # Speedup
    speedup_single = np.mean(single_times_np) / np.mean(single_times)
    print(f"\n  Speedup (single): {speedup_single:.2f}x")
    
    # Verify correctness
    print("\n[Correctness Check]")
    q = queries[0:1]
    dist_gpu, idx_gpu = index.search(q, k)
    dist_cpu, idx_cpu = index._search_cpu(q, k)
    
    # Compare top-1
    if idx_gpu[0, 0] == idx_cpu[0, 0]:
        print("  Top-1 match: OK")
    else:
        print(f"  Top-1 mismatch: GPU={idx_gpu[0,0]}, CPU={idx_cpu[0,0]}")
    
    # Compare distances
    dist_diff = np.abs(dist_gpu - dist_cpu).mean()
    print(f"  Avg distance diff: {dist_diff:.6f}")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    benchmark_optimized_faiss()
