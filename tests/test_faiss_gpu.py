"""
FAISS GPU Integration Test Suite
Run with: pytest tests/test_faiss_gpu.py -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from vulkan_backend import VulkanCompute
from memory_store import MemoryStore


class TestFAISSGPU:
    
    @pytest.fixture
    def gpu(self):
        return VulkanCompute()
    
    def test_distance_l2(self, gpu):
        queries = np.random.randn(5, 128).astype(np.float32)
        database = np.random.randn(100, 128).astype(np.float32)
        
        distances = gpu.faiss_compute_distances(queries, database, distance_type='l2')
        
        assert distances.shape == (5, 100)
        assert np.all(distances >= 0)
        
        expected = np.sqrt(np.sum((queries[0:1] - database) ** 2, axis=1))
        np.testing.assert_allclose(distances[0], expected, rtol=1e-5)
    
    def test_distance_cosine(self, gpu):
        queries = np.random.randn(3, 64).astype(np.float32)
        database = np.random.randn(50, 64).astype(np.float32)
        
        distances = gpu.faiss_compute_distances(queries, database, distance_type='cosine')
        
        assert distances.shape == (3, 50)
        
        q_norm = queries[0] / (np.linalg.norm(queries[0]) + 1e-8)
        db_norms = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-8)
        expected = 1.0 - (db_norms @ q_norm)
        np.testing.assert_allclose(distances[0], expected, rtol=1e-5)
    
    def test_topk_selection(self, gpu):
        distances = np.random.randn(10, 1000).astype(np.float32)
        k = 20
        
        topk_indices, topk_distances = gpu.faiss_topk(distances, k)
        
        assert topk_indices.shape == (10, k)
        assert topk_distances.shape == (10, k)
        assert topk_indices.dtype == np.uint32
        
        for i in range(10):
            assert len(np.unique(topk_indices[i])) == k
            sorted_dists = np.sort(distances[i])
            np.testing.assert_allclose(
                np.sort(topk_distances[i]), sorted_dists[:k], rtol=1e-5
            )
    
    def test_topk_k_exceeds_database(self, gpu):
        distances = np.random.randn(5, 50).astype(np.float32)
        k = 100
        
        topk_indices, topk_distances = gpu.faiss_topk(distances, k)
        
        assert topk_indices.shape == (5, 50)
    
    def test_end_to_end_pipeline(self, gpu):
        np.random.seed(42)
        queries = np.random.randn(10, 256).astype(np.float32)
        database = np.random.randn(10000, 256).astype(np.float32)
        k = 10
        
        distances = gpu.faiss_compute_distances(queries, database, distance_type='cosine')
        topk_indices, topk_distances = gpu.faiss_topk(distances, k)
        
        assert topk_indices.shape == (10, k)
        
        for i in range(10):
            for j in range(k):
                idx = topk_indices[i, j]
                dist = topk_distances[i, j]
                expected = distances[i, idx]
                assert abs(dist - expected) < 1e-5
    
    def test_gpu_vs_numpy_correctness(self, gpu):
        np.random.seed(42)
        queries = np.random.randn(5, 128).astype(np.float32)
        database = np.random.randn(200, 128).astype(np.float32)
        k = 15
        
        gpu_distances = gpu.faiss_compute_distances(queries, database, distance_type='cosine')
        
        q_norms = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        db_norms = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-8)
        cpu_distances = 1.0 - (q_norms @ db_norms.T)
        
        np.testing.assert_allclose(gpu_distances, cpu_distances, rtol=1e-4)
        
        gpu_indices, _ = gpu.faiss_topk(gpu_distances, k)
        
        for i in range(5):
            cpu_topk = np.argsort(cpu_distances[i])[:k]
            np.testing.assert_array_equal(gpu_indices[i], cpu_topk)


class TestMemoryStoreIntegration:
    
    @pytest.fixture
    def memory(self, tmp_path):
        db_path = tmp_path / "test_memory.db"
        store = MemoryStore(
            db_path=str(db_path),
            max_memories=1000,
            embedding_dim=384
        )
        yield store
        store.clear()
    
    def test_store_and_retrieve(self, memory):
        np.random.seed(42)
        
        embeddings = []
        for i in range(100):
            emb = np.random.randn(384).astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-8
            embeddings.append(emb)
            memory.store(emb, f"Memory {i}")
        
        query = embeddings[0]
        results = memory.retrieve(query, k=5, include_identity=False)
        
        assert len(results) > 0
        assert "Memory 0" in results
    
    def test_gpu_fallback_on_error(self, memory):
        original_gpu = memory.gpu
        memory.gpu = None
        
        np.random.seed(42)
        emb = np.random.randn(384).astype(np.float32)
        emb /= np.linalg.norm(emb) + 1e-8
        memory.store(emb, "Test memory")
        
        results = memory.retrieve(emb, k=1, include_identity=False)
        assert "Test memory" in results
        
        memory.gpu = original_gpu
    
    def test_large_scale_retrieval(self, memory):
        np.random.seed(42)
        
        for i in range(1000):
            emb = np.random.randn(384).astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-8
            memory.store(emb, f"Memory {i}")
        
        query = np.random.randn(384).astype(np.float32)
        query /= np.linalg.norm(query) + 1e-8
        
        results = memory.retrieve(query, k=10, include_identity=False)
        assert len(results) == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
