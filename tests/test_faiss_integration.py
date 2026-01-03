"""
Test suite for GPU FAISS integration
Run with: uv run pytest tests/test_faiss_integration.py -v
"""
import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vulkan_backend import VulkanCompute, VULKAN_AVAILABLE


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestFAISSIntegration:
    """Tests for FAISS GPU integration"""
    
    @pytest.fixture
    def gpu(self):
        """Initialize Vulkan compute backend"""
        try:
            return VulkanCompute()
        except Exception as e:
            pytest.skip(f"Failed to initialize Vulkan: {e}")
    
    def test_faiss_distance_l2(self, gpu):
        """Test L2 distance computation"""
        queries = np.random.randn(5, 128).astype(np.float32)
        database = np.random.randn(100, 128).astype(np.float32)
        
        distances = gpu.faiss_compute_distances(queries, database, distance_type='l2')
        
        assert distances.shape == (5, 100)
        assert distances.dtype == np.float32
        assert np.all(distances >= 0)
        
        expected = np.sqrt(np.sum((queries[0:1] - database) ** 2, axis=1))
        np.testing.assert_allclose(distances[0], expected, rtol=1e-5)
    
    def test_faiss_distance_cosine(self, gpu):
        """Test cosine distance computation"""
        queries = np.random.randn(3, 64).astype(np.float32)
        database = np.random.randn(50, 64).astype(np.float32)
        
        distances = gpu.faiss_compute_distances(queries, database, distance_type='cosine')
        
        assert distances.shape == (3, 50)
        assert np.all(distances >= -1) and np.all(distances <= 2)
        
        q_norm = queries[0] / (np.linalg.norm(queries[0]) + 1e-8)
        db_norms = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-8)
        expected = 1.0 - (db_norms @ q_norm)
        np.testing.assert_allclose(distances[0], expected, rtol=1e-5)
    
    def test_faiss_distance_dot(self, gpu):
        """Test dot product distance computation"""
        queries = np.random.randn(2, 32).astype(np.float32)
        database = np.random.randn(20, 32).astype(np.float32)
        
        distances = gpu.faiss_compute_distances(queries, database, distance_type='dot')
        
        assert distances.shape == (2, 20)
        
        expected = -(queries[0] @ database.T)
        np.testing.assert_allclose(distances[0], expected, rtol=1e-5)
    
    def test_faiss_distance_single_query(self, gpu):
        """Test single query vector (1D input)"""
        query = np.random.randn(128).astype(np.float32)
        database = np.random.randn(100, 128).astype(np.float32)
        
        distances = gpu.faiss_compute_distances(query, database, distance_type='cosine')
        
        assert distances.shape == (1, 100)
    
    def test_faiss_topk(self, gpu):
        """Test top-k selection"""
        distances = np.random.randn(10, 1000).astype(np.float32)
        k = 20
        
        topk_indices, topk_distances = gpu.faiss_topk(distances, k)
        
        assert topk_indices.shape == (10, k)
        assert topk_distances.shape == (10, k)
        assert topk_indices.dtype == np.uint32
        assert topk_distances.dtype == np.float32
        
        for i in range(10):
            assert len(np.unique(topk_indices[i])) == k
            sorted_dists = np.sort(distances[i])
            np.testing.assert_allclose(
                np.sort(topk_distances[i]), sorted_dists[:k], rtol=1e-5
            )
    
    def test_faiss_topk_k_larger_than_database(self, gpu):
        """Test top-k when k exceeds database size"""
        distances = np.random.randn(5, 50).astype(np.float32)
        k = 100
        
        topk_indices, topk_distances = gpu.faiss_topk(distances, k)
        
        assert topk_indices.shape == (5, 50)
        assert topk_distances.shape == (5, 50)
    
    def test_faiss_end_to_end(self, gpu):
        """Test complete FAISS pipeline"""
        queries = np.random.randn(10, 256).astype(np.float32)
        database = np.random.randn(1000, 256).astype(np.float32)  # Reduced size for faster tests
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
    
    def test_faiss_correctness_vs_numpy(self, gpu):
        """Validate GPU results match NumPy reference implementation"""
        np.random.seed(42)
        queries = np.random.randn(5, 128).astype(np.float32)
        database = np.random.randn(200, 128).astype(np.float32)
        k = 15
        
        gpu_distances = gpu.faiss_compute_distances(queries, database, distance_type='cosine')
        
        q_norms = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
        db_norms = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-8)
        cpu_distances = 1.0 - (q_norms @ db_norms.T)
        
        np.testing.assert_allclose(gpu_distances, cpu_distances, rtol=1e-4)
        
        gpu_indices, gpu_dists = gpu.faiss_topk(gpu_distances, k)
        
        for i in range(5):
            cpu_topk = np.argsort(cpu_distances[i])[:k]
            np.testing.assert_array_equal(gpu_indices[i], cpu_topk)
    
    def test_faiss_batch_queries(self, gpu):
        """Test batch query processing"""
        queries = np.random.randn(20, 64).astype(np.float32)
        database = np.random.randn(500, 64).astype(np.float32)
        k = 5
        
        distances = gpu.faiss_compute_distances(queries, database, distance_type='cosine')
        topk_indices, topk_distances = gpu.faiss_topk(distances, k)
        
        assert topk_indices.shape == (20, k)
        assert topk_distances.shape == (20, k)
        
        # Verify each query has correct top-k
        for i in range(20):
            assert len(np.unique(topk_indices[i])) == k
            # Verify distances are sorted (ascending for cosine distance)
            assert np.all(np.diff(topk_distances[i]) >= -1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

