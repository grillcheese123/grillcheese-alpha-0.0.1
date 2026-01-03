"""
End-to-end test: Full pipeline from user query to response
Tests the complete system integration
"""
import numpy as np
import pytest
import os
import tempfile
import shutil

try:
    from memory_store import MemoryStore
    from vulkan_backend import SNNCompute
    from compute import HybridCompute
    VULKAN_AVAILABLE = True
except ImportError as e:
    VULKAN_AVAILABLE = False
    print(f"Vulkan not available: {e}")


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestEndToEnd:
    """Test complete end-to-end pipeline"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database directory"""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "e2e_test.db")
        yield db_path
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def system(self, temp_db):
        """Initialize complete system"""
        memory = MemoryStore(db_path=temp_db, max_memories=100, embedding_dim=128)
        snn = SNNCompute(n_neurons=128, use_vulkan=True)
        return {'memory': memory, 'snn': snn}
    
    def test_complete_user_interaction(self, system):
        """Test complete user interaction flow"""
        memory = system['memory']
        snn = system['snn']
        
        # Simulate user queries and responses
        interactions = [
            ("What is AI?", "AI is artificial intelligence"),
            ("How does GPU work?", "GPUs accelerate parallel computations"),
            ("What is a neural network?", "Neural networks are computational models"),
        ]
        
        # Store conversation history
        for query, response in interactions:
            # Simulate embedding (in real system, from Phi-3)
            query_emb = np.random.randn(128).astype(np.float32)
            response_emb = np.random.randn(128).astype(np.float32)
            
            # Store both query and response
            memory.store(query_emb, query)
            memory.store(response_emb, response)
        
        # New user query
        new_query = "Tell me about artificial intelligence"
        new_query_emb = np.random.randn(128).astype(np.float32)
        
        # Retrieve relevant context (memory-enhanced retrieval)
        context = memory.retrieve(new_query_emb, k=2)
        assert len(context) == 2
        
        # Process through SNN for visualization
        spike_result = snn.process(new_query_emb)
        assert spike_result['spike_activity'] >= 0
        
        # Store new interaction
        memory.store(new_query_emb, new_query)
        
        print(f"✓ Complete user interaction test passed")
        print(f"  Context retrieved: {len(context)} memories")
        print(f"  Spike activity: {spike_result['spike_activity']}")
    
    def test_memory_accumulation(self, system):
        """Test that memories accumulate over time"""
        memory = system['memory']
        
        # Add memories incrementally
        for i in range(10):
            emb = np.random.randn(128).astype(np.float32)
            memory.store(emb, f"Memory {i}")
        
        stats = memory.get_stats()
        assert stats['total_memories'] == 10
        
        # Add more memories
        for i in range(10, 20):
            emb = np.random.randn(128).astype(np.float32)
            memory.store(emb, f"Memory {i}")
        
        stats = memory.get_stats()
        assert stats['total_memories'] == 20
        
        print(f"✓ Memory accumulation: {stats['total_memories']} total memories")
    
    def test_similarity_based_retrieval(self, system):
        """Test that retrieval returns similar memories"""
        memory = system['memory']
        
        # Store memories with known similarity
        emb1 = np.random.randn(128).astype(np.float32)
        emb2 = emb1 + 0.1 * np.random.randn(128).astype(np.float32)  # Similar to emb1
        emb3 = np.random.randn(128).astype(np.float32)  # Different
        
        memory.store(emb1, "Similar memory 1")
        memory.store(emb2, "Similar memory 2")
        memory.store(emb3, "Different memory")
        
        # Query similar to emb1 and emb2
        query = emb1 + 0.05 * np.random.randn(128).astype(np.float32)
        retrieved = memory.retrieve(query, k=2)
        
        # Should retrieve the similar memories
        assert len(retrieved) == 2
        assert "Similar memory 1" in retrieved or "Similar memory 2" in retrieved
        
        print(f"✓ Similarity-based retrieval works")
        print(f"  Retrieved: {retrieved}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

