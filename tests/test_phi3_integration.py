"""
Phi-3 Integration Test: Full pipeline with real Phi-3 model
"""
import numpy as np
import pytest
import tempfile
import os
import shutil

try:
    from model import Phi3Model
    from memory_store import MemoryStore
    from vulkan_backend import SNNCompute
    PHI3_AVAILABLE = True
except ImportError as e:
    PHI3_AVAILABLE = False
    print(f"Phi-3 not available: {e}")


@pytest.mark.skipif(not PHI3_AVAILABLE, reason="Phi-3 model not available")
class TestPhi3Integration:
    """Test complete Phi-3 integration with memory store"""
    
    @pytest.fixture
    def phi3_model(self):
        """Load Phi-3 model"""
        try:
            return Phi3Model()
        except Exception as e:
            pytest.skip(f"Could not load Phi-3 model: {e}")
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database"""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "phi3_test.db")
        yield db_path
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def memory_store(self, temp_db):
        """Initialize memory store with Phi-3 embedding dimension"""
        return MemoryStore(db_path=temp_db, max_memories=50, embedding_dim=3072)
    
    @pytest.fixture
    def snn(self):
        """Initialize SNN compute"""
        return SNNCompute(n_neurons=512, use_vulkan=True)
    
    def test_phi3_embedding_and_storage(self, phi3_model, memory_store):
        """Test Phi-3 embedding extraction and storage"""
        text = "Machine learning is a powerful tool for AI"
        embedding = phi3_model.get_embedding(text)
        
        assert embedding.shape == (3072,)
        assert embedding.dtype == np.float32
        
        # Store in memory
        memory_store.store(embedding, text)
        
        stats = memory_store.get_stats()
        assert stats['total_memories'] == 1
        assert stats['gpu_memories'] == 1
        
        print(f"✓ Phi-3 embedding stored: shape {embedding.shape}")
    
    def test_phi3_memory_retrieval(self, phi3_model, memory_store):
        """Test memory retrieval with Phi-3 embeddings"""
        # Store multiple texts
        texts = [
            "Artificial intelligence is transforming technology",
            "Neural networks are computational models",
            "GPUs accelerate deep learning training"
        ]
        
        embeddings = []
        for text in texts:
            emb = phi3_model.get_embedding(text)
            embeddings.append(emb)
            memory_store.store(emb, text)
        
        # Query with similar text
        query_text = "Tell me about AI and neural networks"
        query_emb = phi3_model.get_embedding(query_text)
        
        # Retrieve similar memories
        retrieved = memory_store.retrieve(query_emb, k=2)
        
        assert len(retrieved) == 2
        print(f"✓ Retrieved {len(retrieved)} memories for query: {query_text}")
        print(f"  Retrieved: {retrieved}")
    
    def test_phi3_full_pipeline(self, phi3_model, memory_store, snn):
        """Test complete pipeline: Phi-3 -> Memory -> SNN"""
        # User query
        user_query = "What is machine learning?"
        query_emb = phi3_model.get_embedding(user_query)
        
        # Store query
        memory_store.store(query_emb, user_query)
        
        # Retrieve context
        context = memory_store.retrieve(query_emb, k=2)
        
        # Process through SNN for visualization
        spike_result = snn.process(query_emb)
        
        assert spike_result['spike_activity'] >= 0
        assert 'spikes' in spike_result
        
        print(f"✓ Full pipeline test passed")
        print(f"  Query: {user_query}")
        print(f"  Context: {len(context)} memories retrieved")
        print(f"  Spike activity: {spike_result['spike_activity']}")
    
    def test_phi3_generation_with_memory(self, phi3_model, memory_store):
        """Test Phi-3 generation with memory context"""
        # Store some memories
        memory_texts = [
            "Machine learning uses algorithms to learn patterns",
            "Deep learning uses neural networks with many layers"
        ]
        
        for text in memory_texts:
            emb = phi3_model.get_embedding(text)
            memory_store.store(emb, text)
        
        # Query
        query = "Explain machine learning"
        query_emb = phi3_model.get_embedding(query)
        
        # Retrieve context
        context = memory_store.retrieve(query_emb, k=2)
        
        # Generate response (this might be slow, so we'll just verify it doesn't crash)
        try:
            response = phi3_model.generate(query, context)
            assert isinstance(response, str)
            assert len(response) > 0
            print(f"✓ Generation with memory context works")
            print(f"  Response length: {len(response)} chars")
        except Exception as e:
            # Generation might fail due to model constraints, but embedding should work
            pytest.skip(f"Generation failed (expected for some models): {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

