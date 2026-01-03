"""
Full pipeline test: MemoryStore + Phi-3 embeddings + GPU operations
Tests the complete memory-enhanced inference system
"""
import numpy as np
import pytest
import os
import tempfile
import shutil

try:
    from memory_store import MemoryStore
    from vulkan_backend import VulkanCompute, SNNCompute
    VULKAN_AVAILABLE = True
except ImportError as e:
    VULKAN_AVAILABLE = False
    print(f"Vulkan not available: {e}")

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestFullPipeline:
    """Test the complete GPU-accelerated memory pipeline"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database directory"""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_memories.db")
        yield db_path
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def memory_store(self, temp_db):
        """Initialize MemoryStore with temporary database"""
        # Use smaller embedding dim for faster tests
        store = MemoryStore(db_path=temp_db, max_memories=100, embedding_dim=128)
        yield store
        # Cleanup
        if os.path.exists(temp_db):
            try:
                os.remove(temp_db)
            except:
                pass
    
    @pytest.fixture
    def snn_compute(self):
        """Initialize SNN compute"""
        return SNNCompute(n_neurons=100, use_vulkan=True)
    
    def test_memory_store_basic(self, memory_store):
        """Test basic memory store operations"""
        # Store a memory
        emb1 = np.random.randn(128).astype(np.float32)
        memory_store.store(emb1, "Test memory 1: Hello world")
        
        # Store another memory
        emb2 = np.random.randn(128).astype(np.float32)
        memory_store.store(emb2, "Test memory 2: GPU acceleration")
        
        # Verify stats
        stats = memory_store.get_stats()
        assert stats['total_memories'] == 2
        assert stats['gpu_memories'] == 2
        assert stats['embedding_dim'] == 128
        
        print("✓ Memory store basic operations work")
    
    def test_memory_retrieval(self, memory_store):
        """Test memory retrieval with GPU operations"""
        # Store multiple memories
        embeddings = []
        texts = []
        for i in range(5):
            emb = np.random.randn(128).astype(np.float32)
            text = f"Memory {i}: This is test content {i}"
            embeddings.append(emb)
            texts.append(text)
            memory_store.store(emb, text)
        
        # Retrieve top-3 similar memories
        query = embeddings[0]  # Query similar to first memory
        retrieved = memory_store.retrieve(query, k=3)
        
        assert len(retrieved) == 3
        assert texts[0] in retrieved  # Should retrieve the most similar
        
        print(f"✓ Retrieved {len(retrieved)} memories")
        print(f"  Retrieved: {retrieved}")
    
    def test_snn_processing(self, snn_compute):
        """Test SNN spike generation for embeddings"""
        # Create test embedding
        embedding = np.random.randn(100).astype(np.float32)
        
        # Process through SNN
        result = snn_compute.process(embedding)
        
        assert 'spike_activity' in result
        assert result['spike_activity'] >= 0
        
        print(f"✓ SNN processing: {result['spike_activity']} spikes")
    
    def test_full_memory_pipeline(self, memory_store, snn_compute):
        """Test complete pipeline: store -> retrieve -> SNN processing"""
        # Simulate user queries
        queries = [
            "What is machine learning?",
            "How does GPU acceleration work?",
            "Explain neural networks"
        ]
        
        # Store embeddings (simulate Phi-3 embeddings)
        stored_embeddings = []
        for i, query in enumerate(queries):
            # Simulate embedding (128-dim for test)
            emb = np.random.randn(128).astype(np.float32)
            stored_embeddings.append(emb)
            memory_store.store(emb, query)
        
        # Retrieve context for a new query
        query_embedding = stored_embeddings[0]  # Similar to first query
        context = memory_store.retrieve(query_embedding, k=2)
        
        assert len(context) == 2
        
        # Process through SNN for visualization
        spike_result = snn_compute.process(query_embedding)
        
        assert spike_result['spike_activity'] >= 0
        
        print(f"✓ Full pipeline test passed")
        print(f"  Context: {context}")
        print(f"  Spike activity: {spike_result['spike_activity']}")
    
    def test_memory_persistence(self, temp_db):
        """Test that memories persist across store instances"""
        # Store in first instance
        store1 = MemoryStore(db_path=temp_db, embedding_dim=128)
        emb1 = np.random.randn(128).astype(np.float32)
        store1.store(emb1, "Persistent memory 1")
        
        emb2 = np.random.randn(128).astype(np.float32)
        store1.store(emb2, "Persistent memory 2")
        
        # Create new instance (should load from DB)
        store2 = MemoryStore(db_path=temp_db, embedding_dim=128)
        
        # Verify memories loaded
        stats = store2.get_stats()
        assert stats['total_memories'] == 2
        assert stats['gpu_memories'] == 2
        
        # Should be able to retrieve
        retrieved = store2.retrieve(emb1, k=1)
        assert len(retrieved) == 1
        
        print("✓ Memory persistence works")
    
    def test_large_memory_bank(self, temp_db):
        """Test memory store with larger number of memories"""
        store = MemoryStore(db_path=temp_db, max_memories=1000, embedding_dim=128)
        
        # Store 50 memories
        embeddings = []
        for i in range(50):
            emb = np.random.randn(128).astype(np.float32)
            embeddings.append(emb)
            store.store(emb, f"Large memory bank test {i}")
        
        stats = store.get_stats()
        assert stats['total_memories'] == 50
        assert stats['gpu_memories'] == 50
        
        # Retrieve from large bank
        query = embeddings[10]
        retrieved = store.retrieve(query, k=5)
        assert len(retrieved) == 5
        
        print(f"✓ Large memory bank test: {stats['total_memories']} memories")
    
    def test_round_robin_overflow(self, temp_db):
        """Test round-robin behavior when max_memories is exceeded"""
        store = MemoryStore(db_path=temp_db, max_memories=5, embedding_dim=128)
        
        # Store more than max_memories
        for i in range(10):
            emb = np.random.randn(128).astype(np.float32)
            store.store(emb, f"Round-robin test {i}")
        
        stats = store.get_stats()
        # DB should have all 10, but GPU buffer only holds max_memories
        assert stats['total_memories'] == 10
        assert stats['gpu_memories'] <= 5  # GPU buffer limited
        
        print(f"✓ Round-robin overflow handled: {stats['gpu_memories']} GPU memories")


@pytest.mark.skipif(not VULKAN_AVAILABLE or not TORCH_AVAILABLE, 
                    reason="Vulkan or PyTorch not available")
class TestWithRealEmbeddings:
    """Test with actual Phi-3 embeddings (requires model download)"""
    
    @pytest.fixture
    def phi3_model(self):
        """Load Phi-3 model (skip if not available)"""
        try:
            from model import Phi3Model
            model = Phi3Model()
            return model
        except Exception as e:
            pytest.skip(f"Could not load Phi-3 model: {e}")
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database directory"""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "test_memories_phi3.db")
        yield db_path
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def memory_store_phi3(self, temp_db):
        """Initialize MemoryStore with Phi-3 embedding dimension"""
        store = MemoryStore(db_path=temp_db, max_memories=50, embedding_dim=3072)
        yield store
    
    def test_phi3_embedding_extraction(self, phi3_model):
        """Test embedding extraction from Phi-3"""
        text = "What is artificial intelligence?"
        embedding = phi3_model.get_embedding(text)
        
        assert embedding.shape == (3072,)
        assert embedding.dtype == np.float32
        
        print(f"✓ Phi-3 embedding extracted: shape {embedding.shape}")
    
    def test_phi3_memory_pipeline(self, phi3_model, memory_store_phi3, temp_db):
        """Test full pipeline with real Phi-3 embeddings"""
        # Store some memories with real embeddings
        texts = [
            "Machine learning is a subset of AI",
            "GPUs accelerate neural network training",
            "Transformers revolutionized NLP"
        ]
        
        stored_embeddings = []
        for text in texts:
            emb = phi3_model.get_embedding(text)
            stored_embeddings.append(emb)
            memory_store_phi3.store(emb, text)
        
        # Retrieve similar memories
        query_text = "Tell me about AI"
        query_emb = phi3_model.get_embedding(query_text)
        context = memory_store_phi3.retrieve(query_emb, k=2)
        
        assert len(context) == 2
        
        print(f"✓ Phi-3 memory pipeline test passed")
        print(f"  Query: {query_text}")
        print(f"  Retrieved context: {context}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

