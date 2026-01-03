"""
Test system identity storage and retrieval
"""
import numpy as np
import pytest
import tempfile
import os
import shutil

try:
    from memory_store import MemoryStore
    from identity import DEFAULT_IDENTITY
    VULKAN_AVAILABLE = True
except ImportError as e:
    VULKAN_AVAILABLE = False
    print(f"Vulkan not available: {e}")


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestIdentity:
    """Test system identity storage"""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database"""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "identity_test.db")
        yield db_path
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def memory_store(self, temp_db):
        """Initialize memory store"""
        return MemoryStore(db_path=temp_db, max_memories=100, embedding_dim=64)
    
    def test_store_identity(self, memory_store):
        """Test storing system identity"""
        identity_text = "I am a helpful AI assistant"
        identity_emb = np.random.randn(64).astype(np.float32)
        
        memory_store.store_identity(identity_emb, identity_text)
        
        assert memory_store.identity_index >= 0
        assert memory_store.get_identity() == identity_text
        assert memory_store.memory_texts[memory_store.identity_index] == identity_text
        
        print(f"✓ Identity stored at index {memory_store.identity_index}")
    
    def test_identity_always_included(self, memory_store):
        """Test that identity is always included in retrieval"""
        identity_text = "I am GrillCheese AI"
        identity_emb = np.random.randn(64).astype(np.float32)
        memory_store.store_identity(identity_emb, identity_text)
        
        # Store some regular memories
        for i in range(5):
            emb = np.random.randn(64).astype(np.float32)
            memory_store.store(emb, f"Regular memory {i}")
        
        # Retrieve memories
        query = np.random.randn(64).astype(np.float32)
        retrieved = memory_store.retrieve(query, k=3, include_identity=True)
        
        # Identity should be first
        assert len(retrieved) > 0
        assert retrieved[0] == identity_text
        
        print(f"✓ Identity always included: {retrieved[0][:30]}...")
    
    def test_identity_update(self, memory_store):
        """Test updating identity"""
        identity1 = "I am AI assistant v1"
        emb1 = np.random.randn(64).astype(np.float32)
        memory_store.store_identity(emb1, identity1)
        
        idx1 = memory_store.identity_index
        
        identity2 = "I am AI assistant v2"
        emb2 = np.random.randn(64).astype(np.float32)
        memory_store.store_identity(emb2, identity2)
        
        # Should update at same index
        assert memory_store.identity_index == idx1
        assert memory_store.get_identity() == identity2
        
        print(f"✓ Identity updated at index {idx1}")
    
    def test_identity_persistence(self, temp_db):
        """Test identity persists across sessions"""
        # Store identity in first session
        store1 = MemoryStore(db_path=temp_db, embedding_dim=64)
        identity_text = "I am persistent"
        identity_emb = np.random.randn(64).astype(np.float32)
        store1.store_identity(identity_emb, identity_text)
        
        # Load in second session
        store2 = MemoryStore(db_path=temp_db, embedding_dim=64)
        
        assert store2.identity_index >= 0
        assert store2.memory_texts[store2.identity_index] == identity_text
        
        print("✓ Identity persists across sessions")
    
    def test_default_identity(self, temp_db):
        """Test default identity from identity.py"""
        store = MemoryStore(db_path=temp_db, embedding_dim=64, identity=DEFAULT_IDENTITY)
        
        assert store.get_identity() == DEFAULT_IDENTITY
        print(f"✓ Default identity loaded: {DEFAULT_IDENTITY[:50]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

