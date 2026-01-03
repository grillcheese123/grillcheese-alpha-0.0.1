"""
Tests for MemoryStore class
"""
import os
import pytest
import numpy as np
import tempfile
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from memory_store import MemoryStore
from config import MemoryConfig


class TestMemoryStoreInit:
    """Tests for MemoryStore initialization"""
    
    def test_init_creates_database(self):
        """MemoryStore should create database file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=384)
            assert os.path.exists(db_path)
    
    def test_init_with_identity(self):
        """MemoryStore should accept identity text"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            identity = "I am a test AI"
            memory = MemoryStore(db_path=db_path, embedding_dim=384, identity=identity)
            assert memory.identity_text == identity
    
    def test_init_empty_state(self):
        """New MemoryStore should have empty state"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=384)
            assert memory.num_memories == 0
            assert len(memory.memory_texts) == 0


class TestMemoryStoreStore:
    """Tests for MemoryStore.store method"""
    
    def test_store_single_memory(self):
        """Should store a single memory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=384)
            
            embedding = np.random.randn(384).astype(np.float32)
            text = "Test memory"
            
            memory.store(embedding, text)
            
            assert memory.num_memories == 1
            assert text in memory.memory_texts
    
    def test_store_multiple_memories(self):
        """Should store multiple memories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=384)
            
            for i in range(5):
                embedding = np.random.randn(384).astype(np.float32)
                memory.store(embedding, f"Memory {i}")
            
            assert memory.num_memories == 5
    
    def test_store_validates_embedding_dim(self):
        """Should reject embeddings with wrong dimension"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=384)
            
            wrong_embedding = np.random.randn(100).astype(np.float32)
            
            with pytest.raises(ValueError, match="dimension mismatch"):
                memory.store(wrong_embedding, "Test")
    
    def test_store_with_metadata(self):
        """Should store memory with metadata"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=384)
            
            embedding = np.random.randn(384).astype(np.float32)
            metadata = {"source": "test", "important": True}
            
            memory.store(embedding, "Test", metadata=metadata)
            assert memory.num_memories == 1


class TestMemoryStoreRetrieve:
    """Tests for MemoryStore.retrieve method"""
    
    def test_retrieve_returns_list(self):
        """Retrieve should return a list"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=384)
            
            embedding = np.random.randn(384).astype(np.float32)
            result = memory.retrieve(embedding, k=3)
            
            assert isinstance(result, list)
    
    def test_retrieve_empty_store(self):
        """Retrieve from empty store should return empty list"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=384)
            
            embedding = np.random.randn(384).astype(np.float32)
            result = memory.retrieve(embedding, k=3)
            
            assert result == []
    
    def test_retrieve_returns_similar_memories(self):
        """Retrieve should return stored memories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=384)
            
            # Store a memory
            embedding = np.random.randn(384).astype(np.float32)
            memory.store(embedding, "Test memory")
            
            # Retrieve with same embedding
            result = memory.retrieve(embedding, k=1, include_identity=False)
            
            assert len(result) >= 1
            assert "Test memory" in result
    
    def test_retrieve_respects_k_limit(self):
        """Retrieve should return at most k memories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=384)
            
            # Store 10 memories
            for i in range(10):
                embedding = np.random.randn(384).astype(np.float32)
                memory.store(embedding, f"Memory {i}")
            
            # Retrieve only 3
            embedding = np.random.randn(384).astype(np.float32)
            result = memory.retrieve(embedding, k=3, include_identity=False)
            
            assert len(result) <= 3


class TestMemoryStoreIdentity:
    """Tests for identity storage and retrieval"""
    
    def test_store_identity(self):
        """Should store identity memory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=384)
            
            identity_text = "I am GrillCheese AI"
            embedding = np.random.randn(384).astype(np.float32)
            
            memory.store_identity(embedding, identity_text)
            
            assert memory.identity_text == identity_text
            assert memory.identity_index >= 0
    
    def test_get_identity(self):
        """Should retrieve identity text"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            identity_text = "I am a test AI"
            memory = MemoryStore(db_path=db_path, embedding_dim=384, identity=identity_text)
            
            assert memory.get_identity() == identity_text
    
    def test_retrieve_includes_identity(self):
        """Retrieve should include identity when requested"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=384)
            
            # Store identity
            identity_text = "I am GrillCheese"
            embedding = np.random.randn(384).astype(np.float32)
            memory.store_identity(embedding, identity_text)
            
            # Store regular memory
            memory.store(np.random.randn(384).astype(np.float32), "Regular memory")
            
            # Retrieve with identity
            result = memory.retrieve(embedding, k=3, include_identity=True)
            
            assert identity_text in result


class TestMemoryStoreStats:
    """Tests for get_stats method"""
    
    def test_stats_returns_dict(self):
        """get_stats should return a dictionary"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=384)
            
            stats = memory.get_stats()
            
            assert isinstance(stats, dict)
    
    def test_stats_contains_required_keys(self):
        """Stats should contain required keys"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=384)
            
            stats = memory.get_stats()
            
            assert 'total_memories' in stats
            assert 'gpu_memories' in stats
            assert 'max_memories' in stats
            assert 'embedding_dim' in stats
    
    def test_stats_reflect_stored_memories(self):
        """Stats should reflect actual stored memories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=384)
            
            # Store 3 memories
            for i in range(3):
                embedding = np.random.randn(384).astype(np.float32)
                memory.store(embedding, f"Memory {i}")
            
            stats = memory.get_stats()
            
            assert stats['total_memories'] == 3
            assert stats['gpu_memories'] == 3


class TestMemoryStoreClear:
    """Tests for clear method"""
    
    def test_clear_removes_all_memories(self):
        """Clear should remove all memories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            memory = MemoryStore(db_path=db_path, embedding_dim=384)
            
            # Store some memories
            for i in range(5):
                embedding = np.random.randn(384).astype(np.float32)
                memory.store(embedding, f"Memory {i}")
            
            assert memory.num_memories == 5
            
            # Clear
            memory.clear()
            
            assert memory.num_memories == 0
            assert len(memory.memory_texts) == 0
            
            stats = memory.get_stats()
            assert stats['total_memories'] == 0


class TestMemoryStorePersistence:
    """Tests for persistence across instances"""
    
    def test_memories_persist(self):
        """Memories should persist across MemoryStore instances"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_memories.db")
            
            # Create and store
            memory1 = MemoryStore(db_path=db_path, embedding_dim=384)
            embedding = np.random.randn(384).astype(np.float32)
            memory1.store(embedding, "Persistent memory")
            
            # Create new instance
            memory2 = MemoryStore(db_path=db_path, embedding_dim=384)
            
            assert memory2.num_memories == 1
            assert "Persistent memory" in memory2.memory_texts

