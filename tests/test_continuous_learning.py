"""
Tests for Continuous Learning Module
"""
import asyncio
import os
import pytest
import tempfile
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from learning.events import EventBus, Event
from learning.stdp_learner import STDPLearner
from learning.continuous_learner import (
    ContinuousLearner, LearningConfig, ContentItem, 
    ContentCategory, ProcessingPriority
)


class TestEventBus:
    """Tests for EventBus"""
    
    def test_subscribe_and_emit(self):
        """Should call handler when event emitted"""
        bus = EventBus()
        received = []
        
        def handler(event):
            received.append(event)
        
        bus.subscribe('test_event', handler)
        bus.emit('test_event', {'value': 42})
        
        assert len(received) == 1
        assert received[0].data['value'] == 42
    
    def test_multiple_subscribers(self):
        """Should call all subscribers"""
        bus = EventBus()
        counts = [0, 0]
        
        def handler1(event):
            counts[0] += 1
        
        def handler2(event):
            counts[1] += 1
        
        bus.subscribe('test', handler1)
        bus.subscribe('test', handler2)
        bus.emit('test', {})
        
        assert counts == [1, 1]
    
    def test_unsubscribe(self):
        """Should not call unsubscribed handler"""
        bus = EventBus()
        count = [0]
        
        def handler(event):
            count[0] += 1
        
        bus.subscribe('test', handler)
        bus.emit('test', {})
        assert count[0] == 1
        
        bus.unsubscribe('test', handler)
        bus.emit('test', {})
        assert count[0] == 1  # Not called again
    
    def test_get_stats(self):
        """Should track statistics"""
        bus = EventBus()
        bus.subscribe('event1', lambda e: None)
        bus.subscribe('event2', lambda e: None)
        bus.emit('event1', {})
        bus.emit('event1', {})
        
        stats = bus.get_stats()
        assert stats['total_events'] == 2
        assert 'event1' in stats['subscribers']


class TestSTDPLearner:
    """Tests for STDP Learner"""
    
    def test_init(self):
        """Should initialize with default parameters"""
        stdp = STDPLearner()
        assert stdp.lr_plus == 0.01
        assert stdp.lr_minus == 0.012
        assert stdp.window == 5
    
    def test_process_empty_sequence(self):
        """Should handle empty sequence"""
        stdp = STDPLearner()
        result = stdp.process_sequence([])
        assert result.get('updates', 0) == 0
    
    def test_process_sequence_creates_weights(self):
        """Should create token weights"""
        stdp = STDPLearner()
        token_ids = [1, 2, 3, 4, 5]
        
        stdp.process_sequence(token_ids)
        
        assert len(stdp.token_weights) > 0
    
    def test_repeated_tokens_strengthen(self):
        """Repeated co-occurring tokens should strengthen association"""
        stdp = STDPLearner()
        
        # Process same pattern multiple times
        for _ in range(10):
            stdp.process_sequence([1, 2, 3])
        
        # Weight for token 3 should be higher (follows 1, 2)
        assert stdp.token_weights.get(3, 0) > 0
    
    def test_process_embedding_pair(self):
        """Should create associations between embedding pairs"""
        stdp = STDPLearner()
        
        emb1_indices = [1, 2, 3]
        emb2_indices = [4, 5, 6]
        
        result = stdp.process_embedding_pair(emb1_indices, emb2_indices)
        
        assert result['updates'] > 0
        assert len(stdp.associations) > 0
    
    def test_get_modulations(self):
        """Should return modulation factors"""
        stdp = STDPLearner()
        
        # Process to create weights
        stdp.process_sequence([1, 2, 3])
        
        mods = stdp.get_modulations([1, 2, 3, 4])
        
        assert len(mods) == 4
        assert all(m >= 1.0 for m in mods)
    
    def test_save_load_state(self):
        """Should persist and restore state"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stdp_state.json")
            
            # Create and train
            stdp1 = STDPLearner()
            stdp1.process_sequence([1, 2, 3, 4, 5])
            stdp1.save_state(path)
            
            # Load into new instance
            stdp2 = STDPLearner()
            stdp2.load_state(path)
            
            # Should have same weights
            assert stdp1.token_weights == stdp2.token_weights
    
    def test_get_stats(self):
        """Should return statistics"""
        stdp = STDPLearner()
        stdp.process_sequence([1, 2, 3])
        
        stats = stdp.get_stats()
        
        assert 'total_updates' in stats
        assert 'active_tokens' in stats


class TestContentItem:
    """Tests for ContentItem"""
    
    def test_create_item(self):
        """Should create content item"""
        item = ContentItem(
            text="Test content",
            category=ContentCategory.CONVERSATION,
            priority=ProcessingPriority.HIGH,
            source="test"
        )
        
        assert item.text == "Test content"
        assert item.category == ContentCategory.CONVERSATION
        assert item.content_hash != ""
    
    def test_hash_is_deterministic(self):
        """Same content should have same hash"""
        item1 = ContentItem(
            text="Same text",
            category=ContentCategory.CONVERSATION,
            priority=ProcessingPriority.MEDIUM,
            source="test"
        )
        item2 = ContentItem(
            text="Same text",
            category=ContentCategory.CONVERSATION,
            priority=ProcessingPriority.MEDIUM,
            source="test"
        )
        
        assert item1.content_hash == item2.content_hash


class TestLearningConfig:
    """Tests for LearningConfig"""
    
    def test_default_config(self):
        """Should have sensible defaults"""
        config = LearningConfig()
        
        assert config.stdp_lr_plus > 0
        assert config.queue_size > 0
        assert config.save_interval_sec > 0


class MockEmbedder:
    """Mock embedder for testing"""
    embedding_dim = 384
    
    def get_embedding(self, text: str) -> np.ndarray:
        # Deterministic hash-based embedding
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(384).astype(np.float32)


class MockMemoryStore:
    """Mock memory store for testing"""
    def __init__(self):
        self.memories = []
    
    def store(self, embedding, text):
        self.memories.append({'embedding': embedding, 'text': text})
    
    def retrieve(self, embedding, k=3):
        return [m['text'] for m in self.memories[:k]]


class MockSNNCompute:
    """Mock SNN for testing"""
    def process(self, embedding):
        return {
            'spike_activity': float(np.abs(embedding).sum()),
            'spikes': np.zeros(100),
            'firing_rate': 0.1
        }


class TestContinuousLearner:
    """Tests for ContinuousLearner"""
    
    @pytest.fixture
    def learner(self):
        """Create learner with mocks"""
        return ContinuousLearner(
            memory_store=MockMemoryStore(),
            snn_compute=MockSNNCompute(),
            embedder=MockEmbedder(),
            config=LearningConfig(state_dir=tempfile.mkdtemp())
        )
    
    def test_init(self, learner):
        """Should initialize correctly"""
        assert learner.memory is not None
        assert learner.snn is not None
        assert learner.stdp is not None
        assert learner.event_bus is not None
    
    def test_learn_from_conversation(self, learner):
        """Should process conversation"""
        result = learner.learn_from_conversation(
            user_text="Hello, how are you?",
            response_text="I'm doing well, thanks!"
        )
        
        assert result['success']
        assert 'stdp_updates' in result
        assert 'spike_activity' in result
    
    def test_queue_content(self, learner):
        """Should queue content items"""
        item = ContentItem(
            text="Test content",
            category=ContentCategory.LOCAL_FILE,
            priority=ProcessingPriority.MEDIUM,
            source="test"
        )
        
        success = learner.queue_content(item)
        assert success
        assert learner.content_queue.qsize() == 1
    
    def test_duplicate_content_rejected(self, learner):
        """Should reject duplicate content"""
        item = ContentItem(
            text="Duplicate content",
            category=ContentCategory.LOCAL_FILE,
            priority=ProcessingPriority.MEDIUM,
            source="test"
        )
        
        learner.queue_content(item)
        learner.processed_hashes.add(item.content_hash)
        
        # Second attempt should fail
        success = learner.queue_content(item)
        assert not success
    
    def test_get_stats(self, learner):
        """Should return statistics"""
        learner.learn_from_conversation("Hello", "Hi")
        
        stats = learner.get_stats()
        
        assert 'items_processed' in stats
        assert 'conversations_learned' in stats
        assert 'stdp_stats' in stats
        assert stats['conversations_learned'] == 1
    
    @pytest.mark.asyncio
    async def test_start_stop(self, learner):
        """Should start and stop cleanly"""
        await learner.start()
        assert learner.is_running
        
        await learner.stop()
        assert not learner.is_running
    
    @pytest.mark.asyncio
    async def test_process_queued_item(self, learner):
        """Should process queued items"""
        await learner.start()
        
        item = ContentItem(
            text="Learn this content",
            category=ContentCategory.LOCAL_FILE,
            priority=ProcessingPriority.HIGH,
            source="test"
        )
        learner.queue_content(item)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        await learner.stop()
        
        assert learner.stats['items_processed'] >= 1


class TestIntegration:
    """Integration tests with real components"""
    
    @pytest.mark.skipif(
        not os.path.exists("models"),
        reason="Models directory not found"
    )
    def test_with_real_embedder(self):
        """Test with real GGUF embedder if available"""
        try:
            from model_gguf import Phi3GGUF
            from memory_store import MemoryStore
            from vulkan_backend import SNNCompute
            from config import find_gguf_model
            
            model_path = find_gguf_model()
            if not model_path:
                pytest.skip("GGUF model not available")
            
            with tempfile.TemporaryDirectory() as tmpdir:
                embedder = Phi3GGUF(model_path=model_path, n_gpu_layers=0)
                memory = MemoryStore(
                    db_path=os.path.join(tmpdir, "test.db"),
                    embedding_dim=embedder.embedding_dim
                )
                snn = SNNCompute(n_neurons=100, use_vulkan=False)
                
                learner = ContinuousLearner(
                    memory_store=memory,
                    snn_compute=snn,
                    embedder=embedder,
                    config=LearningConfig(state_dir=tmpdir)
                )
                
                result = learner.learn_from_conversation(
                    "What is machine learning?",
                    "Machine learning is a subset of AI..."
                )
                
                assert result['success']
                
        except ImportError as e:
            pytest.skip(f"Required module not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

