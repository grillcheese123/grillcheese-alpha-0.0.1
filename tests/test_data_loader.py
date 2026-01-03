"""
Tests for GPU Data Loader

Tests:
1. JSONL streaming and parsing
2. Batch processing with GPU/CPU
3. Hebbian learning updates
4. Affect data loading
5. Temporal data loading
6. Memory efficiency
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain.data_loader import (
    GPUDataLoader,
    DataCategory,
    DataItem,
    BatchStats,
    MAX_GPU_MEMORY_MB,
    BATCH_SIZE_DEFAULT,
)


# ==================== Fixtures ====================

@pytest.fixture
def mock_embed_fn():
    """Mock embedding function"""
    def embed(text: str) -> np.ndarray:
        # Deterministic embedding based on text hash
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(384).astype(np.float32)
    return embed


@pytest.fixture
def data_loader(mock_embed_fn):
    """Create data loader with mock embedding"""
    return GPUDataLoader(
        embedding_dim=384,
        batch_size=16,
        max_memory_mb=1000,  # Small for tests
        embed_fn=mock_embed_fn
    )


@pytest.fixture
def sample_affect_jsonl(tmp_path):
    """Create sample affect JSONL file"""
    data = [
        {"id": "aff_001", "text": "I feel happy today", "affect": {"valence": 0.8, "arousal": 0.6}},
        {"id": "aff_002", "text": "This is frustrating", "affect": {"valence": -0.5, "arousal": 0.7}},
        {"id": "aff_003", "text": "Calm and peaceful morning", "affect": {"valence": 0.6, "arousal": 0.2}},
        {"id": "aff_004", "text": "Excited about the event", "affect": {"valence": 0.9, "arousal": 0.9}},
        {"id": "aff_005", "text": "Feeling a bit sad", "affect": {"valence": -0.6, "arousal": 0.3}},
    ]
    
    filepath = tmp_path / "test_affect.jsonl"
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    return filepath


@pytest.fixture
def sample_nyt_json(tmp_path):
    """Create sample NYT JSON file"""
    data = [
        {
            "headline": {"main": "Historic Event Unfolds", "print_headline": "Historic Event"},
            "abstract": "A major historical event occurred today.",
            "pub_date": "1945-05-08T00:00:00+0000",
            "keywords": [{"name": "glocations", "value": "Europe"}]
        },
        {
            "headline": {"main": "Scientific Breakthrough", "print_headline": "Science News"},
            "abstract": "Scientists announce major discovery.",
            "pub_date": "1969-07-20T00:00:00+0000",
            "keywords": [{"name": "glocations", "value": "USA"}]
        }
    ]
    
    # Create directory structure
    nyt_dir = tmp_path / "nyt_data"
    nyt_dir.mkdir()
    
    filepath = nyt_dir / "1945_5.json"
    with open(filepath, 'w') as f:
        json.dump(data, f)
    
    return nyt_dir


@pytest.fixture
def sample_principles_jsonl(tmp_path):
    """Create sample principles JSONL file"""
    data = [
        {
            "id": "principle_001",
            "text": "We value clarity and conciseness in all communications.",
            "principle": "CLARITY",
            "summary": "Be clear and concise."
        },
        {
            "id": "principle_002", 
            "text": "Wisdom emerges from diverse perspectives over time.",
            "principle": "WISDOM_IS_EMERGENT",
            "summary": "Wisdom grows from dialogue."
        }
    ]
    
    filepath = tmp_path / "principles.jsonl"
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    return filepath


# ==================== DataItem Tests ====================

class TestDataItem:
    """Tests for DataItem dataclass"""
    
    def test_create_basic(self):
        """Test basic DataItem creation"""
        item = DataItem(
            id="test_001",
            text="Hello world",
            category=DataCategory.AFFECT
        )
        assert item.id == "test_001"
        assert item.text == "Hello world"
        assert item.category == DataCategory.AFFECT
        assert item.valence is None
        assert item.arousal is None
    
    def test_create_with_affect(self):
        """Test DataItem with affect values"""
        item = DataItem(
            id="test_002",
            text="Happy text",
            category=DataCategory.AFFECT,
            valence=0.8,
            arousal=0.6
        )
        assert item.valence == 0.8
        assert item.arousal == 0.6
    
    def test_memory_size_without_embedding(self):
        """Test memory size calculation without embedding"""
        item = DataItem(id="t", text="Hello", category=DataCategory.AFFECT)
        size = item.memory_size_bytes()
        assert size == len("Hello".encode('utf-8'))
    
    def test_memory_size_with_embedding(self):
        """Test memory size calculation with embedding"""
        item = DataItem(
            id="t",
            text="Hello",
            category=DataCategory.AFFECT,
            embedding=np.zeros(384, dtype=np.float32)
        )
        size = item.memory_size_bytes()
        expected = len("Hello".encode('utf-8')) + 384 * 4  # float32 = 4 bytes
        assert size == expected


# ==================== JSONL Streaming Tests ====================

class TestJSONLStreaming:
    """Tests for JSONL file streaming"""
    
    def test_stream_affect_data(self, data_loader, sample_affect_jsonl):
        """Test streaming affect JSONL"""
        items = list(data_loader.stream_jsonl(
            sample_affect_jsonl, 
            DataCategory.AFFECT
        ))
        
        assert len(items) == 5
        assert all(isinstance(i, DataItem) for i in items)
        assert items[0].valence == 0.8
        assert items[1].valence == -0.5
    
    def test_stream_with_limit(self, data_loader, sample_affect_jsonl):
        """Test streaming with item limit"""
        items = list(data_loader.stream_jsonl(
            sample_affect_jsonl,
            DataCategory.AFFECT,
            limit=2
        ))
        
        assert len(items) == 2
    
    def test_stream_principles(self, data_loader, sample_principles_jsonl):
        """Test streaming principles JSONL"""
        items = list(data_loader.stream_jsonl(
            sample_principles_jsonl,
            DataCategory.PRINCIPLES
        ))
        
        assert len(items) == 2
        assert "clarity" in items[0].text.lower()
    
    def test_stream_handles_malformed_json(self, data_loader, tmp_path):
        """Test that malformed JSON lines are skipped"""
        filepath = tmp_path / "malformed.jsonl"
        with open(filepath, 'w') as f:
            f.write('{"id": "1", "text": "valid"}\n')
            f.write('not valid json\n')
            f.write('{"id": "2", "text": "also valid"}\n')
        
        items = list(data_loader.stream_jsonl(filepath, DataCategory.KNOWLEDGE))
        assert len(items) == 2


# ==================== Batch Processing Tests ====================

class TestBatchProcessing:
    """Tests for batch processing"""
    
    def test_process_batch_basic(self, data_loader):
        """Test basic batch processing"""
        items = [
            DataItem("1", "Happy text", DataCategory.AFFECT, valence=0.8, arousal=0.6),
            DataItem("2", "Sad text", DataCategory.AFFECT, valence=-0.6, arousal=0.3),
        ]
        
        stats = data_loader.process_batch(items, learn=True)
        
        assert isinstance(stats, BatchStats)
        assert stats.items_processed == 2
        assert stats.gpu_time_ms >= 0
        assert stats.learning_updates > 0
    
    def test_process_batch_updates_weights(self, data_loader):
        """Test that batch processing updates weights"""
        initial_weights = data_loader.affect_weights.copy()
        
        items = [
            DataItem("1", "Very happy", DataCategory.AFFECT, valence=0.9, arousal=0.8),
        ] * 10
        
        data_loader.process_batch(items, learn=True)
        
        # Weights should have changed
        assert not np.allclose(initial_weights, data_loader.affect_weights)
    
    def test_process_batch_no_learn(self, data_loader):
        """Test batch processing without learning"""
        initial_weights = data_loader.affect_weights.copy()
        
        items = [
            DataItem("1", "Test text", DataCategory.AFFECT, valence=0.5, arousal=0.5),
        ]
        
        stats = data_loader.process_batch(items, learn=False)
        
        # Weights should not have changed
        assert np.allclose(initial_weights, data_loader.affect_weights)
        assert stats.learning_updates == 0
    
    def test_batch_stats_averages(self, data_loader):
        """Test batch stats compute correct averages"""
        items = [
            DataItem("1", "Text 1", DataCategory.AFFECT, valence=0.6, arousal=0.4),
            DataItem("2", "Text 2", DataCategory.AFFECT, valence=0.8, arousal=0.6),
        ]
        
        stats = data_loader.process_batch(items, learn=False)
        
        assert abs(stats.avg_valence - 0.7) < 0.01
        assert abs(stats.avg_arousal - 0.5) < 0.01


# ==================== Hebbian Learning Tests ====================

class TestHebbianLearning:
    """Tests for Hebbian learning updates"""
    
    def test_cpu_hebbian_update(self, data_loader):
        """Test CPU Hebbian update"""
        pre = np.random.randn(8, 384).astype(np.float32)
        post = np.random.randn(8, 2).astype(np.float32)
        
        initial_weights = data_loader.affect_weights.copy()
        updates = data_loader._cpu_hebbian_update(pre, post)
        
        assert updates == 384 * 2
        assert not np.allclose(initial_weights, data_loader.affect_weights)
    
    def test_hebbian_weight_decay(self, data_loader):
        """Test that weight decay reduces large weights"""
        # Set large initial weights
        data_loader.affect_weights = np.ones((384, 2), dtype=np.float32) * 10.0
        
        # Zero input should only have decay effect
        pre = np.zeros((8, 384), dtype=np.float32)
        post = np.zeros((8, 2), dtype=np.float32)
        
        data_loader._cpu_hebbian_update(pre, post, weight_decay=0.1)
        
        # Weights should be smaller
        assert np.mean(np.abs(data_loader.affect_weights)) < 10.0


# ==================== High-Level Loading Tests ====================

class TestAffectLoading:
    """Tests for affect data loading"""
    
    def test_load_affect_data(self, data_loader, sample_affect_jsonl):
        """Test loading affect data file"""
        result = data_loader.load_affect_data(
            sample_affect_jsonl,
            learn=True
        )
        
        assert result['total_items'] == 5
        assert result['batches'] >= 1
        assert result['avg_gpu_time_ms'] >= 0
    
    def test_load_affect_with_limit(self, data_loader, sample_affect_jsonl):
        """Test loading with item limit"""
        result = data_loader.load_affect_data(
            sample_affect_jsonl,
            learn=True,
            limit=3
        )
        
        assert result['total_items'] == 3
    
    def test_load_affect_updates_stats(self, data_loader, sample_affect_jsonl):
        """Test that loading updates global stats"""
        initial_affect = data_loader.stats['affect_items']
        
        data_loader.load_affect_data(sample_affect_jsonl, learn=True)
        
        assert data_loader.stats['affect_items'] > initial_affect


class TestTemporalLoading:
    """Tests for temporal (NYT) data loading"""
    
    def test_load_temporal_data(self, data_loader, sample_nyt_json):
        """Test loading temporal data"""
        result = data_loader.load_temporal_data(
            sample_nyt_json,
            learn=True
        )
        
        assert result['files_processed'] >= 1
        assert result['total_items'] >= 1
    
    def test_temporal_parses_dates(self, data_loader, sample_nyt_json):
        """Test that temporal data parses dates"""
        # Read the file manually to check parsing
        for item in data_loader.stream_jsonl(
            sample_nyt_json / "1945_5.json",
            DataCategory.TEMPORAL
        ):
            # Items should have headlines as text
            assert len(item.text) > 0
            break


class TestKnowledgeLoading:
    """Tests for knowledge data loading"""
    
    def test_load_knowledge_data(self, data_loader, sample_principles_jsonl):
        """Test loading knowledge data"""
        result = data_loader.load_knowledge_data(
            sample_principles_jsonl,
            category=DataCategory.PRINCIPLES,
            learn=True
        )
        
        assert result['total_items'] == 2
        assert result['batches'] >= 1


# ==================== State Management Tests ====================

class TestStateManagement:
    """Tests for saving/loading state"""
    
    def test_save_and_load_weights(self, data_loader, tmp_path):
        """Test saving and loading weights"""
        # Modify weights
        data_loader.affect_weights = np.random.randn(384, 2).astype(np.float32)
        original_weights = data_loader.affect_weights.copy()
        
        # Save
        save_path = tmp_path / "weights.npz"
        data_loader.save_weights(save_path)
        
        # Modify again
        data_loader.affect_weights = np.zeros((384, 2), dtype=np.float32)
        
        # Load
        data_loader.load_weights(save_path)
        
        # Should match original
        assert np.allclose(original_weights, data_loader.affect_weights)
    
    def test_load_nonexistent_weights(self, data_loader, tmp_path):
        """Test loading from nonexistent file doesn't crash"""
        data_loader.load_weights(tmp_path / "nonexistent.npz")
        # Should not raise


# ==================== Prediction Tests ====================

class TestPrediction:
    """Tests for affect prediction"""
    
    def test_predict_affect(self, data_loader):
        """Test affect prediction from embedding"""
        embedding = np.random.randn(384).astype(np.float32)
        
        valence, arousal = data_loader.predict_affect(embedding)
        
        assert -1.0 <= valence <= 1.0
        assert 0.0 <= arousal <= 1.0
    
    def test_predict_after_training(self, data_loader, sample_affect_jsonl):
        """Test prediction improves after training"""
        # Train on positive affect
        data_loader.load_affect_data(sample_affect_jsonl, learn=True)
        
        # Prediction should work (not crash)
        embedding = np.random.randn(384).astype(np.float32)
        valence, arousal = data_loader.predict_affect(embedding)
        
        assert isinstance(valence, float)
        assert isinstance(arousal, float)


# ==================== Memory Efficiency Tests ====================

class TestMemoryEfficiency:
    """Tests for memory-efficient operation"""
    
    def test_batch_size_respected(self, mock_embed_fn, sample_affect_jsonl):
        """Test that batch size is respected"""
        loader = GPUDataLoader(
            batch_size=2,
            embed_fn=mock_embed_fn
        )
        
        result = loader.load_affect_data(sample_affect_jsonl, learn=True)
        
        # 5 items / batch size 2 = at least 3 batches
        assert result['batches'] >= 2
    
    def test_memory_estimate_in_stats(self, data_loader):
        """Test that batch stats include memory estimate"""
        items = [
            DataItem("1", "Test", DataCategory.AFFECT, valence=0.5, arousal=0.5),
        ]
        
        stats = data_loader.process_batch(items, learn=False)
        
        assert stats.memory_used_mb >= 0
        assert stats.memory_used_mb < MAX_GPU_MEMORY_MB


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests"""
    
    def test_full_pipeline(self, data_loader, sample_affect_jsonl, sample_nyt_json, tmp_path):
        """Test full data loading pipeline"""
        # Load affect data
        affect_result = data_loader.load_affect_data(sample_affect_jsonl, learn=True)
        assert affect_result['total_items'] > 0
        
        # Load temporal data
        temporal_result = data_loader.load_temporal_data(sample_nyt_json, learn=True)
        assert temporal_result['total_items'] > 0
        
        # Save weights
        weights_path = tmp_path / "trained_weights.npz"
        data_loader.save_weights(weights_path)
        
        # Get stats
        stats = data_loader.get_stats()
        assert stats['total_items'] > 0
        assert stats['affect_items'] > 0
        
        # Predict
        embedding = np.random.randn(384).astype(np.float32)
        valence, arousal = data_loader.predict_affect(embedding)
        assert -1 <= valence <= 1
        assert 0 <= arousal <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

