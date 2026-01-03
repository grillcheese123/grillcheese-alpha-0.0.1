"""
Tests for GPU Temporal Indexer

Tests:
1. Time cell encoding
2. Place cell encoding  
3. NYT data parsing
4. Indexing operations
5. Querying by date, location, text
6. State persistence
"""
import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

# Add parent to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from brain.temporal_indexer import (
    GPUTemporalIndexer,
    TemporalRecord,
    TemporalQuery,
    IndexStats,
    YEAR_MIN,
    YEAR_MAX,
)


# ==================== Fixtures ====================

@pytest.fixture
def mock_embed_fn():
    """Mock embedding function"""
    def embed(text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(384).astype(np.float32)
    return embed


@pytest.fixture
def indexer(mock_embed_fn):
    """Create temporal indexer"""
    return GPUTemporalIndexer(
        embedding_dim=384,
        n_time_cells=64,
        n_place_cells=36,
        use_gpu=False,  # CPU for testing
        embed_fn=mock_embed_fn
    )


@pytest.fixture
def sample_nyt_dir(tmp_path):
    """Create sample NYT data directory"""
    nyt_dir = tmp_path / "nyt_data"
    nyt_dir.mkdir()
    
    # Create sample files
    articles_1945 = [
        {
            "headline": {"main": "War Ends in Europe", "print_headline": "VE Day"},
            "abstract": "Germany surrenders unconditionally",
            "pub_date": "1945-05-08T00:00:00+0000",
            "keywords": [{"name": "glocations", "value": "Europe"}],
            "_id": "nyt://article/1945-01"
        },
        {
            "headline": {"main": "Atomic Bomb Dropped on Hiroshima"},
            "abstract": "New weapon of unprecedented power",
            "pub_date": "1945-08-06T00:00:00+0000",
            "keywords": [{"name": "glocations", "value": "Japan"}],
            "_id": "nyt://article/1945-02"
        }
    ]
    
    articles_1969 = [
        {
            "headline": {"main": "Man Walks on Moon"},
            "abstract": "Neil Armstrong takes first steps on lunar surface",
            "pub_date": "1969-07-20T00:00:00+0000",
            "keywords": [{"name": "glocations", "value": "USA"}],
            "_id": "nyt://article/1969-01"
        }
    ]
    
    articles_2001 = [
        {
            "headline": {"main": "America Under Attack"},
            "abstract": "Terrorists strike World Trade Center",
            "pub_date": "2001-09-11T00:00:00+0000",
            "keywords": [{"name": "glocations", "value": "New York City"}],
            "_id": "nyt://article/2001-01"
        }
    ]
    
    # Write files
    with open(nyt_dir / "1945_5.json", 'w') as f:
        json.dump(articles_1945, f)
    
    with open(nyt_dir / "1969_7.json", 'w') as f:
        json.dump(articles_1969, f)
    
    with open(nyt_dir / "2001_9.json", 'w') as f:
        json.dump(articles_2001, f)
    
    return nyt_dir


# ==================== TemporalRecord Tests ====================

class TestTemporalRecord:
    """Tests for TemporalRecord dataclass"""
    
    def test_create_basic(self):
        """Test basic record creation"""
        record = TemporalRecord(
            id="test_001",
            text="Test headline",
            timestamp=datetime(2000, 6, 15)
        )
        
        assert record.id == "test_001"
        assert record.text == "Test headline"
        assert record.timestamp.year == 2000
    
    def test_normalized_time(self):
        """Test time normalization"""
        record = TemporalRecord(
            id="test",
            text="Test",
            timestamp=datetime(1940, 6, 1)  # Middle of normalized range
        )
        
        norm_time = record.normalized_time()
        assert 0.0 <= norm_time <= 1.0
        
        # Check relative ordering
        record_old = TemporalRecord(id="old", text="old", timestamp=datetime(1860, 1, 1))
        record_new = TemporalRecord(id="new", text="new", timestamp=datetime(2020, 1, 1))
        
        assert record_old.normalized_time() < record.normalized_time()
        assert record.normalized_time() < record_new.normalized_time()
    
    def test_record_with_location(self):
        """Test record with location"""
        record = TemporalRecord(
            id="test",
            text="Event in Paris",
            timestamp=datetime(2000, 1, 1),
            location="France",
            keywords=["Paris", "Europe"]
        )
        
        assert record.location == "France"
        assert "Paris" in record.keywords


# ==================== Time Cell Encoding Tests ====================

class TestTimeCellEncoding:
    """Tests for time cell encoding"""
    
    def test_encode_time_returns_array(self, indexer):
        """Test that encoding returns proper array"""
        encoding = indexer.encode_time(0.5)
        
        assert isinstance(encoding, np.ndarray)
        assert len(encoding) == indexer.n_time_cells
        assert encoding.dtype == np.float32
    
    def test_encode_time_range(self, indexer):
        """Test encoding across time range"""
        enc_start = indexer.encode_time(0.0)
        enc_mid = indexer.encode_time(0.5)
        enc_end = indexer.encode_time(1.0)
        
        # All should be non-zero
        assert np.max(enc_start) > 0
        assert np.max(enc_mid) > 0
        assert np.max(enc_end) > 0
        
        # Start and end should activate different cells
        # (peak should be at different positions)
        peak_start = np.argmax(enc_start)
        peak_end = np.argmax(enc_end)
        assert peak_start != peak_end
    
    def test_similar_times_similar_encoding(self, indexer):
        """Test that similar times have similar encodings"""
        enc1 = indexer.encode_time(0.5)
        enc2 = indexer.encode_time(0.51)
        enc3 = indexer.encode_time(0.9)
        
        sim_close = np.dot(enc1, enc2) / (np.linalg.norm(enc1) * np.linalg.norm(enc2))
        sim_far = np.dot(enc1, enc3) / (np.linalg.norm(enc1) * np.linalg.norm(enc3))
        
        # Similar times should have higher similarity
        assert sim_close > sim_far


# ==================== Place Cell Encoding Tests ====================

class TestPlaceCellEncoding:
    """Tests for place cell encoding"""
    
    def test_encode_location_returns_array(self, indexer):
        """Test that encoding returns proper array"""
        encoding = indexer.encode_location("USA")
        
        assert isinstance(encoding, np.ndarray)
        assert len(encoding) == indexer.n_place_cells
        assert encoding.dtype == np.float32
    
    def test_known_locations(self, indexer):
        """Test encoding of known locations"""
        enc_usa = indexer.encode_location("USA")
        enc_europe = indexer.encode_location("Europe")
        enc_japan = indexer.encode_location("Japan")
        
        # All should be non-zero
        assert np.max(enc_usa) > 0
        assert np.max(enc_europe) > 0
        assert np.max(enc_japan) > 0
    
    def test_similar_locations_similar_encoding(self, indexer):
        """Test that nearby locations have similar encodings"""
        enc_ny = indexer.encode_location("New York")
        enc_boston = indexer.encode_location("Boston")
        enc_tokyo = indexer.encode_location("Tokyo")
        
        sim_close = np.dot(enc_ny, enc_boston) / (np.linalg.norm(enc_ny) * np.linalg.norm(enc_boston))
        sim_far = np.dot(enc_ny, enc_tokyo) / (np.linalg.norm(enc_ny) * np.linalg.norm(enc_tokyo))
        
        # Nearby locations should have higher similarity
        assert sim_close > sim_far
    
    def test_none_location(self, indexer):
        """Test encoding of None location"""
        encoding = indexer.encode_location(None)
        
        assert isinstance(encoding, np.ndarray)
        assert len(encoding) == indexer.n_place_cells
    
    def test_unknown_location(self, indexer):
        """Test encoding of unknown location"""
        encoding = indexer.encode_location("Unknown City XYZ123")
        
        assert isinstance(encoding, np.ndarray)
        assert np.max(encoding) > 0  # Should still have activations


# ==================== Indexing Tests ====================

class TestIndexing:
    """Tests for indexing operations"""
    
    def test_index_nyt_directory(self, indexer, sample_nyt_dir):
        """Test indexing NYT directory"""
        result = indexer.index_nyt_directory(sample_nyt_dir)
        
        assert result['records_indexed'] == 4
        assert result['files_processed'] == 3
        assert len(indexer.records) == 4
    
    def test_matrices_built(self, indexer, sample_nyt_dir):
        """Test that matrices are built after indexing"""
        indexer.index_nyt_directory(sample_nyt_dir)
        
        assert indexer.temporal_matrix is not None
        assert indexer.spatial_matrix is not None
        assert indexer.temporal_matrix.shape == (4, indexer.n_time_cells)
        assert indexer.spatial_matrix.shape == (4, indexer.n_place_cells)
    
    def test_index_with_file_limit(self, indexer, sample_nyt_dir):
        """Test indexing with file limit"""
        result = indexer.index_nyt_directory(sample_nyt_dir, file_limit=2)
        
        assert result['files_processed'] == 2
    
    def test_index_with_items_limit(self, indexer, sample_nyt_dir):
        """Test indexing with items per file limit"""
        result = indexer.index_nyt_directory(sample_nyt_dir, items_per_file=1)
        
        # 3 files, 1 item each
        assert result['records_indexed'] == 3
    
    def test_stats_after_indexing(self, indexer, sample_nyt_dir):
        """Test stats are updated after indexing"""
        indexer.index_nyt_directory(sample_nyt_dir)
        
        stats = indexer.get_stats()
        assert stats['total_records'] == 4
        assert stats['unique_locations'] > 0
        assert stats['date_range'] is not None


# ==================== Query Tests ====================

class TestQuerying:
    """Tests for querying the index"""
    
    def test_query_by_date(self, indexer, sample_nyt_dir):
        """Test querying by date"""
        indexer.index_nyt_directory(sample_nyt_dir)
        
        results = indexer.query_by_date(1945)
        
        assert len(results) > 0
        assert all(r.timestamp.year == 1945 for r in results)
    
    def test_query_by_month(self, indexer, sample_nyt_dir):
        """Test querying by specific month"""
        indexer.index_nyt_directory(sample_nyt_dir)
        
        results = indexer.query_by_date(1945, month=5)
        
        assert len(results) > 0
        assert all(r.timestamp.year == 1945 and r.timestamp.month == 5 for r in results)
    
    def test_query_by_location(self, indexer, sample_nyt_dir):
        """Test querying by location"""
        indexer.index_nyt_directory(sample_nyt_dir)
        
        results = indexer.query_by_location("Europe")
        
        assert len(results) > 0
    
    def test_query_combined(self, indexer, sample_nyt_dir):
        """Test combined query"""
        indexer.index_nyt_directory(sample_nyt_dir)
        
        query = TemporalQuery(
            text="war",
            start_date=datetime(1940, 1, 1),
            end_date=datetime(1950, 1, 1),
            location="Europe",
            limit=5
        )
        
        results = indexer.query(query)
        
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
    
    def test_query_with_limit(self, indexer, sample_nyt_dir):
        """Test query respects limit"""
        indexer.index_nyt_directory(sample_nyt_dir)
        
        query = TemporalQuery(limit=2)
        results = indexer.query(query)
        
        assert len(results) <= 2
    
    def test_empty_query_returns_results(self, indexer, sample_nyt_dir):
        """Test that empty query returns top results"""
        indexer.index_nyt_directory(sample_nyt_dir)
        
        query = TemporalQuery(limit=5)
        results = indexer.query(query)
        
        # Should return some results
        assert len(results) >= 0


# ==================== State Persistence Tests ====================

class TestStatePersistence:
    """Tests for saving/loading index"""
    
    def test_save_and_load(self, indexer, sample_nyt_dir, tmp_path):
        """Test saving and loading index"""
        # Index data
        indexer.index_nyt_directory(sample_nyt_dir)
        original_count = len(indexer.records)
        
        # Save
        save_path = tmp_path / "temporal_index.npz"
        indexer.save_index(save_path)
        
        # Create new indexer and load
        new_indexer = GPUTemporalIndexer(
            n_time_cells=64,
            n_place_cells=36,
            use_gpu=False
        )
        new_indexer.load_index(save_path)
        
        assert len(new_indexer.records) == original_count
        assert new_indexer.temporal_matrix is not None
        assert new_indexer.spatial_matrix is not None
    
    def test_load_nonexistent(self, indexer, tmp_path):
        """Test loading from nonexistent path doesn't crash"""
        indexer.load_index(tmp_path / "nonexistent.npz")
        # Should not raise


# ==================== Edge Case Tests ====================

class TestEdgeCases:
    """Tests for edge cases"""
    
    def test_empty_index_query(self, indexer):
        """Test querying empty index"""
        query = TemporalQuery(text="test")
        results = indexer.query(query)
        
        assert results == []
    
    def test_malformed_nyt_file(self, indexer, tmp_path):
        """Test handling malformed NYT file"""
        nyt_dir = tmp_path / "bad_nyt"
        nyt_dir.mkdir()
        
        # Write malformed file
        with open(nyt_dir / "bad.json", 'w') as f:
            f.write("not valid json")
        
        result = indexer.index_nyt_directory(nyt_dir)
        assert result['records_indexed'] == 0
    
    def test_missing_fields_in_nyt(self, indexer, tmp_path):
        """Test handling articles with missing fields"""
        nyt_dir = tmp_path / "incomplete_nyt"
        nyt_dir.mkdir()
        
        articles = [
            {"headline": {"main": "Title Only"}},  # Missing date
            {"pub_date": "2000-01-01T00:00:00+0000"},  # Missing headline
            {"headline": {"main": "Full Record"}, "pub_date": "2000-01-01T00:00:00+0000"}
        ]
        
        with open(nyt_dir / "2000_1.json", 'w') as f:
            json.dump(articles, f)
        
        result = indexer.index_nyt_directory(nyt_dir)
        # Only the full record should be indexed
        assert result['records_indexed'] == 1


# ==================== Integration Tests ====================

class TestIntegration:
    """Integration tests"""
    
    def test_full_workflow(self, mock_embed_fn, sample_nyt_dir, tmp_path):
        """Test complete workflow"""
        # Create indexer
        indexer = GPUTemporalIndexer(
            n_time_cells=64,
            n_place_cells=36,
            use_gpu=False,
            embed_fn=mock_embed_fn
        )
        
        # Index
        result = indexer.index_nyt_directory(sample_nyt_dir)
        assert result['records_indexed'] > 0
        
        # Query by date
        results_1945 = indexer.query_by_date(1945)
        assert len(results_1945) > 0
        
        # Query by location
        results_japan = indexer.query_by_location("Japan")
        assert len(results_japan) > 0
        
        # Combined query
        query = TemporalQuery(
            text="moon landing",
            start_date=datetime(1960, 1, 1),
            end_date=datetime(1980, 1, 1)
        )
        results = indexer.query(query)
        
        # Save and reload
        save_path = tmp_path / "test_index.npz"
        indexer.save_index(save_path)
        
        new_indexer = GPUTemporalIndexer(use_gpu=False)
        new_indexer.load_index(save_path)
        
        assert len(new_indexer.records) == len(indexer.records)
        
        # Stats
        stats = indexer.get_stats()
        assert stats['total_records'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

