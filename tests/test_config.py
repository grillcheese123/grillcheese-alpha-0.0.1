"""
Tests for config module
"""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ModelConfig,
    MemoryConfig,
    SNNConfig,
    ServerConfig,
    LogConfig,
    find_gguf_model,
    BASE_DIR
)


class TestModelConfig:
    """Tests for ModelConfig"""
    
    def test_embedding_dimensions_valid(self):
        """Embedding dimensions should be positive integers"""
        assert ModelConfig.PHI3_EMBEDDING_DIM > 0
        assert ModelConfig.SENTENCE_TRANSFORMER_DIM > 0
        assert ModelConfig.EMBEDDING_DIM > 0
    
    def test_generation_settings_valid(self):
        """Generation settings should be in valid ranges"""
        assert ModelConfig.MAX_NEW_TOKENS_GPU > 0
        assert ModelConfig.MAX_NEW_TOKENS_CPU > 0
        assert 0 < ModelConfig.TEMPERATURE <= 2.0
        assert 0 < ModelConfig.TOP_P <= 1.0
        assert ModelConfig.MAX_CONTEXT_ITEMS > 0
    
    def test_gguf_model_paths_exist(self):
        """GGUF model paths should be a non-empty list"""
        assert len(ModelConfig.GGUF_MODEL_PATHS) > 0
        for path in ModelConfig.GGUF_MODEL_PATHS:
            assert isinstance(path, str)


class TestMemoryConfig:
    """Tests for MemoryConfig"""
    
    def test_memory_settings_valid(self):
        """Memory settings should be positive"""
        assert MemoryConfig.MAX_MEMORIES > 0
        assert MemoryConfig.EMBEDDING_DIM > 0
        assert MemoryConfig.DEFAULT_K > 0
        assert MemoryConfig.GPU_BUFFER_SIZE > 0
    
    def test_db_path_valid(self):
        """DB path should be a valid string"""
        assert isinstance(MemoryConfig.DB_PATH, str)
        assert len(MemoryConfig.DB_PATH) > 0


class TestSNNConfig:
    """Tests for SNNConfig"""
    
    def test_neuron_count_valid(self):
        """Neuron count should be positive"""
        assert SNNConfig.N_NEURONS > 0
    
    def test_lif_parameters_valid(self):
        """LIF parameters should be in valid ranges"""
        assert SNNConfig.DT > 0
        assert SNNConfig.TAU_MEM > 0
        assert SNNConfig.V_THRESH > 0
        assert SNNConfig.TIMESTEPS > 0
        assert SNNConfig.INPUT_SCALE > 0


class TestServerConfig:
    """Tests for ServerConfig"""
    
    def test_server_settings_valid(self):
        """Server settings should be valid"""
        assert isinstance(ServerConfig.HOST, str)
        assert isinstance(ServerConfig.PORT, int)
        assert 1 <= ServerConfig.PORT <= 65535
        assert ServerConfig.WS_PATH.startswith("/")


class TestLogConfig:
    """Tests for LogConfig"""
    
    def test_log_settings_valid(self):
        """Log settings should be valid"""
        assert LogConfig.LEVEL in ["DEBUG", "INFO", "WARNING", "ERROR"]
        assert isinstance(LogConfig.FORMAT, str)
        # ASCII-safe symbols
        assert LogConfig.CHECK.isascii()
        assert LogConfig.CROSS.isascii()
        assert LogConfig.WARNING.isascii()


class TestFindGgufModel:
    """Tests for find_gguf_model function"""
    
    def test_returns_string_or_none(self):
        """Should return string path or None"""
        result = find_gguf_model()
        assert result is None or isinstance(result, str)
    
    def test_returned_path_exists_if_not_none(self):
        """If a path is returned, it should exist"""
        result = find_gguf_model()
        if result is not None:
            assert Path(result).exists()


class TestBasePaths:
    """Tests for base path configuration"""
    
    def test_base_dir_exists(self):
        """BASE_DIR should exist"""
        assert BASE_DIR.exists()
        assert BASE_DIR.is_dir()

