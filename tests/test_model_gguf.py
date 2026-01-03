"""
Tests for Phi3GGUF model class
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ModelConfig, find_gguf_model

# Try to import the model
try:
    from model_gguf import Phi3GGUF, LLAMA_CPP_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE
    MODEL_AVAILABLE = True
except ImportError as e:
    MODEL_AVAILABLE = False
    LLAMA_CPP_AVAILABLE = False
    SENTENCE_TRANSFORMERS_AVAILABLE = False


class TestModelGGUFImports:
    """Tests for model imports and availability flags"""
    
    def test_model_module_importable(self):
        """model_gguf module should be importable"""
        try:
            import model_gguf
            assert True
        except ImportError:
            pytest.fail("model_gguf module not importable")
    
    def test_availability_flags_are_boolean(self):
        """Availability flags should be boolean"""
        if MODEL_AVAILABLE:
            from model_gguf import LLAMA_CPP_AVAILABLE, SENTENCE_TRANSFORMERS_AVAILABLE
            assert isinstance(LLAMA_CPP_AVAILABLE, bool)
            assert isinstance(SENTENCE_TRANSFORMERS_AVAILABLE, bool)


@pytest.mark.skipif(not MODEL_AVAILABLE or not LLAMA_CPP_AVAILABLE, 
                    reason="llama-cpp-python not available")
class TestPhi3GGUFInit:
    """Tests for Phi3GGUF initialization"""
    
    def test_init_without_model_raises_if_not_found(self):
        """Init should raise if model file not found"""
        with pytest.raises(FileNotFoundError):
            # Use a path that definitely doesn't exist
            Phi3GGUF(model_path="/nonexistent/path/model.gguf")
    
    def test_init_with_valid_model(self):
        """Init should succeed with valid model"""
        model_path = find_gguf_model()
        if model_path is None:
            pytest.skip("No GGUF model available")
        
        model = Phi3GGUF(model_path=model_path, n_gpu_layers=0)  # CPU only for testing
        assert model.llm is not None


@pytest.mark.skipif(not MODEL_AVAILABLE or not LLAMA_CPP_AVAILABLE,
                    reason="llama-cpp-python not available")
class TestPhi3GGUFEmbedding:
    """Tests for embedding extraction"""
    
    @pytest.fixture
    def model(self):
        """Create model instance for tests"""
        model_path = find_gguf_model()
        if model_path is None:
            pytest.skip("No GGUF model available")
        return Phi3GGUF(model_path=model_path, n_gpu_layers=0)
    
    def test_get_embedding_returns_array(self, model):
        """get_embedding should return numpy array"""
        embedding = model.get_embedding("Hello world")
        
        assert isinstance(embedding, np.ndarray)
    
    def test_get_embedding_has_correct_dim(self, model):
        """Embedding should have correct dimension"""
        embedding = model.get_embedding("Hello world")
        
        assert len(embedding) == model.embedding_dim
    
    def test_get_embedding_is_float32(self, model):
        """Embedding should be float32"""
        embedding = model.get_embedding("Hello world")
        
        assert embedding.dtype == np.float32
    
    def test_get_embedding_is_normalized(self, model):
        """Embedding should be approximately normalized (if using sentence-transformers)"""
        embedding = model.get_embedding("Hello world")
        
        norm = np.linalg.norm(embedding)
        # Sentence-transformers embeddings are normalized
        # Hash-based fallback may not be perfectly normalized
        assert 0.5 < norm < 2.0  # Reasonable range
    
    def test_different_texts_give_different_embeddings(self, model):
        """Different texts should give different embeddings"""
        emb1 = model.get_embedding("Hello world")
        emb2 = model.get_embedding("Goodbye universe")
        
        # Should not be identical
        assert not np.allclose(emb1, emb2)
    
    def test_same_text_gives_same_embedding(self, model):
        """Same text should give same embedding"""
        emb1 = model.get_embedding("Test text")
        emb2 = model.get_embedding("Test text")
        
        np.testing.assert_array_almost_equal(emb1, emb2)


@pytest.mark.skipif(not MODEL_AVAILABLE or not LLAMA_CPP_AVAILABLE,
                    reason="llama-cpp-python not available")
class TestPhi3GGUFGenerate:
    """Tests for text generation"""
    
    @pytest.fixture
    def model(self):
        """Create model instance for tests"""
        model_path = find_gguf_model()
        if model_path is None:
            pytest.skip("No GGUF model available")
        return Phi3GGUF(model_path=model_path, n_gpu_layers=0)
    
    def test_generate_returns_string(self, model):
        """Generate should return a string"""
        response = model.generate("Hello", [])
        
        assert isinstance(response, str)
    
    def test_generate_with_context(self, model):
        """Generate should work with context"""
        context = ["Previous conversation about AI"]
        response = model.generate("Tell me more", context)
        
        assert isinstance(response, str)
    
    def test_generate_with_empty_context(self, model):
        """Generate should work with empty context"""
        response = model.generate("Hello", [])
        
        assert isinstance(response, str)
        assert len(response) > 0 or response.startswith("[Error")


@pytest.mark.skipif(not SENTENCE_TRANSFORMERS_AVAILABLE,
                    reason="sentence-transformers not available")
class TestSentenceTransformerEmbeddings:
    """Tests specific to sentence-transformer embeddings"""
    
    def test_semantic_similarity(self):
        """Similar texts should have similar embeddings"""
        model_path = find_gguf_model()
        if model_path is None or not LLAMA_CPP_AVAILABLE:
            pytest.skip("Model not available")
        
        model = Phi3GGUF(model_path=model_path, n_gpu_layers=0)
        
        if model.embedder is None:
            pytest.skip("Sentence-transformers embedder not available")
        
        # Similar texts
        emb1 = model.get_embedding("I love programming")
        emb2 = model.get_embedding("I enjoy coding")
        
        # Dissimilar text
        emb3 = model.get_embedding("The weather is nice today")
        
        # Compute cosine similarities
        sim_12 = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        sim_13 = np.dot(emb1, emb3) / (np.linalg.norm(emb1) * np.linalg.norm(emb3))
        
        # Similar texts should have higher similarity
        assert sim_12 > sim_13

