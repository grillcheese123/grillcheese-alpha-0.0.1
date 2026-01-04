"""
Test script for multimodal integration with existing GrillCheese system
Run this to validate the integration works with your FAISS shaders
"""

import sys
import io
import numpy as np
from pathlib import Path

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def test_encoder_dimension():
    """Test that encoder produces 384-dim embeddings"""
    print("\n1. Testing encoder dimension...")
    
    try:
        # Add parent directory to path
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from learning.multimodal_encoder import MultimodalEncoder
        
        encoder = MultimodalEncoder(Path("./models"))
        loaded = encoder.load_text_encoder()
        
        if not loaded:
            print("  ⚠ Text encoder not loaded (will download on first use)")
            return False
            
        # Test encoding
        text = "Hello world"
        embedding = encoder.encode_text(text)
        
        if embedding.shape[0] == 384:
            print(f"  ✓ Embedding dimension: {embedding.shape[0]} (matches config)")
            return True
        else:
            print(f"  ✗ Embedding dimension: {embedding.shape[0]} (expected 384)")
            return False
            
    except Exception as e:
        print(f"  ✗ Encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_integration():
    """Test integration with existing MemoryStore"""
    print("\n2. Testing memory store integration...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from learning.multimodal_encoder import MultimodalEncoder
        from memory_store import MemoryStore
        from config import MemoryConfig
        
        # Initialize
        encoder = MultimodalEncoder(Path("./models"))
        encoder.load_text_encoder()
        
        # Create test memory store
        db_path = Path("./test_integration.db")
        if db_path.exists():
            db_path.unlink()
            
        memory = MemoryStore(db_path=str(db_path))
        
        # Test storing
        text = "This is a test memory"
        embedding = encoder.encode_text(text)
        memory.store(embedding, text, is_protected=False)
        print("  ✓ Stored memory with encoder embedding")
        
        # Test retrieval
        query_text = "test memory"
        query_embedding = encoder.encode_text(query_text)
        results = memory.retrieve(query_embedding, k=1)
        
        if len(results) > 0:
            print(f"  ✓ Retrieved: '{results[0]}'")
            
            # Cleanup
            db_path.unlink()
            return True
        else:
            print("  ✗ No results retrieved")
            return False
            
    except Exception as e:
        print(f"  ✗ Memory integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multilingual():
    """Test multilingual encoding"""
    print("\n3. Testing multilingual encoding...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from learning.multimodal_encoder import MultimodalEncoder
        from learning.multilingual_utils import MultilingualProcessor
        
        encoder = MultimodalEncoder(Path("./models"))
        encoder.load_text_encoder()
        
        lang_processor = MultilingualProcessor()
        
        test_texts = [
            ("Hello world", "en"),
            ("Hola mundo", "es"),
            ("Bonjour le monde", "fr"),
            ("你好世界", "zh"),
        ]
        
        all_passed = True
        for text, expected_lang in test_texts:
            detected_lang = lang_processor.detect_language(text)
            embedding = encoder.encode_text(text)
            
            if detected_lang == expected_lang and embedding.shape[0] == 384:
                print(f"  ✓ [{detected_lang}] '{text}' → 384-dim")
            else:
                print(f"  ✗ [{detected_lang}] '{text}' (expected {expected_lang})")
                all_passed = False
                
        return all_passed
        
    except Exception as e:
        print(f"  ✗ Multilingual test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_faiss_shaders():
    """Test that FAISS shaders work with encoder embeddings"""
    print("\n4. Testing FAISS shader integration...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from learning.multimodal_encoder import MultimodalEncoder
        from memory_store import MemoryStore
        
        encoder = MultimodalEncoder(Path("./models"))
        encoder.load_text_encoder()
        
        # Create test database
        db_path = Path("./test_faiss.db")
        if db_path.exists():
            db_path.unlink()
            
        memory = MemoryStore(db_path=str(db_path))
        
        # Add multiple memories
        test_memories = [
            "The cat sat on the mat",
            "Dogs are loyal animals",
            "Photosynthesis converts light to energy",
            "Python is a programming language",
            "The weather is sunny today"
        ]
        
        for text in test_memories:
            embedding = encoder.encode_text(text)
            memory.store(embedding, text)
            
        print(f"  ✓ Stored {len(test_memories)} memories")
        
        # Test retrieval with GPU FAISS shaders
        query = "Tell me about cats"
        query_embedding = encoder.encode_text(query)
        results = memory.retrieve(query_embedding, k=3)
        
        print(f"  ✓ Retrieved {len(results)} results")
        print(f"    Top result: '{results[0]}'")
        
        # Verify GPU was used
        if memory._use_gpu_similarity:
            print("  ✓ GPU FAISS shaders active")
        else:
            print("  ⚠ Using CPU fallback")
            
        # Cleanup
        db_path.unlink()
        return True
        
    except Exception as e:
        print(f"  ✗ FAISS shader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_distillation():
    """Test knowledge distillation with encoder"""
    print("\n5. Testing knowledge distillation...")
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        
        from learning.multimodal_encoder import MultimodalEncoder
        from learning.multilingual_utils import MultilingualProcessor
        from learning.knowledge_distillation import KnowledgeDistillation
        from memory_store import MemoryStore
        
        # Initialize components
        encoder = MultimodalEncoder(Path("./models"))
        encoder.load_text_encoder()
        
        lang_processor = MultilingualProcessor()
        
        db_path = Path("./test_distillation.db")
        if db_path.exists():
            db_path.unlink()
            
        memory = MemoryStore(db_path=str(db_path))
        
        distillation = KnowledgeDistillation(
            memory_store=memory,
            encoder=encoder,
            lang_processor=lang_processor,
            quality_threshold=0.7
        )
        
        # Test interaction distillation
        knowledge = distillation.distill_interaction(
            user_message="What is AI?",
            assistant_response="Artificial Intelligence is the simulation of human intelligence by machines.",
            quality_score=0.85
        )
        
        if knowledge:
            print(f"  ✓ Distilled interaction (quality: {knowledge.quality_score:.2f})")
            print(f"  ✓ Language detected: {knowledge.language}")
            print(f"  ✓ Embedding shape: {knowledge.embedding.shape}")
            
            # Cleanup
            db_path.unlink()
            return True
        else:
            print("  ✗ Distillation failed")
            return False
            
    except Exception as e:
        print(f"  ✗ Distillation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all integration tests"""
    print("="*60)
    print("Multimodal Integration Tests - GrillCheese")
    print("="*60)
    
    results = []
    
    results.append(("Encoder Dimension", test_encoder_dimension()))
    results.append(("Memory Integration", test_memory_integration()))
    results.append(("Multilingual Support", test_multilingual()))
    results.append(("FAISS Shaders", test_faiss_shaders()))
    results.append(("Knowledge Distillation", test_distillation()))
    
    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8s} {name}")
        
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print("="*60)
    print(f"Total: {passed}/{total} tests passed")
    print("="*60)
    
    if passed == total:
        print("\n✓ All tests passed! Integration successful.")
        print("\nNext steps:")
        print("1. Update cli/cli.py with multimodal support")
        print("2. Test with real conversations")
        print("3. Tune quality thresholds")
    else:
        print("\n⚠ Some tests failed. Check errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
