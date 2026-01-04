"""
Test script for multimodal knowledge distillation system.
Validates all components without requiring actual model files.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from learning.multimodal_encoder import MultimodalEncoder
        print("✓ MultimodalEncoder")
    except Exception as e:
        print(f"✗ MultimodalEncoder: {e}")
        return False
        
    try:
        from learning.multilingual_utils import MultilingualProcessor
        print("✓ MultilingualProcessor")
    except Exception as e:
        print(f"✗ MultilingualProcessor: {e}")
        return False
        
    try:
        from learning.knowledge_distillation import KnowledgeDistillation, DistilledKnowledge
        print("✓ KnowledgeDistillation")
    except Exception as e:
        print(f"✗ KnowledgeDistillation: {e}")
        return False
        
    try:
        from learning.multimodal_memory_store import MultimodalMemoryStore
        print("✓ MultimodalMemoryStore")
    except Exception as e:
        print(f"✗ MultimodalMemoryStore: {e}")
        return False
        
    return True

def test_language_detection():
    """Test language detection"""
    print("\nTesting language detection...")
    
    try:
        from learning.multilingual_utils import MultilingualProcessor
        
        processor = MultilingualProcessor()
        
        test_cases = [
            ("Hello world", "en"),
            ("Hola mundo", "es"),
            ("Bonjour le monde", "fr"),
            ("こんにちは世界", "ja"),
            ("你好世界", "zh")
        ]
        
        for text, expected in test_cases:
            detected = processor.detect_language(text)
            status = "✓" if detected == expected else "⚠"
            print(f"  {status} '{text}' → {detected} (expected {expected})")
            
        return True
    except Exception as e:
        print(f"✗ Language detection failed: {e}")
        return False

def test_memory_store():
    """Test memory store operations"""
    print("\nTesting memory store...")
    
    try:
        from learning.multimodal_memory_store import MultimodalMemoryStore
        import numpy as np
        
        # Create temporary database
        db_path = Path("./test_memories.db")
        if db_path.exists():
            db_path.unlink()
            
        store = MultimodalMemoryStore(db_path, embedding_dim=128)
        print("  ✓ Database created")
        
        # Add memories
        embedding1 = np.random.randn(128).astype(np.float32)
        mem_id1 = store.add_memory(
            text="Test memory in English",
            embedding=embedding1,
            language="en",
            quality_score=0.8
        )
        print(f"  ✓ Added memory {mem_id1}")
        
        embedding2 = np.random.randn(128).astype(np.float32)
        mem_id2 = store.add_memory(
            text="Memoria de prueba en español",
            embedding=embedding2,
            language="es",
            quality_score=0.9,
            protected=True
        )
        print(f"  ✓ Added protected memory {mem_id2}")
        
        # Retrieve
        query_embedding = np.random.randn(128).astype(np.float32)
        results = store.retrieve(embedding=query_embedding, k=2)
        print(f"  ✓ Retrieved {len(results)} memories")
        
        # Stats
        stats = store.get_stats()
        print(f"  ✓ Stats: {stats['total_memories']} total, {stats['protected_memories']} protected")
        
        # Cleanup
        db_path.unlink()
        print("  ✓ Cleanup successful")
        
        return True
    except Exception as e:
        print(f"✗ Memory store test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_distillation_logic():
    """Test distillation without actual encoders"""
    print("\nTesting distillation logic...")
    
    try:
        from learning.knowledge_distillation import DistilledKnowledge
        from datetime import datetime
        import numpy as np
        
        # Create knowledge object
        knowledge = DistilledKnowledge(
            content="Test knowledge",
            modality="text",
            embedding=np.random.randn(128).astype(np.float32),
            language="en",
            quality_score=0.85,
            source="test"
        )
        print("  ✓ Created DistilledKnowledge object")
        
        # Verify attributes
        assert knowledge.content == "Test knowledge"
        assert knowledge.modality == "text"
        assert knowledge.language == "en"
        assert knowledge.quality_score == 0.85
        assert knowledge.timestamp is not None
        print("  ✓ All attributes correct")
        
        return True
    except Exception as e:
        print(f"✗ Distillation logic test failed: {e}")
        return False

def test_multilingual_utils():
    """Test multilingual utilities"""
    print("\nTesting multilingual utilities...")
    
    try:
        from learning.multilingual_utils import MultilingualProcessor
        
        processor = MultilingualProcessor()
        
        # Test normalization
        cjk_text = "你 好 世 界"
        normalized = processor.normalize_text(cjk_text, "zh")
        print(f"  ✓ CJK normalization: '{cjk_text}' → '{normalized}'")
        
        # Test language names
        name = processor.get_language_name("es")
        assert name == "Spanish"
        print(f"  ✓ Language name: es → {name}")
        
        # Test script detection
        script = MultilingualProcessor.get_script("Hello")
        assert script == "Latin"
        print(f"  ✓ Script detection: 'Hello' → {script}")
        
        script = MultilingualProcessor.get_script("你好")
        assert script == "Han"
        print(f"  ✓ Script detection: '你好' → {script}")
        
        return True
    except Exception as e:
        print(f"✗ Multilingual utils test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("Multimodal Knowledge Distillation - Component Tests")
    print("="*60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("Language Detection", test_language_detection()))
    results.append(("Memory Store", test_memory_store()))
    results.append(("Distillation Logic", test_distillation_logic()))
    results.append(("Multilingual Utils", test_multilingual_utils()))
    
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
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
