"""
Integration test: Quick smoke test of full pipeline
Runs fast and verifies all components work together
"""
import numpy as np
import pytest
import tempfile
import os
import shutil

try:
    from memory_store import MemoryStore
    from vulkan_backend import SNNCompute
    VULKAN_AVAILABLE = True
except ImportError:
    VULKAN_AVAILABLE = False


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
def test_integration_smoke():
    """Quick smoke test: verify all components work together"""
    # Setup
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "smoke_test.db")
    
    try:
        # Initialize components
        memory = MemoryStore(db_path=db_path, max_memories=10, embedding_dim=64)
        snn = SNNCompute(n_neurons=64, use_vulkan=True)
        
        # Test 1: Store memory
        emb1 = np.random.randn(64).astype(np.float32)
        memory.store(emb1, "Test memory")
        
        # Test 2: Retrieve memory
        retrieved = memory.retrieve(emb1, k=1)
        assert len(retrieved) == 1
        
        # Test 3: SNN processing
        result = snn.process(emb1)
        assert 'spike_activity' in result
        
        # Test 4: Multiple operations
        for i in range(5):
            emb = np.random.randn(64).astype(np.float32)
            memory.store(emb, f"Memory {i}")
        
        stats = memory.get_stats()
        assert stats['total_memories'] == 6  # 1 + 5
        
        print("✓ Integration smoke test passed!")
        
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

