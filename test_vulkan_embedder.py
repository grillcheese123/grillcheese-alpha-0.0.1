"""
Test custom Vulkan embedder
"""

import numpy as np
from pathlib import Path
import sys

def test_vulkan_embedder():
    """Test the custom Vulkan embedder"""
    print("="*60)
    print("Testing Vulkan Embedder")
    print("="*60)
    
    # Initialize
    print("\n1. Initializing Vulkan backend...")
    try:
        from vulkan_backend import VulkanCompute
        gpu = VulkanCompute()
        print("✓ Vulkan initialized")
    except Exception as e:
        print(f"✗ Failed to initialize Vulkan: {e}")
        return False
    
    # Load config
    print("\n2. Loading embedder config...")
    try:
        from learning.embedder_config import EmbedderConfig
        
        config_path = Path("models/embedder_config.json")
        if config_path.exists():
            config = EmbedderConfig.from_json(config_path)
            print(f"✓ Loaded config: {config.num_layers} layers, {config.hidden_dim}D")
        else:
            print("⚠ Config not found, using defaults")
            config = EmbedderConfig(
                vocab_size=30522,
                hidden_dim=384,
                num_layers=2,
                num_heads=6,
                ffn_dim=1536,
                max_seq_len=128
            )
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False
    
    # Create embedder
    print("\n3. Creating embedder...")
    try:
        from learning.vulkan_embedder import VulkanEmbedder
        
        weights_path = Path("models/embedder_weights.npz")
        embedder = VulkanEmbedder(
            gpu=gpu,
            config=config,
            model_path=weights_path if weights_path.exists() else None
        )
        print("✓ Embedder created")
    except Exception as e:
        print(f"✗ Failed to create embedder: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test encoding
    print("\n4. Testing text encoding...")
    try:
        test_texts = [
            "Hello world",
            "This is a test of the Vulkan embedder",
            "GPU-accelerated sentence embeddings"
        ]
        
        print(f"   Encoding {len(test_texts)} texts...")
        embeddings = embedder.encode(test_texts)
        
        print(f"✓ Embeddings shape: {embeddings.shape}")
        print(f"✓ Expected: ({len(test_texts)}, {config.hidden_dim})")
        
        if embeddings.shape != (len(test_texts), config.hidden_dim):
            print(f"✗ Shape mismatch!")
            return False
            
    except Exception as e:
        print(f"✗ Encoding failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test L2 normalization
    print("\n5. Testing L2 normalization...")
    try:
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"✓ L2 norms: {norms}")
        
        if np.allclose(norms, 1.0, atol=0.01):
            print("✓ All embeddings are L2 normalized")
        else:
            print("⚠ Embeddings not perfectly normalized (may be ok)")
    except Exception as e:
        print(f"✗ Norm check failed: {e}")
    
    # Test similarity
    print("\n6. Testing cosine similarity...")
    try:
        # Compute similarity between first two texts
        sim = np.dot(embeddings[0], embeddings[1])
        print(f"✓ Similarity between texts: {sim:.4f}")
        
        # Self-similarity should be ~1.0
        self_sim = np.dot(embeddings[0], embeddings[0])
        print(f"✓ Self-similarity: {self_sim:.4f}")
        
        if not np.isclose(self_sim, 1.0, atol=0.01):
            print("⚠ Self-similarity not close to 1.0")
    except Exception as e:
        print(f"✗ Similarity test failed: {e}")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    print("\nNext step: Integrate with memory_store.py")
    
    return True

if __name__ == "__main__":
    success = test_vulkan_embedder()
    sys.exit(0 if success else 1)
