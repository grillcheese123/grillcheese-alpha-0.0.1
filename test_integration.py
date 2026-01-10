"""
Integration Test: Full Capsule Memory Pipeline

Tests the complete flow:
1. Create encoder (VulkanCapsuleTransformer)
2. Create memory store (CA3MemoryStore)
3. Add memories with cognitive features
4. Query and retrieve relevant memories
5. Forward pass with memory injection
"""

import time
import numpy as np
from vulkan_capsule_transformer import (
    VulkanCapsuleTransformer,
    CapsuleTransformerConfig,
    CapsuleMemory,
    CognitiveFeatures,
    MemoryType
)
from ca3_memory_store import CA3MemoryStore


def test_full_integration():
    """Test complete capsule memory pipeline"""
    print("\n" + "="*60)
    print("INTEGRATION TEST: Full Capsule Memory Pipeline")
    print("="*60)
    
    # 1. Initialize encoder
    print("\n[1] Initializing VulkanCapsuleTransformer...")
    config = CapsuleTransformerConfig(
        hidden_dim=384,
        num_layers=6,
        num_heads=6,
        capsule_dim=32,
        injection_layers=(4, 5),
        injection_strength=0.1
    )
    encoder = VulkanCapsuleTransformer(config=config)
    print(f"    Encoder ready: {config.hidden_dim}D -> {config.capsule_dim}D capsules")
    
    # 2. Initialize memory store
    print("\n[2] Initializing CA3MemoryStore...")
    memory_store = CA3MemoryStore(encoder, capacity=1000)
    print(f"    Memory store ready: capacity={memory_store.capacity}")
    
    # 3. Add test memories
    print("\n[3] Adding test memories...")
    
    test_memories = [
        # Identity memories (protected)
        {
            "content": "I am GrillCheese AI, a privacy-focused local AI assistant.",
            "memory_type": MemoryType.SELF_STATE,
            "domain": "identity",
            "protected": True,
            "cognitive": CognitiveFeatures(plasticity_gain=0.2, stability=0.95, consolidation_priority=1.0)
        },
        {
            "content": "I run entirely on the user's computer using Vulkan GPU acceleration.",
            "memory_type": MemoryType.SELF_STATE,
            "domain": "architecture",
            "protected": True,
            "cognitive": CognitiveFeatures(plasticity_gain=0.3, stability=0.9, consolidation_priority=0.9)
        },
        
        # Technical memories
        {
            "content": "Pattern separation in the dentate gyrus transforms similar inputs into distinct representations.",
            "memory_type": MemoryType.CONCEPT,
            "domain": "neuroscience",
            "cognitive": CognitiveFeatures(plasticity_gain=0.7, stability=0.6, consolidation_priority=0.8)
        },
        {
            "content": "FAISS is a library for efficient similarity search developed by Facebook Research.",
            "memory_type": MemoryType.CONCEPT,
            "domain": "technology",
            "cognitive": CognitiveFeatures(plasticity_gain=0.6, stability=0.7, consolidation_priority=0.7)
        },
        {
            "content": "Vulkan provides low-level GPU access for compute shaders.",
            "memory_type": MemoryType.CONCEPT,
            "domain": "technology",
            "cognitive": CognitiveFeatures(plasticity_gain=0.5, stability=0.8, consolidation_priority=0.8)
        },
        
        # Episodic memories
        {
            "content": "Implemented Flash Attention 2 with fused RoPE for efficient transformer attention.",
            "memory_type": MemoryType.EPISODE,
            "domain": "development",
            "cognitive": CognitiveFeatures(plasticity_gain=0.8, stability=0.5, consolidation_priority=0.7)
        },
        {
            "content": "Achieved 130% improvement in pattern separation using 2% DG sparsity.",
            "memory_type": MemoryType.EPISODE,
            "domain": "testing",
            "cognitive": CognitiveFeatures(plasticity_gain=0.9, stability=0.4, consolidation_priority=0.6)
        },
    ]
    
    for mem_data in test_memories:
        mem = CapsuleMemory(
            memory_id=f"{mem_data['memory_type'].value.lower()}_{len(memory_store.memories)}",
            memory_type=mem_data["memory_type"],
            domain=mem_data["domain"],
            content=mem_data["content"],
            cognitive_features=mem_data["cognitive"],
            protected=mem_data.get("protected", False)
        )
        memory_store.add_memory(mem)
    
    print(f"    Added {len(memory_store.memories)} memories")
    print(f"    Protected: {len(memory_store.protected_memories)}")
    
    # 4. Test retrieval
    print("\n[4] Testing memory retrieval...")
    
    test_queries = [
        "What AI am I talking to?",
        "How does pattern separation work?",
        "What GPU technologies are available?"
    ]
    
    for query in test_queries:
        print(f"\n    Query: '{query}'")
        start = time.perf_counter()
        results = memory_store.query(query, k=3)
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"    Retrieved {len(results)} memories in {elapsed:.1f}ms:")
        for mem, dist in results:
            print(f"      [{mem.memory_type.value}:{mem.domain}] dist={dist:.3f}")
            print(f"        {mem.content[:60]}...")
    
    # 5. Test forward pass with memory injection
    print("\n[5] Testing forward pass with memory injection...")
    
    query = "Tell me about your architecture"
    print(f"    Query: '{query}'")
    
    # Retrieve memories
    retrieved = memory_store.query(query, k=5)
    retrieved_memories = [mem for mem, _ in retrieved]
    
    # Encode query
    encoded = encoder.tokenizer.encode(query, max_length=64, padding=True)
    input_ids = encoded['input_ids'].reshape(1, -1)
    attention_mask = encoded['attention_mask'].reshape(1, -1)
    
    # Forward without injection
    start = time.perf_counter()
    emb_without = encoder.forward(input_ids, attention_mask)
    time_without = (time.perf_counter() - start) * 1000
    
    # Forward with injection
    start = time.perf_counter()
    emb_with = encoder.forward(input_ids, attention_mask, inject_memories=retrieved_memories)
    time_with = (time.perf_counter() - start) * 1000
    
    # Measure difference
    diff = np.linalg.norm(emb_with - emb_without)
    
    print(f"    Forward without injection: {time_without:.1f}ms")
    print(f"    Forward with injection:    {time_with:.1f}ms")
    print(f"    Embedding difference:      {diff:.4f}")
    print(f"    Embedding norm (with):     {np.linalg.norm(emb_with):.4f}")
    
    # 6. Test memory statistics
    print("\n[6] Memory store statistics...")
    stats = memory_store.get_stats()
    for key, val in stats.items():
        print(f"    {key}: {val}")
    
    # 7. Test persistence
    print("\n[7] Testing persistence...")
    save_path = "test_memory_store.json"
    memory_store.save(save_path)
    print(f"    Saved to {save_path}")
    
    # Load into new store
    new_store = CA3MemoryStore(encoder, capacity=1000)
    new_store.load(save_path)
    print(f"    Loaded {len(new_store.memories)} memories from disk")
    
    # Verify retrieval works after load
    results = new_store.query("What AI am I?", k=1)
    if results:
        mem, dist = results[0]
        print(f"    Verification: Retrieved '{mem.domain}' memory after reload")
    
    # Cleanup
    import os
    os.remove(save_path)
    
    print("\n" + "="*60)
    print("INTEGRATION TEST PASSED")
    print("="*60)


if __name__ == "__main__":
    test_full_integration()
