# GrillCheese AI: Capsule Memory System

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│  VulkanCapsuleTransformer           │
│  ┌───────────────────────────────┐  │
│  │ 1. Tokenize (SentencePiece)   │  │
│  │ 2. Token Embeddings           │  │
│  │ 3. 6x Transformer Layers      │  │
│  │    └─ Memory Injection @4-5   │◄─┼── Retrieved Memories
│  │ 4. Pooling (mean)             │  │
│  │ 5. Capsule Projection 384→32D │  │
│  │ 6. DG Expansion 32→128D (2%)  │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  CA3MemoryStore (FAISS/numpy)       │
│  • Add: Store with DG vector        │
│  • Query: kNN retrieval             │
│  • Consolidate: Importance-based    │
└─────────────────────────────────────┘
```

## 32D Capsule Structure

```
┌────────────────────────────────────────────┐
│ Dims 0-27: Semantic Content (projected)    │
├────────────────────────────────────────────┤
│ Dim 28: plasticity_gain (learning rate)    │
│ Dim 29: consolidation_priority (replay)    │
│ Dim 30: stability (forgetting resistance)  │
│ Dim 31: stress_link (emotional tag)        │
└────────────────────────────────────────────┘
```

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| **Text Encoding** | ~1100-1300ms | Bottleneck (CPU forward) |
| Forward Pass (CPU) | 80-160ms | 6 layers, 384D |
| FAISS Search | <1ms | Numpy, 7 memories |
| FAISS Search | ~34ms | Numpy, 100K memories |
| Memory Injection | +10ms | Residual addition |
| DG Expansion | <1ms | Sparse projection |

### Bottleneck Analysis

The main bottleneck is **text encoding** (~1.1s), which includes:
1. Tokenization: ~5ms
2. Forward pass: ~80-160ms × 1 (but includes Python overhead)
3. Full pipeline overhead

**Solution**: GPU-accelerated forward pass (in progress)

## Files

### Core Modules
| File | Lines | Description |
|------|-------|-------------|
| `vulkan_capsule_transformer.py` | ~1050 | RoPE transformer, capsule encoding, DG, injection |
| `ca3_memory_store.py` | ~650 | FAISS/numpy memory store with consolidation |
| `tokenizer.py` | ~200 | SentencePiece with byte-level fallback |

### Shaders (93 total, GPU compute)
| Shader | Purpose |
|--------|---------|
| `rope.glsl` | Rotary Position Embeddings |
| `flash-attention2-rope.glsl` | Flash Attention 2 + RoPE |
| `capsule-project.glsl` | 384D→32D + cognitive features |
| `dg-sparse-expand.glsl` | 32D→128D sparse (2% active) |
| `memory-inject-residual.glsl` | Inject to layers 4-5 |
| `faiss-distance.glsl` | L2/cosine/dot distances |
| `faiss-topk.glsl` | Top-k selection |

## Empirical Validation

| Test | Result | Significance |
|------|--------|--------------|
| DG Sparsity (2%) | **+130.5%** | Pattern separation vs dense |
| Cognitive Features | **+48.2%** | Inter/intra cluster ratio |
| Pattern Separation | **+46.7%** | Overlap reduction |
| Retrieval Accuracy | **+68.4%** | Domain accuracy vs random |
| Capsule Compression | 12x | 384D→32D, 67% correlation |

## Usage

### Quick Start
```python
from vulkan_capsule_transformer import VulkanCapsuleTransformer
from ca3_memory_store import CA3MemoryStore

# Initialize
encoder = VulkanCapsuleTransformer()
store = CA3MemoryStore(encoder, capacity=100000)

# Add memory
from vulkan_capsule_transformer import CapsuleMemory, MemoryType, CognitiveFeatures

memory = encoder.create_memory(
    content="I am GrillCheese AI",
    memory_type=MemoryType.SELF_STATE,
    domain="identity",
    cognitive_features=CognitiveFeatures(stability=0.95),
    protected=True
)
store.add_memory(memory)

# Query
results = store.query("What AI am I?", k=5)
for mem, dist in results:
    print(f"{mem.domain}: {mem.content[:50]}...")
```

### Memory Injection
```python
# Retrieve relevant memories
retrieved = store.query("Tell me about your architecture", k=5)
memories = [mem for mem, _ in retrieved]

# Encode with injection at layers 4-5
encoded = encoder.tokenizer.encode("Tell me about your architecture")
embedding = encoder.forward(
    encoded['input_ids'].reshape(1, -1),
    encoded['attention_mask'].reshape(1, -1),
    inject_memories=memories
)
```

### Persistence
```python
store.save("memories.json")
store.load("memories.json")
```

## Optimization Roadmap

### Phase 1: GPU Forward Pass (Next)
- Persistent GPU buffers for weights
- Chain shader dispatches without CPU readback
- Expected: 80-160ms → 15-25ms

### Phase 2: Batch Encoding
- Batch multiple texts in single forward pass
- Amortize GPU dispatch overhead
- Expected: 10ms/text in batch

### Phase 3: FAISS Shader Optimization
- Parallel reduction for topk
- Tiled matrix multiplication for distances
- Expected: Beat numpy at >50K vectors

## Dependencies

```
numpy
vulkan (via vulkan-python)
sentencepiece (optional)
faiss-cpu (optional, numpy fallback available)
```
