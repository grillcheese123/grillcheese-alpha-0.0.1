# Capsule Memory Integration Guide

## Overview

The capsule memory system is a bio-inspired 32-dimensional memory architecture based on hippocampal circuits (DG, CA3). It's integrated as an optional plugin that can be enabled via configuration.

## Architecture

```
Text Input
    ↓
[CapsuleEncoder] → 32D cognitive features
    ↓
[DentateGyrus] → 128D sparse expansion (2% sparsity)
    ↓
[CA3Memory] → FAISS kNN search
    ↓
Memory Retrieval
```

## Components

### 1. CapsuleMemory
- 32D cognitive feature vector
- Memory types: CONCEPT, EPISODE, SELF_STATE, TASK, TOOL
- Cognitive features: plasticity_gain, consolidation_priority, stability, stress_link

### 2. CapsuleEncoder
- Encodes text → 32D capsule vector
- Uses BGE embeddings (if enabled) or hash-based fallback
- Blends semantic content (90%) with cognitive features (10%)

### 3. DentateGyrus
- Sparse expansion: 32D → 128D
- 2% sparsity (~3 active neurons)
- Pattern separation for distinct representations

### 4. CA3Memory
- FAISS-based autoassociative memory
- Pattern completion via kNN search
- Falls back to numpy if FAISS unavailable

### 5. CapsuleMemoryStore
- Main memory store with consolidation
- Importance-based forgetting
- Protected memory support

## Usage

### Enable Capsule Memory

1. **Via Configuration**:
```python
# In config.py
MemoryConfig.USE_CAPSULE_MEMORY = True
```

2. **Via Module System**:
```json
// In modules_config.json
{
  "defaults": {
    "memory_backend": "CapsuleMemoryBackend"
  }
}
```

### Load Identity Dataset

```python
from memory.identity_loader import load_identity_dataset
from memory.capsule_store import CapsuleMemoryStore

# Load identity capsules
identity_path = "data/identity/capsule.jsonl"
identity_memories = load_identity_dataset(identity_path)

# Add to store
store = CapsuleMemoryStore()
store.add_batch(identity_memories)
```

### Query Memory

```python
# Query with text
results = store.query("What is pattern separation?", k=32)

for memory, distance in results:
    print(f"[{memory.memory_type.value}:{memory.domain}]")
    print(f"  {memory.content}")
    print(f"  Distance: {distance:.3f}")
```

### Store New Memory

```python
from memory.capsule_memory import CapsuleMemory, MemoryType

memory = CapsuleMemory(
    memory_id="episode_001",
    memory_type=MemoryType.EPISODE,
    domain="development",
    content="Successfully integrated capsule memory system",
    plasticity_gain=0.9,
    consolidation_priority=0.85,
    stability=0.7,
    stress_link=0.3,
    protected=False
)

store.add_memory(memory)
```

## Integration with GrillCheese

The capsule memory backend implements `BaseMemoryBackend`, so it works seamlessly with the existing system:

```python
# Via module registry
from modules.registry import ModuleRegistry

registry = ModuleRegistry()
registry.load_all_modules()

memory = registry.get_active_memory_backend()

# Use as normal
memory.store(embedding, text, metadata={
    'memory_type': 'EPISODE',
    'domain': 'conversation',
    'plasticity_gain': 0.8
})
```

## Performance

- **Encoding**: 50-150 memories/sec (CPU), 150-600/sec (GPU target)
- **Retrieval**: 1-5ms at 100K memories (FAISS), 0.5-2ms at 1M (HNSW)
- **Memory**: ~128 bytes per memory (DG vector) + metadata

## Configuration

```python
# config.py
MemoryConfig.USE_CAPSULE_MEMORY = False  # Enable/disable
MemoryConfig.CAPSULE_MEMORY_CAPACITY = 100000
MemoryConfig.CAPSULE_IDENTITY_PATH = "data/identity/capsule.jsonl"
```

## Notes

- Capsule memory works best with semantic embeddings enabled
- Falls back to hash-based encoding if embeddings disabled
- FAISS recommended for large-scale (>10K memories)
- Protected memories never get consolidated/forgotten
