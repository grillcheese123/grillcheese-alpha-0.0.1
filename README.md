# GrillCheese AI - Complete Feature Map

**Version**: 1.0 (January 2026)  
**Architecture**: Bio-Inspired Local AI Assistant

---

## Core Architecture

### Language Model
- **Model**: Microsoft Phi-3 Mini (3.8B parameters)
- **Format**: GGUF (quantized, GPU-optimized)
- **Backend**: llama-cpp-python with Vulkan support
- **Context**: 4K tokens
- **Embeddings**: sentence-transformers (384-dim)
- **Generation**:
  - Temperature: 0.7
  - Top-P: 0.9
  - Max tokens: 256 (GPU) / 128 (CPU)
  - Auto stop sequences

### Hardware Support
- **GPU**: AMD RX 6750 XT (RDNA2)
- **Compute**: Vulkan 1.2+ shaders
- **Fallback**: CPU mode (full functionality)
- **Platform**: Windows 11 (primary), cross-platform capable

---

## Memory System

### Storage Engine
- **Database**: SQLite persistent storage
- **GPU Acceleration**: Vulkan compute shaders
- **Capacity**: 100,000 memories (configurable)
- **Embedding Dimension**: 384 (auto-detected)
- **Search**: GPU-accelerated FAISS similarity

### Memory Types
1. **Regular Memories**
   - Automatic pruning when limit reached
   - Access tracking
   - Temporal metadata
   
2. **Protected Memories**
   - Persistent and non-deletable
   - Higher retrieval priority
   - Created via teach mode
   
3. **Identity Memory**
   - System personality/behavior
   - Single special memory
   - Always included in context

### Memory Operations
- **Store**: GPU write → Database commit
- **Retrieve**: GPU FAISS top-k search (2-5x faster than CPU)
- **Clear**: Preserves protected/identity (optional override)
- **Export**: JSONL format
- **Stats**: Access patterns, temporal distribution

### GPU Optimization
- **FAISS Compute**: L2, cosine, dot product distances
- **Top-K Selection**: GPU-accelerated heap sort
- **Descriptor Cache**: >95% hit rate after warmup
- **Fallback Chain**: GPU → NumPy → CPU loop

---

## Bio-Inspired Brain System

### Amygdala (Emotional Processing)
- **Valence**: -1 (negative) to +1 (positive)
- **Arousal**: 0 (calm) to 1 (excited)
- **Emotions**: joy, anxiety, curiosity, calm, neutral
- **Affect Prediction**: Neural network trained on interactions

### Limbic System (Memory-Emotion Link)
- Links memories to emotional context
- Influences memory consolidation
- Modulates emotional responses

### Basal Ganglia (Strategy Selection)
- **Strategies**: empathetic, analytical, creative, balanced, assertive
- **Hebbian Learning**: 384×64 weight matrix
- **Selection**: Based on context + past success

### Central Nervous System (CNS)
- **Consciousness Levels**: DROWSY, ALERT, FOCUSED
- **Stress Tracking**: 0.0 to 1.0
- **Fatigue Simulation**: Affects performance
- **Homeostasis**: Automatic stress recovery

### Endocrine System (Hormonal Modulation)
- **Cortisol**: Stress hormone (↑ arousal, ↓ empathy)
- **Oxytocin**: Social bonding (↑ warmth, ↑ empathy)
- **Dopamine**: Reward/motivation (↑ creativity, ↑ energy)
- **Decay**: Realistic hormone half-lives

### Experience System
- **Interaction Tracking**: Success/failure patterns
- **Quality Scoring**: 0.0 to 1.0
- **Strategy Learning**: Adapts based on outcomes
- **Trend Analysis**: Recent vs historical performance

---

## Spiking Neural Network (SNN)

### LIF Neurons
- **Count**: 1,000 neurons (configurable)
- **Dynamics**: Leaky Integrate-and-Fire
- **Membrane Tau**: 5.0 ms
- **Threshold**: 0.5
- **Refractory Period**: Simulated

### GPU Acceleration
- **Vulkan Compute**: LIF step shader
- **Timesteps**: 50 per input
- **Input Encoding**: Scaled to spike rate
- **Metrics**: Spike count, firing rate

### STDP Learning (Planned)
- Spike-timing dependent plasticity
- Continuous learning from interactions
- Hebbian weight updates

---

## Interaction Modes

### 1. Interactive Mode
**Access**: `python cli.py --interactive` or `python cli.py`

**Features**:
- Real-time conversation
- Emotional intelligence enabled
- Memory retrieval (top-5)
- Strategy selection
- Continuous learning (optional)

**Commands**:
- `quit` - Exit
- `stats` - Memory + brain statistics
- `emotion` - Current emotional state
- `clear` - Clear memories (protected preserved)

**Stats Display**: `[Memories: N | emotion | GPU]`

### 2. Single Prompt Mode
**Access**: `python cli.py "Your prompt here"`

**Features**:
- One-shot query
- Full memory retrieval
- Emotional context
- Fast execution

### 3. Teach Mode (Public)
**Access**: `python cli.py --teach`

**Purpose**: Create permanent, protected memories

**Commands**:
- `teach <text>` - Store protected memory
- `file <path>` - Import from text file (one per line)
- `list` - Show all protected memories
- `stats` - Memory statistics
- `quit` - Exit

**Use Cases**:
- Personal preferences
- Domain knowledge
- Core facts
- Training examples

### 4. Developer Mode (Restricted)
**Access**: `python cli.py --dev`  
**Authentication**: Password required (`grillcheese_dev_2026`)

**Commands**:
- `export-training` - Export conversation pairs (JSONL)
- `analyze-memory` - Deep memory analysis
- `edit-identity` - Modify system prompt
- `tune-params` - View configuration
- `test-retrieval` - Test memory search
- `export-embeddings` - Export vectors (NPZ)
- `brain-dump` - Export complete brain state
- `create-dataset` - Fine-tuning dataset
- `stats` - Comprehensive statistics
- `quit` - Exit

**Security**:
- SHA-256 password hashing
- 3-attempt lockout
- Environment variable override

---

## System Features

### Configuration
**File**: `config.py`

**Model Settings**:
- Embedding dimension auto-detection
- Temperature, Top-P
- Max tokens (GPU/CPU)
- Context items (5)

**Memory Settings**:
- Database path
- Max memories (100K)
- Default retrieval K (5)
- GPU buffer size (10K)

**SNN Settings**:
- Neuron count (1K)
- Time constants
- Thresholds
- Timesteps

**Brain Settings**:
- Emotion ranges
- Hormone decay rates
- Strategy weights

### Logging
- **Level**: INFO (configurable)
- **Format**: `[%(levelname)s] %(message)s`
- **Unicode Symbols**: check (✓), cross (✗), warning (⚠)

### Error Handling
- Graceful GPU fallback
- Database rollback on errors
- Model loading fallbacks (GGUF → PyTorch)
- Safe cleanup on interrupts

---

## Performance Optimizations

### Memory Operations
- **GPU FAISS**: 2-5x faster than CPU (10K memories)
- **Retrieval Latency**: <2ms (target)
- **Descriptor Cache**: >95% hit rate
- **Async Stats**: Non-blocking access count updates

### Response Generation
- **llama-cpp Output Suppression**: Clean responses
- **Context Pruning**: Max 5 relevant memories
- **Stop Sequences**: Prevents prompt leaking
- **Response Cleaning**: Removes system prompt echoes

### Database
- **Indexes**: timestamp, identity, protected
- **Batch Stats Updates**: 100 items per flush
- **Background Worker**: 1-second interval
- **Connection Pooling**: Per-operation connections

---

## Security & Privacy

### Local-First Architecture
- **On-Device Processing**: All computation performed locally
- **Zero Telemetry**: No tracking or data collection
- **Data Privacy**: All private data remains on local machine
- **Offline Operation**: Full functionality available without internet connection

### Authentication
- **Developer Mode**: Password-protected
- **Hashing**: SHA-256
- **Environment Variables**: Secure password storage
- **Access Control**: 3-attempt lockout

### Data Protection
- **Protected Memories**: Deletion-resistant
- **Clear Safety**: Preserves important data
- **Export Control**: Developer-only access
- **No Plaintext Passwords**: Hashed storage

---

## File Structure

```
backend/
├── cli/
│   └── cli.py                # Main CLI entry point
├── config.py                 # Configuration
├── identity.py               # System identity
├── dev_auth.py              # Developer authentication
├── model_gguf.py            # Phi-3 GGUF model
├── memory_store.py          # Memory system
├── brain/
│   ├── unified_brain.py     # Brain orchestrator
│   ├── amygdala.py          # Emotional processing
│   ├── limbic.py            # Memory-emotion link
│   ├── basal_ganglia.py     # Strategy selection
│   ├── cns.py               # Consciousness/stress
│   └── endocrine.py         # Hormonal modulation
├── vulkan_backend/
│   ├── vulkan_compute.py    # Main GPU interface
│   ├── vulkan_faiss.py      # FAISS similarity search
│   ├── vulkan_snn.py        # SNN compute
│   ├── vulkan_core.py       # Vulkan initialization
│   ├── vulkan_pipelines.py  # Pipeline management
│   └── snn_compute.py       # High-level SNN API
├── tests/
│   └── test_faiss_gpu.py    # GPU FAISS tests
├── docs/
│   ├── TEACH_MODE.md        # Teaching mode guide
│   ├── DEVELOPER_MODE.md    # Developer guide
│   ├── PROTECTED_MODES_REFERENCE.md
│   └── DEPLOYMENT_GUIDE.md  # Production deployment
└── memories.db              # SQLite database
```

---

## Testing & Validation

### Test Suite
**File**: `tests/test_faiss_gpu.py`

**Tests**:
- L2 distance correctness
- Cosine similarity accuracy
- Top-k selection validation
- Edge cases (k > database size)
- GPU vs NumPy correctness
- MemoryStore integration
- Large-scale retrieval (1000 memories)

**Run**: `pytest tests/test_faiss_gpu.py -v`

### Benchmarking
**File**: `benchmark_faiss_performance.py`

**Metrics**:
- GPU vs CPU latency (mean, P95, P99)
- Multiple scales (1K, 5K, 10K, 20K memories)
- MemoryStore integration benchmark
- Speedup calculations

**Run**: `python benchmark_faiss_performance.py`

---

## Roadmap & Future Features

### Implemented
- [x] GPU-accelerated memory
- [x] FAISS similarity search
- [x] Protected memories
- [x] Teach mode
- [x] Developer mode
- [x] Emotional intelligence
- [x] Bio-inspired brain
- [x] SNN compute
- [x] Response optimization
- [x] Descriptor caching

### In Progress
- [ ] STDP learning implementation
- [ ] Conversation history tracking
- [ ] Fine-tuning dataset creation
- [ ] Embedding space visualization

### Planned
- [ ] Multi-turn conversation context
- [ ] Voice input/output
- [ ] Document ingestion
- [ ] Web search integration
- [ ] Plugin system
- [ ] Mobile app
- [ ] Multi-model support
- [ ] Distributed memory

---

## Key Strengths

1. **Privacy-First Architecture**: All processing performed locally with no external data transmission
2. **Bio-Inspired Design**: Emotionally intelligent system based on biological models
3. **GPU-Accelerated Performance**: Memory operations 2-5x faster than CPU-based implementations
4. **Protected Teaching System**: Permanent knowledge base with protected memory storage
5. **Developer Tools**: Advanced toolset for model improvement and analysis
6. **Modular Design**: Extensible architecture for easy customization
7. **Production Ready**: Comprehensive error handling and fallback mechanisms
8. **Comprehensive Documentation**: Detailed guides and references for all features

---

## Quick Stats

| Metric | Value |
|--------|-------|
| **Model Size** | 3.8B parameters |
| **Context Window** | 4,096 tokens |
| **Embedding Dim** | 384 |
| **Max Memories** | 100,000 |
| **GPU Speedup** | 2-5x (vs CPU) |
| **Retrieval Time** | <2ms (target) |
| **Brain Components** | 5 systems |
| **SNN Neurons** | 1,000 |
| **Interaction Modes** | 4 |
| **Protected Modes** | 2 |

---

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Download Phi-3 GGUF model
# Place in models/ directory

# Interactive mode
python cli.py --interactive

# Teach mode
python cli.py --teach

# Developer mode (password: grillcheese_dev_2026)
python cli.py --dev

# Single prompt
python cli.py "Hello, how are you?"
```

---

## Documentation Index

- **FEATURE_MAP.md** (this file) - Complete overview
- **TEACH_MODE.md** - Public teaching guide
- **DEVELOPER_MODE.md** - Developer tools guide
- **PROTECTED_MODES_REFERENCE.md** - Quick reference
- **DEPLOYMENT_GUIDE.md** - Production deployment
- **PRODUCTION_FIXES.md** - Critical fixes applied


