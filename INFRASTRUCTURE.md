# GrillCheese Backend Infrastructure Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Core Components](#core-components)
4. [Data Schemas](#data-schemas)
5. [GPU/Vulkan Backend](#gpuvulkan-backend)
6. [API Interfaces](#api-interfaces)
7. [Configuration](#configuration)
8. [State Management](#state-management)
9. [Data Flow](#data-flow)

---

## System Overview

GrillCheese is a bio-inspired AI assistant with:
- **Persistent Memory**: GPU-accelerated episodic memory using FAISS similarity search
- **Emotional Intelligence**: Multi-region brain architecture (Amygdala, Limbic System, Thalamus, etc.)
- **GPU Acceleration**: Vulkan compute shaders for neural operations
- **Continuous Learning**: STDP-based temporal association learning
- **Multi-modal Support**: Text, embeddings, and future vision/audio

### Key Technologies
- **Language Model**: Phi-3-mini (GGUF format via llama-cpp-python)
- **Embeddings**: Sentence-transformers (all-MiniLM-L6-v2, 384-dim)
- **GPU Backend**: Vulkan compute shaders (78+ shaders)
- **Database**: SQLite for persistent storage
- **Web Framework**: FastAPI with WebSocket support
- **Neural Networks**: Spiking Neural Networks (SNN) + Feedforward Networks (FNN)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Server                          │
│                      (main.py, cli/cli.py)                     │
└────────────────────────────┬────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  Phi3GGUF     │    │ MemoryStore   │    │ UnifiedBrain │
│  Model        │    │ (Hippocampus) │    │ (Orchestrator)│
└───────┬───────┘    └───────┬───────┘    └───────┬───────┘
        │                    │                     │
        │                    │                     │
        ▼                    ▼                     ▼
┌───────────────────────────────────────────────────────────────┐
│                    Vulkan GPU Backend                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Vulkan   │  │ Vulkan   │  │ Vulkan   │  │ Vulkan    │     │
│  │ SNN      │  │ FAISS    │  │ Memory   │  │ Attention │     │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ Vulkan   │  │ Vulkan   │  │ Vulkan    │                   │
│  │ FNN      │  │ Cells    │  │ Compute   │                   │
│  └──────────┘  └──────────┘  └──────────┘                   │
└───────────────────────────────────────────────────────────────┘
        │                    │                     │
        ▼                    ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  SQLite DB    │    │ Brain State   │    │ Learning      │
│  (memories.db)│    │ (JSON files)  │    │ State (JSON)  │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## Core Components

### 1. Model Layer (`model_gguf.py`, `model.py`)

**Purpose**: Language model inference and embedding generation

**Components**:
- `Phi3GGUF`: GGUF format model (primary)
- `Phi3Model`: PyTorch fallback
- Sentence-transformers for embeddings

**Schema**:
```python
class Phi3GGUF:
    llm: Llama                    # llama-cpp-python instance
    embedder: SentenceTransformer  # all-MiniLM-L6-v2
    embedding_dim: int = 384       # Fixed embedding dimension
    
    def get_embedding(text: str) -> np.ndarray[384]
    def generate(prompt: str, **kwargs) -> str
```

**Configuration** (`config.py`):
```python
class ModelConfig:
    EMBEDDING_DIM = 384
    PHI3_EMBEDDING_DIM = 3072  # Hidden states (not used for memory)
    SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
    MAX_NEW_TOKENS_GPU = 256
    MAX_NEW_TOKENS_CPU = 128
    TEMPERATURE = 0.7
    TOP_P = 0.9
```

---

### 2. Memory Store (`memory_store.py`)

**Purpose**: Persistent episodic memory with GPU-accelerated similarity search

**Architecture**:
- SQLite database for persistence
- GPU buffers for fast similarity search (FAISS)
- Async statistics tracking

**Database Schema**:
```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    embedding BLOB NOT NULL,           -- np.float32[384] serialized
    text TEXT NOT NULL,
    timestamp TEXT NOT NULL,            -- ISO format
    access_count INTEGER DEFAULT 0,
    last_accessed TEXT,                 -- ISO format
    metadata TEXT,                      -- JSON string
    is_identity INTEGER DEFAULT 0,     -- Boolean flag
    is_protected INTEGER DEFAULT 0     -- Boolean flag
);

CREATE INDEX idx_timestamp ON memories(timestamp);
CREATE INDEX idx_identity ON memories(is_identity);
CREATE INDEX idx_protected ON memories(is_protected);
```

**In-Memory Structure**:
```python
class MemoryStore:
    memory_keys: np.ndarray      # [max_memories, 384] - Query embeddings
    memory_values: np.ndarray    # [max_memories, 384] - Stored embeddings
    memory_texts: List[str]      # Parallel array of text
    identity_index: int          # Index of identity memory
    next_write_index: int        # Circular buffer write position
    num_memories: int            # Current count
    embedding_dim: int = 384
    max_memories: int = 100000
```

**Key Methods**:
- `store(embedding, text, metadata=None)`: Store new memory
- `retrieve(query_embedding, k=3)`: GPU-accelerated similarity search
- `set_identity(text)`: Store identity memory (always retrieved)
- `get_stats()`: Access statistics

---

### 3. Unified Brain (`brain/unified_brain.py`)

**Purpose**: Orchestrates all bio-inspired brain components

**Component Pipeline**:
```
Input Text/Embedding
    ↓
1. Thalamus (Sensory Gating)
    ↓
2. Amygdala (Emotional Processing)
    ↓
3. Limbic System (Memory-Emotion Integration)
    ↓
4. Central Nervous System (Consciousness/Stress)
    ↓
5. Endocrine System (Hormonal Homeostasis)
    ↓
6. Basal Ganglia (Action Selection)
    ↓
Output: Strategy, Modulations, Emotional State
```

**Brain Components**:

#### 3.1 Amygdala (`brain/amygdala.py`)
**Purpose**: Emotional processing and affect prediction

**Schema**:
```python
@dataclass
class EmotionalState:
    arousal: float          # 0.0 (calm) to 1.0 (excited)
    valence: float         # -1.0 (negative) to 1.0 (positive)
    dominant_emotion: str  # "happy", "sad", "angry", etc.
    confidence: float      # 0.0 to 1.0
    timestamp: float       # Unix timestamp

class Amygdala:
    affect_weights: np.ndarray  # [embedding_dim, 64] - Learned associations
    emotion_decay: float = 0.85
    sensitivity: float = 1.2
    current_state: EmotionalState
```

**State Persistence** (`brain_state/amygdala.json`):
```json
{
    "affect_weights": [[...]],  // Serialized numpy array
    "current_state": {
        "arousal": 0.5,
        "valence": 0.0,
        "dominant_emotion": "neutral",
        "confidence": 0.5
    },
    "calibration_stats": {...}
}
```

#### 3.2 Limbic System (`brain/limbic_system.py`)
**Purpose**: Integrates memory retrieval with emotional context

**Schema**:
```python
class LimbicSystem:
    amygdala: Amygdala
    memory_store: MemoryStore
    embedding_dim: int
    
    def integrate(memory_context, emotional_state) -> Dict
```

#### 3.3 Thalamus (`brain/thalamus.py`)
**Purpose**: Sensory gating and attention routing

**Schema**:
```python
class Thalamus:
    gate_weights: np.ndarray      # [num_routes, embedding_dim]
    gate_threshold: float = 0.3
    num_routes: int = 4
    
    def gate(embedding) -> Tuple[bool, np.ndarray]
```

#### 3.4 Basal Ganglia (`brain/basal_ganglia.py`)
**Purpose**: Action selection and response strategy

**Schema**:
```python
class BasalGanglia:
    num_regions: int = 4
    selection_temperature: float = 1.0
    strategy_weights: Dict[str, float]
    
    def select_action(region_activations) -> str
```

**Strategies**:
- `empathetic`: Emotional support
- `informative`: Factual information
- `creative`: Creative responses
- `analytical`: Deep analysis

#### 3.5 Endocrine System (`brain/endocrine.py`)
**Purpose**: Hormonal homeostasis and long-term state

**Schema**:
```python
class EndocrineSystem:
    hormones: Dict[HormoneType, float]
    
    class HormoneType(Enum):
        CORTISOL = "cortisol"      # Stress
        DOPAMINE = "dopamine"      # Reward
        SEROTONIN = "serotonin"    # Mood
        OXYTOCIN = "oxytocin"      # Social bonding
```

**State Persistence** (`brain_state/endocrine.json`):
```json
{
    "hormones": {
        "cortisol": 0.3,
        "dopamine": 0.5,
        "serotonin": 0.6,
        "oxytocin": 0.4
    },
    "last_update": "2024-01-01T00:00:00"
}
```

#### 3.6 Central Nervous System (`brain/cns.py`)
**Purpose**: Consciousness level and stress management

**Schema**:
```python
class CentralNervousSystem:
    class ConsciousnessLevel(Enum):
        DEEP_SLEEP = 0
        LIGHT_SLEEP = 1
        DROWSY = 2
        ALERT = 3
        HYPER_ALERT = 4
    
    consciousness: ConsciousnessLevel
    stress_level: float        # 0.0 to 1.0
    fatigue: float            # 0.0 to 1.0
    stress_recovery_rate: float = 0.02
```

**State Persistence** (`brain_state/cns.json`):
```json
{
    "consciousness": "ALERT",
    "stress_level": 0.3,
    "fatigue": 0.2,
    "last_update": "2024-01-01T00:00:00"
}
```

#### 3.7 GPU Brain Compute (`brain/gpu_brain.py`)
**Purpose**: GPU-accelerated neural computations

**Components**:
- Place cells: Spatial memory encoding
- Time cells: Temporal memory encoding
- Hebbian learning: Weight updates
- STDP learning: Temporal associations
- Theta-gamma encoding: Phase-coupled oscillations

**Schema**:
```python
class GPUBrainCompute:
    vulkan: VulkanCompute
    use_vulkan: bool
    
    def compute_place_cells(position, centers, field_width, max_rate)
    def compute_time_cells(elapsed_time, preferences, temporal_width)
    def hebbian_update(pre_activations, post_activations, weights)
    def stdp_update(pre_activations, post_activations, weights, traces)
    def compute_theta_gamma_encoding(positions, embedding_dim)

class GPUSpatialMemory:
    n_place_cells: int = 1000
    n_time_cells: int = 100
    place_centers: np.ndarray      # [n_place_cells, spatial_dims]
    time_preferences: np.ndarray   # [n_time_cells]
    position: np.ndarray           # Current position [spatial_dims]
    elapsed_time: float
```

---

### 4. Learning System (`learning/`)

#### 4.1 Continuous Learner (`learning/continuous_learner.py`)
**Purpose**: Background learning from conversations and content

**Schema**:
```python
@dataclass
class LearningConfig:
    stdp_lr_plus: float = 0.01
    stdp_lr_minus: float = 0.012
    stdp_time_window: int = 5
    queue_size: int = 1000
    batch_size: int = 10
    process_interval_sec: float = 1.0
    vocab_dir: Optional[str] = None
    state_dir: str = "learning_state"
    save_interval_sec: float = 300.0

@dataclass
class ContentItem:
    text: str
    category: ContentCategory      # CONVERSATION, LOCAL_FILE, RSS_FEED
    priority: ProcessingPriority   # CRITICAL, HIGH, MEDIUM, LOW, BACKGROUND
    source: str
    timestamp: datetime
    content_hash: str
    processed: bool = False

class ContinuousLearner:
    memory: MemoryStore
    snn: SNNCompute
    embedder: Model
    stdp: STDPLearner
    content_queue: asyncio.Queue[ContentItem]
    event_bus: EventBus
```

**State Persistence** (`learning_state/`):
- `stdp_state.json`: STDP learner weights and associations
- `processed_hashes.json`: Deduplication tracking
- `stats.json`: Learning statistics

#### 4.2 STDP Learner (`learning/stdp_learner.py`)
**Purpose**: Spike-Timing Dependent Plasticity for temporal associations

**Schema**:
```python
class STDPLearner:
    token_weights: Dict[int, float]           # Token salience
    associations: Dict[tuple, float]          # (token_i, token_j) -> strength
    spike_traces: Dict[int, float]            # Last spike time
    current_time: float
    
    def process_sequence(token_ids, spikes)
    def process_embedding_pair(emb1_indices, emb2_indices, relevance)
    def get_modulations(token_ids) -> np.ndarray
```

**State Schema**:
```json
{
    "token_weights": {
        "12345": 0.7,
        "67890": 0.5
    },
    "associations": {
        "12345:67890": 0.6
    },
    "stats": {
        "total_updates": 1000,
        "ltp_events": 800,
        "ltd_events": 200
    }
}
```

---

### 5. Vulkan GPU Backend (`vulkan_backend/`)

**Purpose**: GPU-accelerated compute operations using Vulkan shaders

#### 5.1 Core Components

**VulkanCompute** (`vulkan_backend/vulkan_compute.py`):
```python
class VulkanCompute:
    core: VulkanCore              # Low-level Vulkan operations
    snn: VulkanSNN                # Spiking neural networks
    faiss: VulkanFAISS            # Similarity search
    fnn: VulkanFNN                # Feedforward networks
    attention: VulkanAttention    # Attention mechanisms
    memory: VulkanMemory          # Memory operations
    cells: VulkanCells            # Place/time cells
    
    def create_buffer(data_or_size, usage='storage')
    def upload_buffer(buffer, memory, data)
    def download_buffer(memory, size, dtype)
    def dispatch_compute(pipeline, layout, descriptor_set, workgroups)
```

**VulkanCore** (`vulkan_backend/vulkan_core.py`):
- Device initialization
- Buffer management
- Pipeline creation
- Descriptor set management
- Command buffer recording

**Shader Organization** (`shaders/`):
- `snn-*.glsl`: Spiking neural network operations
- `faiss-*.glsl`: FAISS similarity search
- `hebbian-learning.glsl`: Hebbian weight updates
- `stdp-learning.glsl`: STDP learning
- `place-cells.glsl`: Place cell computations
- `time-cells.glsl`: Time cell computations
- `theta-gamma-encoding.glsl`: Phase-coupled encoding
- `attention-*.glsl`: Attention mechanisms
- `fnn-*.glsl`: Feedforward network operations

#### 5.2 FAISS Integration (`vulkan_backend/vulkan_faiss.py`)
**Purpose**: GPU-accelerated similarity search

**Schema**:
```python
class VulkanFAISS:
    def compute_distances(queries, database, distance_type='cosine')
        # Returns: distances [num_queries, num_database]
    
    def topk(distances, k)
        # Returns: (indices [num_queries, k], values [num_queries, k])
```

**Shaders**:
- `faiss-distance.glsl`: L2, cosine, dot product distances
- `faiss-topk.glsl`: Top-k selection

#### 5.3 SNN Compute (`vulkan_backend/snn_compute.py`)
**Purpose**: Spiking neural network simulation

**Schema**:
```python
class SNNCompute:
    n_neurons: int = 1000
    dt: float = 0.01              # 10ms timestep
    tau_mem: float = 5.0          # Membrane time constant
    v_thresh: float = 0.5         # Spike threshold
    
    def forward(embeddings) -> Tuple[np.ndarray, np.ndarray]
        # Returns: (spike_times, spike_counts)
```

---

## Data Schemas

### Memory Entry Schema
```python
{
    "id": int,                    # Database primary key
    "embedding": np.ndarray[384], # float32 embedding vector
    "text": str,                  # Memory text content
    "timestamp": str,             # ISO 8601 timestamp
    "access_count": int,          # Number of retrievals
    "last_accessed": str,         # ISO 8601 timestamp
    "metadata": dict,             # Optional metadata
    "is_identity": bool,          # Identity memory flag
    "is_protected": bool          # Protected from deletion
}
```

### Brain Processing Result Schema
```python
{
    "emotional_state": {
        "arousal": float,
        "valence": float,
        "dominant_emotion": str,
        "confidence": float
    },
    "strategy": str,              # Selected response strategy
    "modulations": {
        "attention_scale": float,
        "creativity": float,
        "empathy": float,
        "analytical_depth": float
    },
    "memory_context": List[str],  # Retrieved memory texts
    "should_respond": bool,       # Go/no-go decision
    "spatial_context": {
        "place_activations": np.ndarray,  # [n_place_cells]
        "time_activations": np.ndarray     # [n_time_cells]
    },
    "thalamus_gate": bool,        # Whether input passed gating
    "hormonal_state": Dict[str, float],
    "consciousness": str,
    "stress_level": float
}
```

### Training Data Schema (`brain/data_loader.py`)
```python
@dataclass
class DataItem:
    id: str
    text: str
    category: DataCategory        # AFFECT, CONVERSATION, INSTRUCTION
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray[384]]
    valence: Optional[float]      # For affect data
    arousal: Optional[float]     # For affect data
    timestamp: Optional[datetime]

@dataclass
class BatchStats:
    batch_id: int
    items_processed: int
    gpu_time_ms: float
    learning_updates: int
    avg_valence: float
    avg_arousal: float
```

### Temporal Record Schema (`brain/temporal_indexer.py`)
```python
@dataclass
class TemporalRecord:
    id: str
    text: str
    timestamp: datetime
    location: Optional[str]
    keywords: List[str]
    text_embedding: Optional[np.ndarray[384]]
    temporal_encoding: Optional[np.ndarray]
    spatial_encoding: Optional[np.ndarray]
    source_file: str
    metadata: Dict[str, Any]
```

---

## API Interfaces

### FastAPI Server (`main.py`)

**WebSocket Endpoint**: `/ws`

**Message Format**:
```json
{
    "prompt": "User message text",
    "context": {...}  // Optional
}
```

**Response Format**:
```json
{
    "response": "AI response text",
    "spike_activity": 1234,
    "memories_used": ["memory1", "memory2"],
    "emotional_state": {...},
    "strategy": "empathetic"
}
```

### CLI Interface (`cli/cli.py`)

**Commands**:
- `python cli.py "prompt"`: Single query
- `python cli.py`: Interactive mode
- `python cli.py --stats`: Show memory statistics
- `python cli.py --clear`: Clear all memories
- `python cli.py --learning`: Enable continuous learning
- `python cli.py --calibrate`: Train amygdala affect prediction

---

## Configuration

### Configuration File (`config.py`)

**ModelConfig**:
```python
EMBEDDING_DIM = 384
PHI3_EMBEDDING_DIM = 3072
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
MAX_NEW_TOKENS_GPU = 256
MAX_NEW_TOKENS_CPU = 128
TEMPERATURE = 0.7
TOP_P = 0.9
```

**MemoryConfig**:
```python
DB_PATH = "memories.db"
MAX_MEMORIES = 100000
EMBEDDING_DIM = 384
DEFAULT_K = 3
GPU_BUFFER_SIZE = 10000
```

**SNNConfig**:
```python
N_NEURONS = 1000
DT = 0.01
TAU_MEM = 5.0
V_THRESH = 0.5
TIMESTEPS = 50
INPUT_SCALE = 20.0
```

**ServerConfig**:
```python
HOST = "127.0.0.1"
PORT = 8080
WS_PATH = "/ws"
```

---

## State Management

### Persistent State Files

**Brain State** (`brain_state/`):
- `amygdala.json`: Affect weights and emotional state
- `limbic_system.json`: Memory-emotion associations
- `thalamus.json`: Gate weights
- `basal_ganglia.json`: Strategy weights
- `endocrine.json`: Hormone levels
- `cns.json`: Consciousness and stress
- `experiences.json`: Significant experiences
- `brain_stats.json`: Overall statistics

**Learning State** (`learning_state/`):
- `stdp_state.json`: STDP weights and associations
- `processed_hashes.json`: Content deduplication
- `stats.json`: Learning statistics

**Database**:
- `memories.db`: SQLite database with all memories

### State Loading/Saving

All brain components support:
- `save_state(path)`: Persist to JSON
- `load_state(path)`: Load from JSON
- Automatic state persistence on shutdown
- Periodic state saves during operation

---

## Data Flow

### 1. User Input Processing

```
User Input (text)
    ↓
Phi3GGUF.get_embedding() → embedding[384]
    ↓
MemoryStore.retrieve(embedding, k=3) → memory_context
    ↓
UnifiedBrain.process(text, embedding, context)
    ├─→ Thalamus.gate(embedding) → gate_decision
    ├─→ Amygdala.process(text, embedding) → emotional_state
    ├─→ LimbicSystem.integrate(memory_context, emotional_state)
    ├─→ CNS.update(stress, fatigue)
    ├─→ EndocrineSystem.update(hormones)
    ├─→ BasalGanglia.select_action(activations) → strategy
    └─→ GPUBrainCompute (place/time cells, spatial memory)
    ↓
Result: {strategy, modulations, emotional_state, ...}
    ↓
Phi3GGUF.generate(prompt_with_context) → response_text
    ↓
MemoryStore.store(response_embedding, response_text)
    ↓
User receives response
```

### 2. Learning Flow

```
Conversation Pair (user_text, assistant_text)
    ↓
ContinuousLearner.learn_from_conversation()
    ├─→ Get embeddings for both texts
    ├─→ STDPLearner.process_embedding_pair(emb1, emb2)
    │   └─→ Update token associations
    ├─→ MemoryStore.store() for both texts
    └─→ Update statistics
    ↓
Background Processing (async)
    ├─→ Process content queue
    ├─→ Update STDP weights
    └─→ Save state periodically
```

### 3. GPU Computation Flow

```
CPU: Prepare data (numpy arrays)
    ↓
VulkanCompute.create_buffer() → GPU buffer
    ↓
VulkanCompute.upload_buffer() → Transfer to GPU
    ↓
VulkanCompute.dispatch_compute() → Execute shader
    ├─→ Shader reads from input buffers
    ├─→ Shader performs computation
    └─→ Shader writes to output buffers
    ↓
VulkanCompute.download_buffer() → Transfer back to CPU
    ↓
CPU: Process results (numpy arrays)
```

---

## Performance Characteristics

### Memory Usage
- **Memory Store**: ~150MB per 10k memories (384-dim embeddings)
- **Brain State**: ~50MB (weights and state)
- **GPU Buffers**: ~500MB-2GB depending on batch size
- **Model**: ~2.3GB (Phi-3-mini GGUF Q4)

### GPU Acceleration
- **FAISS Search**: 10-100x faster than CPU
- **SNN Forward**: 50-200x faster than CPU
- **Hebbian Learning**: 20-50x faster than CPU
- **Place/Time Cells**: 100-500x faster than CPU

### Latency
- **Embedding Generation**: 10-50ms (sentence-transformers)
- **Memory Retrieval**: 1-5ms (GPU FAISS)
- **Brain Processing**: 5-20ms (GPU-accelerated)
- **Text Generation**: 100-500ms (depends on length)

---

## Extension Points

### Adding New Brain Regions
1. Create new class in `brain/` directory
2. Implement `process()` method
3. Add to `UnifiedBrain.__init__()`
4. Integrate into `UnifiedBrain.process()` pipeline
5. Add state persistence (`save_state()`, `load_state()`)

### Adding New Shaders
1. Create `.glsl` file in `shaders/` directory
2. Add pipeline creation in `vulkan_backend/vulkan_pipelines.py`
3. Add method to appropriate Vulkan module (e.g., `VulkanSNN`, `VulkanFAISS`)
4. Expose via `VulkanCompute` public API

### Adding New Learning Algorithms
1. Create new learner class in `learning/` directory
2. Implement learning interface (process, update, save_state)
3. Integrate into `ContinuousLearner`
4. Add configuration to `LearningConfig`

---

## Security Considerations

1. **Memory Protection**: `is_protected` flag prevents deletion of important memories
2. **Identity Memory**: Always retrieved, cannot be deleted
3. **Input Validation**: All user inputs sanitized before processing
4. **State Persistence**: JSON files validated on load
5. **GPU Memory**: Bounded buffers prevent OOM attacks

---

## Troubleshooting

### Common Issues

1. **GPU Not Available**: Falls back to CPU automatically
2. **Memory Full**: Oldest memories evicted (except protected/identity)
3. **State Corruption**: Backup state files created on each save
4. **Shader Compilation Errors**: Check Vulkan drivers and SPIR-V compiler

### Debugging

- Enable debug logging: `LogConfig.LEVEL = "DEBUG"`
- Check GPU status: `VulkanCompute.get_stats()`
- Monitor memory: `MemoryStore.get_stats()`
- Brain diagnostics: `UnifiedBrain.get_stats()`

---

## Future Enhancements

1. **Multi-modal Support**: Vision and audio embeddings
2. **Distributed Memory**: Multi-node memory sharing
3. **Advanced Learning**: Reinforcement learning, meta-learning
4. **Real-time Adaptation**: Online learning from user feedback
5. **Memory Compression**: Quantization and pruning for efficiency

---

## References

- **Vulkan Specification**: https://www.khronos.org/vulkan/
- **FAISS Documentation**: https://github.com/facebookresearch/faiss
- **Phi-3 Model**: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
- **Sentence-Transformers**: https://www.sbert.net/

---

*Last Updated: 2024-01-08*
