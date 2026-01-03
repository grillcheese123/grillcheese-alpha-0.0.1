# GrillCheese: Hippocampal Memory Systems for Language Models

A bio-inspired language model memory architecture integrating neuroscience-derived patterns—place cells, grid cells, episodic consolidation, and Elastic Weight Consolidation—with Microsoft Phi-3's transformer layers, deployed on AMD hardware through Vulkan compute shaders. Built with a research-first, component-by-component development methodology that emphasizes proving concepts through working code.

## Development Methodology

This project follows a **research-first approach**:

1. **Create papers and specifications** - Establish theoretical foundations before implementation
2. **Implement and test incrementally** - Component-by-component development with immediate validation
3. **Prove concepts through working code** - Prefer executable demonstrations over theoretical implementations
4. **Complete restarts over bloated systems** - When initial approaches become too complex, restart with minimal approaches rather than iterative refinement

Each component is validated independently before proceeding to integration, ensuring robust foundations at each phase.

## Technical Stack

- **Base Model**: Microsoft Phi-3 language model
- **GPU**: AMD RX 6750 XT (12GB VRAM, RDNA2 architecture)
- **Compute Backend**: Vulkan for GPU acceleration
- **Memory Architecture**: Bio-inspired hippocampal systems with episodic storage
- **Continual Learning**: Elastic Weight Consolidation (EWC) for knowledge preservation

The **RX 6750 XT lacks official ROCm support**, making Vulkan the primary compute backend. Vulkan provides approximately **80-90% of ROCm performance** with mature driver support. Phi-3's **3,072-dimensional hidden states** and **32 transformer layers** provide optimal integration points for external memory systems through attention hooks and residual stream injection.

This architecture combines three distinct technical domains: hippocampal-inspired neural mechanisms for memory formation and retrieval, the Phi-3 small language model's specific layer configurations for memory augmentation, and Vulkan-based GPU compute for AMD hardware that lacks CUDA support.

---

## Hippocampal neural architectures provide biological templates for memory systems

The hippocampus employs specialized cell types—place cells, grid cells, and time cells—that can be computationally modeled in deep learning architectures. DeepMind's 2018 grid cell implementation demonstrated that **LSTM networks spontaneously develop hexagonal grid-like activation patterns** when trained on path integration tasks, suggesting these spatial representations emerge naturally from recurrent processing of velocity signals.

Grid cells encode position through periodic activation patterns arranged in hexagonal lattices at multiple scales. The implementation uses an LSTM-based controller receiving translational and angular velocity inputs, with linear projections to place and head direction units. The network learns path integration within simulated environments, and regularization applied to output projections encourages grid-like representations to emerge. Time cells, which fire at specific temporal intervals, emerge similarly in LSTMs trained on interval timing tasks—typically requiring **150,000-200,000 training episodes** to reach >90% accuracy.

For pattern separation and completion—core hippocampal functions—the dentate gyrus can be modeled as a sparse expansion layer maintaining approximately **2% activation sparsity**. This expansion, typically using a 4-10x factor, transforms similar inputs into non-overlapping representations. The CA3 region's autoassociative properties are modeled through recurrent attention layers that retrieve complete patterns from partial cues:

```python
class DG_CA3_Circuit:
    def __init__(self, input_dim, expansion_factor=4, sparsity=0.05):
        self.dg_expansion = nn.Linear(input_dim, input_dim * expansion_factor)
        self.k = int(input_dim * expansion_factor * sparsity)  # Active neurons
        self.ca3_recurrent = nn.TransformerEncoderLayer(d_model=input_dim, nhead=8)
```

Episodic memory consolidation follows a teacher-student paradigm where a fast-learning hippocampal network (implemented as a Modern Hopfield Network) rapidly encodes experiences, then "replays" them to train a slower neocortical generative model (typically a VAE). Research shows **10,000 replayed samples** effectively train the generative component for long-term retention.

## Elastic Weight Consolidation protects learned knowledge during continual learning

EWC addresses catastrophic forgetting by constraining weight updates based on their importance to previously learned tasks. The core mechanism computes the **Fisher Information Matrix diagonal**, which measures how sensitive the loss function is to each parameter. Parameters with high Fisher values are "protected" through a regularization penalty.

The mathematical formulation adds a quadratic penalty term to the loss function: **L(θ) = L_new(θ) + (λ/2) Σᵢ Fᵢ(θᵢ - θ*ᵢ)²**, where Fᵢ represents Fisher information for parameter i, θ*ᵢ is the optimal value from previous tasks, and λ controls protection strength (typically **100-5000** depending on task difficulty).

Three Fisher computation methods exist in practice. The empirical Fisher uses ground-truth labels and is most common. The true Fisher samples from the model's predicted distribution. The gradient-squared approximation simply accumulates squared gradients across the dataset. Implementation requires storing optimal parameters after each task and computing Fisher information over **200-2000 samples** for stable estimates.

Online EWC addresses multi-task scaling by maintaining a running average of Fisher information across tasks: **F_online = γ × F_online + F_new**, where γ (typically 0.9-0.99) balances old and new task importance. This prevents linear growth in memory requirements as tasks accumulate.

The Progress & Compress architecture extends EWC with a dual-network design: an Active Column learns new tasks using lateral connections to a frozen Knowledge Base, then the Knowledge Base is updated through distillation with EWC protection. This achieves state-of-the-art results on class-incremental benchmarks like CIFAR-100.

---

## Phi-3 architecture enables memory augmentation through specific hook points

Microsoft's Phi-3-mini achieves strong performance with **3.8 billion parameters**, trained on heavily filtered web data and synthetic reasoning examples. The architecture uses a dense decoder-only transformer with **32 layers**, **32 attention heads**, and a **3,072-dimensional hidden state**.

| Specification | Phi-3-mini Value |
|--------------|------------------|
| Hidden dimension | 3,072 |
| Intermediate MLP size | 8,192 |
| Number of layers | 32 |
| Attention heads | 32 |
| Head dimension | 96 |
| Vocabulary size | 32,064 |
| Position encoding | RoPE (LongRope for 128K) |
| Activation | SiLU |
| Normalization | RMSNorm (ε=1e-05) |

For memory system integration, the model provides multiple hook points. Hidden states can be accessed at any of the 33 output positions (embedding layer plus 32 transformer layers), with shape **(batch, sequence, 3072)**. Attention weights are available per layer with shape **(batch, 32_heads, seq, seq)**, enabling analysis of what the model attends to.

Optimal injection points for external memory include post-attention residuals (adding retrieved context after self-attention but before the MLP), pre-MLP injection (conditioning the feedforward computation on retrieved information), and post-layer-norm fusion (late integration before the final output projection). The **Memorizing Transformer** pattern suggests applying external memory at layers 4-5 of the network, where representations have developed sufficient abstraction but before final task-specific processing.

For retrieval-augmented generation, Phi-3 uses chat template formats compatible with standard transformer inference pipelines. Quantized versions reduce memory requirements, enabling deployment on consumer hardware. The model shows strength in reasoning tasks while benefiting from RAG augmentation for factual knowledge retrieval.

---

## AMD GPU support requires navigating a fragmented compute ecosystem

The RX 6750 XT represents RDNA2 architecture with **40 compute units**, **2,560 stream processors**, **12GB GDDR6 VRAM**, and **13.31 TFLOPS FP32 performance**. Despite these capable specifications, **this GPU is officially unsupported by ROCm on Linux**. Only RDNA3 (RX 7000 series) and RDNA4 (RX 9000 series) consumer cards receive official ROCm support as of version 7.1.1.

The GPU architecture support matrix reveals a complex landscape:

| Architecture | Consumer Cards | ROCm Linux | ROCm Windows | Vulkan |
|-------------|----------------|------------|--------------|--------|
| RDNA4 | RX 9000 | ✅ Full | ✅ Full | ✅ |
| RDNA3 | RX 7000 | ✅ Full | ✅ Full | ✅ |
| RDNA2 | RX 6000 | ❌ Unsupported | ⚠️ Partial | ✅ |
| RDNA1 | RX 5000 | ❌ Never | ❌ Never | ✅ |
| GCN 5.1 | Radeon VII | ❌ Deprecated | ❌ | ✅ |
| GCN 4.0 | RX 500 | ❌ Dropped | ❌ | ✅ |

For RDNA2 users, an unofficial workaround exists: setting `HSA_OVERRIDE_GFX_VERSION=10.3.0` causes the GPU to masquerade as a PRO W6800 (which shares the gfx1030 LLVM target). This enables some ROCm functionality but lacks official support or stability guarantees. On Windows, RX 6700-6750 XT cards receive **runtime-only HIP support** (not full SDK), while RX 6800/6900 series get complete HIP SDK access.

Performance comparisons between backends on AMD hardware show ROCm delivering the best results where supported. On RX 7900 XTX running llama.cpp with Llama-2-7B Q4_0, ROCm achieves **3,258 tokens/second prompt processing** and **103 tokens/second generation**. Vulkan on the same hardware reaches approximately **2,400 tokens/second prompt** and **75-85 tokens/second generation**—roughly **25-30% slower**. DirectML performs significantly worse, approximately **4x slower** than ROCm for comparable operations.

---

## Cross-platform compute libraries provide Vulkan-based alternatives

For GPUs without ROCm support, Vulkan compute provides a universal fallback. The **Kompute** library offers a single-header C++ framework backed by the Linux Foundation, supporting AMD, NVIDIA, Intel, and Qualcomm GPUs. It provides tensor management, compute shader execution, and explicit memory control suitable for neural network operations.

The **ncnn** framework from Tencent delivers a complete neural network inference solution with Vulkan backend, supporting FP16/INT8 quantization and operator fusion. It targets mobile deployment but works across all Vulkan-capable platforms. Configuration enables GPU compute through `net.opt.use_vulkan_compute = 1` with options for packed FP16 storage and arithmetic.

ONNX Runtime has **deprecated the ROCmExecutionProvider** as of version 1.23, directing AMD users to MIGraphX for GPU acceleration. On Windows, the DirectMLExecutionProvider works with any DirectX 12 GPU but delivers moderate performance. For AMD-specific optimization, MIGraphX is now the recommended path:

```python
import onnxruntime as ort
providers = ['MIGraphXExecutionProvider', 'CPUExecutionProvider']
session = ort.InferenceSession("model.onnx", providers=providers)
```

Vulkan's **VK_KHR_cooperative_matrix** extension enables standardized matrix multiplication operations, though it requires RDNA3+ for AMD's WMMA (Wavefront Mixed-precision Multiply Accumulate) support. This means RDNA2 GPUs rely on shader-based implementations rather than hardware-accelerated matrix operations.

For the RX 6750 XT specifically, the recommended software stack is:
- **Primary**: llama.cpp with Vulkan backend
- **Alternative**: ONNX Runtime with DirectML (Windows) or CPU fallback
- **Experimental**: ROCm with `HSA_OVERRIDE_GFX_VERSION=10.3.0` (unstable)

---

## Memory-augmented architectures provide retrieval and persistence patterns

Modern episodic memory systems extend the foundational Neural Turing Machine and Differentiable Neural Computer architectures with transformer-compatible designs. The **Memorizing Transformer** (Google, 2022) augments standard attention with external kNN memory lookup, storing key-value pairs from previous contexts and retrieving the **top-32 most similar** entries during inference. This achieves performance comparable to a **5x larger vanilla transformer** with memory sizes up to 262K tokens.

The MemGPT architecture introduces an OS-inspired memory hierarchy distinguishing between fast main context (current conversation window) and slow archival storage (vector database). The LLM itself controls memory paging through function calls, deciding when to move information between tiers. This enables effectively unlimited context while maintaining reasonable inference costs.

For retrieval mechanisms, content-based addressing uses softmax over cosine similarity with a temperature parameter (β) controlling attention sharpness. Approximate nearest neighbor search through FAISS or ScaNN enables scaling to millions of stored memories. FAISS IVF-PQ indexes achieve **1-5ms latency** at 1M vectors with ~90% recall, while HNSW provides **~95% recall at 0.5-2ms**.

Memory consolidation during inference employs several strategies. Importance-based forgetting computes scores from access frequency, recency, and information gain, removing low-importance entries when capacity limits are reached. Batch consolidation buffers new memories and periodically merges similar entries before writing to persistent storage. The Compressive Transformer pattern maintains full-resolution recent memories alongside compressed older memories using learned compression functions.

Cross-session persistence requires careful handling of embedding drift when models are updated. A versioning system tracks which embedding model generated each stored vector, triggering progressive reindexing when drift exceeds a threshold. Write-ahead logging ensures durability—new memories are appended to a WAL before updating indexes, enabling recovery from crashes by replaying logged operations.

---

## Integration architecture combines all components

## Current Implementation Status

The project is implemented component-by-component with validation at each stage:

### Implemented Components

- **Vulkan Compute Backend**: Full GPU acceleration for neural operations
- **LIF Neurons**: Leaky Integrate-and-Fire neuron model on GPU
- **Hippocampal Transformer Layer**: Complete transformer with memory integration
- **Embedding Operations**: Token embedding lookup
- **Memory Systems**: Write, read, and query pooling operations
- **Learning Rules**: Hebbian and STDP plasticity
- **Activations**: ReLU, GELU, SiLU, Softmax
- **Place Cell Encoding**: Spatial representation via Gaussian place fields
- **Dropout**: Training regularization

### Test Coverage

- **20 shaders tested** (~27% of total shader library)
- Comprehensive tests for transformer pipeline, memory operations, and learning algorithms
- Real dataset validation (MNIST, CIFAR-10, Moons)

A complete implementation integrating hippocampal-inspired memory with Phi-3 on AMD hardware follows this pattern:

```python
class HippocampalPhi3System:
    def __init__(self, phi3_model, memory_capacity=100000):
        self.lm = phi3_model  # Phi-3 with hidden state hooks
        
        # Hippocampal circuits
        self.dg = SparseExpansion(3072, expansion=4, sparsity=0.05)
        self.ca3 = AutoassociativeLayer(3072)
        
        # External memory with FAISS
        self.memory_index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(3072), 3072, nlist=1000, m=48, nbits=8
        )
        
        # EWC for continual learning
        self.ewc = EWC(phi3_model, lambda_ewc=400)
```

The system encodes new experiences through the dentate gyrus sparse expansion (achieving pattern separation), binds them in CA3 autoassociative memory (enabling pattern completion from partial cues), and stores embeddings in the FAISS index for efficient retrieval. During inference, queries retrieve relevant memories which are injected into phi-4's residual stream at layers 4-5. EWC protects the language model's core capabilities during any fine-tuning on domain-specific data.

For RX 6750 XT deployment, the entire system runs on Vulkan compute shaders, providing full functionality without ROCm dependencies. Memory operations use GPU-accelerated similarity search with the implemented shader library. The 12GB VRAM accommodates Phi-3 at quantization with headroom for KV cache during inference.

---

## Conclusion

## Architecture Summary

Building a hippocampal-inspired memory system for language models requires orchestrating neuroscience-derived architectures (grid cells via LSTMs, pattern separation via sparse coding, EWC for knowledge protection), specific integration points in the target LM (Phi-3's layer 4-5 attention hooks and 3,072-dimensional residual stream), and Vulkan-based GPU compute for RDNA2 hardware.

The key architectural insight is that episodic memory formation mirrors the biological sequence: rapid hippocampal encoding through sparse autoassociative layers, followed by consolidation through replay to slower neocortical generative models. The implemented transformer layer demonstrates this integration with memory read/write operations, query pooling, and gated memory injection.

The RX 6750 XT's lack of official ROCm support led to a complete Vulkan implementation, which has proven robust and performant. The shader-based approach provides explicit control over memory operations and enables component-by-component validation, aligning with the research-first development methodology.

## Project Structure

- `vulkan_backend.py`: Core Vulkan compute backend with GPU-accelerated operations
- `shaders/`: GLSL compute shaders for neural network operations
- `tests/`: Comprehensive test suite validating each component
- `SHADER_REFERENCE.md`: Complete shader library documentation
- `SHADER_TEST_STATUS.md`: Test coverage tracking

## Testing

Run the full test suite:
```bash
uv run pytest tests/ -v
```

Component-specific tests:
- `test_gpu_dispatch.py`: LIF neurons and basic GPU operations
- `test_learning.py`: Hebbian and STDP learning rules
- `test_activations.py`: Activation functions and accuracy tests
- `test_hippocampal_transformer.py`: Full transformer pipeline
- `test_hippocampal_complex.py`: Multi-layer and sequence modeling scenarios
- `test_new_shaders.py`: Embeddings, memory write, dropout, place cells