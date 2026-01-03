# Shader Test Status

## ✅ Tested & Implemented

### 1. Neuron Models
- ✅ `lif-neuron` - Tested in `test_gpu_dispatch.py`

### 2. Synaptic & Learning
- ✅ `hebbian-learning` - Tested in `test_learning.py` and `test_learning_datasets.py`
- ✅ `stdp-learning` - Tested in `test_learning.py` and `test_learning_datasets.py`

### 3. Feed-Forward (FFN) & Activations
- ✅ `fnn-linear` - Tested in `test_hippocampal_transformer.py`
- ✅ `fnn-layernorm` - Tested in `test_hippocampal_transformer.py`
- ✅ `activation-relu` - Tested in `test_activations.py`
- ✅ `activation-gelu` - Tested in `test_activations.py`
- ✅ `activation-silu` - Tested in `test_activations.py`
- ✅ `activation-softmax` - Tested in `test_activations.py` and `test_hippocampal_transformer.py`

### 4. Attention Mechanisms
- ✅ `attention-scores` - Tested in `test_hippocampal_transformer.py`
- ✅ `attention-mask` - Tested in `test_hippocampal_transformer.py`
- ✅ `attention-output` - Tested in `test_hippocampal_transformer.py`
- ✅ `attention-concat-heads` - Tested in `test_hippocampal_transformer.py`

### 5. Memory & Retrieval (RAG)
- ✅ `memory-query-pooling` - Tested in `test_hippocampal_transformer.py`
- ✅ `memory-read` - Tested in `test_hippocampal_transformer.py`
- ✅ `memory-inject-gate` - Tested in `test_hippocampal_transformer.py`

---

## ❌ Not Yet Tested (Priority Order)

### High Priority (Core Functionality)

#### 1. Neuron Models
- ❌ `gif-neuron` - Gated Integrate-and-Fire (complex adaptive dynamics)
- ❌ `gif-prosody` - Prosody-Modulated GIF (Auditory/Speech processing)

#### 2. Synaptic & Learning
- ❌ `synapsis-forward` - Linear projection (Spike → Current)
- ❌ `synapsis-stdp-trace` - Update eligibility traces
- ❌ `synapsis-stdp-update` - STDP weight update
- ❌ `fisher-info` - Accumulate Fisher Information (Continual Learning/EWC)
- ❌ `fisher-natural-gradient` - Natural Gradient Descent

#### 3. Encodings & Embeddings
- ❌ `embedding-lookup` - Dense vector retrieval from token IDs
- ❌ `place-cell` - Gaussian place field generation (Spatial encoding)
- ❌ `time-cell` - Temporal Gaussian/Sequential fields (Temporal encoding)
- ❌ `theta-gamma-encoding` - Phase-amplitude coupling positional encoding
- ❌ `semantic-encoder` - Fuses token embeddings + Place cells

#### 4. Attention Mechanisms
- ❌ `attention-prosody-modulation` - Biases scores using prosody features

#### 5. Feed-Forward (FFN) & Activations
- ❌ `fnn-dropout` - Training regularization (Inverted dropout)
- ❌ `activation-softplus` - Softplus activation (shader exists)

#### 6. Memory & Retrieval (RAG)
- ❌ `memory-write` - Store Key-Value pairs (Episodic storage)
- ❌ `memory-context-aggregate` - Weighted sum of memories
- ❌ `memory-inject-concat` - Simple additive memory injection (alternative to gate)

### Medium Priority (Advanced Features)

#### 7. FAISS (Similarity Search)
- ❌ `faiss-distance` - Pairwise distance matrix (L2, Cosine, IP)
- ❌ `faiss-topk` - Select K nearest neighbors (Selection sort)
- ❌ `faiss-quantize` - Vector Quantization (VQ) (Compression)
- ❌ `faiss-kmeans-update` - Centroid update step (Clustering)
- ❌ `faiss-ivf-filter` - IVF filtering

#### 8. Adaptive Experts (NLMS)
- ❌ `nlms-predict` - Linear prediction (Online expert)
- ❌ `nlms-update` - Normalized LMS update (Online learning)
- ❌ `nlms-ensemble` - RMSE-weighted gating (Expert fusion)
- ❌ `nlms-metrics` - NLMS metrics computation
- ❌ `domain-classifier` - Classify input domain (Routing)
- ❌ `domain-router` - Domain routing
- ❌ `domain-predict` - Domain prediction
- ❌ `domain-combine-experts` - Combine expert predictions

#### 9. Preprocessing & Whitening
- ❌ `whitening-transform` - Online standardization (Input normalization)
- ❌ `whitening-batch-stats` - Batch statistics for whitening
- ❌ `whitening-apply` - Apply whitening transformation
- ❌ `fft-butterfly` - Radix-2 FFT (Frequency analysis)
- ❌ `fft-bitrev` - Bit-reversal for FFT
- ❌ `fft-magnitude` - FFT magnitude computation
- ❌ `fft-normalize` - FFT normalization
- ❌ `fft-power-spectrum` - Power spectrum computation
- ❌ `bridge-spike-to-continuous` - Rate/Phase decoding (SNN → ANN)
- ❌ `bridge-continuous-to-spike` - Continuous to spike encoding
- ❌ `bridge-temporal-weights` - Temporal weight bridging

### Lower Priority (Specialized/Advanced)

#### 10. Loss Functions
- ❌ `loss-cross-entropy` - Cross-entropy loss (3-pass algorithm)
- ❌ `loss-fn-bce` - Binary Cross-Entropy loss

#### 11. Decoding
- ❌ `decode-greedy` - Greedy decoding
- ❌ `decode-sample` - Sampling-based decoding

#### 12. SNN-Specific
- ❌ `snn-matmul` - SNN matrix multiplication
- ❌ `snn-softmax` - SNN softmax
- ❌ `snn-rmsnorm` - SNN RMS normalization
- ❌ `snn-time-expand` - Time dimension expansion
- ❌ `snn-expand-time-dim` - Expand time dimension
- ❌ `snn-readout` - SNN readout layer
- ❌ `snn-expert-readout` - Expert readout for SNN

#### 13. Backward Pass (Gradients)
- ❌ `fnn-linear-backward` - Linear layer backward pass
- ❌ `activation-gelu-backward` - GELU backward pass

#### 14. Other
- ❌ `fnn-residual` - Residual connection
- ❌ `fisher-ewc-penalty` - EWC penalty computation
- ❌ `fisher-normalize` - Fisher information normalization
- ❌ `hybrid-blend` - Hybrid blending

---

## Summary Statistics

- **Tested**: 16 shaders
- **Untested**: ~59 shaders
- **Total**: ~75 shaders
- **Test Coverage**: ~21%

## Recommended Testing Order

1. **Core SNN Operations** (Complete SNN pipeline)
   - `synapsis-forward`
   - `synapsis-stdp-trace`
   - `synapsis-stdp-update`
   - `gif-neuron`

2. **Embeddings & Encodings** (Essential for transformer input)
   - `embedding-lookup`
   - `place-cell`
   - `time-cell`
   - `semantic-encoder`

3. **Memory Operations** (Complete RAG pipeline)
   - `memory-write`
   - `memory-context-aggregate`
   - `memory-inject-concat`

4. **Attention Extensions**
   - `attention-prosody-modulation`

5. **Regularization & Training**
   - `fnn-dropout`
   - `loss-cross-entropy`
   - `loss-fn-bce`

6. **Advanced Features**
   - FAISS shaders (similarity search)
   - NLMS shaders (adaptive experts)
   - FFT shaders (frequency analysis)
   - Whitening shaders (preprocessing)

