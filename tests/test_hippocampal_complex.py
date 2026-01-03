import numpy as np
import pytest

try:
    from vulkan_backend import VulkanCompute
    VULKAN_AVAILABLE = True
except Exception as e:
    VULKAN_AVAILABLE = False
    print(f"Vulkan not available: {e}")


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestHippocampalComplex:
    """Complex scenarios for hippocampal transformer"""
    
    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        return VulkanCompute()
    
    def create_transformer_layer_weights(self, dim, num_heads, head_dim, ffn_dim, num_memories):
        """Create weights for a single transformer layer"""
        qkv_dim = num_heads * head_dim
        
        return {
            'gamma1': np.ones(dim, dtype=np.float32),
            'beta1': np.zeros(dim, dtype=np.float32),
            'gamma2': np.ones(dim, dtype=np.float32),
            'beta2': np.zeros(dim, dtype=np.float32),
            'W_qkv': np.random.randn(3 * qkv_dim, dim).astype(np.float32) * 0.1,
            'b_qkv': np.zeros(3 * qkv_dim, dtype=np.float32),
            'W_query': np.random.randn(dim, dim).astype(np.float32) * 0.1,
            'b_query': np.zeros(dim, dtype=np.float32),
            'memory_keys': np.random.randn(num_memories, dim).astype(np.float32) * 0.1,
            'memory_values': np.random.randn(num_memories, dim).astype(np.float32) * 0.1,
            'W_gate': np.random.randn(dim, dim * 2).astype(np.float32) * 0.1,
            'b_gate': np.zeros(dim, dtype=np.float32),
            'W_mem_proj': np.random.randn(dim, dim).astype(np.float32) * 0.1,
            'W_ffn1': np.random.randn(ffn_dim, dim).astype(np.float32) * 0.1,
            'b_ffn1': np.zeros(ffn_dim, dtype=np.float32),
            'W_ffn2': np.random.randn(dim, ffn_dim).astype(np.float32) * 0.1,
            'b_ffn2': np.zeros(dim, dtype=np.float32),
        }
    
    def transformer_layer_forward(self, gpu, x, weights, num_heads, head_dim, ffn_dim, num_memories, eps=1e-5):
        """Forward pass through a single transformer layer"""
        # LayerNorm 1
        x_norm1 = gpu.layernorm(x, weights['gamma1'], weights['beta1'], eps=eps)
        
        # QKV Projection
        qkv = gpu.linear(x_norm1, weights['W_qkv'], weights['b_qkv'])
        qkv_dim = num_heads * head_dim
        Q = qkv[:, :, :qkv_dim].reshape(x.shape[0], x.shape[1], num_heads, head_dim)
        K = qkv[:, :, qkv_dim:2*qkv_dim].reshape(x.shape[0], x.shape[1], num_heads, head_dim)
        V = qkv[:, :, 2*qkv_dim:].reshape(x.shape[0], x.shape[1], num_heads, head_dim)
        
        # Attention Scores
        attn_scores = gpu.attention_scores(Q, K, num_heads, head_dim)
        
        # Memory Query Pooling
        memory_queries = gpu.memory_query_pooling(x_norm1, weights['W_query'], weights['b_query'])
        
        # Memory Read
        memory_context = gpu.memory_read(memory_queries, weights['memory_keys'], weights['memory_values'])
        
        # Causal Mask
        attn_scores_masked = gpu.attention_mask(attn_scores, use_causal=True)
        
        # Softmax
        attn_scores_flat = attn_scores_masked.reshape(x.shape[0] * num_heads, x.shape[1], x.shape[1])
        attn_weights = gpu.activation_softmax(attn_scores_flat, axis=-1)
        attn_weights = attn_weights.reshape(x.shape[0], num_heads, x.shape[1], x.shape[1])
        
        # Attention Output
        attn_output = gpu.attention_output(attn_weights, V, num_heads, head_dim)
        attn_output_concat = gpu.attention_concat_heads(attn_output)
        
        # Memory Inject
        x_augmented = gpu.memory_inject_gate(
            attn_output_concat, memory_context,
            weights['W_gate'], weights['b_gate'], weights['W_mem_proj']
        )
        
        # Residual 1
        x_residual = x + x_augmented
        
        # LayerNorm 2
        x_norm2 = gpu.layernorm(x_residual, weights['gamma2'], weights['beta2'], eps=eps)
        
        # FFN
        ffn_output = gpu.linear(x_norm2, weights['W_ffn1'], weights['b_ffn1'])
        ffn_activated = gpu.activation_gelu(ffn_output)
        ffn_output2 = gpu.linear(ffn_activated, weights['W_ffn2'], weights['b_ffn2'])
        
        # Residual 2
        output = x_residual + ffn_output2
        
        return output
    
    def test_multi_layer_transformer(self, gpu):
        """Test multiple stacked transformer layers"""
        batch_size = 4
        seq_len = 16
        dim = 128
        num_heads = 8
        head_dim = 16
        ffn_dim = 256
        num_memories = 20
        num_layers = 3
        
        print(f"\n✓ Testing Multi-Layer Transformer:")
        print(f"  Layers: {num_layers}, Batch: {batch_size}, Seq: {seq_len}, Dim: {dim}")
        
        # Create input
        x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        x_initial = x.copy()
        
        # Create weights for each layer
        layer_weights = []
        for i in range(num_layers):
            weights = self.create_transformer_layer_weights(dim, num_heads, head_dim, ffn_dim, num_memories)
            layer_weights.append(weights)
        
        # Forward through all layers
        current_x = x
        for i, weights in enumerate(layer_weights):
            current_x = self.transformer_layer_forward(
                gpu, current_x, weights, num_heads, head_dim, ffn_dim, num_memories
            )
            print(f"  ✓ Layer {i+1} output: {current_x.shape}, mean: {current_x.mean():.4f}, std: {current_x.std():.4f}")
        
        # Verify output shape
        assert current_x.shape == x.shape
        assert not np.allclose(current_x, x_initial, atol=1e-5)
        
        # Verify output statistics are reasonable
        assert np.isfinite(current_x).all(), "Output contains NaN or Inf"
        assert current_x.std() > 0.01, "Output variance too low"
        
        print(f"  ✓ Multi-layer transformer completed successfully!")
    
    def test_long_sequence_transformer(self, gpu):
        """Test transformer with longer sequences"""
        batch_size = 2
        seq_len = 64  # Longer sequence
        dim = 128
        num_heads = 8
        head_dim = 16
        ffn_dim = 256
        num_memories = 30
        
        print(f"\n✓ Testing Long Sequence Transformer:")
        print(f"  Batch: {batch_size}, Seq: {seq_len}, Dim: {dim}")
        
        # Create input
        x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        
        # Create weights
        weights = self.create_transformer_layer_weights(dim, num_heads, head_dim, ffn_dim, num_memories)
        
        # Forward pass
        output = self.transformer_layer_forward(
            gpu, x, weights, num_heads, head_dim, ffn_dim, num_memories
        )
        
        # Verify
        assert output.shape == x.shape
        assert np.isfinite(output).all()
        
        # Check attention patterns (causal mask should work correctly)
        # Memory should be retrieved for each batch
        memory_queries = gpu.memory_query_pooling(
            gpu.layernorm(x, weights['gamma1'], weights['beta1']),
            weights['W_query'], weights['b_query']
        )
        memory_context = gpu.memory_read(memory_queries, weights['memory_keys'], weights['memory_values'])
        
        assert memory_context.shape == (batch_size, dim)
        assert np.isfinite(memory_context).all()
        
        print(f"  ✓ Long sequence transformer completed successfully!")
    
    def test_sequence_modeling_task(self, gpu):
        """Test transformer on a simple sequence modeling task (next token prediction)"""
        batch_size = 4
        seq_len = 32
        dim = 128
        num_heads = 8
        head_dim = 16
        ffn_dim = 256
        num_memories = 25
        vocab_size = 1000  # Vocabulary size
        num_layers = 2
        
        print(f"\n✓ Testing Sequence Modeling Task:")
        print(f"  Task: Next token prediction")
        print(f"  Vocab: {vocab_size}, Layers: {num_layers}, Seq: {seq_len}")
        
        # Create input embeddings (simulate token embeddings)
        # In real scenario, these would come from embedding lookup
        input_embeddings = np.random.randn(batch_size, seq_len, dim).astype(np.float32) * 0.1
        
        # Create layer weights
        layer_weights = []
        for i in range(num_layers):
            weights = self.create_transformer_layer_weights(dim, num_heads, head_dim, ffn_dim, num_memories)
            layer_weights.append(weights)
        
        # Forward through transformer layers
        hidden = input_embeddings
        for i, weights in enumerate(layer_weights):
            hidden = self.transformer_layer_forward(
                gpu, hidden, weights, num_heads, head_dim, ffn_dim, num_memories
            )
            print(f"  ✓ Layer {i+1} hidden states: {hidden.shape}")
        
        # Output projection to vocabulary (simulate output head)
        W_out = np.random.randn(vocab_size, dim).astype(np.float32) * 0.1
        b_out = np.zeros(vocab_size, dtype=np.float32)
        
        # Project to vocabulary logits
        # Take last token for next token prediction
        last_hidden = hidden[:, -1, :]  # (batch, dim)
        logits = gpu.linear(last_hidden.reshape(batch_size, 1, dim), W_out, b_out)
        logits = logits.reshape(batch_size, vocab_size)
        
        print(f"  ✓ Output logits: {logits.shape}")
        assert logits.shape == (batch_size, vocab_size)
        
        # Apply softmax to get probabilities
        logits_reshaped = logits.reshape(batch_size, 1, vocab_size)
        probs = gpu.activation_softmax(logits_reshaped, axis=-1)
        probs = probs.reshape(batch_size, vocab_size)
        
        # Verify probabilities sum to 1
        prob_sums = probs.sum(axis=-1)
        np.testing.assert_allclose(prob_sums, np.ones(batch_size), rtol=1e-4, atol=1e-4)
        
        # Get predictions (argmax)
        predictions = np.argmax(probs, axis=-1)
        print(f"  ✓ Predictions: {predictions}")
        assert predictions.shape == (batch_size,)
        assert np.all((predictions >= 0) & (predictions < vocab_size))
        
        print(f"  ✓ Sequence modeling task completed successfully!")
    
    def test_memory_retrieval_scenarios(self, gpu):
        """Test different memory retrieval scenarios"""
        batch_size = 3
        seq_len = 24
        dim = 128
        num_heads = 8
        head_dim = 16
        ffn_dim = 256
        num_memories = 50  # Larger memory bank
        
        print(f"\n✓ Testing Memory Retrieval Scenarios:")
        print(f"  Memory bank size: {num_memories}")
        
        # Create input
        x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        
        # Create weights with larger memory bank
        weights = self.create_transformer_layer_weights(dim, num_heads, head_dim, ffn_dim, num_memories)
        
        # Test 1: Standard memory retrieval
        x_norm = gpu.layernorm(x, weights['gamma1'], weights['beta1'])
        memory_queries = gpu.memory_query_pooling(x_norm, weights['W_query'], weights['b_query'])
        memory_context = gpu.memory_read(memory_queries, weights['memory_keys'], weights['memory_values'])
        
        assert memory_context.shape == (batch_size, dim)
        print(f"  ✓ Standard retrieval: {memory_context.shape}")
        
        # Test 2: Multiple sequential retrievals (simulating multiple queries)
        memory_contexts = []
        for i in range(3):
            # Create different query projections
            W_query_i = np.random.randn(dim, dim).astype(np.float32) * 0.1
            b_query_i = np.zeros(dim, dtype=np.float32)
            queries_i = gpu.memory_query_pooling(x_norm, W_query_i, b_query_i)
            context_i = gpu.memory_read(queries_i, weights['memory_keys'], weights['memory_values'])
            memory_contexts.append(context_i)
        
        # Verify all retrievals are different
        for i in range(len(memory_contexts)):
            for j in range(i + 1, len(memory_contexts)):
                assert not np.allclose(memory_contexts[i], memory_contexts[j], atol=1e-5), \
                    f"Memory contexts {i} and {j} are too similar"
        
        print(f"  ✓ Multiple retrievals: {len(memory_contexts)} different contexts")
        
        # Test 3: Memory with different temperature settings
        queries = memory_queries
        context_temp1 = gpu.memory_read(queries, weights['memory_keys'], weights['memory_values'], temperature=1.0)
        context_temp2 = gpu.memory_read(queries, weights['memory_keys'], weights['memory_values'], temperature=10.0)
        
        # Higher temperature should make attention more uniform
        assert not np.allclose(context_temp1, context_temp2, atol=1e-5)
        print(f"  ✓ Temperature effects: different contexts with different temperatures")
        
        print(f"  ✓ Memory retrieval scenarios completed successfully!")
    
    def test_batch_processing(self, gpu):
        """Test transformer with larger batch sizes"""
        batch_size = 8  # Larger batch
        seq_len = 32
        dim = 128
        num_heads = 8
        head_dim = 16
        ffn_dim = 256
        num_memories = 30
        
        print(f"\n✓ Testing Batch Processing:")
        print(f"  Batch size: {batch_size}, Seq: {seq_len}, Dim: {dim}")
        
        # Create input
        x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        
        # Create weights
        weights = self.create_transformer_layer_weights(dim, num_heads, head_dim, ffn_dim, num_memories)
        
        # Forward pass
        output = self.transformer_layer_forward(
            gpu, x, weights, num_heads, head_dim, ffn_dim, num_memories
        )
        
        # Verify
        assert output.shape == x.shape
        assert np.isfinite(output).all()
        
        # Check that each batch element is processed independently
        # (outputs should be different for different inputs)
        for i in range(batch_size - 1):
            assert not np.allclose(output[i], output[i+1], atol=1e-5), \
                f"Batch elements {i} and {i+1} produced identical outputs"
        
        print(f"  ✓ Batch processing completed successfully!")
    
    def test_gradient_flow_simulation(self, gpu):
        """Simulate gradient flow through multiple layers (forward passes only)"""
        batch_size = 4
        seq_len = 16
        dim = 128
        num_heads = 8
        head_dim = 16
        ffn_dim = 256
        num_memories = 20
        num_layers = 4
        
        print(f"\n✓ Testing Gradient Flow Simulation:")
        print(f"  Layers: {num_layers}, Simulating forward passes")
        
        # Create input
        x = np.random.randn(batch_size, seq_len, dim).astype(np.float32)
        
        # Create weights for all layers
        layer_weights = []
        for i in range(num_layers):
            weights = self.create_transformer_layer_weights(dim, num_heads, head_dim, ffn_dim, num_memories)
            layer_weights.append(weights)
        
        # Forward through all layers
        activations = [x]
        current_x = x
        
        for i, weights in enumerate(layer_weights):
            current_x = self.transformer_layer_forward(
                gpu, current_x, weights, num_heads, head_dim, ffn_dim, num_memories
            )
            activations.append(current_x)
            
            # Check activation statistics (gradient flow indicators)
            mean_act = current_x.mean()
            std_act = current_x.std()
            max_act = current_x.max()
            min_act = current_x.min()
            
            print(f"  Layer {i+1}: mean={mean_act:.4f}, std={std_act:.4f}, range=[{min_act:.4f}, {max_act:.4f}]")
            
            # Verify activations are reasonable (not exploding or vanishing)
            assert abs(mean_act) < 10.0, f"Activation mean too large at layer {i+1}"
            assert std_act > 0.01, f"Activation std too small at layer {i+1}"
            assert np.isfinite(current_x).all(), f"Non-finite values at layer {i+1}"
        
        # Verify all activations have correct shapes
        for i, act in enumerate(activations):
            assert act.shape == x.shape, f"Activation shape mismatch at layer {i}"
        
        print(f"  ✓ Gradient flow simulation completed successfully!")

