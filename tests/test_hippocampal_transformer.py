import numpy as np
import pytest

try:
    from vulkan_backend import VulkanCompute
    VULKAN_AVAILABLE = True
except Exception as e:
    VULKAN_AVAILABLE = False
    print(f"Vulkan not available: {e}")


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestHippocampalTransformer:
    """Test the complete hippocampal transformer layer pipeline"""
    
    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        return VulkanCompute()
    
    @pytest.fixture
    def transformer_config(self):
        """Configuration for transformer"""
        return {
            'batch_size': 2,
            'seq_len': 8,
            'dim': 64,
            'num_heads': 4,
            'head_dim': 16,
            'ffn_dim': 128,
            'num_memories': 10,
            'eps': 1e-5
        }
    
    @pytest.fixture
    def transformer_weights(self, transformer_config):
        """Initialize transformer weights"""
        cfg = transformer_config
        dim = cfg['dim']
        num_heads = cfg['num_heads']
        head_dim = cfg['head_dim']
        ffn_dim = cfg['ffn_dim']
        
        # LayerNorm parameters
        gamma1 = np.ones(dim, dtype=np.float32)
        beta1 = np.zeros(dim, dtype=np.float32)
        gamma2 = np.ones(dim, dtype=np.float32)
        beta2 = np.zeros(dim, dtype=np.float32)
        
        # QKV projection weights (3 * num_heads * head_dim, dim)
        qkv_dim = num_heads * head_dim
        W_qkv = np.random.randn(3 * qkv_dim, dim).astype(np.float32) * 0.1
        b_qkv = np.zeros(3 * qkv_dim, dtype=np.float32)
        
        # Memory query weights
        W_query = np.random.randn(dim, dim).astype(np.float32) * 0.1
        b_query = np.zeros(dim, dtype=np.float32)
        
        # Memory keys and values
        memory_keys = np.random.randn(cfg['num_memories'], dim).astype(np.float32) * 0.1
        memory_values = np.random.randn(cfg['num_memories'], dim).astype(np.float32) * 0.1
        
        # Memory gate weights (dim, dim * 2)
        W_gate = np.random.randn(dim, dim * 2).astype(np.float32) * 0.1
        b_gate = np.zeros(dim, dtype=np.float32)
        W_mem_proj = np.random.randn(dim, dim).astype(np.float32) * 0.1
        
        # FFN weights
        W_ffn1 = np.random.randn(ffn_dim, dim).astype(np.float32) * 0.1
        b_ffn1 = np.zeros(ffn_dim, dtype=np.float32)
        W_ffn2 = np.random.randn(dim, ffn_dim).astype(np.float32) * 0.1
        b_ffn2 = np.zeros(dim, dtype=np.float32)
        
        return {
            'gamma1': gamma1, 'beta1': beta1,
            'gamma2': gamma2, 'beta2': beta2,
            'W_qkv': W_qkv, 'b_qkv': b_qkv,
            'W_query': W_query, 'b_query': b_query,
            'memory_keys': memory_keys, 'memory_values': memory_values,
            'W_gate': W_gate, 'b_gate': b_gate, 'W_mem_proj': W_mem_proj,
            'W_ffn1': W_ffn1, 'b_ffn1': b_ffn1,
            'W_ffn2': W_ffn2, 'b_ffn2': b_ffn2
        }
    
    def test_hippocampal_transformer_full_pipeline(self, gpu, transformer_config, transformer_weights):
        """Test the complete hippocampal transformer layer"""
        cfg = transformer_config
        weights = transformer_weights
        
        # Create input
        x = np.random.randn(cfg['batch_size'], cfg['seq_len'], cfg['dim']).astype(np.float32)
        x_initial = x.copy()
        
        print(f"\n✓ Testing Hippocampal Transformer Layer:")
        print(f"  Input shape: {x.shape}")
        print(f"  Config: batch={cfg['batch_size']}, seq={cfg['seq_len']}, dim={cfg['dim']}, heads={cfg['num_heads']}")
        
        # Step 1: LayerNorm 1
        x_norm1 = gpu.layernorm(x, weights['gamma1'], weights['beta1'], eps=cfg['eps'])
        print(f"  ✓ LayerNorm 1: {x_norm1.shape}")
        assert x_norm1.shape == x.shape
        
        # Step 2: QKV Projection
        qkv = gpu.linear(x_norm1, weights['W_qkv'], weights['b_qkv'])
        print(f"  ✓ QKV Projection: {qkv.shape}")
        assert qkv.shape == (cfg['batch_size'], cfg['seq_len'], 3 * cfg['num_heads'] * cfg['head_dim'])
        
        # Split QKV
        qkv_dim = cfg['num_heads'] * cfg['head_dim']
        Q = qkv[:, :, :qkv_dim].reshape(cfg['batch_size'], cfg['seq_len'], cfg['num_heads'], cfg['head_dim'])
        K = qkv[:, :, qkv_dim:2*qkv_dim].reshape(cfg['batch_size'], cfg['seq_len'], cfg['num_heads'], cfg['head_dim'])
        V = qkv[:, :, 2*qkv_dim:].reshape(cfg['batch_size'], cfg['seq_len'], cfg['num_heads'], cfg['head_dim'])
        
        # Step 3: Attention Scores
        attn_scores = gpu.attention_scores(Q, K, cfg['num_heads'], cfg['head_dim'])
        print(f"  ✓ Attention Scores: {attn_scores.shape}")
        assert attn_scores.shape == (cfg['batch_size'], cfg['num_heads'], cfg['seq_len'], cfg['seq_len'])
        
        # Step 4: Memory Query Pooling
        memory_queries = gpu.memory_query_pooling(x_norm1, weights['W_query'], weights['b_query'])
        print(f"  ✓ Memory Queries: {memory_queries.shape}")
        assert memory_queries.shape == (cfg['batch_size'], cfg['dim'])
        
        # Step 5: Memory Read
        memory_context = gpu.memory_read(memory_queries, weights['memory_keys'], weights['memory_values'])
        print(f"  ✓ Memory Context: {memory_context.shape}")
        assert memory_context.shape == (cfg['batch_size'], cfg['dim'])
        
        # Step 6: Causal Mask
        attn_scores_masked = gpu.attention_mask(attn_scores, use_causal=True)
        print(f"  ✓ Causal Mask Applied")
        
        # Verify causal mask: upper triangle should be masked
        for b in range(cfg['batch_size']):
            for h in range(cfg['num_heads']):
                for i in range(cfg['seq_len']):
                    for j in range(i + 1, cfg['seq_len']):
                        assert attn_scores_masked[b, h, i, j] < -1e8, f"Causal mask failed at [{b},{h},{i},{j}]"
        
        # Step 7: Softmax on attention scores
        attn_scores_flat = attn_scores_masked.reshape(cfg['batch_size'] * cfg['num_heads'], cfg['seq_len'], cfg['seq_len'])
        attn_weights = gpu.activation_softmax(attn_scores_flat, axis=-1)
        attn_weights = attn_weights.reshape(cfg['batch_size'], cfg['num_heads'], cfg['seq_len'], cfg['seq_len'])
        print(f"  ✓ Softmax: {attn_weights.shape}")
        
        # Verify softmax: rows should sum to 1
        row_sums = attn_weights.sum(axis=-1)
        np.testing.assert_allclose(row_sums, np.ones_like(row_sums), rtol=1e-5, atol=1e-5)
        
        # Step 8: Attention Output
        attn_output = gpu.attention_output(attn_weights, V, cfg['num_heads'], cfg['head_dim'])
        print(f"  ✓ Attention Output: {attn_output.shape}")
        assert attn_output.shape == (cfg['batch_size'], cfg['seq_len'], cfg['num_heads'], cfg['head_dim'])
        
        # Step 9: Concatenate Heads
        attn_output_concat = gpu.attention_concat_heads(attn_output)
        print(f"  ✓ Concatenate Heads: {attn_output_concat.shape}")
        assert attn_output_concat.shape == (cfg['batch_size'], cfg['seq_len'], cfg['dim'])
        
        # Step 10: Memory Inject Gate
        x_augmented = gpu.memory_inject_gate(
            attn_output_concat, memory_context,
            weights['W_gate'], weights['b_gate'], weights['W_mem_proj']
        )
        print(f"  ✓ Memory Injection: {x_augmented.shape}")
        assert x_augmented.shape == x.shape
        
        # Step 11: Residual Connection
        x_residual = x + x_augmented
        print(f"  ✓ Residual Connection: {x_residual.shape}")
        
        # Step 12: LayerNorm 2
        x_norm2 = gpu.layernorm(x_residual, weights['gamma2'], weights['beta2'], eps=cfg['eps'])
        print(f"  ✓ LayerNorm 2: {x_norm2.shape}")
        assert x_norm2.shape == x.shape
        
        # Step 13: Feed Forward Network
        ffn_output = gpu.linear(x_norm2, weights['W_ffn1'], weights['b_ffn1'])
        print(f"  ✓ FFN Layer 1: {ffn_output.shape}")
        assert ffn_output.shape == (cfg['batch_size'], cfg['seq_len'], cfg['ffn_dim'])
        
        # Apply GELU activation
        ffn_activated = gpu.activation_gelu(ffn_output)
        print(f"  ✓ GELU Activation: {ffn_activated.shape}")
        
        # FFN Layer 2
        ffn_output2 = gpu.linear(ffn_activated, weights['W_ffn2'], weights['b_ffn2'])
        print(f"  ✓ FFN Layer 2: {ffn_output2.shape}")
        assert ffn_output2.shape == x.shape
        
        # Step 14: Final Residual
        output = x_residual + ffn_output2
        print(f"  ✓ Final Output: {output.shape}")
        assert output.shape == x.shape
        
        # Verify output is different from input
        assert not np.allclose(output, x_initial, atol=1e-5)
        
        print(f"\n✓ Hippocampal Transformer pipeline completed successfully!")
    
    def test_individual_components(self, gpu, transformer_config):
        """Test individual components of the transformer"""
        cfg = transformer_config
        
        # Test LayerNorm
        x = np.random.randn(cfg['batch_size'], cfg['seq_len'], cfg['dim']).astype(np.float32)
        gamma = np.ones(cfg['dim'], dtype=np.float32)
        beta = np.zeros(cfg['dim'], dtype=np.float32)
        
        x_norm = gpu.layernorm(x, gamma, beta)
        assert x_norm.shape == x.shape
        
        # Test Linear
        W = np.random.randn(cfg['dim'], cfg['dim']).astype(np.float32) * 0.1
        b = np.zeros(cfg['dim'], dtype=np.float32)
        y = gpu.linear(x, W, b)
        assert y.shape == x.shape
        
        # Test Attention Scores
        num_heads = cfg['num_heads']
        head_dim = cfg['head_dim']
        Q = np.random.randn(cfg['batch_size'], cfg['seq_len'], num_heads, head_dim).astype(np.float32)
        K = np.random.randn(cfg['batch_size'], cfg['seq_len'], num_heads, head_dim).astype(np.float32)
        scores = gpu.attention_scores(Q, K, num_heads, head_dim)
        assert scores.shape == (cfg['batch_size'], num_heads, cfg['seq_len'], cfg['seq_len'])
        
        # Test Memory Query Pooling
        W_query = np.random.randn(cfg['dim'], cfg['dim']).astype(np.float32) * 0.1
        b_query = np.zeros(cfg['dim'], dtype=np.float32)
        queries = gpu.memory_query_pooling(x, W_query, b_query)
        assert queries.shape == (cfg['batch_size'], cfg['dim'])
        
        # Test Memory Read
        num_memories = cfg['num_memories']
        memory_keys = np.random.randn(num_memories, cfg['dim']).astype(np.float32)
        memory_values = np.random.randn(num_memories, cfg['dim']).astype(np.float32)
        context = gpu.memory_read(queries, memory_keys, memory_values)
        assert context.shape == (cfg['batch_size'], cfg['dim'])
        
        print("\n✓ All individual components tested successfully!")

