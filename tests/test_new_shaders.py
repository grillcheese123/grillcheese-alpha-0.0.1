import numpy as np
import pytest

try:
    from vulkan_backend import VulkanCompute
    VULKAN_AVAILABLE = True
except Exception as e:
    VULKAN_AVAILABLE = False
    print(f"Vulkan not available: {e}")


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestEmbeddingLookup:
    """Test embedding lookup shader"""
    
    @pytest.fixture
    def gpu(self):
        return VulkanCompute()
    
    def test_embedding_lookup_basic(self, gpu):
        """Test basic embedding lookup"""
        batch_size = 2
        seq_len = 5
        vocab_size = 100
        embedding_dim = 64
        
        # Create embedding table
        embedding_table = np.random.randn(vocab_size, embedding_dim).astype(np.float32) * 0.1
        
        # Create token IDs
        token_ids = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.uint32)
        
        # Lookup embeddings
        embeddings = gpu.embedding_lookup(token_ids, embedding_table)
        
        print(f"\n✓ Embedding Lookup:")
        print(f"  Input tokens: {token_ids}")
        print(f"  Output shape: {embeddings.shape}")
        print(f"  Embedding mean: {embeddings.mean():.4f}, std: {embeddings.std():.4f}")
        
        assert embeddings.shape == (batch_size, seq_len, embedding_dim)
        
        # Verify embeddings match expected values
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = token_ids[b, s]
                expected_emb = embedding_table[token_id]
                actual_emb = embeddings[b, s]
                np.testing.assert_allclose(actual_emb, expected_emb, rtol=1e-5, atol=1e-5)
    
    def test_embedding_lookup_invalid_tokens(self, gpu):
        """Test embedding lookup with invalid token IDs"""
        vocab_size = 50
        embedding_dim = 32
        embedding_table = np.random.randn(vocab_size, embedding_dim).astype(np.float32)
        
        # Create token IDs with some invalid values
        token_ids = np.array([[0, 10, 60, 30, 5]], dtype=np.uint32)  # 60 is out of range
        
        embeddings = gpu.embedding_lookup(token_ids, embedding_table)
        
        # Invalid tokens should result in zero embeddings
        assert embeddings[0, 2].sum() == 0.0  # Token 60 should be zeroed
        assert embeddings[0, 0].sum() != 0.0  # Valid tokens should have values


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestMemoryWrite:
    """Test memory write operations"""
    
    @pytest.fixture
    def gpu(self):
        return VulkanCompute()
    
    def test_memory_write_overwrite(self, gpu):
        """Test overwriting memory"""
        num_memories = 10
        key_dim = 32
        value_dim = 64
        write_index = 3
        
        # Initialize memory
        memory_keys = np.random.randn(num_memories, key_dim).astype(np.float32)
        memory_values = np.random.randn(num_memories, value_dim).astype(np.float32)
        
        old_key = memory_keys[write_index].copy()
        old_value = memory_values[write_index].copy()
        
        # Create new key-value pair
        new_key = np.random.randn(key_dim).astype(np.float32)
        new_value = np.random.randn(value_dim).astype(np.float32)
        
        # Write to memory
        updated_keys, updated_values = gpu.memory_write(
            new_key, new_value, memory_keys, memory_values, 
            write_index, write_mode=0  # overwrite
        )
        
        print(f"\n✓ Memory Write (Overwrite):")
        print(f"  Write index: {write_index}")
        print(f"  Old key sum: {old_key.sum():.4f}, New key sum: {updated_keys[write_index].sum():.4f}")
        
        # Verify the written memory is updated
        np.testing.assert_allclose(updated_keys[write_index], new_key, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(updated_values[write_index], new_value, rtol=1e-5, atol=1e-5)
        
        # Verify other memories are unchanged
        for i in range(num_memories):
            if i != write_index:
                np.testing.assert_allclose(updated_keys[i], memory_keys[i], rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(updated_values[i], memory_values[i], rtol=1e-5, atol=1e-5)
    
    def test_memory_write_blend(self, gpu):
        """Test blending with existing memory"""
        num_memories = 5
        key_dim = 16
        value_dim = 32
        write_index = 2
        blend_factor = 0.3
        
        # Initialize memory
        memory_keys = np.random.randn(num_memories, key_dim).astype(np.float32)
        memory_values = np.random.randn(num_memories, value_dim).astype(np.float32)
        
        old_key = memory_keys[write_index].copy()
        old_value = memory_values[write_index].copy()
        
        # Create new key-value pair
        new_key = np.random.randn(key_dim).astype(np.float32)
        new_value = np.random.randn(value_dim).astype(np.float32)
        
        # Write to memory with blending
        updated_keys, updated_values = gpu.memory_write(
            new_key, new_value, memory_keys, memory_values,
            write_index, write_mode=1, blend_factor=blend_factor  # blend
        )
        
        print(f"\n✓ Memory Write (Blend):")
        print(f"  Blend factor: {blend_factor}")
        
        # Verify blended values
        expected_key = blend_factor * new_key + (1.0 - blend_factor) * old_key
        expected_value = blend_factor * new_value + (1.0 - blend_factor) * old_value
        
        np.testing.assert_allclose(updated_keys[write_index], expected_key, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(updated_values[write_index], expected_value, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestDropout:
    """Test dropout regularization"""
    
    @pytest.fixture
    def gpu(self):
        return VulkanCompute()
    
    def test_dropout_training(self, gpu):
        """Test dropout in training mode"""
        input_data = np.ones((10, 20), dtype=np.float32) * 5.0
        dropout_prob = 0.2
        
        # Apply dropout in training mode
        output = gpu.dropout(input_data, dropout_prob=dropout_prob, is_training=True, seed=42)
        
        print(f"\n✓ Dropout (Training):")
        print(f"  Input mean: {input_data.mean():.4f}")
        print(f"  Output mean: {output.mean():.4f}")
        print(f"  Output std: {output.std():.4f}")
        print(f"  Zeros ratio: {(output == 0).sum() / output.size:.2%}")
        
        # In training mode with inverted dropout, mean should be preserved approximately
        # (some elements zeroed, others scaled up)
        assert output.shape == input_data.shape
        assert output.mean() > 0  # Some values should remain
        
        # With inverted dropout, the expected mean is approximately the input mean
        # (scaled by 1/(1-p) for kept elements, but some are zeroed)
        expected_mean = input_data.mean() * (1 - dropout_prob)
        assert abs(output.mean() - expected_mean) < input_data.mean() * 0.3
    
    def test_dropout_inference(self, gpu):
        """Test dropout in inference mode"""
        input_data = np.random.randn(5, 10).astype(np.float32)
        
        # Apply dropout in inference mode (should pass through unchanged)
        output = gpu.dropout(input_data, dropout_prob=0.5, is_training=False)
        
        print(f"\n✓ Dropout (Inference):")
        print(f"  Input shape: {input_data.shape}")
        print(f"  Output matches input: {np.allclose(output, input_data)}")
        
        # In inference mode, output should match input exactly
        np.testing.assert_allclose(output, input_data, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestPlaceCell:
    """Test place cell encoding"""
    
    @pytest.fixture
    def gpu(self):
        return VulkanCompute()
    
    def test_place_cell_2d(self, gpu):
        """Test 2D place cell encoding"""
        n_neurons = 20
        spatial_dims = 2
        
        # Create place field centers in 2D space
        field_centers = np.random.randn(n_neurons, spatial_dims).astype(np.float32) * 2.0
        
        # Agent position
        agent_position = np.array([1.0, 0.5], dtype=np.float32)
        
        # Generate firing rates
        rates = gpu.place_cell(
            agent_position, field_centers,
            field_width=1.0, max_rate=20.0, baseline_rate=0.1, spatial_dims=2
        )
        
        print(f"\n✓ Place Cell (2D):")
        print(f"  Agent position: {agent_position}")
        print(f"  Number of neurons: {n_neurons}")
        print(f"  Firing rates shape: {rates.shape}")
        print(f"  Rate range: [{rates.min():.2f}, {rates.max():.2f}] Hz")
        print(f"  Mean rate: {rates.mean():.2f} Hz")
        
        assert rates.shape == (n_neurons,)
        assert np.all(rates >= 0)  # Rates should be non-negative
        assert np.all(rates <= 20.1)  # Should not exceed max_rate + small epsilon
        
        # Neurons closer to agent should fire more
        # Find neuron closest to agent
        distances = np.linalg.norm(field_centers - agent_position, axis=1)
        closest_neuron = np.argmin(distances)
        print(f"  Closest neuron {closest_neuron} rate: {rates[closest_neuron]:.2f} Hz")
        assert rates[closest_neuron] > rates.mean()  # Closest should fire more than average
    
    def test_place_cell_3d(self, gpu):
        """Test 3D place cell encoding"""
        n_neurons = 15
        spatial_dims = 3
        
        # Create place field centers in 3D space
        field_centers = np.random.randn(n_neurons, spatial_dims).astype(np.float32) * 2.0
        
        # Agent position in 3D
        agent_position = np.array([0.5, -0.3, 1.0], dtype=np.float32)
        
        # Generate firing rates
        rates = gpu.place_cell(
            agent_position, field_centers,
            field_width=1.5, max_rate=25.0, baseline_rate=0.1, spatial_dims=3
        )
        
        print(f"\n✓ Place Cell (3D):")
        print(f"  Agent position: {agent_position}")
        print(f"  Rate range: [{rates.min():.2f}, {rates.max():.2f}] Hz")
        
        assert rates.shape == (n_neurons,)
        assert np.all(rates >= 0)
        assert np.all(rates <= 25.1)

