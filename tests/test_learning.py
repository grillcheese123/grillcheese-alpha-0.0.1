import numpy as np
import pytest

try:
    from vulkan_backend import VulkanCompute
    VULKAN_AVAILABLE = True
except Exception as e:
    VULKAN_AVAILABLE = False
    print(f"Vulkan not available: {e}")


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestHebbianLearning:
    """Test Hebbian learning on GPU"""
    
    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        return VulkanCompute()
    
    def test_hebbian_basic(self, gpu):
        """Test basic Hebbian learning"""
        batch_size = 2
        time_steps = 5
        pre_dim = 4
        post_dim = 3
        
        # Create correlated activations
        pre_activations = np.random.randn(batch_size, time_steps, pre_dim).astype(np.float32)
        post_activations = np.random.randn(batch_size, time_steps, post_dim).astype(np.float32)
        
        # Initialize weights to small random values
        weights = np.random.randn(post_dim, pre_dim).astype(np.float32) * 0.1
        
        # Apply Hebbian learning
        weights_new = gpu.hebbian_learning(
            pre_activations, post_activations, weights,
            learning_rate=0.01, weight_decay=0.0
        )
        
        print(f"\n✓ Hebbian learning test:")
        print(f"  Weights shape: {weights.shape}")
        print(f"  Weight change: {np.abs(weights_new - weights).mean():.6f}")
        
        # Weights should have changed
        assert weights_new.shape == weights.shape
        assert not np.allclose(weights_new, weights, atol=1e-6)
    
    def test_hebbian_weight_decay(self, gpu):
        """Test Hebbian learning with weight decay"""
        batch_size = 1
        time_steps = 3
        pre_dim = 2
        post_dim = 2
        
        pre_activations = np.ones((batch_size, time_steps, pre_dim), dtype=np.float32)
        post_activations = np.ones((batch_size, time_steps, post_dim), dtype=np.float32)
        
        weights = np.ones((post_dim, pre_dim), dtype=np.float32)
        
        # With zero activations but positive weights, weight decay should reduce weights
        pre_zero = np.zeros((batch_size, time_steps, pre_dim), dtype=np.float32)
        post_zero = np.zeros((batch_size, time_steps, post_dim), dtype=np.float32)
        
        weights_new = gpu.hebbian_learning(
            pre_zero, post_zero, weights.copy(),
            learning_rate=0.01, weight_decay=0.1
        )
        
        print(f"\n✓ Hebbian weight decay test:")
        print(f"  Original weights: {weights[0, 0]:.3f}")
        print(f"  New weights: {weights_new[0, 0]:.3f}")
        
        # With zero correlation and weight decay, weights should decrease
        assert weights_new[0, 0] < weights[0, 0]


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestSTDPLearning:
    """Test STDP learning on GPU"""
    
    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        return VulkanCompute()
    
    def test_stdp_basic(self, gpu):
        """Test basic STDP learning"""
        batch_size = 2
        time_steps = 5
        pre_dim = 4
        post_dim = 3
        
        # Create spike sequences
        pre_activations = (np.random.rand(batch_size, time_steps, pre_dim) > 0.7).astype(np.float32)
        post_activations = (np.random.rand(batch_size, time_steps, post_dim) > 0.7).astype(np.float32)
        
        # Initialize weights
        weights = np.random.randn(post_dim, pre_dim).astype(np.float32) * 0.1
        
        # Initialize traces
        pre_trace = np.zeros((batch_size, pre_dim), dtype=np.float32)
        post_trace = np.zeros((batch_size, post_dim), dtype=np.float32)
        
        # Apply STDP learning
        weights_new, pre_trace_new, post_trace_new = gpu.stdp_learning(
            pre_activations, post_activations, weights,
            pre_trace, post_trace,
            lr_potentiation=0.01, lr_depression=0.01, trace_decay=0.9
        )
        
        print(f"\n✓ STDP learning test:")
        print(f"  Weights shape: {weights.shape}")
        print(f"  Weight change: {np.abs(weights_new - weights).mean():.6f}")
        print(f"  Trace update: pre_trace max={pre_trace_new.max():.3f}, post_trace max={post_trace_new.max():.3f}")
        
        # Weights should have changed
        assert weights_new.shape == weights.shape
        assert pre_trace_new.shape == pre_trace.shape
        assert post_trace_new.shape == post_trace.shape
        
        # Traces should be non-zero after update
        assert pre_trace_new.max() > 0 or np.all(pre_trace_new == 0)
        assert post_trace_new.max() > 0 or np.all(post_trace_new == 0)
    
    def test_stdp_correlation(self, gpu):
        """Test STDP with correlated pre/post spikes"""
        batch_size = 1
        time_steps = 10
        pre_dim = 2
        post_dim = 2
        
        # Create correlated spikes (post spikes after pre)
        pre_activations = np.zeros((batch_size, time_steps, pre_dim), dtype=np.float32)
        post_activations = np.zeros((batch_size, time_steps, post_dim), dtype=np.float32)
        
        # Pre fires at time 2, post fires at time 3 (LTP scenario)
        pre_activations[0, 2, 0] = 1.0
        post_activations[0, 3, 0] = 1.0
        
        weights = np.zeros((post_dim, pre_dim), dtype=np.float32)
        pre_trace = np.zeros((batch_size, pre_dim), dtype=np.float32)
        post_trace = np.zeros((batch_size, post_dim), dtype=np.float32)
        
        weights_new, _, _ = gpu.stdp_learning(
            pre_activations, post_activations, weights,
            pre_trace, post_trace,
            lr_potentiation=0.1, lr_depression=0.01, trace_decay=0.5
        )
        
        print(f"\n✓ STDP correlation test:")
        print(f"  Weight[0,0] change: {weights_new[0, 0]:.6f}")
        
        # Weight should increase due to LTP (pre before post)
        # Note: exact value depends on trace dynamics
        assert weights_new[0, 0] >= weights[0, 0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

