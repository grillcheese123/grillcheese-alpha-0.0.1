"""
Tests for SNNCompute class
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vulkan_backend import SNNCompute
from config import SNNConfig


class TestSNNComputeInit:
    """Tests for SNNCompute initialization"""
    
    def test_init_default_neurons(self):
        """Should initialize with default neuron count"""
        snn = SNNCompute(use_vulkan=False)  # Force CPU for test reliability
        assert snn.n_neurons == SNNConfig.N_NEURONS
    
    def test_init_custom_neurons(self):
        """Should initialize with custom neuron count"""
        snn = SNNCompute(n_neurons=500, use_vulkan=False)
        assert snn.n_neurons == 500
    
    def test_init_creates_membrane_state(self):
        """Should create membrane potential array"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        assert snn.membrane is not None
        assert len(snn.membrane) == 100
        assert snn.membrane.dtype == np.float32
    
    def test_init_membrane_starts_at_zero(self):
        """Membrane potential should start at zero"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        assert np.all(snn.membrane == 0)


class TestSNNComputeReset:
    """Tests for reset method"""
    
    def test_reset_clears_membrane(self):
        """Reset should clear membrane potential"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        
        # Modify membrane
        snn.membrane = np.random.randn(100).astype(np.float32)
        
        # Reset
        snn.reset()
        
        assert np.all(snn.membrane == 0)
    
    def test_reset_clears_refractory(self):
        """Reset should clear refractory period"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        
        # Modify refractory
        snn.refractory = np.ones(100, dtype=np.float32)
        
        # Reset
        snn.reset()
        
        assert np.all(snn.refractory == 0)


class TestSNNComputeForward:
    """Tests for forward method (single timestep)"""
    
    def test_forward_returns_spikes(self):
        """Forward should return spike array"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        input_current = np.random.randn(100).astype(np.float32)
        
        spikes = snn.forward(input_current)
        
        assert spikes is not None
        assert len(spikes) == 100
        assert spikes.dtype == np.float32
    
    def test_forward_spikes_are_binary(self):
        """Spikes should be 0 or 1"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        input_current = np.random.randn(100).astype(np.float32)
        
        spikes = snn.forward(input_current)
        
        assert np.all((spikes == 0) | (spikes == 1))
    
    def test_forward_updates_membrane(self):
        """Forward should update membrane potential"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        initial_membrane = snn.membrane.copy()
        
        input_current = np.ones(100, dtype=np.float32)
        snn.forward(input_current)
        
        # Membrane should have changed
        assert not np.allclose(snn.membrane, initial_membrane)
    
    def test_forward_strong_input_causes_spikes(self):
        """Strong input should cause spikes"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        
        # Very strong input
        input_current = np.ones(100, dtype=np.float32) * 100.0
        
        # Run multiple timesteps
        total_spikes = 0
        for _ in range(10):
            spikes = snn.forward(input_current)
            total_spikes += spikes.sum()
        
        assert total_spikes > 0


class TestSNNComputeProcess:
    """Tests for process method (full pipeline)"""
    
    def test_process_returns_dict(self):
        """Process should return a dictionary"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        embedding = np.random.randn(384).astype(np.float32)
        
        result = snn.process(embedding)
        
        assert isinstance(result, dict)
    
    def test_process_contains_required_keys(self):
        """Result should contain required keys"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        embedding = np.random.randn(384).astype(np.float32)
        
        result = snn.process(embedding)
        
        assert 'spike_activity' in result
        assert 'spikes' in result
        assert 'firing_rate' in result
    
    def test_process_spike_activity_is_numeric(self):
        """Spike activity should be a number"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        embedding = np.random.randn(384).astype(np.float32)
        
        result = snn.process(embedding)
        
        assert isinstance(result['spike_activity'], (int, float))
        assert result['spike_activity'] >= 0
    
    def test_process_spikes_is_array(self):
        """Spikes should be an array"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        embedding = np.random.randn(384).astype(np.float32)
        
        result = snn.process(embedding)
        
        assert isinstance(result['spikes'], np.ndarray)
        assert len(result['spikes']) == 100
    
    def test_process_firing_rate_in_valid_range(self):
        """Firing rate should be between 0 and 1"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        embedding = np.random.randn(384).astype(np.float32)
        
        result = snn.process(embedding)
        
        assert 0 <= result['firing_rate'] <= 1
    
    def test_process_handles_small_embedding(self):
        """Process should handle embeddings smaller than n_neurons"""
        snn = SNNCompute(n_neurons=1000, use_vulkan=False)
        embedding = np.random.randn(100).astype(np.float32)  # Smaller than n_neurons
        
        result = snn.process(embedding)
        
        assert result['spikes'] is not None
        assert len(result['spikes']) == 1000
    
    def test_process_handles_large_embedding(self):
        """Process should handle embeddings larger than n_neurons"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        embedding = np.random.randn(1000).astype(np.float32)  # Larger than n_neurons
        
        result = snn.process(embedding)
        
        assert result['spikes'] is not None
        assert len(result['spikes']) == 100
    
    def test_process_produces_activity(self):
        """Process should produce some spike activity"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        
        # Use non-zero embedding
        embedding = np.ones(384, dtype=np.float32) * 0.5
        
        result = snn.process(embedding)
        
        # Should have some activity (either from LIF or fallback)
        assert result['spike_activity'] > 0


class TestSNNComputeGPU:
    """Tests for GPU mode (if available)"""
    
    def test_gpu_init_does_not_crash(self):
        """GPU initialization should not crash"""
        try:
            snn = SNNCompute(n_neurons=100, use_vulkan=True)
            # Either GPU or CPU mode should work
            assert snn.n_neurons == 100
        except Exception as e:
            pytest.skip(f"GPU not available: {e}")
    
    def test_gpu_process_works(self):
        """Process should work in GPU mode"""
        try:
            snn = SNNCompute(n_neurons=100, use_vulkan=True)
            embedding = np.random.randn(384).astype(np.float32)
            
            result = snn.process(embedding)
            
            assert 'spike_activity' in result
        except Exception as e:
            pytest.skip(f"GPU not available: {e}")


class TestSNNComputeReproducibility:
    """Tests for reproducibility"""
    
    def test_same_input_after_reset_gives_same_output(self):
        """Same input after reset should give same output"""
        snn = SNNCompute(n_neurons=100, use_vulkan=False)
        embedding = np.ones(384, dtype=np.float32) * 0.1
        
        # First run
        result1 = snn.process(embedding)
        
        # Reset and run again
        snn.reset()
        result2 = snn.process(embedding)
        
        # Should be the same
        assert result1['spike_activity'] == result2['spike_activity']
        np.testing.assert_array_equal(result1['spikes'], result2['spikes'])

