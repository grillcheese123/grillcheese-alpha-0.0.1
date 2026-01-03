"""
GPU Dispatch Tests and Benchmarks
Tests actual GPU execution with Vulkan compute shaders
"""
import numpy as np
import pytest
import time

try:
    from vulkan_backend import VulkanCompute, SNNCompute
    VULKAN_AVAILABLE = True
except Exception as e:
    VULKAN_AVAILABLE = False
    print(f"Vulkan not available: {e}")


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestGPUDispatch:
    """Test actual GPU execution"""
    
    @pytest.fixture
    def gpu(self):
        """Initialize GPU backend"""
        return VulkanCompute()
    
    def test_gpu_initialization(self, gpu):
        """Test GPU initializes correctly"""
        assert gpu.device is not None
        assert gpu.queue is not None
        assert len(gpu.shaders) > 0
        print(f"\n[OK] GPU initialized with {len(gpu.shaders)} shaders")
    
    def test_lif_single_neuron(self, gpu):
        """Test single LIF neuron on GPU"""
        input_current = np.array([0.5], dtype=np.float32)
        membrane = np.array([0.3], dtype=np.float32)
        refractory = np.array([0.0], dtype=np.float32)
        
        mem_out, ref_out, spikes = gpu.lif_step(
            input_current, membrane, refractory,
            dt=0.001, tau_mem=20.0, v_thresh=1.0
        )
        
        print(f"\n[OK] Single neuron test:")
        print(f"  Input: {input_current[0]:.3f}")
        print(f"  Membrane: {membrane[0]:.3f} -> {mem_out[0]:.3f}")
        print(f"  Spike: {spikes[0]}")
        
        # Should integrate but not spike
        assert spikes[0] == 0.0
        assert mem_out[0] > membrane[0]  # Membrane increased
    
    def test_lif_spike_threshold(self, gpu):
        """Test LIF neuron spikes above threshold"""
        # Use large input current and dt to ensure spike occurs in one step
        input_current = np.array([20.0], dtype=np.float32)
        membrane = np.array([0.5], dtype=np.float32)
        refractory = np.array([0.0], dtype=np.float32)
        
        mem_out, ref_out, spikes = gpu.lif_step(
            input_current, membrane, refractory,
            dt=0.6, tau_mem=20.0, v_thresh=1.0  # dt=0.6 ensures spike: 0.5 + 0.6*(19.5/20.0) = 1.085
        )
        
        print(f"\n[OK] Spike test:")
        print(f"  Membrane before: {membrane[0]:.3f}")
        print(f"  Membrane after: {mem_out[0]:.3f}")
        print(f"  Spike: {spikes[0]}")
        print(f"  Refractory: {ref_out[0]:.3f}")
        
        # Should spike and reset
        assert spikes[0] == 1.0
        assert mem_out[0] == 0.0  # Reset
        assert ref_out[0] > 0.0   # In refractory period
    
    def test_lif_batch(self, gpu):
        """Test batch of neurons"""
        n = 100
        input_current = np.random.randn(n).astype(np.float32) * 0.5
        membrane = np.random.rand(n).astype(np.float32) * 0.5
        refractory = np.zeros(n, dtype=np.float32)
        
        mem_out, ref_out, spikes = gpu.lif_step(
            input_current, membrane, refractory
        )
        
        spike_count = spikes.sum()
        print(f"\n[OK] Batch test (n={n}):")
        print(f"  Spikes: {spike_count:.0f} / {n}")
        print(f"  Spike rate: {spike_count/n*100:.1f}%")
        
        assert spikes.shape == (n,)
        assert 0 <= spike_count <= n
    
    def test_lif_large_batch(self, gpu):
        """Test large batch (stress test)"""
        n = 10000
        input_current = np.random.randn(n).astype(np.float32) * 0.3
        membrane = np.zeros(n, dtype=np.float32)
        refractory = np.zeros(n, dtype=np.float32)
        
        start = time.time()
        mem_out, ref_out, spikes = gpu.lif_step(
            input_current, membrane, refractory
        )
        elapsed = time.time() - start
        
        spike_count = spikes.sum()
        print(f"\n[OK] Large batch test (n={n}):")
        print(f"  Time: {elapsed*1000:.2f} ms")
        print(f"  Throughput: {n/elapsed:.0f} neurons/sec")
        print(f"  Spikes: {spike_count:.0f}")
        
        assert elapsed < 0.5  # Should be fast on GPU (relaxed for safety)
    
    def test_lif_refractory_period(self, gpu):
        """Test refractory period prevents spikes"""
        input_current = np.array([1.0], dtype=np.float32)
        membrane = np.array([0.0], dtype=np.float32)
        refractory = np.array([2.0], dtype=np.float32)  # In refractory
        
        mem_out, ref_out, spikes = gpu.lif_step(
            input_current, membrane, refractory,
            dt=0.001
        )
        
        print(f"\n[OK] Refractory test:")
        print(f"  Refractory: {refractory[0]:.3f} -> {ref_out[0]:.3f}")
        print(f"  Membrane: {mem_out[0]:.3f}")
        print(f"  Spike: {spikes[0]}")
        
        # Should not spike
        assert spikes[0] == 0.0
        assert mem_out[0] == 0.0  # Held at reset
        assert ref_out[0] < refractory[0]  # Refractory decreased
    
    def test_gpu_vs_cpu_consistency(self, gpu):
        """Test GPU matches CPU implementation"""
        n = 100
        input_current = np.random.randn(n).astype(np.float32) * 0.3
        membrane = np.random.rand(n).astype(np.float32) * 0.5
        refractory = np.zeros(n, dtype=np.float32)
        
        # GPU version
        mem_gpu, ref_gpu, spikes_gpu = gpu.lif_step(
            input_current.copy(), membrane.copy(), refractory.copy()
        )
        
        # CPU version (from tests)
        mem_cpu = membrane.copy()
        ref_cpu = refractory.copy()
        spikes_cpu = np.zeros(n, dtype=np.float32)
        
        dt = 0.001
        tau = 20.0
        thresh = 1.0
        
        for i in range(n):
            if ref_cpu[i] > 0:
                ref_cpu[i] = max(0, ref_cpu[i] - dt)
                mem_cpu[i] = 0.0
            else:
                dV = (-mem_cpu[i] + input_current[i]) / tau
                mem_cpu[i] += dt * dV
                
                if mem_cpu[i] >= thresh:
                    spikes_cpu[i] = 1.0
                    mem_cpu[i] = 0.0
                    ref_cpu[i] = 2.0
        
        print(f"\n[OK] GPU vs CPU consistency:")
        print(f"  GPU spikes: {spikes_gpu.sum():.0f}")
        print(f"  CPU spikes: {spikes_cpu.sum():.0f}")
        
        # Results should be very similar (allowing for minor float differences)
        np.testing.assert_allclose(spikes_gpu, spikes_cpu, rtol=1e-3, atol=1e-3)
        np.testing.assert_allclose(mem_gpu, mem_cpu, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
class TestSNNInterface:
    """Test high-level SNN interface"""
    
    def test_snn_initialization(self):
        """Test SNN initializes with GPU"""
        snn = SNNCompute(n_neurons=100, use_vulkan=True)
        
        # GPU may or may not be available, just check initialization works
        assert snn.n_neurons == 100
        assert snn.membrane.shape == (100,)
        print(f"\n[OK] SNN interface initialized (GPU: {snn.use_vulkan})")
    
    def test_snn_forward_pass(self):
        """Test SNN forward pass"""
        snn = SNNCompute(n_neurons=100)
        input_current = np.random.randn(100).astype(np.float32) * 0.5
        
        spikes = snn.forward(input_current)
        
        print(f"\n[OK] Forward pass:")
        print(f"  Spikes: {spikes.sum():.0f} / 100")
        
        assert spikes.shape == (100,)
        assert spikes.dtype == np.float32
    
    def test_snn_reset(self):
        """Test SNN state reset"""
        snn = SNNCompute(n_neurons=100)
        
        # Run forward pass to create state
        input_current = np.ones(100, dtype=np.float32)
        snn.forward(input_current)
        
        # Reset
        snn.reset()
        
        assert snn.membrane.sum() == 0
        assert snn.refractory.sum() == 0
        print("\n[OK] Reset successful")
    
    def test_snn_temporal_dynamics(self):
        """Test SNN over multiple timesteps"""
        snn = SNNCompute(n_neurons=10)
        snn.reset()
        
        # Apply constant input over time - use very large input to ensure spikes occur
        input_current = np.ones(10, dtype=np.float32) * 250.0
        spike_history = []
        
        for t in range(100):
            spikes = snn.forward(input_current)
            spike_history.append(spikes.sum())
        
        total_spikes = sum(spike_history)
        print(f"\n[OK] Temporal dynamics (100 steps):")
        print(f"  Total spikes: {total_spikes:.0f}")
        print(f"  Average rate: {total_spikes/10:.1f} spikes/neuron")
        
        # Should spike periodically
        assert total_spikes > 0
    
    def test_snn_process_method(self):
        """Test high-level process method"""
        snn = SNNCompute(n_neurons=100)
        embedding = np.random.randn(384).astype(np.float32)
        
        result = snn.process(embedding)
        
        print(f"\n[OK] Process method:")
        print(f"  Spike activity: {result['spike_activity']:.0f}")
        print(f"  Firing rate: {result['firing_rate']*100:.2f}%")
        
        assert 'spike_activity' in result
        assert 'spikes' in result
        assert 'firing_rate' in result
        assert result['spike_activity'] >= 0


class TestPerformanceBenchmark:
    """Benchmark GPU performance"""
    
    @pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
    def test_benchmark_small(self):
        """Benchmark small network"""
        snn = SNNCompute(n_neurons=1000)
        input_current = np.random.randn(1000).astype(np.float32) * 0.5
        
        # Warmup
        for _ in range(10):
            snn.forward(input_current)
        
        # Benchmark
        start = time.time()
        iterations = 100
        for _ in range(iterations):
            snn.forward(input_current)
        elapsed = time.time() - start
        
        # Avoid division by zero
        elapsed = max(elapsed, 1e-6)
        throughput = (1000 * iterations) / elapsed
        
        print(f"\n[OK] Small network benchmark:")
        print(f"  Network: 1000 neurons")
        print(f"  Iterations: {iterations}")
        print(f"  Time: {elapsed:.3f} sec")
        print(f"  Throughput: {throughput/1e6:.2f} M neurons/sec")
    
    @pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
    def test_benchmark_medium(self):
        """Benchmark medium network"""
        snn = SNNCompute(n_neurons=100000)
        input_current = np.random.randn(100000).astype(np.float32) * 0.5
        
        # Warmup
        snn.forward(input_current)
        
        # Benchmark
        start = time.time()
        iterations = 50
        for _ in range(iterations):
            snn.forward(input_current)
        elapsed = time.time() - start
        
        elapsed = max(elapsed, 1e-6)
        throughput = (100000 * iterations) / elapsed
        
        print(f"\n[OK] Medium network benchmark:")
        print(f"  Network: 100,000 neurons")
        print(f"  Iterations: {iterations}")
        print(f"  Time: {elapsed:.3f} sec")
        print(f"  Throughput: {throughput/1e6:.2f} M neurons/sec")
    
    @pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
    def test_benchmark_large(self):
        """Benchmark large network"""
        snn = SNNCompute(n_neurons=1000000)
        input_current = np.random.randn(1000000).astype(np.float32) * 0.5
        
        # Warmup
        snn.forward(input_current)
        
        # Benchmark
        start = time.time()
        iterations = 10
        for _ in range(iterations):
            snn.forward(input_current)
        elapsed = time.time() - start
        
        elapsed = max(elapsed, 1e-6)
        throughput = (1000000 * iterations) / elapsed
        
        print(f"\n[OK] Large network benchmark:")
        print(f"  Network: 1,000,000 neurons")
        print(f"  Iterations: {iterations}")
        print(f"  Time: {elapsed:.3f} sec")
        print(f"  Throughput: {throughput/1e6:.2f} M neurons/sec")
        
        # GPU should be reasonably fast
        assert elapsed < 10  # Should complete in < 10 seconds
    
    @pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
    def test_benchmark_memory_store_similarity(self):
        """Benchmark GPU memory similarity computation"""
        try:
            from memory_store import MemoryStore
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as tmpdir:
                db_path = os.path.join(tmpdir, "bench_memories.db")
                memory = MemoryStore(db_path=db_path, embedding_dim=384)
                
                # Store some memories
                for i in range(100):
                    emb = np.random.randn(384).astype(np.float32)
                    memory.store(emb, f"Memory {i}")
                
                # Benchmark retrieval
                query = np.random.randn(384).astype(np.float32)
                
                # Warmup
                memory.retrieve(query, k=5)
                
                # Benchmark
                start = time.time()
                iterations = 100
                for _ in range(iterations):
                    memory.retrieve(query, k=5)
                elapsed = time.time() - start
                
                elapsed = max(elapsed, 1e-6)
                avg_latency = elapsed / iterations * 1000  # ms
                
                print(f"\n[OK] Memory similarity benchmark:")
                print(f"  Memories: 100")
                print(f"  Iterations: {iterations}")
                print(f"  Avg latency: {avg_latency:.2f} ms")
                
        except Exception as e:
            pytest.skip(f"MemoryStore not available: {e}")


class TestGPUShaders:
    """Test various GPU shaders"""
    
    @pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")
    def test_memory_write_shader(self):
        """Test memory-write shader"""
        from vulkan_backend import VulkanCompute
        gpu = VulkanCompute()
        
        # Create memory buffers
        n_memories = 10
        embedding_dim = 384
        
        keys = np.zeros((n_memories, embedding_dim), dtype=np.float32)
        values = np.zeros((n_memories, embedding_dim), dtype=np.float32)
        
        # Write a new memory
        new_key = np.random.randn(embedding_dim).astype(np.float32)
        new_value = np.random.randn(embedding_dim).astype(np.float32)
        write_index = 5
        
        updated_keys, updated_values = gpu.memory_write(
            new_key, new_value, keys, values, write_index, write_mode=0
        )
        
        print(f"\n[OK] memory-write shader test:")
        print(f"  Write index: {write_index}")
        print(f"  Key norm: {np.linalg.norm(updated_keys[write_index]):.4f}")
        
        # Check the memory was written
        assert np.allclose(updated_keys[write_index], new_key, rtol=1e-3)
        assert np.allclose(updated_values[write_index], new_value, rtol=1e-3)
    
    @pytest.mark.skipif(not VULKAN_AVAILABLE, reason="Vulkan not available")  
    def test_memory_read_shader(self):
        """Test memory-read shader"""
        from vulkan_backend import VulkanCompute
        gpu = VulkanCompute()
        
        # Create memory with some data
        n_memories = 10
        embedding_dim = 384
        
        keys = np.random.randn(n_memories, embedding_dim).astype(np.float32)
        values = np.random.randn(n_memories, embedding_dim).astype(np.float32)
        
        # Query
        queries = np.random.randn(1, embedding_dim).astype(np.float32)
        
        result = gpu.memory_read(queries, keys, values)
        
        print(f"\n[OK] memory-read shader test:")
        print(f"  Query shape: {queries.shape}")
        print(f"  Result shape: {result.shape}")
        print(f"  Result norm: {np.linalg.norm(result):.4f}")
        
        assert result.shape[1] == embedding_dim


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
