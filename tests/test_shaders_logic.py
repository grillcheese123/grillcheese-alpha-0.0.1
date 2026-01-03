import numpy as np
import pytest

class TestLIFLogic:
    """Test LIF neuron logic (CPU reference)"""
    
    def lif_step(self, membrane, input_current, threshold=1.0, decay=0.9):
        """CPU implementation - will verify shader matches this"""
        membrane = membrane * decay + input_current
        spike = (membrane >= threshold).astype(np.float32)
        membrane = membrane * (1 - spike)  # Reset on spike
        return membrane, spike
    
    def test_no_spike(self):
        """Below threshold - should integrate"""
        membrane, spike = self.lif_step(
            np.array([0.5]),
            np.array([0.2])
        )
        assert spike[0] == 0.0
        assert membrane[0] == pytest.approx(0.65, abs=0.01)
    
    def test_spike_and_reset(self):
        """Above threshold - should spike and reset"""
        membrane, spike = self.lif_step(
            np.array([0.95]),
            np.array([0.15])  # Changed: 0.95*0.9+0.15 = 1.005 > 1.0
        )
        assert spike[0] == 1.0
        assert membrane[0] == 0.0
    
    def test_multiple_neurons(self):
        """Test batch of neurons"""
        membrane = np.array([0.3, 0.95, 0.5, 1.2])
        input_current = np.array([0.1, 0.15, 0.6, 0.0])  # Fixed: 0.95*0.9+0.15=1.005
        
        membrane, spike = self.lif_step(membrane, input_current)
        
        assert spike[0] == 0.0  # 0.3*0.9+0.1 = 0.37 < 1.0
        assert spike[1] == 1.0  # 0.95*0.9+0.15 = 1.005 >= 1.0 ✓
        assert spike[2] == 1.0  # 0.5*0.9+0.6 = 1.05 >= 1.0
        assert spike[3] == 1.0  # 1.2*0.9+0.0 = 1.08 >= 1.0

class TestBridgeLogic:
    """Test SNN-ANN bridge"""
    
    def continuous_to_spike(self, embedding, timesteps=10):
        """Rate encoding: higher values = more spikes"""
        # Normalize to [0, 1]
        normalized = (embedding - embedding.min()) / (embedding.max() - embedding.min() + 1e-8)
        # Broadcast to timesteps
        spike_prob = np.tile(normalized.reshape(-1, 1), (1, timesteps))
        # Stochastic spikes
        spikes = (np.random.random(spike_prob.shape) < spike_prob).astype(np.float32)
        return spikes
    
    def spike_to_continuous(self, spike_train):
        """Decode spike rate to continuous value"""
        return spike_train.mean(axis=1)
    
    def test_high_value_more_spikes(self):
        """High embedding values should produce more spikes"""
        embedding = np.array([0.1, 0.9])
        spikes = self.continuous_to_spike(embedding, timesteps=100)
        
        rate_low = spikes[0].mean()
        rate_high = spikes[1].mean()
        
        assert rate_high > rate_low
    
    def test_roundtrip_preserves_order(self):
        """Encoding then decoding should preserve relative magnitudes"""
        original = np.array([0.2, 0.8, 0.5, 0.1, 0.9])
        spikes = self.continuous_to_spike(original, timesteps=1000)
        recovered = self.spike_to_continuous(spikes)
        
        # Check order preserved
        assert np.corrcoef(original, recovered)[0, 1] > 0.9


class TestSTDPLogic:
    """Test STDP learning rule"""
    
    def stdp_update(self, weights, pre_spikes, post_spikes, lr=0.01, tau=20.0):
        """
        Simple STDP: strengthen when pre→post, weaken when post→pre
        """
        # Outer product: pre_spikes (N,) × post_spikes (M,) = (N, M)
        dw = lr * np.outer(pre_spikes, post_spikes)
        weights = weights + dw
        return weights
    
    def test_potentiation(self):
        """Pre fires with post - weight increases"""
        weights = np.ones((3, 3)) * 0.5
        pre_spikes = np.array([1.0, 0.0, 0.0])
        post_spikes = np.array([1.0, 0.0, 0.0])
        
        new_weights = self.stdp_update(weights, pre_spikes, post_spikes, lr=0.1)
        
        # Weight[0,0] should increase
        assert new_weights[0, 0] > weights[0, 0]
        # Other weights unchanged (no pre or post spike)
        assert new_weights[1, 1] == weights[1, 1]
    
    def test_no_change_without_correlation(self):
        """No spikes = no weight change"""
        weights = np.ones((3, 3)) * 0.5
        pre_spikes = np.zeros(3)
        post_spikes = np.zeros(3)
        
        new_weights = self.stdp_update(weights, pre_spikes, post_spikes)
        
        np.testing.assert_array_equal(new_weights, weights)


class TestMemoryLogic:
    """Test memory storage logic"""
    memory_bank = []
    
    def write(self, embedding, metadata):
        """Store embedding with metadata"""
        self.memory_bank.append((embedding.copy(), metadata))
    
    def read(self, query, k=3):
        """Retrieve top-k similar memories"""
        if not self.memory_bank:
            return []
        
        # Cosine similarity
        similarities = []
        for emb, meta in self.memory_bank:
            sim = np.dot(query, emb) / (np.linalg.norm(query) * np.linalg.norm(emb))
            similarities.append((sim, emb, meta))
        
        # Sort and return top-k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return similarities[:k]
    
    def test_write_and_read(self):
        mem = TestMemoryLogic()
        
        emb1 = np.random.randn(768)
        emb2 = np.random.randn(768)
        
        mem.write(emb1, "first memory")
        mem.write(emb2, "second memory")
        
        # Query with emb1 - should retrieve itself
        results = mem.read(emb1, k=1)
        
        assert len(results) == 1
        assert results[0][2] == "first memory"
        assert results[0][0] > 0.99  # Very high similarity to itself
    
    def test_retrieval_order(self):
        """More similar memories should rank higher"""
        mem = TestMemoryLogic()
        
        query = np.random.randn(768)
        similar = query + np.random.randn(768) * 0.1  # Very similar
        dissimilar = np.random.randn(768)  # Random
        
        mem.write(dissimilar, "dissimilar")
        mem.write(similar, "similar")
        
        results = mem.read(query, k=2)
        
        # First result should be the similar one
        assert results[0][2] == "similar"
        assert results[0][0] > results[1][0]


class TestFAISSLogic:
    """Test FAISS-style operations"""
    
    def cosine_similarity(self, query, database):
        """Compute cosine similarity between query and database"""
        # Normalize
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        db_norm = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-8)
        
        # Dot product
        return np.dot(db_norm, query_norm)
    
    def topk(self, scores, k):
        """Return indices of top-k scores"""
        return np.argsort(scores)[-k:][::-1]
    
    def test_cosine_similarity(self):
        query = np.array([1.0, 0.0, 0.0])
        database = np.array([
            [1.0, 0.0, 0.0],  # Identical
            [0.0, 1.0, 0.0],  # Orthogonal
            [1.0, 1.0, 0.0],  # 45 degrees
        ])
        
        scores = self.cosine_similarity(query, database)
        
        assert scores[0] == pytest.approx(1.0, abs=0.01)  # Identical
        assert scores[1] == pytest.approx(0.0, abs=0.01)  # Orthogonal
        assert scores[2] == pytest.approx(0.707, abs=0.01)  # 45 deg
    
    def test_topk(self):
        scores = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
        indices = self.topk(scores, k=3)
        
        # Should be [1, 3, 4] (0.9, 0.7, 0.5)
        assert indices[0] == 1
        assert indices[1] == 3
        assert indices[2] == 4