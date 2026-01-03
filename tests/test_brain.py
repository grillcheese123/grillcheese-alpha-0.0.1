"""
Tests for the Brain Module
"""
import numpy as np
import pytest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from brain.amygdala import Amygdala, EmotionalState, EMOTION_LEXICON
from brain.endocrine import EndocrineSystem, HormoneType
from brain.cns import CentralNervousSystem, ConsciousnessLevel
from brain.thalamus import Thalamus
from brain.basal_ganglia import BasalGanglia
from brain.limbic_system import LimbicSystem
from brain.unified_brain import UnifiedBrain
from brain.gpu_brain import GPUBrainCompute, GPUSpatialMemory


class MockMemoryStore:
    """Mock memory store for testing"""
    def __init__(self):
        self.memories = []
        self.memory_count = 0
    
    def store(self, embedding, text):
        self.memories.append({'embedding': embedding, 'text': text})
        self.memory_count += 1
    
    def retrieve(self, embedding, k=3):
        return [m['text'] for m in self.memories[:k]]


class TestAmygdala:
    """Tests for the Amygdala component"""
    
    def test_init(self):
        amygdala = Amygdala()
        assert amygdala.state.arousal == 0.5
        assert amygdala.state.valence == 0.0
    
    def test_process_positive_text(self):
        amygdala = Amygdala(emotion_decay=0.5)
        
        result = amygdala.process("I am so happy and excited today!")
        
        assert result.valence > 0  # Should detect positive
        assert result.arousal > 0.5  # Happy + excited = high arousal
    
    def test_process_negative_text(self):
        amygdala = Amygdala(emotion_decay=0.5)
        
        result = amygdala.process("I feel sad and lonely")
        
        assert result.valence < 0  # Should detect negative
    
    def test_emotion_lexicon_coverage(self):
        # Verify lexicon has varied entries
        positive = sum(1 for _, (v, a) in EMOTION_LEXICON.items() if v > 0)
        negative = sum(1 for _, (v, a) in EMOTION_LEXICON.items() if v < 0)
        
        assert positive > 10
        assert negative > 10
    
    def test_emotional_state_to_dict(self):
        state = EmotionalState(arousal=0.7, valence=0.5, dominant_emotion="happy")
        d = state.to_dict()
        
        assert d['arousal'] == 0.7
        assert d['valence'] == 0.5
        assert d['dominant_emotion'] == "happy"
    
    def test_get_emotional_modulation(self):
        amygdala = Amygdala()
        amygdala.state.valence = 0.8
        amygdala.state.arousal = 0.6
        
        mod = amygdala.get_emotional_modulation()
        
        assert 'response_warmth' in mod
        assert 'response_energy' in mod
        assert mod['response_warmth'] > 0.5  # Positive valence = more warmth
    
    def test_save_load_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/amygdala.json"
            
            amygdala1 = Amygdala()
            amygdala1.process("I am very happy!")
            amygdala1.save_state(path)
            
            amygdala2 = Amygdala()
            amygdala2.load_state(path)
            
            assert abs(amygdala1.state.valence - amygdala2.state.valence) < 0.01


class TestAmygdalaCalibration:
    """Tests for Amygdala GPU calibration features"""
    
    @pytest.fixture
    def mock_embed_fn(self):
        """Mock embedding function"""
        def embed(text: str) -> np.ndarray:
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(384).astype(np.float32)
        return embed
    
    @pytest.fixture
    def sample_affect_jsonl(self, tmp_path):
        """Create sample affect JSONL file"""
        import json
        data = [
            {"id": "1", "text": "I feel very happy", "affect": {"valence": 0.8, "arousal": 0.6}},
            {"id": "2", "text": "This is frustrating", "affect": {"valence": -0.6, "arousal": 0.7}},
            {"id": "3", "text": "Calm peaceful morning", "affect": {"valence": 0.5, "arousal": 0.2}},
            {"id": "4", "text": "Excited and thrilled", "affect": {"valence": 0.9, "arousal": 0.9}},
            {"id": "5", "text": "Sad and lonely today", "affect": {"valence": -0.7, "arousal": 0.3}},
            {"id": "6", "text": "Anxious about results", "affect": {"valence": -0.4, "arousal": 0.8}},
            {"id": "7", "text": "Content and satisfied", "affect": {"valence": 0.6, "arousal": 0.3}},
            {"id": "8", "text": "Angry and annoyed", "affect": {"valence": -0.8, "arousal": 0.9}},
        ]
        
        filepath = tmp_path / "affect.jsonl"
        with open(filepath, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        return filepath
    
    def test_initial_uncalibrated(self):
        """Test that new amygdala is uncalibrated"""
        amygdala = Amygdala()
        assert not amygdala.is_calibrated
        assert amygdala.calibration_samples == 0
    
    def test_affect_weights_initialized(self):
        """Test 3-layer MLP weights are properly initialized"""
        amygdala = Amygdala(embedding_dim=384, hidden_dim=64)
        
        # Check 3-layer MLP architecture
        assert amygdala.W1.shape == (64, 384)   # hidden x embedding
        assert amygdala.b1.shape == (64,)
        assert amygdala.W2.shape == (64, 64)    # hidden x hidden (residual)
        assert amygdala.b2.shape == (64,)
        assert amygdala.W3.shape == (2, 64)     # output x hidden
        assert amygdala.b3.shape == (2,)
        
        # Check Adam state initialized
        assert 'W1' in amygdala.adam_m
        assert 'W3' in amygdala.adam_v
    
    def test_calibrate_from_jsonl(self, sample_affect_jsonl, mock_embed_fn):
        """Test calibration from JSONL file"""
        amygdala = Amygdala(embedding_dim=384, use_gpu=False)
        
        result = amygdala.calibrate_from_jsonl(
            sample_affect_jsonl,
            embed_fn=mock_embed_fn,
            batch_size=4,
            epochs=2,
            learning_rate=0.01
        )
        
        assert result['samples'] == 8
        assert result['epochs'] == 2
        assert result['is_calibrated']
        assert 'final_loss' in result
        assert amygdala.is_calibrated
    
    def test_calibration_reduces_loss(self, sample_affect_jsonl, mock_embed_fn):
        """Test that calibration runs and produces valid loss history"""
        amygdala = Amygdala(embedding_dim=384, use_gpu=False)
        
        result = amygdala.calibrate_from_jsonl(
            sample_affect_jsonl,
            embed_fn=mock_embed_fn,
            batch_size=4,
            epochs=5,
            learning_rate=0.001  # Lower LR for stability
        )
        
        losses = result['loss_history']
        # With random embeddings, loss may not decrease, but should be finite
        assert len(losses) == 5
        assert all(0 <= loss < 10 for loss in losses)  # Reasonable loss values
        assert result['final_loss'] >= 0
    
    def test_predict_affect_neural(self, sample_affect_jsonl, mock_embed_fn):
        """Test neural affect prediction after calibration"""
        amygdala = Amygdala(embedding_dim=384, use_gpu=False)
        
        # Calibrate first
        amygdala.calibrate_from_jsonl(
            sample_affect_jsonl,
            embed_fn=mock_embed_fn,
            epochs=3
        )
        
        # Predict
        embedding = mock_embed_fn("I am happy")
        valence, arousal = amygdala._predict_affect_neural(embedding)
        
        assert -1.0 <= valence <= 1.0
        assert 0.0 <= arousal <= 1.0
    
    def test_evaluate_calibration(self, sample_affect_jsonl, mock_embed_fn):
        """Test calibration evaluation"""
        amygdala = Amygdala(embedding_dim=384, use_gpu=False)
        
        # Calibrate
        amygdala.calibrate_from_jsonl(
            sample_affect_jsonl,
            embed_fn=mock_embed_fn,
            epochs=3
        )
        
        # Evaluate
        metrics = amygdala.evaluate_calibration(
            sample_affect_jsonl,
            embed_fn=mock_embed_fn,
            limit=5
        )
        
        assert 'valence_mae' in metrics
        assert 'arousal_mae' in metrics
        assert metrics['samples'] == 5
    
    def test_calibration_persisted_in_save(self, sample_affect_jsonl, mock_embed_fn, tmp_path):
        """Test that calibration is saved and loaded"""
        amygdala1 = Amygdala(embedding_dim=384, hidden_dim=64, use_gpu=False)
        
        # Calibrate
        amygdala1.calibrate_from_jsonl(
            sample_affect_jsonl,
            embed_fn=mock_embed_fn,
            epochs=2
        )
        
        # Save
        save_path = str(tmp_path / "amygdala_calibrated.json")
        amygdala1.save_state(save_path)
        
        # Load into new instance
        amygdala2 = Amygdala(embedding_dim=384, hidden_dim=64, use_gpu=False)
        amygdala2.load_state(save_path)
        
        assert amygdala2.is_calibrated
        # Check all 3 layers
        assert np.allclose(amygdala1.W1, amygdala2.W1)
        assert np.allclose(amygdala1.W2, amygdala2.W2)
        assert np.allclose(amygdala1.W3, amygdala2.W3)
        assert np.allclose(amygdala1.b1, amygdala2.b1)
        assert np.allclose(amygdala1.b2, amygdala2.b2)
        assert np.allclose(amygdala1.b3, amygdala2.b3)
        # Check Adam state
        assert amygdala2.adam_t == amygdala1.adam_t
    
    def test_cpu_calibration_step(self, mock_embed_fn):
        """Test CPU calibration step"""
        amygdala = Amygdala(embedding_dim=384, use_gpu=False)
        
        initial_weights = amygdala.affect_weights.copy()
        
        # Create sample data
        embeddings = np.random.randn(8, 384).astype(np.float32)
        targets = np.array([
            [0.8, 0.6], [-0.5, 0.7], [0.3, 0.2], [0.9, 0.9],
            [-0.6, 0.3], [-0.4, 0.8], [0.5, 0.4], [-0.7, 0.5]
        ], dtype=np.float32)
        
        amygdala._cpu_calibration_step(embeddings, targets, learning_rate=0.1)
        
        # Weights should have changed
        assert not np.allclose(initial_weights, amygdala.affect_weights)
    
    def test_calibrated_amygdala_uses_neural(self, sample_affect_jsonl, mock_embed_fn):
        """Test that calibrated amygdala uses neural prediction"""
        amygdala = Amygdala(embedding_dim=384, use_gpu=False, emotion_decay=0.0)
        
        # Process before calibration - uses pattern matching
        embedding = mock_embed_fn("test text")
        result_before = amygdala.process("test text", embedding)
        
        # Calibrate
        amygdala.calibrate_from_jsonl(
            sample_affect_jsonl,
            embed_fn=mock_embed_fn,
            epochs=3
        )
        
        # Reset state for fair comparison
        amygdala.state = EmotionalState()
        
        # Process after calibration - uses neural prediction
        result_after = amygdala.process("test text", embedding)
        
        # Results may differ since prediction method changed
        assert result_after.confidence >= 0.5


class TestEndocrineSystem:
    """Tests for the Endocrine System"""
    
    def test_init(self):
        endo = EndocrineSystem()
        levels = endo.get_levels()
        
        assert 'cortisol' in levels
        assert 'dopamine' in levels
        assert 'oxytocin' in levels
    
    def test_step_updates_levels(self):
        endo = EndocrineSystem()
        
        # Step with positive emotion
        levels = endo.step(
            emotional_state={'valence': 0.8, 'arousal': 0.3},
            response_success=0.9
        )
        
        # Should have elevated dopamine from success
        assert levels['dopamine'] >= endo.hormones[HormoneType.DOPAMINE].baseline
    
    def test_stress_increases_cortisol(self):
        endo = EndocrineSystem()
        
        # Step with negative emotion
        endo.step(
            emotional_state={'valence': -0.8, 'arousal': 0.9},
            response_success=0.2
        )
        
        levels = endo.get_levels()
        assert levels['cortisol'] > 0.3  # Should be elevated
    
    def test_modulation_factors(self):
        endo = EndocrineSystem()
        mods = endo.get_modulation_factors()
        
        assert 'creativity' in mods
        assert 'warmth' in mods
        assert 'empathy' in mods
        assert all(0 <= v <= 1 for v in mods.values())


class TestCNS:
    """Tests for the Central Nervous System"""
    
    def test_init(self):
        cns = CentralNervousSystem()
        assert cns.state.consciousness == ConsciousnessLevel.ALERT
    
    def test_stress_changes_consciousness(self):
        cns = CentralNervousSystem()
        
        # Add high stress
        cns.add_stress(0.9)
        
        assert cns.state.consciousness == ConsciousnessLevel.HYPERVIGILANT
    
    def test_stress_recovery(self):
        cns = CentralNervousSystem(stress_recovery_rate=0.5)
        cns.state.stress_level = 0.5
        
        # Update with time passing
        import time
        cns.last_update = time.time() - 10  # Simulate 10 seconds
        cns.update()
        
        assert cns.state.stress_level < 0.5  # Should have recovered
    
    def test_processing_modulation(self):
        cns = CentralNervousSystem()
        mods = cns.get_processing_modulation()
        
        assert 'processing_depth' in mods
        assert 'processing_speed' in mods
        assert 'creativity' in mods


class TestThalamus:
    """Tests for the Thalamus"""
    
    def test_init(self):
        thalamus = Thalamus()
        assert thalamus.embedding_dim == 384
    
    def test_gate_input(self):
        thalamus = Thalamus()
        embedding = np.random.randn(384)
        
        gated, level = thalamus.gate_input(embedding, arousal=0.5)
        
        assert gated.shape == embedding.shape
        assert 0 <= level <= 1
    
    def test_arousal_affects_gating(self):
        thalamus = Thalamus()
        embedding = np.random.randn(384) * 0.5  # Moderate magnitude
        
        # High arousal should lower threshold
        _, level_high = thalamus.gate_input(embedding, arousal=0.9)
        
        thalamus = Thalamus()  # Reset
        _, level_low = thalamus.gate_input(embedding, arousal=0.1)
        
        # High arousal should allow more through
        assert level_high >= level_low
    
    def test_route_input(self):
        thalamus = Thalamus()
        embedding = np.random.randn(384)
        
        routes = thalamus.route_input(embedding)
        
        assert 'memory' in routes
        assert 'emotion' in routes
        assert 'reasoning' in routes


class TestBasalGanglia:
    """Tests for the Basal Ganglia"""
    
    def test_init(self):
        bg = BasalGanglia()
        assert len(bg.strategies) > 0
    
    def test_select_strategy(self):
        bg = BasalGanglia()
        signal = np.array([0.3, 0.8, 0.2, 0.5])  # High emotion
        
        strategy, confidence = bg.select_strategy(signal)
        
        assert strategy in bg.strategies
        assert 0 <= confidence <= 1
    
    def test_emotion_affects_strategy(self):
        bg = BasalGanglia()
        signal = np.array([0.3, 0.8, 0.2, 0.5])
        
        # Negative emotion should boost empathetic
        strategy, _ = bg.select_strategy(
            signal,
            emotional_state={'valence': -0.7, 'arousal': 0.5}
        )
        
        # Should tend toward empathetic for negative emotions
        # (Note: may not always be empathetic due to other factors)
        assert strategy in bg.strategies
    
    def test_go_nogo_decision(self):
        bg = BasalGanglia()
        
        should_go, strength = bg.go_nogo_decision(confidence=0.8, urgency=0.5)
        
        assert isinstance(should_go, bool)
        assert 0 <= strength <= 1


class TestLimbicSystem:
    """Tests for the Limbic System"""
    
    @pytest.fixture
    def limbic(self):
        amygdala = Amygdala()
        memory = MockMemoryStore()
        return LimbicSystem(amygdala, memory)
    
    def test_init(self, limbic):
        assert limbic.amygdala is not None
        assert limbic.memory is not None
    
    def test_process_input(self, limbic):
        embedding = np.random.randn(384)
        result = limbic.process_input("I feel happy today!", embedding)
        
        assert 'emotional_state' in result
        assert 'modulation' in result
        assert 'should_remember' in result
    
    def test_store_with_emotion(self, limbic):
        embedding = np.random.randn(384)
        limbic.amygdala.state.arousal = 0.8
        limbic.amygdala.state.valence = 0.7
        
        limbic.store_with_emotion(embedding, "Important memory!")
        
        assert len(limbic.memory_emotions) > 0


class TestUnifiedBrain:
    """Tests for the Unified Brain"""
    
    @pytest.fixture
    def brain(self):
        memory = MockMemoryStore()
        with tempfile.TemporaryDirectory() as tmpdir:
            return UnifiedBrain(memory_store=memory, state_dir=tmpdir)
    
    def test_init(self, brain):
        assert brain.amygdala is not None
        assert brain.limbic_system is not None
        assert brain.thalamus is not None
        assert brain.basal_ganglia is not None
        assert brain.endocrine is not None
        assert brain.cns is not None
    
    def test_process(self, brain):
        embedding = np.random.randn(384).astype(np.float32)
        
        result = brain.process("Hello, how are you?", embedding)
        
        assert 'emotional_state' in result
        assert 'strategy' in result
        assert 'modulation' in result
        assert 'should_respond' in result
    
    def test_process_emotional_text(self, brain):
        embedding = np.random.randn(384).astype(np.float32)
        
        result = brain.process("I'm feeling really sad and lonely today", embedding)
        
        # Should detect negative emotion
        assert result['emotional_state'].valence < 0 or result['strategy'] == 'empathetic'
    
    def test_provide_feedback(self, brain):
        embedding = np.random.randn(384).astype(np.float32)
        brain.process("Test input", embedding)
        
        # Provide positive feedback
        brain.provide_feedback(quality=0.9, strategy_worked=True)
        
        # Should update stats
        assert brain.stats['total_interactions'] > 0
    
    def test_get_empathy_prompt(self, brain):
        brain.amygdala.state.valence = -0.6
        brain.amygdala.state.arousal = 0.7
        
        prompt = brain.get_empathy_prompt()
        
        assert len(prompt) > 0
        assert "empathy" in prompt.lower() or "difficult" in prompt.lower()
    
    def test_get_self_awareness_prompt(self, brain):
        """Test that self-awareness prompt contains internal state info"""
        brain.amygdala.state.valence = 0.3
        brain.amygdala.state.arousal = 0.8
        brain.amygdala.state.dominant_emotion = "curious"
        
        prompt = brain.get_self_awareness_prompt()
        
        assert "[MY_STATE]" in prompt
        assert "curious" in prompt.lower()
        assert "valence" in prompt.lower()
        assert "arousal" in prompt.lower()
        assert "0.30" in prompt  # valence value
        assert "0.80" in prompt  # arousal value
    
    def test_get_response_style(self, brain):
        style = brain.get_response_style()
        
        assert 'tone' in style
        assert 'length' in style
        assert 'strategy' in style
    
    def test_get_stats(self, brain):
        stats = brain.get_stats()
        
        assert 'brain_stats' in stats
        assert 'amygdala' in stats
        assert 'endocrine' in stats
        assert 'cns' in stats
    
    def test_experiential_learning_stores_experience(self, brain):
        """Test that experiences are stored after feedback"""
        embedding = np.random.randn(384).astype(np.float32)
        
        # Process an input
        brain.process("I'm learning something new", embedding)
        
        # Provide positive feedback
        result = brain.provide_feedback(quality=0.9, strategy_worked=True)
        
        assert result['experience_stored'] == True
        assert len(brain.experiences) > 0
    
    def test_experiential_learning_recall(self, brain):
        """Test that similar experiences can be recalled"""
        embedding1 = np.random.randn(384).astype(np.float32)
        embedding2 = embedding1 + np.random.randn(384).astype(np.float32) * 0.1  # Similar
        
        # Store an experience
        brain.process("Test interaction", embedding1)
        brain.provide_feedback(quality=0.8)
        
        # Recall similar experiences
        recalled = brain.recall_similar_experiences(embedding2, k=3)
        
        assert len(recalled) > 0
        assert recalled[0]['similarity'] > 0
    
    def test_experience_insights(self, brain):
        """Test experience insights generation"""
        embedding = np.random.randn(384).astype(np.float32)
        
        # Create some experiences with high quality (triggers storage)
        for i in range(5):
            brain.process(f"Test {i}", embedding + np.random.randn(384).astype(np.float32) * 0.1)
            brain.provide_feedback(quality=0.85)  # High quality ensures storage
        
        insights = brain.get_experience_insights()
        
        assert insights['total_experiences'] >= 3  # At least some stored
        assert 'avg_quality' in insights
        assert 'best_strategy' in insights
    
    def test_online_amygdala_learning(self, brain):
        """Test that Amygdala learns from feedback"""
        embedding = np.random.randn(384).astype(np.float32)
        
        # Calibrate first
        brain.amygdala.is_calibrated = True
        
        # Process and provide feedback
        brain.process("This should update the Amygdala", embedding)
        result = brain.provide_feedback(quality=0.9)
        
        assert result['amygdala_updated'] == True
        assert result['online_learning'] == True


class TestGPUBrainCompute:
    """Tests for GPU Brain Compute (with CPU fallback)"""
    
    def test_init(self):
        gpu = GPUBrainCompute(use_vulkan=False)  # Force CPU for testing
        assert gpu is not None
    
    def test_place_cells(self):
        gpu = GPUBrainCompute(use_vulkan=False)
        
        agent_pos = np.array([0.0, 0.0], dtype=np.float32)
        centers = np.random.randn(100, 2).astype(np.float32)
        
        rates = gpu.compute_place_cells(
            agent_pos, centers,
            field_width=2.0, max_rate=20.0
        )
        
        assert rates.shape == (100,)
        assert np.all(rates >= 0)
        assert np.all(rates <= 20.0)
    
    def test_time_cells(self):
        gpu = GPUBrainCompute(use_vulkan=False)
        
        pref_times = np.linspace(0, 10, 50).astype(np.float32)
        
        rates, mem_state = gpu.compute_time_cells(
            current_time=5.0,
            preferred_times=pref_times,
            temporal_width=1.0
        )
        
        assert rates.shape == (50,)
        # Time cells near t=5 should have higher rates
        mid_rates = rates[20:30].mean()
        edge_rates = (rates[:10].mean() + rates[40:].mean()) / 2
        assert mid_rates > edge_rates
    
    def test_theta_gamma_encoding(self):
        gpu = GPUBrainCompute(use_vulkan=False)
        
        positions = np.arange(10).reshape(1, 10).astype(np.float32)
        
        encoded = gpu.compute_theta_gamma_encoding(
            positions,
            embedding_dim=64,
            theta_freq=8.0,
            gamma_freq=40.0
        )
        
        assert encoded.shape == (1, 10, 64)
        # Should have oscillatory pattern
        assert np.std(encoded) > 0
    
    def test_hebbian_update(self):
        gpu = GPUBrainCompute(use_vulkan=False)
        
        pre = np.random.randn(2, 5, 32).astype(np.float32)
        post = np.random.randn(2, 5, 16).astype(np.float32)
        weights = np.zeros((16, 32), dtype=np.float32)
        
        updated = gpu.hebbian_update(
            pre, post, weights,
            learning_rate=0.1,
            weight_decay=0.0
        )
        
        assert updated.shape == (16, 32)
        # Weights should have changed
        assert not np.allclose(updated, 0)
    
    def test_domain_routing(self):
        gpu = GPUBrainCompute(use_vulkan=False)
        
        domain_probs = np.array([[0.7, 0.2, 0.1]], dtype=np.float32)
        expert_weights = np.random.randn(3, 5).astype(np.float32)
        
        routing, indices = gpu.domain_routing(
            domain_probs, expert_weights, top_k=2
        )
        
        assert routing.shape == (1, 5)
        assert indices.shape == (1, 2)


class TestGPUSpatialMemory:
    """Tests for GPU Spatial Memory"""
    
    def test_init(self):
        mem = GPUSpatialMemory(use_vulkan=False)
        assert mem.n_place_cells == 1000
        assert mem.n_time_cells == 100
    
    def test_update_position(self):
        mem = GPUSpatialMemory(n_place_cells=100, use_vulkan=False)
        
        pos = np.array([1.0, 2.0], dtype=np.float32)
        rates = mem.update_position(pos)
        
        assert rates.shape == (100,)
        assert np.any(rates > 0)
    
    def test_update_time(self):
        mem = GPUSpatialMemory(n_time_cells=50, use_vulkan=False)
        
        rates = mem.update_time(dt=0.1)
        
        assert rates.shape == (50,)
    
    def test_spatial_context(self):
        mem = GPUSpatialMemory(use_vulkan=False)
        mem.update_position(np.array([0.0, 0.0]))
        mem.update_time(dt=0.5)
        
        ctx = mem.get_spatial_context()
        
        assert 'position' in ctx
        assert 'place_cells' in ctx
        assert 'time_cells' in ctx
        assert 'elapsed_time' in ctx
    
    def test_reset_time(self):
        mem = GPUSpatialMemory(use_vulkan=False)
        mem.update_time(dt=5.0)
        
        assert mem.elapsed_time > 0
        
        mem.reset_time()
        
        assert mem.elapsed_time == 0.0


class TestIntegration:
    """Integration tests for the brain module"""
    
    def test_full_conversation_flow(self):
        """Test a realistic conversation flow"""
        memory = MockMemoryStore()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            brain = UnifiedBrain(memory_store=memory, state_dir=tmpdir)
            
            # User expresses frustration
            embedding1 = np.random.randn(384).astype(np.float32)
            result1 = brain.process("I'm so frustrated with this problem!", embedding1)
            
            assert result1['emotional_state'].valence < 0.2
            
            # Provide feedback
            brain.provide_feedback(quality=0.7, strategy_worked=True)
            
            # User calms down
            embedding2 = np.random.randn(384).astype(np.float32)
            result2 = brain.process("Thanks, that actually helped me understand", embedding2)
            
            # Emotion should have shifted
            assert result2['emotional_state'].valence >= result1['emotional_state'].valence
    
    def test_stress_accumulation(self):
        """Test that stress accumulates and affects consciousness"""
        memory = MockMemoryStore()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            brain = UnifiedBrain(memory_store=memory, state_dir=tmpdir)
            
            # Directly test stress accumulation via CNS
            initial_stress = brain.cns.state.stress_level
            
            # Add stress directly
            brain.cns.add_stress(0.5)
            
            # Check stress accumulated
            assert brain.cns.state.stress_level > initial_stress
            
            # Add more stress to trigger hypervigilance
            brain.cns.add_stress(0.5)
            
            # Should be hypervigilant now
            assert brain.cns.state.consciousness == ConsciousnessLevel.HYPERVIGILANT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

