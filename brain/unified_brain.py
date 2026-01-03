"""
Unified Brain - Orchestrates all brain components

The UnifiedBrain integrates all bio-inspired components into a cohesive
cognitive architecture that processes inputs through:

1. Thalamus (sensory gating)
2. Amygdala (emotional processing)
3. Limbic System (memory-emotion integration)
4. Central Nervous System (consciousness/stress)
5. Endocrine System (hormonal homeostasis)
6. Basal Ganglia (action selection)

This creates an AI with genuine emotional intelligence and adaptive behavior.
"""
import json
import logging
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from .amygdala import Amygdala, EmotionalState
from .limbic_system import LimbicSystem
from .thalamus import Thalamus
from .basal_ganglia import BasalGanglia
from .endocrine import EndocrineSystem
from .cns import CentralNervousSystem, ConsciousnessLevel
from .gpu_brain import GPUBrainCompute, GPUSpatialMemory

logger = logging.getLogger(__name__)

# Default affect training data path
DEFAULT_AFFECT_DATA = Path(__file__).parent.parent.parent.parent / "data_learning" / "jsonl" / "amygdala_affect.jsonl"


class UnifiedBrain:
    """
    UnifiedBrain - Complete bio-inspired cognitive architecture
    
    Orchestrates all brain components to create an emotionally intelligent,
    adaptive, and empathetic AI system.
    
    Usage:
        brain = UnifiedBrain(memory_store=memory, embedding_dim=384)
        result = brain.process(text, embedding)
        # result contains emotional state, selected strategy, modulations, etc.
    """
    
    def __init__(
        self,
        memory_store,
        embedding_dim: int = 384,
        state_dir: Optional[str] = None,
        use_gpu: bool = True
    ):
        """
        Initialize the unified brain
        
        Args:
            memory_store: MemoryStore instance (hippocampal function)
            embedding_dim: Dimension of embeddings
            state_dir: Directory for persisting brain state
            use_gpu: Whether to use GPU acceleration for brain computations
        """
        self.embedding_dim = embedding_dim
        self.state_dir = Path(state_dir) if state_dir else Path("brain_state")
        self.state_dir.mkdir(exist_ok=True)
        self.use_gpu = use_gpu
        
        # Initialize brain components
        logger.info("Initializing unified brain architecture...")
        
        # 0. GPU Brain Compute - Accelerated neural computations
        self.gpu_brain = GPUBrainCompute(use_vulkan=use_gpu)
        
        # 0b. GPU Spatial Memory - Place cells and time cells
        self.spatial_memory = GPUSpatialMemory(
            n_place_cells=1000,
            n_time_cells=100,
            spatial_dims=2,  # 2D semantic space
            use_vulkan=use_gpu
        )
        
        # 1. Amygdala - Emotional processing
        self.amygdala = Amygdala(
            embedding_dim=embedding_dim,
            emotion_decay=0.85,
            sensitivity=1.2
        )
        
        # 2. Limbic System - Memory-emotion integration
        self.limbic_system = LimbicSystem(
            amygdala=self.amygdala,
            memory_store=memory_store,
            embedding_dim=embedding_dim
        )
        
        # 3. Thalamus - Sensory gating and routing
        self.thalamus = Thalamus(
            embedding_dim=embedding_dim,
            num_routes=4,
            base_gate_threshold=0.3
        )
        
        # 4. Basal Ganglia - Action selection
        self.basal_ganglia = BasalGanglia(
            num_regions=4,
            selection_temperature=1.0
        )
        
        # 5. Endocrine System - Hormonal homeostasis
        self.endocrine = EndocrineSystem()
        
        # 6. Central Nervous System - Consciousness/stress
        self.cns = CentralNervousSystem(
            stress_recovery_rate=0.02,
            fatigue_rate=0.005
        )
        
        # Hebbian weights for learning associations (GPU-accelerated)
        # Shape: [post_dim, pre_dim] = [embedding_dim, 64]
        # Pre = emotional signal (64), Post = embedding (embedding_dim)
        self._hebbian_weights = np.random.randn(embedding_dim, 64).astype(np.float32) * 0.01
        
        # Brain-wide state
        self.last_process_time = time.time()
        self.interaction_count = 0
        
        # Statistics
        self.stats = {
            'total_interactions': 0,
            'positive_interactions': 0,
            'negative_interactions': 0,
            'empathetic_responses': 0,
            'informative_responses': 0,
            'gpu_operations': 0,
            'experiences_learned': 0,
            'online_learning_updates': 0
        }
        
        # === EXPERIENTIAL LEARNING ===
        # Store significant experiences for learning
        self.experiences = []  # List of experience dicts
        self.experience_embeddings = np.zeros((0, embedding_dim), dtype=np.float32)
        self.max_experiences = 1000  # Limit for memory efficiency
        
        # Current interaction buffer (for learning from feedback)
        self._current_interaction = None
        
        # Load persisted state if available
        self._load_state()
        
        gpu_status = "GPU" if self.gpu_brain.use_vulkan else "CPU"
        logger.info(f"[OK] Unified brain initialized ({gpu_status} mode)")
    
    def process(
        self,
        text: str,
        embedding: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process input through the complete brain architecture
        
        This is the main entry point that coordinates all brain components.
        Uses GPU-accelerated computations where available.
        
        Args:
            text: Input text from user
            embedding: Text embedding vector
            context: Optional additional context
            
        Returns:
            Comprehensive processing result including:
            - emotional_state: Current emotional assessment
            - strategy: Selected response strategy
            - modulations: Factors for response generation
            - memory_context: Retrieved relevant memories
            - should_respond: Go/no-go decision
            - spatial_context: Place/time cell activations
        """
        self.interaction_count += 1
        self.stats['total_interactions'] += 1
        
        # Get current emotional state for modulation
        current_emotion = self.amygdala.get_state()
        emotional_dict = current_emotion.to_dict()
        
        # 0. GPU SPATIAL MEMORY: Update place/time cells
        # Map embedding to 2D semantic space position using PCA-like projection
        semantic_position = self._embedding_to_spatial(embedding)
        place_activations = self.spatial_memory.update_position(semantic_position)
        
        # Update time cells (track elapsed time in conversation)
        dt = time.time() - self.last_process_time
        self.last_process_time = time.time()
        time_activations = self.spatial_memory.update_time(min(dt, 1.0))
        
        self.stats['gpu_operations'] += 2
        
        # 1. THALAMUS: Gate and route input
        thalamic_result = self.thalamus.process(
            embedding,
            emotional_state=emotional_dict
        )
        gated_embedding = thalamic_result['gated_embedding']
        routes = thalamic_result['routes']
        
        # 2. LIMBIC SYSTEM: Process emotion and retrieve memories
        limbic_result = self.limbic_system.process_input(text, embedding)
        emotional_state = limbic_result['emotional_state']
        memory_context = limbic_result['memory_context']
        
        # 2b. EXPERIENTIAL RECALL: Inform processing with past experiences
        similar_experiences = self.recall_similar_experiences(embedding, k=3, min_quality=0.5)
        if similar_experiences:
            # Use past successful experiences to inform current processing
            avg_past_quality = np.mean([e.get('quality', 0.5) for e in similar_experiences])
            best_past_strategy = max(similar_experiences, key=lambda e: e.get('quality', 0))
            
            # Modulate arousal based on familiarity (familiar = calmer)
            familiarity = similar_experiences[0].get('similarity', 0)
            if familiarity > 0.8:
                emotional_state.arousal *= 0.9  # Slightly calmer for familiar situations
        
        # 3. CNS: Update consciousness and stress
        cns_state = self.cns.update(
            arousal=emotional_state.arousal,
            valence=emotional_state.valence,
            interaction_occurred=True
        )
        
        # 4. ENDOCRINE: Update hormonal state
        hormone_levels = self.endocrine.step(
            emotional_state=emotional_state.to_dict(),
            interaction_quality=limbic_result['modulation'].get('response_warmth', 0.5)
        )
        
        # 5. BASAL GANGLIA: Select strategy and make go/no-go decision
        urgency = 0.5 + emotional_state.arousal * 0.3  # Higher arousal = more urgent
        inhibition = max(0, -emotional_state.valence) * 0.3  # Negative = more cautious
        
        bg_result = self.basal_ganglia.process(
            region_activations=routes,
            emotional_state=emotional_state.to_dict(),
            urgency=urgency,
            inhibition=inhibition
        )
        
        # 6. GPU HEBBIAN LEARNING: Update association weights
        if self.interaction_count % 5 == 0:  # Every 5 interactions
            self._update_hebbian_weights(embedding, emotional_state)
            self.stats['gpu_operations'] += 1
        
        # 7. Compile modulation factors from all systems
        amygdala_mod = self.amygdala.get_emotional_modulation()
        endocrine_mod = self.endocrine.get_modulation_factors()
        cns_mod = self.cns.get_processing_modulation()
        strategy_mod = self.basal_ganglia.get_strategy_modulation()
        
        # Combine modulations
        combined_modulation = self._combine_modulations(
            amygdala_mod, endocrine_mod, cns_mod, strategy_mod
        )
        
        # Boost modulation based on place cell activity (familiar contexts)
        place_activity = float(np.mean(place_activations))
        if place_activity > 5.0:  # Strong place field activation
            combined_modulation['stability'] *= 1.2
            combined_modulation['warmth'] *= 1.1
        
        # 8. Store memory if emotionally salient
        if limbic_result['should_remember'] > 0.5:
            self.limbic_system.store_with_emotion(embedding, text, emotional_state)
        
        # Track statistics
        if emotional_state.valence > 0.3:
            self.stats['positive_interactions'] += 1
        elif emotional_state.valence < -0.3:
            self.stats['negative_interactions'] += 1
        
        if bg_result['strategy'] == 'empathetic':
            self.stats['empathetic_responses'] += 1
        elif bg_result['strategy'] == 'informative':
            self.stats['informative_responses'] += 1
        
        # Store current interaction for experiential learning
        self._current_interaction = {
            'text': text,
            'embedding': embedding.copy(),
            'emotional_state': emotional_state.to_dict(),
            'strategy': bg_result['strategy'],
            'modulation': combined_modulation.copy(),
            'timestamp': time.time()
        }
        
        return {
            # Core outputs
            'emotional_state': emotional_state,
            'strategy': bg_result['strategy'],
            'confidence': bg_result['confidence'],
            'should_respond': bg_result['should_respond'],
            'response_strength': bg_result['response_strength'],
            
            # Context
            'memory_context': memory_context,
            'memories': limbic_result['memories'],
            
            # Modulations for response generation
            'modulation': combined_modulation,
            
            # Detailed state (for debugging/visualization)
            'consciousness_level': cns_state.consciousness.name,
            'stress_level': cns_state.stress_level,
            'hormone_levels': hormone_levels,
            'gate_level': thalamic_result['gate_level'],
            'routes': routes,
            
            # GPU-computed spatial context
            'spatial_context': {
                'place_activity': place_activity,
                'time_activity': float(np.mean(time_activations)),
                'semantic_position': semantic_position.tolist()
            }
        }
    
    def _embedding_to_spatial(self, embedding: np.ndarray) -> np.ndarray:
        """Map embedding to 2D semantic space for place cells"""
        # Simple projection: use first 2 PCA-like dimensions
        # In practice, this maps high-dimensional semantics to 2D space
        norm = np.linalg.norm(embedding) + 1e-8
        normalized = embedding / norm
        
        # Project to 2D using weighted sum of embedding dimensions
        x = np.sum(normalized[:len(normalized)//2])
        y = np.sum(normalized[len(normalized)//2:])
        
        # Scale to place field range
        return np.array([x * 5.0, y * 5.0], dtype=np.float32)
    
    def _update_hebbian_weights(self, embedding: np.ndarray, emotional_state: EmotionalState) -> None:
        """Update Hebbian weights using GPU acceleration"""
        # Create pre/post activations with correct shapes
        # Pre = emotional signal (64 dims)
        emotional_signal = np.array([
            emotional_state.arousal,
            emotional_state.valence,
            emotional_state.confidence
        ] + [0.0] * 61, dtype=np.float32)  # Pad to 64
        pre = emotional_signal.reshape(1, 1, 64)  # [1, 1, 64]
        
        # Post = embedding (embedding_dim)
        post = embedding.reshape(1, 1, self.embedding_dim)  # [1, 1, embedding_dim]
        
        # GPU Hebbian update
        # Weights shape: [post_dim, pre_dim] = [embedding_dim, 64]
        # But we initialized as [64, embedding_dim], so we need to transpose or fix initialization
        # For now, ensure weights match: [post_dim, pre_dim] = [embedding_dim, 64]
        if self._hebbian_weights.shape != (self.embedding_dim, 64):
            # Reinitialize if shape mismatch
            self._hebbian_weights = np.random.randn(self.embedding_dim, 64).astype(np.float32) * 0.01
        
        self._hebbian_weights = self.gpu_brain.hebbian_update(
            pre_activations=pre,
            post_activations=post,
            weights=self._hebbian_weights,
            learning_rate=0.001,
            weight_decay=0.0001
        )
    
    def _combine_modulations(
        self,
        amygdala_mod: Dict[str, float],
        endocrine_mod: Dict[str, float],
        cns_mod: Dict[str, float],
        strategy_mod: Dict[str, float]
    ) -> Dict[str, float]:
        """Combine modulations from different brain systems"""
        return {
            # Warmth: from amygdala, endocrine (oxytocin), and strategy
            'warmth': (
                amygdala_mod.get('response_warmth', 0.5) * 0.3 +
                endocrine_mod.get('warmth', 0.5) * 0.4 +
                strategy_mod.get('warmth', 0.5) * 0.3
            ),
            
            # Energy: from amygdala, endocrine, and CNS
            'energy': (
                amygdala_mod.get('response_energy', 0.5) * 0.3 +
                endocrine_mod.get('energy', 0.5) * 0.3 +
                cns_mod.get('processing_speed', 0.5) * 0.4
            ),
            
            # Empathy: from amygdala boost and endocrine
            'empathy': (
                amygdala_mod.get('empathy_boost', 0.0) * 0.4 +
                endocrine_mod.get('empathy', 0.5) * 0.6
            ),
            
            # Creativity: from endocrine and CNS
            'creativity': (
                endocrine_mod.get('creativity', 0.5) * 0.5 +
                cns_mod.get('creativity', 0.5) * 0.5
            ),
            
            # Caution: from amygdala and endocrine
            'caution': (
                amygdala_mod.get('response_caution', 0.0) * 0.4 +
                endocrine_mod.get('caution', 0.3) * 0.6
            ),
            
            # Detail level: from CNS and strategy
            'detail_level': (
                cns_mod.get('processing_depth', 0.5) * 0.5 +
                strategy_mod.get('detail_level', 0.5) * 0.5
            ),
            
            # Focus: from endocrine and CNS
            'focus': (
                endocrine_mod.get('focus', 0.5) * 0.5 +
                cns_mod.get('vigilance', 0.5) * 0.5
            ),
            
            # Stability: from endocrine
            'stability': endocrine_mod.get('stability', 0.5)
        }
    
    def provide_feedback(
        self,
        quality: float,
        strategy_worked: bool = True,
        user_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Provide feedback after a response to enable learning
        
        Args:
            quality: Quality score of interaction (0-1)
            strategy_worked: Whether the selected strategy was appropriate
            user_response: Optional user's response text for deeper learning
            
        Returns:
            Learning result with updates made
        """
        learning_result = {
            'experience_stored': False,
            'amygdala_updated': False,
            'strategy_learned': False,
            'online_learning': False
        }
        
        # Update endocrine based on feedback
        self.endocrine.step(
            response_success=quality,
            interaction_quality=quality
        )
        
        # Update CNS stress based on feedback
        if quality < 0.4:
            self.cns.add_stress(0.1)
        elif quality > 0.7:
            self.cns.reduce_stress(0.05)
        
        # Learn strategy preferences
        current_strategy = self.basal_ganglia.current_strategy
        reward = (quality - 0.5) * 2  # Map to -1 to 1
        if not strategy_worked:
            reward *= -0.5
        self.basal_ganglia.learn_from_feedback(current_strategy, reward)
        learning_result['strategy_learned'] = True
        
        # Update social connection
        self.endocrine.update_social_connection(quality)
        
        # === EXPERIENTIAL LEARNING ===
        if self._current_interaction is not None:
            # Store experience if significant
            experience = self._current_interaction.copy()
            experience['quality'] = quality
            experience['strategy_worked'] = strategy_worked
            experience['user_response'] = user_response
            
            # Calculate significance (how memorable is this experience?)
            emotional_intensity = abs(experience['emotional_state']['valence']) + experience['emotional_state']['arousal']
            outcome_surprise = abs(quality - 0.5) * 2  # How different from expected?
            significance = emotional_intensity * 0.5 + outcome_surprise * 0.5
            
            # Store if significant enough
            if significance > 0.3 or quality > 0.8 or quality < 0.3:
                self._store_experience(experience)
                learning_result['experience_stored'] = True
            
            # Online Amygdala learning - adjust emotional predictions
            if self.amygdala.is_calibrated:
                self._online_amygdala_learning(experience, quality)
                learning_result['amygdala_updated'] = True
                learning_result['online_learning'] = True
                self.stats['online_learning_updates'] += 1
            
            self._current_interaction = None
        
        return learning_result
    
    def _store_experience(self, experience: Dict[str, Any]) -> None:
        """Store an experience for future learning and recall"""
        # Add to experiences
        self.experiences.append(experience)
        
        # Add embedding to experience embeddings
        emb = experience['embedding'].reshape(1, -1)
        self.experience_embeddings = np.vstack([self.experience_embeddings, emb])
        
        # Trim if over limit
        if len(self.experiences) > self.max_experiences:
            # Remove oldest, least significant experiences
            self.experiences = self.experiences[-self.max_experiences:]
            self.experience_embeddings = self.experience_embeddings[-self.max_experiences:]
        
        self.stats['experiences_learned'] += 1
        logger.debug(f"Stored experience: {experience.get('text', '')[:50]}... (total: {len(self.experiences)})")
    
    def _online_amygdala_learning(self, experience: Dict[str, Any], quality: float) -> None:
        """
        Online learning for the Amygdala based on real interaction feedback.
        Adjusts affect predictions based on what actually worked.
        """
        embedding = experience['embedding']
        predicted_state = experience['emotional_state']
        
        # Determine target affect based on outcome
        # If high quality + strategy worked -> current emotional approach was good
        # If low quality -> adjust emotional understanding
        
        if quality > 0.7:
            # Reinforce current emotional interpretation
            target_valence = predicted_state['valence']
            target_arousal = predicted_state['arousal']
        elif quality < 0.4:
            # Emotional interpretation may have been off
            # Nudge towards more neutral/appropriate response
            target_valence = predicted_state['valence'] * 0.7  # Dampen extreme predictions
            target_arousal = max(0.4, min(0.7, predicted_state['arousal']))  # Moderate arousal
        else:
            # Moderate quality - small adjustment
            target_valence = predicted_state['valence'] * 0.9
            target_arousal = predicted_state['arousal']
        
        # Create targets array (valence, arousal)
        targets = np.array([[target_valence, target_arousal]], dtype=np.float32)
        
        # Perform small gradient update on Amygdala's MLP
        # This is a simplified online learning step
        self.amygdala._cpu_calibration_step(
            embeddings=embedding.reshape(1, -1),
            targets=targets,
            learning_rate=0.0001  # Very small learning rate for online updates
        )
    
    def recall_similar_experiences(
        self,
        embedding: np.ndarray,
        k: int = 3,
        min_quality: Optional[float] = None
    ) -> list:
        """
        Recall similar past experiences to inform current processing
        
        Args:
            embedding: Current input embedding
            k: Number of experiences to recall
            min_quality: Optional minimum quality filter
            
        Returns:
            List of relevant past experiences
        """
        if len(self.experiences) == 0:
            return []
        
        # Compute similarities
        embedding = embedding.reshape(1, -1)
        similarities = np.dot(self.experience_embeddings, embedding.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Filter and return
        results = []
        for idx in top_indices:
            exp = self.experiences[idx]
            if min_quality is not None and exp.get('quality', 0) < min_quality:
                continue
            results.append({
                **exp,
                'similarity': float(similarities[idx]),
                'embedding': None  # Don't include large embeddings in recall
            })
        
        return results
    
    def get_experience_insights(self) -> Dict[str, Any]:
        """Get insights from accumulated experiences"""
        if len(self.experiences) == 0:
            return {'total_experiences': 0, 'insights': []}
        
        # Analyze experiences
        qualities = [e.get('quality', 0.5) for e in self.experiences]
        strategies = [e.get('strategy', 'unknown') for e in self.experiences]
        
        # Strategy success rates
        strategy_stats = {}
        for exp in self.experiences:
            strat = exp.get('strategy', 'unknown')
            if strat not in strategy_stats:
                strategy_stats[strat] = {'count': 0, 'total_quality': 0}
            strategy_stats[strat]['count'] += 1
            strategy_stats[strat]['total_quality'] += exp.get('quality', 0.5)
        
        for strat in strategy_stats:
            count = strategy_stats[strat]['count']
            strategy_stats[strat]['avg_quality'] = strategy_stats[strat]['total_quality'] / count
        
        # Best strategy
        best_strategy = max(strategy_stats.items(), key=lambda x: x[1].get('avg_quality', 0))[0]
        
        return {
            'total_experiences': len(self.experiences),
            'avg_quality': float(np.mean(qualities)),
            'best_strategy': best_strategy,
            'strategy_stats': strategy_stats,
            'recent_trend': float(np.mean(qualities[-10:])) if len(qualities) >= 10 else float(np.mean(qualities))
        }
    
    def get_empathy_prompt(self) -> str:
        """
        Get an empathy-aware prompt prefix based on current emotional state
        """
        state = self.amygdala.get_state()
        modulation = self.endocrine.get_modulation_factors()
        
        prompt_parts = []
        
        # Add emotional awareness
        if state.valence < -0.4:
            prompt_parts.append("The user seems to be experiencing difficult emotions.")
            prompt_parts.append("Respond with empathy and understanding.")
        elif state.valence > 0.5 and state.arousal > 0.5:
            prompt_parts.append("The user is expressing positive energy.")
            prompt_parts.append("Match their enthusiasm while being helpful.")
        elif state.arousal > 0.7:
            prompt_parts.append("The user seems stressed or urgent.")
            prompt_parts.append("Be clear, direct, and reassuring.")
        
        # Add modulation hints
        if modulation.get('empathy', 0) > 0.7:
            prompt_parts.append("Prioritize emotional support over information.")
        
        if modulation.get('warmth', 0) > 0.7:
            prompt_parts.append("Use a warm, caring tone.")
        
        if self.cns.get_state().consciousness == ConsciousnessLevel.FOCUSED:
            prompt_parts.append("Provide detailed, thorough responses.")
        
        return " ".join(prompt_parts) if prompt_parts else ""
    
    def get_self_awareness_prompt(self) -> str:
        """
        Get a self-awareness prompt describing GrillCheese's current internal state.
        This allows the LLM to accurately report its own emotional/cognitive state.
        """
        emo_state = self.amygdala.get_state()
        cns_state = self.cns.get_state()
        modulation = self.endocrine.get_modulation_factors()
        strategy = self.basal_ganglia.current_strategy
        
        # Describe valence
        if emo_state.valence > 0.5:
            valence_desc = "positive and upbeat"
        elif emo_state.valence > 0.2:
            valence_desc = "mildly positive"
        elif emo_state.valence > -0.2:
            valence_desc = "neutral"
        elif emo_state.valence > -0.5:
            valence_desc = "mildly negative"
        else:
            valence_desc = "quite negative"
        
        # Describe arousal
        if emo_state.arousal > 0.7:
            arousal_desc = "highly alert and energized"
        elif emo_state.arousal > 0.5:
            arousal_desc = "alert and engaged"
        elif emo_state.arousal > 0.3:
            arousal_desc = "calm and attentive"
        else:
            arousal_desc = "very calm and relaxed"
        
        # Get experiential insights
        insights = self.get_experience_insights()
        experience_text = ""
        if insights['total_experiences'] > 0:
            experience_text = f"""
My learning from experience:
- I have learned from {insights['total_experiences']} past interactions
- My average interaction quality: {insights['avg_quality']:.2f}
- My most successful strategy: {insights['best_strategy']}
- Recent performance trend: {insights['recent_trend']:.2f}
- I continuously learn and improve from every conversation"""
        
        # Build minimal state prompt - emotions influence tone, not explicit mentions
        state_prompt = f"""[MY_STATE]
Current state: {emo_state.dominant_emotion} ({valence_desc}, {arousal_desc})
Strategy: {strategy}
Consciousness: {cns_state.consciousness.name}

Note: Let your emotional state influence your tone and empathy naturally. Only explicitly discuss your feelings if the user asks about them.
[/MY_STATE]"""
        
        return state_prompt
    
    def get_response_style(self) -> Dict[str, Any]:
        """
        Get recommended response style based on brain state
        """
        state = self.amygdala.get_state()
        strategy = self.basal_ganglia.current_strategy
        modulation = self._get_current_modulation()
        
        return {
            'tone': self._determine_tone(state, modulation),
            'length': self._determine_length(strategy, modulation),
            'formality': self._determine_formality(modulation),
            'strategy': strategy,
            'emotion_aware': True,
            'empathy_level': modulation.get('empathy', 0.5)
        }
    
    def _determine_tone(
        self,
        state: EmotionalState,
        modulation: Dict[str, float]
    ) -> str:
        """Determine appropriate tone"""
        warmth = modulation.get('warmth', 0.5)
        energy = modulation.get('energy', 0.5)
        
        if warmth > 0.7 and energy < 0.4:
            return 'gentle'
        elif warmth > 0.7 and energy > 0.6:
            return 'enthusiastic'
        elif warmth < 0.4 and energy > 0.6:
            return 'direct'
        elif warmth < 0.4:
            return 'neutral'
        else:
            return 'friendly'
    
    def _determine_length(
        self,
        strategy: str,
        modulation: Dict[str, float]
    ) -> str:
        """Determine appropriate response length"""
        detail = modulation.get('detail_level', 0.5)
        
        if strategy == 'empathetic':
            return 'medium'
        elif strategy == 'informative' and detail > 0.6:
            return 'long'
        elif strategy == 'questioning':
            return 'short'
        else:
            return 'medium'
    
    def _determine_formality(self, modulation: Dict[str, float]) -> str:
        """Determine appropriate formality level"""
        warmth = modulation.get('warmth', 0.5)
        
        if warmth > 0.7:
            return 'casual'
        elif warmth < 0.3:
            return 'formal'
        else:
            return 'balanced'
    
    def _get_current_modulation(self) -> Dict[str, float]:
        """Get current combined modulation"""
        amygdala_mod = self.amygdala.get_emotional_modulation()
        endocrine_mod = self.endocrine.get_modulation_factors()
        cns_mod = self.cns.get_processing_modulation()
        strategy_mod = self.basal_ganglia.get_strategy_modulation()
        
        return self._combine_modulations(
            amygdala_mod, endocrine_mod, cns_mod, strategy_mod
        )
    
    def consolidate(self) -> Dict[str, Any]:
        """
        Perform periodic consolidation (call during idle periods)
        
        - Consolidate emotional memories
        - Update baselines
        - Decay temporary states
        """
        # Consolidate limbic memories
        limbic_result = self.limbic_system.consolidate_memories()
        
        # Save state
        self._save_state()
        
        return {
            'limbic_consolidation': limbic_result,
            'state_saved': True
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive brain statistics"""
        return {
            'brain_stats': self.stats,
            'amygdala': self.amygdala.get_stats(),
            'limbic': self.limbic_system.get_stats(),
            'thalamus': self.thalamus.get_stats(),
            'basal_ganglia': self.basal_ganglia.get_stats(),
            'endocrine': self.endocrine.get_stats(),
            'cns': self.cns.get_stats(),
            'gpu_brain': self.gpu_brain.get_stats(),
            'spatial_memory': self.spatial_memory.get_spatial_context(),
            'experiential_learning': self.get_experience_insights(),
            'interaction_count': self.interaction_count
        }
    
    def _save_state(self) -> None:
        """Save brain state to disk"""
        try:
            self.amygdala.save_state(str(self.state_dir / "amygdala.json"))
            self.endocrine.save_state(str(self.state_dir / "endocrine.json"))
            self.cns.save_state(str(self.state_dir / "cns.json"))
            
            # Save brain-wide stats
            with open(self.state_dir / "brain_stats.json", 'w') as f:
                json.dump(self.stats, f)
            
            # Save experiences (without embeddings to save space)
            experiences_to_save = []
            for exp in self.experiences[-500:]:  # Save last 500
                exp_copy = {k: v for k, v in exp.items() if k != 'embedding'}
                experiences_to_save.append(exp_copy)
            
            with open(self.state_dir / "experiences.json", 'w') as f:
                json.dump(experiences_to_save, f)
            
            # Save experience embeddings as numpy
            if len(self.experience_embeddings) > 0:
                np.save(self.state_dir / "experience_embeddings.npy", self.experience_embeddings[-500:])
            
            logger.debug(f"Brain state saved ({len(self.experiences)} experiences)")
        except Exception as e:
            logger.error(f"Failed to save brain state: {e}")
    
    def _load_state(self) -> None:
        """Load brain state from disk"""
        try:
            self.amygdala.load_state(str(self.state_dir / "amygdala.json"))
            self.endocrine.load_state(str(self.state_dir / "endocrine.json"))
            self.cns.load_state(str(self.state_dir / "cns.json"))
            
            stats_path = self.state_dir / "brain_stats.json"
            if stats_path.exists():
                with open(stats_path, 'r') as f:
                    loaded_stats = json.load(f)
                    # Merge with default stats to handle new fields
                    self.stats.update(loaded_stats)
            
            # Load experiences
            exp_path = self.state_dir / "experiences.json"
            emb_path = self.state_dir / "experience_embeddings.npy"
            if exp_path.exists() and emb_path.exists():
                with open(exp_path, 'r') as f:
                    self.experiences = json.load(f)
                self.experience_embeddings = np.load(emb_path)
                logger.info(f"Loaded {len(self.experiences)} experiences")
            
            logger.info("Brain state loaded")
        except Exception as e:
            logger.debug(f"Could not load brain state: {e}")
    
    def is_amygdala_calibrated(self) -> bool:
        """Check if the Amygdala's affect prediction is calibrated"""
        return self.amygdala.is_calibrated
    
    def calibrate_affect(
        self,
        embed_fn,
        data_path: Optional[Path] = None,
        epochs: int = 20,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Calibrate the Amygdala's affect prediction MLP
        
        This trains the neural network to predict valence/arousal from embeddings,
        enabling accurate emotional understanding.
        
        Args:
            embed_fn: Function to generate embeddings (e.g., phi3.get_embedding)
            data_path: Path to amygdala_affect.jsonl (default: data_learning/jsonl)
            epochs: Training epochs
            learning_rate: Adam learning rate
            batch_size: Batch size for training
            limit: Max training samples
            
        Returns:
            Training result with loss history
        """
        if data_path is None:
            data_path = DEFAULT_AFFECT_DATA
        
        if not data_path.exists():
            logger.warning(f"Affect training data not found: {data_path}")
            return {'error': 'Training data not found', 'path': str(data_path)}
        
        logger.info(f"Calibrating Amygdala affect prediction...")
        logger.info(f"  Data: {data_path}")
        logger.info(f"  Epochs: {epochs}, LR: {learning_rate}, Batch: {batch_size}")
        
        result = self.amygdala.calibrate_from_jsonl(
            filepath=data_path,
            embed_fn=embed_fn,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            limit=limit
        )
        
        # Save calibrated state
        self._save_state()
        
        logger.info(f"[OK] Amygdala calibrated (loss: {result.get('final_loss', 'N/A'):.4f})")
        return result
    
    def get_affect_prediction(self, embedding: np.ndarray) -> Dict[str, float]:
        """
        Get affect prediction for an embedding
        
        Returns:
            Dict with 'valence' (-1 to 1) and 'arousal' (0 to 1)
        """
        valence, arousal = self.amygdala._predict_affect_neural(embedding)
        return {
            'valence': valence,
            'arousal': arousal,
            'calibrated': self.amygdala.is_calibrated
        }

