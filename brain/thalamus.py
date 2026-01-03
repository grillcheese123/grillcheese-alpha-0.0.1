"""
Thalamus - Sensory Gateway and Attention Router

The thalamus acts as a relay station that:
- Gates sensory input based on relevance/salience
- Routes information to appropriate processing systems
- Modulates attention based on emotional state
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class Thalamus:
    """
    Thalamus - Sensory gating and attention routing
    
    Functions:
    1. Input gating: Filter irrelevant information
    2. Attention routing: Direct input to appropriate processors
    3. Arousal modulation: Adjust processing based on arousal level
    4. Salience detection: Identify important information
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        num_routes: int = 4,
        base_gate_threshold: float = 0.3
    ):
        """
        Initialize thalamus
        
        Args:
            embedding_dim: Dimension of input embeddings
            num_routes: Number of processing routes (brain regions)
            base_gate_threshold: Base threshold for gating
        """
        self.embedding_dim = embedding_dim
        self.num_routes = num_routes
        self.base_gate_threshold = base_gate_threshold
        
        # Route names (mapping to brain regions)
        self.route_names = [
            'memory',      # Hippocampal/memory processing
            'emotion',     # Amygdala/limbic processing
            'reasoning',   # Prefrontal/executive processing
            'response'     # Language/output processing
        ]
        
        # Learned route biases (initialized neutral)
        self.route_biases = np.zeros(num_routes)
        
        # Gating state
        self.current_gate_level = 0.5
        self.arousal_modulation = 1.0
        
        # Statistics
        self.stats = {
            'inputs_gated': 0,
            'inputs_passed': 0,
            'route_activations': {name: 0 for name in self.route_names}
        }
        
        logger.info("Thalamus initialized")
    
    def gate_input(
        self,
        embedding: np.ndarray,
        arousal: float = 0.5,
        valence: float = 0.0
    ) -> Tuple[np.ndarray, float]:
        """
        Gate input based on salience and arousal
        
        Args:
            embedding: Input embedding
            arousal: Current arousal level (0-1)
            valence: Current valence (-1 to 1)
            
        Returns:
            Tuple of (gated_embedding, gate_level)
        """
        # Compute salience from embedding magnitude
        magnitude = np.linalg.norm(embedding)
        salience = np.tanh(magnitude / 10.0)  # Normalize
        
        # Adjust threshold based on arousal
        # Higher arousal = lower threshold (more sensitive)
        threshold = self.base_gate_threshold * (1.0 - arousal * 0.3)
        
        # Compute gate level
        gate_level = self._compute_gate(salience, arousal, threshold)
        
        self.current_gate_level = gate_level
        
        # Apply gate to embedding
        gated_embedding = embedding * gate_level
        
        # Track statistics
        if gate_level > 0.5:
            self.stats['inputs_passed'] += 1
        else:
            self.stats['inputs_gated'] += 1
        
        return gated_embedding, gate_level
    
    def _compute_gate(
        self,
        salience: float,
        arousal: float,
        threshold: float
    ) -> float:
        """Compute gate value using soft thresholding"""
        # Sigmoid-like gate
        gate = 1.0 / (1.0 + np.exp(-(salience - threshold) * 5.0))
        
        # Modulate by arousal (higher arousal = wider gate)
        gate = gate * (0.7 + arousal * 0.3)
        
        return float(np.clip(gate, 0.0, 1.0))
    
    def route_input(
        self,
        embedding: np.ndarray,
        emotional_state: Optional[Dict[str, float]] = None,
        context_type: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Route input to processing systems
        
        Args:
            embedding: Gated embedding
            emotional_state: Current emotional state
            context_type: Hint about input type
            
        Returns:
            Dictionary mapping route names to activation weights
        """
        activations = {}
        
        # Base routing from embedding patterns
        # Different embedding patterns suggest different routes
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Simple heuristic routing based on embedding statistics
        mean_val = np.mean(embedding_norm)
        std_val = np.std(embedding_norm)
        max_val = np.max(np.abs(embedding_norm))
        
        # Memory route: Strong when pattern is distinctive
        activations['memory'] = float(np.clip(std_val * 2, 0, 1))
        
        # Emotion route: Strong when emotional state present
        if emotional_state:
            arousal = emotional_state.get('arousal', 0.5)
            valence_mag = abs(emotional_state.get('valence', 0))
            activations['emotion'] = float(np.clip(arousal + valence_mag, 0, 1))
        else:
            activations['emotion'] = 0.3
        
        # Reasoning route: Strong for complex patterns
        activations['reasoning'] = float(np.clip(max_val * 1.5, 0, 1))
        
        # Response route: Always active
        activations['response'] = 0.8
        
        # Apply learned biases
        for i, name in enumerate(self.route_names[:len(self.route_biases)]):
            if name in activations:
                activations[name] = float(np.clip(
                    activations[name] + self.route_biases[i],
                    0, 1
                ))
        
        # Track activations
        for name, weight in activations.items():
            if weight > 0.5:
                self.stats['route_activations'][name] = self.stats['route_activations'].get(name, 0) + 1
        
        return activations
    
    def modulate_by_arousal(self, arousal: float) -> None:
        """
        Adjust thalamic gating based on arousal level
        
        High arousal: Lower thresholds, more passes through
        Low arousal: Higher thresholds, more filtering
        """
        self.arousal_modulation = 0.5 + arousal * 0.5
        
        # Adjust base threshold
        # High arousal = lower threshold (more sensitive)
        self.base_gate_threshold = 0.3 * (1.0 - arousal * 0.4)
    
    def process(
        self,
        embedding: np.ndarray,
        emotional_state: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Full thalamic processing pipeline
        
        Args:
            embedding: Input embedding
            emotional_state: Current emotional state
            
        Returns:
            Dictionary with gated embedding and routing weights
        """
        arousal = emotional_state.get('arousal', 0.5) if emotional_state else 0.5
        valence = emotional_state.get('valence', 0.0) if emotional_state else 0.0
        
        # Gate input
        gated_embedding, gate_level = self.gate_input(embedding, arousal, valence)
        
        # Route to processing systems
        routes = self.route_input(gated_embedding, emotional_state)
        
        return {
            'gated_embedding': gated_embedding,
            'gate_level': gate_level,
            'routes': routes,
            'arousal_modulation': self.arousal_modulation
        }
    
    def learn_routing(
        self,
        route_name: str,
        reward: float
    ) -> None:
        """
        Learn routing preferences from feedback
        
        Args:
            route_name: Route that was used
            reward: Reward signal (-1 to 1)
        """
        if route_name in self.route_names:
            idx = self.route_names.index(route_name)
            # Simple bias update
            self.route_biases[idx] += reward * 0.01
            # Clamp biases
            self.route_biases = np.clip(self.route_biases, -0.3, 0.3)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get thalamus statistics"""
        total = self.stats['inputs_gated'] + self.stats['inputs_passed']
        pass_rate = self.stats['inputs_passed'] / max(1, total)
        
        return {
            **self.stats,
            'pass_rate': pass_rate,
            'current_gate_level': self.current_gate_level,
            'arousal_modulation': self.arousal_modulation,
            'route_biases': self.route_biases.tolist()
        }

