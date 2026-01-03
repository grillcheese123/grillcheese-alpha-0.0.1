"""
Basal Ganglia - Action Selection and Response Gating

The basal ganglia integrates signals from multiple brain regions and
selects appropriate responses through:
- Competitive selection (winner-take-all)
- Response gating (go/no-go)
- Action value learning
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


class BasalGanglia:
    """
    Basal Ganglia - Action selection and output integration
    
    Functions:
    1. Integrate inputs from cortical regions
    2. Select appropriate response strategy
    3. Gate response output (go/no-go)
    4. Learn action values
    
    Implements simplified direct/indirect pathway model
    """
    
    def __init__(
        self,
        num_regions: int = 4,
        selection_temperature: float = 1.0
    ):
        """
        Initialize basal ganglia
        
        Args:
            num_regions: Number of input regions to integrate
            selection_temperature: Temperature for softmax selection
        """
        self.num_regions = num_regions
        self.selection_temperature = selection_temperature
        
        # Region names (inputs from thalamic routing)
        self.region_names = ['memory', 'emotion', 'reasoning', 'response']
        
        # Learned region weights (importance of each region)
        self.region_weights = np.ones(num_regions) / num_regions
        
        # Response strategies
        self.strategies = [
            'informative',    # Provide information
            'empathetic',     # Emotional support
            'questioning',    # Ask clarifying questions
            'action',         # Suggest actions
            'default'         # Standard response
        ]
        self.strategy_biases = np.zeros(len(self.strategies))
        
        # Go/no-go threshold
        self.go_threshold = 0.4
        
        # Current state
        self.current_strategy = 'default'
        self.current_confidence = 0.5
        
        # Statistics
        self.stats = {
            'selections_made': 0,
            'go_decisions': 0,
            'nogo_decisions': 0,
            'strategy_counts': {s: 0 for s in self.strategies}
        }
        
        logger.info("Basal ganglia initialized")
    
    def integrate_inputs(
        self,
        region_activations: Dict[str, float],
        embeddings: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Integrate inputs from multiple brain regions
        
        Args:
            region_activations: Activation weights from thalamic routing
            embeddings: Optional embeddings from each region
            
        Returns:
            Integrated signal vector
        """
        # Weighted combination of region activations
        integrated = np.zeros(self.num_regions)
        
        for i, name in enumerate(self.region_names):
            if name in region_activations:
                activation = region_activations[name]
                weight = self.region_weights[i]
                integrated[i] = activation * weight
        
        # Normalize
        total = np.sum(integrated)
        if total > 0:
            integrated = integrated / total
        
        return integrated
    
    def select_strategy(
        self,
        integrated_signal: np.ndarray,
        emotional_state: Optional[Dict[str, float]] = None
    ) -> Tuple[str, float]:
        """
        Select response strategy based on integrated signal
        
        Args:
            integrated_signal: Integrated input from regions
            emotional_state: Current emotional state
            
        Returns:
            Tuple of (strategy_name, confidence)
        """
        # Compute strategy scores
        scores = np.zeros(len(self.strategies))
        
        # Memory-heavy -> informative
        scores[0] = integrated_signal[0] * 1.5  # informative
        
        # Emotion-heavy -> empathetic
        scores[1] = integrated_signal[1] * 2.0  # empathetic
        
        # Reasoning-heavy -> questioning or action
        scores[2] = integrated_signal[2] * 1.0  # questioning
        scores[3] = integrated_signal[2] * 0.8  # action
        
        # Base score for default
        scores[4] = 0.3  # default
        
        # Apply emotional modulation
        if emotional_state:
            valence = emotional_state.get('valence', 0)
            arousal = emotional_state.get('arousal', 0.5)
            
            # Negative valence boosts empathetic
            if valence < -0.3:
                scores[1] *= 1.5
            
            # High arousal boosts action-oriented
            if arousal > 0.6:
                scores[3] *= 1.3
        
        # Apply learned biases
        scores += self.strategy_biases
        
        # Softmax selection
        scores_exp = np.exp(scores / self.selection_temperature)
        probabilities = scores_exp / np.sum(scores_exp)
        
        # Select strategy (could be probabilistic or argmax)
        selected_idx = np.argmax(probabilities)
        selected_strategy = self.strategies[selected_idx]
        confidence = probabilities[selected_idx]
        
        # Update state
        self.current_strategy = selected_strategy
        self.current_confidence = confidence
        
        # Track statistics
        self.stats['selections_made'] += 1
        self.stats['strategy_counts'][selected_strategy] += 1
        
        return selected_strategy, float(confidence)
    
    def go_nogo_decision(
        self,
        confidence: float,
        urgency: float = 0.5,
        inhibition: float = 0.0
    ) -> Tuple[bool, float]:
        """
        Make go/no-go decision for response
        
        Args:
            confidence: Confidence in selected response
            urgency: How urgent the response is (0-1)
            inhibition: Inhibition signal (0-1)
            
        Returns:
            Tuple of (should_respond, response_strength)
        """
        # Direct pathway (GO) - promotes action
        direct_signal = confidence * (0.5 + urgency * 0.5)
        
        # Indirect pathway (NO-GO) - inhibits action
        indirect_signal = inhibition + (1.0 - confidence) * 0.3
        
        # Net signal
        net_signal = direct_signal - indirect_signal
        
        # Decision
        should_go = net_signal > self.go_threshold
        response_strength = float(np.clip((net_signal + 1) / 2, 0, 1))
        
        # Track statistics
        if should_go:
            self.stats['go_decisions'] += 1
        else:
            self.stats['nogo_decisions'] += 1
        
        return should_go, response_strength
    
    def process(
        self,
        region_activations: Dict[str, float],
        emotional_state: Optional[Dict[str, float]] = None,
        urgency: float = 0.5,
        inhibition: float = 0.0
    ) -> Dict[str, Any]:
        """
        Full basal ganglia processing pipeline
        
        Args:
            region_activations: From thalamic routing
            emotional_state: Current emotional state
            urgency: Response urgency
            inhibition: Inhibition level
            
        Returns:
            Processing result with strategy and go/no-go decision
        """
        # Integrate inputs
        integrated = self.integrate_inputs(region_activations)
        
        # Select strategy
        strategy, confidence = self.select_strategy(integrated, emotional_state)
        
        # Go/no-go decision
        should_respond, strength = self.go_nogo_decision(confidence, urgency, inhibition)
        
        return {
            'integrated_signal': integrated.tolist(),
            'strategy': strategy,
            'confidence': confidence,
            'should_respond': should_respond,
            'response_strength': strength
        }
    
    def learn_from_feedback(
        self,
        strategy: str,
        reward: float
    ) -> None:
        """
        Learn from response feedback
        
        Args:
            strategy: Strategy that was used
            reward: Reward signal (-1 to 1)
        """
        if strategy in self.strategies:
            idx = self.strategies.index(strategy)
            # Simple bias update
            self.strategy_biases[idx] += reward * 0.05
            # Clamp biases
            self.strategy_biases = np.clip(self.strategy_biases, -0.5, 0.5)
    
    def learn_region_importance(
        self,
        region_name: str,
        importance_delta: float
    ) -> None:
        """
        Update region importance weights
        
        Args:
            region_name: Name of region
            importance_delta: Change in importance
        """
        if region_name in self.region_names:
            idx = self.region_names.index(region_name)
            self.region_weights[idx] += importance_delta * 0.1
            # Renormalize
            self.region_weights = np.clip(self.region_weights, 0.1, 1.0)
            self.region_weights = self.region_weights / np.sum(self.region_weights)
    
    def get_strategy_modulation(self) -> Dict[str, float]:
        """
        Get modulation factors based on current strategy
        """
        strategy_modulations = {
            'informative': {
                'detail_level': 0.8,
                'warmth': 0.5,
                'questioning': 0.3
            },
            'empathetic': {
                'detail_level': 0.4,
                'warmth': 0.9,
                'questioning': 0.2
            },
            'questioning': {
                'detail_level': 0.5,
                'warmth': 0.6,
                'questioning': 0.9
            },
            'action': {
                'detail_level': 0.6,
                'warmth': 0.5,
                'questioning': 0.4
            },
            'default': {
                'detail_level': 0.5,
                'warmth': 0.6,
                'questioning': 0.3
            }
        }
        
        return strategy_modulations.get(self.current_strategy, strategy_modulations['default'])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basal ganglia statistics"""
        total_decisions = self.stats['go_decisions'] + self.stats['nogo_decisions']
        go_rate = self.stats['go_decisions'] / max(1, total_decisions)
        
        return {
            **self.stats,
            'go_rate': go_rate,
            'current_strategy': self.current_strategy,
            'current_confidence': self.current_confidence,
            'region_weights': self.region_weights.tolist(),
            'strategy_biases': self.strategy_biases.tolist()
        }

