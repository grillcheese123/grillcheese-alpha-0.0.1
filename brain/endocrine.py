"""
Endocrine System - Hormonal Homeostasis

The endocrine system maintains homeostatic balance through hormones that
regulate the brain's response patterns:

- Cortisol: Stress response (increases caution, reduces creativity)
- Dopamine: Reward/motivation (reinforces successful patterns)
- Norepinephrine: Arousal/alertness (modulates attention)
- Serotonin: Mood stability (smooths emotional responses)
- Oxytocin: Social bonding (increases empathy, warmth)
"""
import json
import logging
import time
import numpy as np
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class HormoneType(Enum):
    """Types of hormones in the endocrine system"""
    CORTISOL = "cortisol"           # Stress response
    DOPAMINE = "dopamine"           # Reward/motivation
    NOREPINEPHRINE = "norepinephrine"  # Arousal/alertness
    SEROTONIN = "serotonin"         # Mood stability
    OXYTOCIN = "oxytocin"           # Social bonding/empathy


@dataclass
class Hormone:
    """Individual hormone with concentration and dynamics"""
    type: HormoneType
    concentration: float = 0.5      # Current level (0-1 normalized)
    baseline: float = 0.5           # Resting level
    half_life: float = 300.0        # Decay half-life in seconds
    max_concentration: float = 1.0
    min_concentration: float = 0.0
    
    def release(self, amount: float) -> float:
        """Release hormone (increase concentration)"""
        self.concentration = min(
            self.max_concentration,
            self.concentration + amount
        )
        return self.concentration
    
    def decay(self, dt: float) -> float:
        """Apply time-based decay toward baseline"""
        decay_factor = np.exp(-dt / self.half_life)
        self.concentration = (
            self.baseline + 
            (self.concentration - self.baseline) * decay_factor
        )
        return self.concentration
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': self.type.value,
            'concentration': self.concentration,
            'baseline': self.baseline
        }


@dataclass 
class SystemMetrics:
    """
    System-level metrics that drive hormone release
    """
    prediction_accuracy: float = 0.5   # How well responses are received
    response_coherence: float = 0.5    # Quality of generated responses
    user_engagement: float = 0.5       # User interaction quality
    stress_level: float = 0.3          # Overall stress
    social_connection: float = 0.5     # Quality of social interaction
    
    def update(
        self,
        accuracy: Optional[float] = None,
        coherence: Optional[float] = None,
        engagement: Optional[float] = None,
        stress: Optional[float] = None,
        connection: Optional[float] = None
    ):
        """Update metrics with exponential moving average"""
        alpha = 0.8  # Smoothing factor
        
        if accuracy is not None:
            self.prediction_accuracy = alpha * self.prediction_accuracy + (1-alpha) * accuracy
        if coherence is not None:
            self.response_coherence = alpha * self.response_coherence + (1-alpha) * coherence
        if engagement is not None:
            self.user_engagement = alpha * self.user_engagement + (1-alpha) * engagement
        if stress is not None:
            self.stress_level = alpha * self.stress_level + (1-alpha) * stress
        if connection is not None:
            self.social_connection = alpha * self.social_connection + (1-alpha) * connection


class EndocrineSystem:
    """
    Endocrine System - Master controller for hormonal homeostasis
    
    Monitors system metrics and releases hormones to maintain optimal
    functioning. Hormones modulate:
    - Response generation (creativity vs caution)
    - Memory formation (which memories to strengthen)
    - Attention routing (what to focus on)
    - Emotional expression (warmth, energy)
    """
    
    def __init__(self):
        """Initialize endocrine system with default hormone levels"""
        self.metrics = SystemMetrics()
        self.last_update_time = time.time()
        
        # Initialize hormones with different baselines
        self.hormones: Dict[HormoneType, Hormone] = {
            HormoneType.CORTISOL: Hormone(
                type=HormoneType.CORTISOL,
                concentration=0.3,
                baseline=0.3,
                half_life=600.0  # Cortisol decays slowly
            ),
            HormoneType.DOPAMINE: Hormone(
                type=HormoneType.DOPAMINE,
                concentration=0.5,
                baseline=0.5,
                half_life=120.0  # Dopamine decays quickly
            ),
            HormoneType.NOREPINEPHRINE: Hormone(
                type=HormoneType.NOREPINEPHRINE,
                concentration=0.4,
                baseline=0.4,
                half_life=60.0   # Very fast decay
            ),
            HormoneType.SEROTONIN: Hormone(
                type=HormoneType.SEROTONIN,
                concentration=0.6,
                baseline=0.6,
                half_life=1800.0  # Very slow decay
            ),
            HormoneType.OXYTOCIN: Hormone(
                type=HormoneType.OXYTOCIN,
                concentration=0.5,
                baseline=0.5,
                half_life=300.0
            )
        }
        
        # Targets for homeostasis
        self.target_stress = 0.3
        self.target_engagement = 0.7
        
        logger.info("Endocrine system initialized")
    
    def step(
        self,
        emotional_state: Optional[Dict[str, float]] = None,
        interaction_quality: Optional[float] = None,
        response_success: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Update endocrine system and release hormones based on current state
        
        Args:
            emotional_state: From amygdala (valence, arousal)
            interaction_quality: Quality of recent interaction (0-1)
            response_success: Success of recent response (0-1)
            
        Returns:
            Dictionary of current hormone levels
        """
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # 1. Apply decay to all hormones
        for hormone in self.hormones.values():
            hormone.decay(dt)
        
        # 2. Update metrics from inputs
        if response_success is not None:
            self.metrics.update(accuracy=response_success)
        if interaction_quality is not None:
            self.metrics.update(engagement=interaction_quality)
        
        # 3. Compute hormone releases based on state
        releases = self._compute_releases(emotional_state or {})
        
        # 4. Apply releases
        for hormone_type, amount in releases.items():
            if hormone_type in self.hormones:
                self.hormones[hormone_type].release(amount)
        
        # 5. Return current levels
        return self.get_levels()
    
    def _compute_releases(self, emotional_state: Dict[str, float]) -> Dict[HormoneType, float]:
        """Compute hormone releases based on current state"""
        releases = {h: 0.0 for h in HormoneType}
        
        valence = emotional_state.get('valence', 0.0)
        arousal = emotional_state.get('arousal', 0.5)
        
        # Cortisol: Release on stress/negative emotions
        if valence < -0.3 or arousal > 0.7:
            stress = max(-valence, arousal - 0.5)
            releases[HormoneType.CORTISOL] = stress * 0.1
            self.metrics.update(stress=stress)
        
        # Dopamine: Release on success/positive feedback
        if self.metrics.prediction_accuracy > 0.6:
            reward = self.metrics.prediction_accuracy - 0.5
            releases[HormoneType.DOPAMINE] = reward * 0.15
        
        # Norepinephrine: Release on high arousal
        if arousal > 0.6:
            releases[HormoneType.NOREPINEPHRINE] = (arousal - 0.5) * 0.2
        
        # Serotonin: Release on stable positive state
        if valence > 0.2 and arousal < 0.6:
            releases[HormoneType.SEROTONIN] = valence * 0.05
        
        # Oxytocin: Release on positive social interaction
        if self.metrics.social_connection > 0.5 and valence > 0:
            releases[HormoneType.OXYTOCIN] = self.metrics.social_connection * 0.1
        
        return releases
    
    def get_levels(self) -> Dict[str, float]:
        """Get current hormone levels as dictionary"""
        return {
            h.value: self.hormones[h].concentration 
            for h in HormoneType
        }
    
    def get_modulation_factors(self) -> Dict[str, float]:
        """
        Get factors for modulating system behavior
        
        Returns modulation factors for:
        - creativity: Higher dopamine, lower cortisol = more creative
        - caution: Higher cortisol = more careful
        - warmth: Higher oxytocin = warmer responses
        - energy: Higher norepinephrine = more energetic
        - stability: Higher serotonin = more stable
        """
        cortisol = self.hormones[HormoneType.CORTISOL].concentration
        dopamine = self.hormones[HormoneType.DOPAMINE].concentration
        norepinephrine = self.hormones[HormoneType.NOREPINEPHRINE].concentration
        serotonin = self.hormones[HormoneType.SEROTONIN].concentration
        oxytocin = self.hormones[HormoneType.OXYTOCIN].concentration
        
        return {
            'creativity': np.clip(dopamine - cortisol * 0.5 + 0.3, 0, 1),
            'caution': np.clip(cortisol * 1.5, 0, 1),
            'warmth': np.clip(oxytocin * 1.2 + serotonin * 0.3, 0, 1),
            'energy': np.clip(norepinephrine * 1.3 + dopamine * 0.2, 0, 1),
            'stability': np.clip(serotonin * 1.2 - cortisol * 0.3, 0, 1),
            'focus': np.clip(norepinephrine * 0.8 + cortisol * 0.2, 0, 1),
            'empathy': np.clip(oxytocin * 1.5, 0, 1)
        }
    
    def update_social_connection(self, quality: float) -> None:
        """Update social connection metric from interaction"""
        self.metrics.update(connection=quality)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get endocrine system statistics"""
        return {
            'levels': self.get_levels(),
            'modulation': self.get_modulation_factors(),
            'metrics': {
                'accuracy': self.metrics.prediction_accuracy,
                'coherence': self.metrics.response_coherence,
                'engagement': self.metrics.user_engagement,
                'stress': self.metrics.stress_level,
                'connection': self.metrics.social_connection
            }
        }
    
    def save_state(self, path: str) -> None:
        """Save endocrine state"""
        state = {
            'hormones': {h.value: self.hormones[h].to_dict() for h in HormoneType},
            'metrics': {
                'accuracy': self.metrics.prediction_accuracy,
                'coherence': self.metrics.response_coherence,
                'engagement': self.metrics.user_engagement,
                'stress': self.metrics.stress_level,
                'connection': self.metrics.social_connection
            }
        }
        with open(path, 'w') as f:
            json.dump(state, f)
        logger.info(f"Endocrine state saved to {path}")
    
    def load_state(self, path: str) -> None:
        """Load endocrine state"""
        if not Path(path).exists():
            return
        
        with open(path, 'r') as f:
            state = json.load(f)
        
        for h_name, h_data in state.get('hormones', {}).items():
            h_type = HormoneType(h_name)
            if h_type in self.hormones:
                self.hormones[h_type].concentration = h_data['concentration']
                self.hormones[h_type].baseline = h_data['baseline']
        
        metrics = state.get('metrics', {})
        self.metrics.prediction_accuracy = metrics.get('accuracy', 0.5)
        self.metrics.response_coherence = metrics.get('coherence', 0.5)
        self.metrics.user_engagement = metrics.get('engagement', 0.5)
        self.metrics.stress_level = metrics.get('stress', 0.3)
        self.metrics.social_connection = metrics.get('connection', 0.5)
        
        logger.info(f"Endocrine state loaded from {path}")

