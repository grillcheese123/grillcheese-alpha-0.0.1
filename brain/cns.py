"""
Central Nervous System (CNS) - Consciousness and State Management

The CNS manages overall system state including:
- Consciousness levels (alert, focused, drowsy)
- Stress response and recovery
- Global arousal modulation
- State transitions
"""
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConsciousnessLevel(Enum):
    """
    Levels of consciousness affecting processing mode
    
    Based on biological consciousness states:
    - DEEP_SLEEP: Minimal processing, memory consolidation only
    - DROWSY: Reduced attention, slow responses
    - ALERT: Normal operation
    - FOCUSED: Enhanced attention, deeper processing
    - HYPERVIGILANT: High stress, rapid but shallow processing
    """
    DEEP_SLEEP = 0
    DROWSY = 1
    ALERT = 2
    FOCUSED = 3
    HYPERVIGILANT = 4


@dataclass
class CNSState:
    """Current state of the central nervous system"""
    consciousness: ConsciousnessLevel = ConsciousnessLevel.ALERT
    stress_level: float = 0.0          # 0-1, current stress
    fatigue_level: float = 0.0         # 0-1, accumulated fatigue
    focus_intensity: float = 0.5       # 0-1, current focus
    last_state_change: float = 0.0     # Timestamp of last state change
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'consciousness': self.consciousness.value,
            'stress_level': self.stress_level,
            'fatigue_level': self.fatigue_level,
            'focus_intensity': self.focus_intensity
        }


class CentralNervousSystem:
    """
    Central Nervous System - Master controller for consciousness and arousal
    
    Manages:
    - Consciousness state transitions
    - Global stress response
    - Fatigue accumulation and recovery
    - Focus modulation
    
    Influences:
    - Response speed vs depth trade-off
    - Memory consolidation priority
    - Attention allocation
    """
    
    def __init__(
        self,
        stress_recovery_rate: float = 0.01,
        fatigue_rate: float = 0.001,
        fatigue_recovery_rate: float = 0.005
    ):
        """
        Initialize CNS
        
        Args:
            stress_recovery_rate: How fast stress decays (per second)
            fatigue_rate: How fast fatigue accumulates (per interaction)
            fatigue_recovery_rate: How fast fatigue recovers during low activity
        """
        self.stress_recovery_rate = stress_recovery_rate
        self.fatigue_rate = fatigue_rate
        self.fatigue_recovery_rate = fatigue_recovery_rate
        
        self.state = CNSState()
        self.last_update = time.time()
        
        # Thresholds for state transitions
        self.stress_thresholds = {
            ConsciousnessLevel.HYPERVIGILANT: 0.8,
            ConsciousnessLevel.FOCUSED: 0.4,
            ConsciousnessLevel.ALERT: 0.0,
            ConsciousnessLevel.DROWSY: -0.3,  # Low stress + high fatigue
        }
        
        # Statistics
        self.stats = {
            'state_changes': 0,
            'peak_stress': 0.0,
            'total_interactions': 0,
            'time_in_states': {level.name: 0.0 for level in ConsciousnessLevel}
        }
        
        logger.info("Central Nervous System initialized")
    
    def update(
        self,
        error_signal: Optional[float] = None,
        arousal: Optional[float] = None,
        valence: Optional[float] = None,
        interaction_occurred: bool = False
    ) -> CNSState:
        """
        Update CNS state based on inputs
        
        Args:
            error_signal: Error from recent processing (0-1)
            arousal: Current emotional arousal (0-1)
            valence: Current emotional valence (-1 to 1)
            interaction_occurred: Whether an interaction just happened
            
        Returns:
            Updated CNS state
        """
        current_time = time.time()
        dt = current_time - self.last_update
        self.last_update = current_time
        
        # Track time in current state
        self.stats['time_in_states'][self.state.consciousness.name] += dt
        
        # 1. Update stress level
        if error_signal is not None:
            # Stress increases with errors
            stress_increase = error_signal * 0.3
            self.state.stress_level = min(1.0, self.state.stress_level + stress_increase)
        
        if arousal is not None and valence is not None:
            # High arousal + negative valence = stress
            if valence < 0 and arousal > 0.5:
                stress_from_emotion = (-valence) * arousal * 0.2
                self.state.stress_level = min(1.0, self.state.stress_level + stress_from_emotion)
        
        # Natural stress recovery
        self.state.stress_level = max(0.0, self.state.stress_level - self.stress_recovery_rate * dt)
        
        # Track peak stress
        if self.state.stress_level > self.stats['peak_stress']:
            self.stats['peak_stress'] = self.state.stress_level
        
        # 2. Update fatigue
        if interaction_occurred:
            self.state.fatigue_level = min(1.0, self.state.fatigue_level + self.fatigue_rate)
            self.stats['total_interactions'] += 1
        else:
            # Recover fatigue during inactivity
            self.state.fatigue_level = max(0.0, self.state.fatigue_level - self.fatigue_recovery_rate * dt)
        
        # 3. Update focus
        if arousal is not None:
            # Focus increases with moderate arousal, decreases at extremes
            optimal_arousal = 0.6
            focus_from_arousal = 1.0 - abs(arousal - optimal_arousal)
            self.state.focus_intensity = 0.7 * self.state.focus_intensity + 0.3 * focus_from_arousal
        
        # Fatigue reduces focus
        self.state.focus_intensity *= (1.0 - self.state.fatigue_level * 0.3)
        
        # 4. Determine consciousness level
        self._update_consciousness()
        
        return self.state
    
    def _update_consciousness(self) -> None:
        """Update consciousness level based on stress and fatigue"""
        old_level = self.state.consciousness
        
        stress = self.state.stress_level
        fatigue = self.state.fatigue_level
        
        # Determine new level
        if stress > 0.8:
            new_level = ConsciousnessLevel.HYPERVIGILANT
        elif fatigue > 0.8:
            new_level = ConsciousnessLevel.DROWSY
        elif stress > 0.5 or self.state.focus_intensity > 0.7:
            new_level = ConsciousnessLevel.FOCUSED
        elif fatigue > 0.5 and stress < 0.2:
            new_level = ConsciousnessLevel.DROWSY
        else:
            new_level = ConsciousnessLevel.ALERT
        
        if new_level != old_level:
            self.state.consciousness = new_level
            self.state.last_state_change = time.time()
            self.stats['state_changes'] += 1
            logger.debug(f"Consciousness changed: {old_level.name} -> {new_level.name}")
    
    def add_stress(self, amount: float) -> None:
        """Directly add stress (for external events)"""
        self.state.stress_level = min(1.0, self.state.stress_level + amount)
        self._update_consciousness()
    
    def reduce_stress(self, amount: float) -> None:
        """Directly reduce stress (for calming events)"""
        self.state.stress_level = max(0.0, self.state.stress_level - amount)
        self._update_consciousness()
    
    def get_processing_modulation(self) -> Dict[str, float]:
        """
        Get modulation factors for processing based on consciousness
        
        Returns factors for:
        - processing_depth: How thorough to be (0-1)
        - processing_speed: How fast to respond (0-1)
        - creativity: How creative/exploratory (0-1)
        - vigilance: How alert to threats/errors (0-1)
        """
        level = self.state.consciousness
        stress = self.state.stress_level
        fatigue = self.state.fatigue_level
        focus = self.state.focus_intensity
        
        modulations = {
            ConsciousnessLevel.DEEP_SLEEP: {
                'processing_depth': 0.1,
                'processing_speed': 0.1,
                'creativity': 0.0,
                'vigilance': 0.0
            },
            ConsciousnessLevel.DROWSY: {
                'processing_depth': 0.4,
                'processing_speed': 0.3,
                'creativity': 0.2,
                'vigilance': 0.2
            },
            ConsciousnessLevel.ALERT: {
                'processing_depth': 0.6,
                'processing_speed': 0.6,
                'creativity': 0.5,
                'vigilance': 0.5
            },
            ConsciousnessLevel.FOCUSED: {
                'processing_depth': 0.9,
                'processing_speed': 0.5,
                'creativity': 0.4,
                'vigilance': 0.7
            },
            ConsciousnessLevel.HYPERVIGILANT: {
                'processing_depth': 0.3,
                'processing_speed': 0.9,
                'creativity': 0.1,
                'vigilance': 1.0
            }
        }
        
        base = modulations[level]
        
        # Adjust based on current state
        result = {
            'processing_depth': base['processing_depth'] * (1.0 - fatigue * 0.3) * focus,
            'processing_speed': base['processing_speed'] * (1.0 + stress * 0.2),
            'creativity': base['creativity'] * (1.0 - stress * 0.5) * (1.0 - fatigue * 0.3),
            'vigilance': base['vigilance'] * (1.0 + stress * 0.3)
        }
        
        # Clamp all values
        return {k: max(0.0, min(1.0, v)) for k, v in result.items()}
    
    def get_state(self) -> CNSState:
        """Get current CNS state"""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get CNS statistics"""
        return {
            **self.stats,
            'current_state': self.state.to_dict(),
            'modulation': self.get_processing_modulation()
        }
    
    def save_state(self, path: str) -> None:
        """Save CNS state"""
        state_dict = {
            'state': self.state.to_dict(),
            'stats': self.stats
        }
        with open(path, 'w') as f:
            json.dump(state_dict, f)
        logger.info(f"CNS state saved to {path}")
    
    def load_state(self, path: str) -> None:
        """Load CNS state"""
        if not Path(path).exists():
            return
        
        with open(path, 'r') as f:
            state_dict = json.load(f)
        
        s = state_dict['state']
        self.state = CNSState(
            consciousness=ConsciousnessLevel(s['consciousness']),
            stress_level=s['stress_level'],
            fatigue_level=s['fatigue_level'],
            focus_intensity=s['focus_intensity']
        )
        self.stats = state_dict['stats']
        logger.info(f"CNS state loaded from {path}")

