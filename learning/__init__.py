"""
GrillCheese Continuous Learning Module
Implements STDP-based learning for persistent memory enhancement
"""
from .events import EventBus, Event
from .stdp_learner import STDPLearner
from .continuous_learner import ContinuousLearner, LearningConfig

__all__ = ['EventBus', 'Event', 'STDPLearner', 'ContinuousLearner', 'LearningConfig']

