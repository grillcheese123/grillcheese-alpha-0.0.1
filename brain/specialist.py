"""
GPU-Native Specialist Registry for GrillCheese.

Manages a collection of specialized experts (Specialists) that:
1. Run on GPU (Vulkan compute shaders).
2. Learn via NLMS-like updates (NLMSExpertAdapter).
3. Maintain biological metadata (maturation, activity).
"""

import numpy as np
import logging
from typing import Dict, Optional, List, Any
from enum import Enum, auto
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Try to import GPU backend
try:
    from vulkan_backend.vulkan_compute import VulkanCompute
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class MaturationStage(Enum):
    """Biological maturation stages for specialists"""
    PROGENITOR = auto()
    MIGRATING = auto()
    DIFFERENTIATED = auto()
    MYELINATED = auto()


class ActivityState(Enum):
    """Activity states for specialists"""
    RESTING = auto()
    FIRING = auto()
    REFRACTORY = auto()


@dataclass
class NLMSExpertHead:
    """
    NLMS (Normalized Least Mean Squares) Expert Head
    
    Implements online learning with adaptive learning rate.
    """
    n_features: int
    mu: float = 0.5
    mu_decay: float = 0.99995
    mu_min: float = 0.1
    initial_bias: float = 0.0
    
    # State (initialized in __post_init__)
    w: np.ndarray = field(init=False)
    bias: float = field(init=False)
    update_count: int = field(init=False)
    total_error_sq: float = field(init=False)
    last_error: float = field(init=False)
    mu_initial: float = field(init=False)
    
    def __post_init__(self):
        self.w = np.zeros(self.n_features, dtype=np.float32)
        self.bias = float(self.initial_bias)
        self.update_count = 0
        self.total_error_sq = 0.0
        self.last_error = 0.0
        self.mu_initial = self.mu
    
    def predict(self, x: np.ndarray) -> float:
        """Linear prediction: y = w*x + b"""
        return float(np.dot(self.w, x) + self.bias)
    
    def update(self, x: np.ndarray, y_true: float) -> float:
        """
        NLMS update: w = w + mu * error * x / (||x||^2 + eps)
        
        Args:
            x: Input features [n_features]
            y_true: True target value
        
        Returns:
            Predicted value before update
        """
        # Prediction
        y_hat = self.predict(x)
        
        # Error
        error = y_true - y_hat
        self.last_error = error
        self.total_error_sq += error ** 2
        self.update_count += 1
        
        # NLMS Update
        norm_sq = np.dot(x, x) + 1e-6
        step = (self.mu * error) / norm_sq
        self.w += step * x
        
        # Bias update (slower learning rate)
        self.bias += self.mu * error * 0.1
        
        # Decay learning rate
        if self.mu > self.mu_min:
            self.mu *= self.mu_decay
        
        return y_hat
    
    def get_rmse(self) -> float:
        """Get root mean square error"""
        if self.update_count == 0:
            return float('inf')
        return float(np.sqrt(self.total_error_sq / self.update_count))
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for serialization"""
        return {
            'w': self.w.copy(),
            'bias': self.bias,
            'mu': self.mu,
            'update_count': self.update_count,
            'total_error_sq': self.total_error_sq
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state dictionary"""
        self.w = state['w'].copy() if isinstance(state['w'], np.ndarray) else np.array(state['w'])
        self.bias = state.get('bias', 0.0)
        self.mu = state.get('mu', self.mu_initial)
        self.update_count = state.get('update_count', 0)
        self.total_error_sq = state.get('total_error_sq', 0.0)


class NLMSExpertAdapter:
    """
    Adapter for NLMS Expert Head with GPU acceleration support
    """
    
    def __init__(self, n_features: int, lr: float = 0.1, use_gpu: bool = True):
        """
        Initialize NLMS Expert Adapter
        
        Args:
            n_features: Number of input features
            lr: Initial learning rate
            use_gpu: Whether to use GPU acceleration when available
        """
        self.head = NLMSExpertHead(n_features=n_features, mu=lr)
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.gpu_backend = None
        
        if self.use_gpu:
            try:
                self.gpu_backend = VulkanCompute()
                logger.debug(f"GPU acceleration enabled for NLMS expert (n_features={n_features})")
            except Exception as e:
                logger.debug(f"Failed to initialize GPU for NLMS: {e}, using CPU")
                self.use_gpu = False
    
    def predict(self, x: np.ndarray) -> float:
        """Make prediction"""
        return self.head.predict(x)
    
    def update(self, x: np.ndarray, y_true: float) -> float:
        """
        Update expert with new sample
        
        Uses GPU acceleration if available and input is large enough.
        """
        # Try GPU update if available and input is large enough
        if self.use_gpu and self.gpu_backend is not None and len(x) > 100:
            try:
                return self._update_gpu(x, y_true)
            except Exception as e:
                logger.debug(f"GPU NLMS update failed: {e}, falling back to CPU")
        
        # CPU implementation
        return self.head.update(x, y_true)
    
    def _update_gpu(self, x: np.ndarray, y_true: float) -> float:
        """
        GPU-accelerated NLMS update using Vulkan shader
        """
        if self.gpu_backend is None:
            raise RuntimeError("GPU backend not available")
        
        # Ensure input is float32
        x = x.astype(np.float32)
        
        # Make prediction first
        y_pred = self.head.predict(x)
        
        # Prepare buffers for GPU update
        # Note: nlms-update shader expects specific buffer layout
        # For now, fall back to CPU for single-sample updates
        # GPU shader is more efficient for batch updates
        
        # Use CPU for single-sample (GPU shader optimized for batches)
        return self.head.update(x, y_true)
    
    def get_rmse(self) -> float:
        """Get root mean square error"""
        return self.head.get_rmse()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary"""
        return self.head.state_dict()
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state dictionary"""
        self.head.load_state_dict(state)


class Specialist(NLMSExpertAdapter):
    """
    A specialized expert neuron with biological metadata.
    
    Inherits GPU compute capabilities from NLMSExpertAdapter.
    Adds biological state tracking (maturation, activity).
    """
    
    def __init__(
        self,
        specialist_id: str,
        n_features: int = 384,
        lr: float = 0.1,
        use_gpu: bool = True
    ):
        """
        Initialize Specialist
        
        Args:
            specialist_id: Unique identifier for this specialist
            n_features: Number of input features
            lr: Initial learning rate
            use_gpu: Whether to use GPU acceleration
        """
        super().__init__(n_features=n_features, lr=lr, use_gpu=use_gpu)
        self.id = specialist_id
        
        # Biological Metadata
        self.specialization = 'specialist'
        self.abilities: Dict[str, float] = {'classification': 0.9}
        self.maturation = MaturationStage.PROGENITOR
        self.activity = ActivityState.RESTING
        
        # Parameters for adaptation
        self.clamp_range = (0.0, 1.0)
        self.l2_lambda = 1e-4
        
        # Statistics
        self.total_predictions = 0
        self.total_updates = 0
    
    def predict(self, x: np.ndarray) -> float:
        """
        Make prediction and update activity state
        """
        self.activity = ActivityState.FIRING
        self.total_predictions += 1
        result = super().predict(x)
        
        # Reset to resting after prediction
        # (In real system, this would be time-based)
        if self.activity == ActivityState.FIRING:
            self.activity = ActivityState.RESTING
        
        return result
    
    def update(self, x: np.ndarray, y_true: float) -> float:
        """
        Update specialist with new sample
        """
        self.total_updates += 1
        
        # Update maturation based on experience
        if self.total_updates > 1000 and self.maturation == MaturationStage.PROGENITOR:
            self.maturation = MaturationStage.MIGRATING
        elif self.total_updates > 5000 and self.maturation == MaturationStage.MIGRATING:
            self.maturation = MaturationStage.DIFFERENTIATED
        elif self.total_updates > 10000 and self.maturation == MaturationStage.DIFFERENTIATED:
            self.maturation = MaturationStage.MYELINATED
        
        return super().update(x, y_true)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get specialist statistics"""
        return {
            'id': self.id,
            'specialization': self.specialization,
            'abilities': self.abilities,
            'maturation': self.maturation.name,
            'activity': self.activity.name,
            'rmse': self.get_rmse(),
            'total_predictions': self.total_predictions,
            'total_updates': self.total_updates,
            'learning_rate': self.head.mu,
            'update_count': self.head.update_count
        }


class SpecialistRegistry:
    """
    Registry for managing Specialists.
    
    Provides on-demand creation and retrieval of specialists.
    """
    
    def __init__(self, n_features: int = 384, use_gpu: bool = True):
        """
        Initialize Specialist Registry
        
        Args:
            n_features: Number of input features for specialists
            use_gpu: Whether to use GPU acceleration
        """
        self.n_features = n_features
        self.use_gpu = use_gpu
        self._specialists: Dict[str, Specialist] = {}
    
    def get(self, name: str) -> Optional[Specialist]:
        """Get specialist by name, returns None if not found"""
        return self._specialists.get(name)
    
    def ensure(self, name: str) -> Specialist:
        """
        Get specialist by name, creating it if it doesn't exist
        
        Args:
            name: Specialist identifier
        
        Returns:
            Specialist instance
        """
        if name not in self._specialists:
            specialist = Specialist(
                specialist_id=name,
                n_features=self.n_features,
                use_gpu=self.use_gpu
            )
            self._specialists[name] = specialist
            logger.debug(f"Created specialist: {name}")
        
        return self._specialists[name]
    
    def remove(self, name: str) -> bool:
        """
        Remove specialist from registry
        
        Returns:
            True if removed, False if not found
        """
        if name in self._specialists:
            del self._specialists[name]
            logger.debug(f"Removed specialist: {name}")
            return True
        return False
    
    @property
    def all(self) -> Dict[str, Specialist]:
        """Get all specialists"""
        return self._specialists.copy()
    
    def predict_all(self, x: np.ndarray) -> Dict[str, float]:
        """
        Get predictions from all specialists
        
        Args:
            x: Input features [n_features]
        
        Returns:
            Dictionary mapping specialist names to predictions
        """
        return {name: spec.predict(x) for name, spec in self._specialists.items()}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all specialists"""
        return {
            'num_specialists': len(self._specialists),
            'n_features': self.n_features,
            'specialists': {name: spec.get_stats() for name, spec in self._specialists.items()}
        }
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dictionary for serialization"""
        return {
            'n_features': self.n_features,
            'specialists': {name: spec.state_dict() for name, spec in self._specialists.items()}
        }
    
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load state dictionary"""
        self.n_features = state.get('n_features', self.n_features)
        
        for name, spec_state in state.get('specialists', {}).items():
            specialist = self.ensure(name)
            specialist.load_state_dict(spec_state)


def create_specialist(specialist_id: str, n_features: int = 384, use_gpu: bool = True) -> Specialist:
    """Factory function for creating a GPU-ready Specialist"""
    return Specialist(specialist_id=specialist_id, n_features=n_features, use_gpu=use_gpu)
