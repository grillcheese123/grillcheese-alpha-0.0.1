"""
Capsule Memory System
Bio-inspired 32-dimensional capsule memory with hippocampal architecture
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Tuple
import numpy as np
import time
import json
import logging

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of capsule memories"""
    CONCEPT = "CONCEPT"
    EPISODE = "EPISODE"
    SELF_STATE = "SELF_STATE"
    TASK = "TASK"
    TOOL = "TOOL"


@dataclass
class CapsuleMemory:
    """32D bio-inspired capsule memory structure"""
    memory_id: str
    memory_type: MemoryType
    domain: str
    content: str
    
    # 32D cognitive features (not the full embedding, just the "controls")
    plasticity_gain: float = 0.5      # 0.0-1.0: learning rate for this memory
    consolidation_priority: float = 0.5  # 0.0-1.0: importance for replay
    stability: float = 0.5            # 0.0-1.0: resistance to forgetting
    stress_link: float = 0.0          # 0.0-1.0: emotional/stress association
    
    # Full 32D capsule vector (computed by encoder)
    capsule_vector: Optional[np.ndarray] = None  # shape: (32,)
    
    # Expanded DG representation (for FAISS storage)
    dg_vector: Optional[np.ndarray] = None  # shape: (128,)
    
    # Metadata
    protected: bool = False
    access_count: int = 0
    last_access: Optional[float] = None
    created: float = field(default_factory=lambda: time.time())
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'memory_id': self.memory_id,
            'memory_type': self.memory_type.value,
            'domain': self.domain,
            'content': self.content,
            'plasticity_gain': self.plasticity_gain,
            'consolidation_priority': self.consolidation_priority,
            'stability': self.stability,
            'stress_link': self.stress_link,
            'capsule_vector': self.capsule_vector.tolist() if self.capsule_vector is not None else None,
            'dg_vector': self.dg_vector.tolist() if self.dg_vector is not None else None,
            'protected': self.protected,
            'access_count': self.access_count,
            'last_access': self.last_access,
            'created': self.created
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CapsuleMemory':
        """Create from dictionary"""
        return cls(
            memory_id=data['memory_id'],
            memory_type=MemoryType(data['memory_type']),
            domain=data['domain'],
            content=data['content'],
            plasticity_gain=data['plasticity_gain'],
            consolidation_priority=data['consolidation_priority'],
            stability=data['stability'],
            stress_link=data['stress_link'],
            capsule_vector=np.array(data['capsule_vector']) if data.get('capsule_vector') else None,
            dg_vector=np.array(data['dg_vector']) if data.get('dg_vector') else None,
            protected=data.get('protected', False),
            access_count=data.get('access_count', 0),
            last_access=data.get('last_access'),
            created=data.get('created', time.time())
        )
