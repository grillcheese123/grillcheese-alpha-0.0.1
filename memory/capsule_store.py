"""
Capsule Memory Store
Manages capsule memories with encoding, storage, retrieval, and consolidation
"""
import time
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np

from memory.capsule_memory import CapsuleMemory, MemoryType
from memory.capsule_encoder import CapsuleEncoder
from memory.dentate_gyrus import DentateGyrus
from memory.ca3_memory import CA3Memory

logger = logging.getLogger(__name__)


class CapsuleMemoryStore:
    """
    Manages capsule memories with encoding, storage, retrieval, and consolidation.
    """
    
    def __init__(
        self,
        encoder: Optional[CapsuleEncoder] = None,
        capacity: int = 100000,
        use_embeddings: bool = False
    ):
        """
        Initialize capsule memory store
        
        Args:
            encoder: Optional CapsuleEncoder instance (creates default if None)
            capacity: Maximum number of memories before consolidation
            use_embeddings: Whether to use sentence-transformers for encoding
        """
        self.encoder = encoder or CapsuleEncoder(use_embeddings=use_embeddings)
        self.dg = DentateGyrus(input_dim=32, expansion_factor=4, sparsity=0.02)
        self.ca3 = CA3Memory(dim=128)
        
        self.memories: List[CapsuleMemory] = []
        self.capacity = capacity
        self.protected_memories: List[CapsuleMemory] = []
    
    def add_memory(self, memory: CapsuleMemory):
        """Add memory with encoding."""
        # Encode to 32D capsule
        features = {
            'plasticity_gain': memory.plasticity_gain,
            'consolidation_priority': memory.consolidation_priority,
            'stability': memory.stability,
            'stress_link': memory.stress_link
        }
        memory.capsule_vector = self.encoder.encode(memory.content, features)
        
        # Expand to 128D DG representation
        memory.dg_vector = self.dg.expand(memory.capsule_vector)
        memory.created = time.time()
        
        if memory.protected:
            self.protected_memories.append(memory)
        
        self.memories.append(memory)
        
        # Rebuild index periodically or if capacity reached
        if len(self.memories) % 100 == 0 or len(self.memories) >= self.capacity:
            self._consolidate()
            self.ca3.build_index(self.memories)
        elif len(self.memories) == 1:
            # Build index on first memory
            self.ca3.build_index(self.memories)
    
    def add_batch(self, memories: List[CapsuleMemory]):
        """Batch addition for efficiency."""
        texts = [m.content for m in memories]
        features = [
            {
                'plasticity_gain': m.plasticity_gain,
                'consolidation_priority': m.consolidation_priority,
                'stability': m.stability,
                'stress_link': m.stress_link
            } for m in memories
        ]
        
        capsules = self.encoder.encode_batch(texts, features)
        dg_vectors = self.dg.expand_batch(capsules)
        
        for i, memory in enumerate(memories):
            memory.capsule_vector = capsules[i]
            memory.dg_vector = dg_vectors[i]
            memory.created = time.time()
            
            if memory.protected:
                self.protected_memories.append(memory)
            
            self.memories.append(memory)
        
        self.ca3.build_index(self.memories)
    
    def query(self, text: str, k: int = 32) -> List[Tuple[CapsuleMemory, float]]:
        """
        Retrieve memories relevant to query.
        
        Args:
            text: Query text
            k: Number of memories to retrieve
        
        Returns:
            List of (CapsuleMemory, distance) sorted by relevance
        """
        # Default cognitive features for query
        query_features = {
            'plasticity_gain': 0.8,
            'consolidation_priority': 0.5,
            'stability': 0.5,
            'stress_link': 0.0
        }
        
        # Encode query
        query_capsule = self.encoder.encode(text, query_features)
        query_dg = self.dg.expand(query_capsule)
        
        # Retrieve via CA3
        results = self.ca3.retrieve(query_dg, k)
        
        return results
    
    def _consolidate(self):
        """
        Importance-based forgetting when capacity exceeded.
        Preserves protected memories and high consolidation_priority memories.
        """
        if len(self.memories) <= self.capacity:
            return
        
        logger.info(f"Consolidating memories: {len(self.memories)} → {int(self.capacity * 0.8)}")
        
        # Compute importance scores
        scores = []
        for m in self.memories:
            if m.protected:
                scores.append(float('inf'))  # Never remove protected
            else:
                # Score: consolidation_priority * stability * recency * access_freq
                recency = 1.0 / (time.time() - m.last_access + 1) if m.last_access else 0.0
                freq = np.log1p(m.access_count)
                score = m.consolidation_priority * m.stability * (recency + freq)
                scores.append(score)
        
        # Remove lowest-scoring memories
        keep_count = int(self.capacity * 0.8)  # Keep 80% after consolidation
        keep_indices = np.argsort(scores)[-keep_count:]
        
        self.memories = [self.memories[i] for i in keep_indices]
        # Rebuild protected list
        self.protected_memories = [m for m in self.memories if m.protected]
    
    def rebuild_index(self):
        """Force index rebuild."""
        self.ca3.build_index(self.memories)
    
    def save(self, path: str):
        """Persist memories to disk."""
        data = {
            'memories': [m.to_dict() for m in self.memories]
        }
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved {len(self.memories)} memories to {path}")
    
    def load(self, path: str):
        """Load memories from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.memories = []
        for m_data in data['memories']:
            memory = CapsuleMemory.from_dict(m_data)
            
            if memory.protected:
                self.protected_memories.append(memory)
            
            self.memories.append(memory)
        
        self.rebuild_index()
        logger.info(f"Loaded {len(self.memories)} memories from {path}")
    
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        return {
            'total_memories': len(self.memories),
            'protected_memories': len(self.protected_memories),
            'capacity': self.capacity,
            'utilization': len(self.memories) / self.capacity if self.capacity > 0 else 0.0
        }
