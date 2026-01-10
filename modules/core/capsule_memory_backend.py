"""
Capsule Memory Backend Plugin
Wraps CapsuleMemoryStore as a BaseMemoryBackend plugin
"""
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from modules.base import BaseMemoryBackend
from memory.capsule_store import CapsuleMemoryStore
from memory.capsule_encoder import CapsuleEncoder
from memory.capsule_memory import CapsuleMemory, MemoryType
from memory.identity_loader import load_identity_dataset
from config import MemoryConfig, ModelConfig

logger = logging.getLogger(__name__)


class CapsuleMemoryBackend(BaseMemoryBackend):
    """
    Capsule memory backend plugin
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize capsule memory backend
        
        Args:
            config: Optional configuration dict
        """
        config = config or {}
        capacity = config.get("capacity", 100000)
        use_embeddings = config.get("use_embeddings", ModelConfig.USE_EMBEDDINGS)
        
        # Create encoder
        encoder = CapsuleEncoder(use_embeddings=use_embeddings)
        
        # Create store
        self._store = CapsuleMemoryStore(
            encoder=encoder,
            capacity=capacity,
            use_embeddings=use_embeddings
        )
        
        # Store for compatibility
        self.identity_text: Optional[str] = None
        self.identity_index: int = -1
        
        # Auto-load identity dataset if path exists
        identity_path = config.get("identity_path", MemoryConfig.CAPSULE_IDENTITY_PATH)
        if identity_path and Path(identity_path).exists():
            try:
                identity_memories = load_identity_dataset(str(identity_path))
                if identity_memories:
                    self._store.add_batch(identity_memories)
                    logger.info(f"Loaded {len(identity_memories)} identity memories from {identity_path}")
                    # Set identity text from first memory
                    if identity_memories:
                        self.identity_text = identity_memories[0].content
            except Exception as e:
                logger.warning(f"Failed to load identity dataset: {e}")
    
    @property
    def embedding_dim(self) -> int:
        """Dimension of embeddings (32D capsule, 128D DG for FAISS)"""
        return 32  # Capsule dimension
    
    def store(
        self,
        embedding: np.ndarray,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store a memory
        
        Args:
            embedding: Embedding vector (ignored, will be recomputed)
            text: Memory text content
            metadata: Optional metadata dict (can contain memory_type, domain, etc.)
        
        Returns:
            memory_id: Memory identifier
        """
        metadata = metadata or {}
        
        # Extract cognitive features from metadata or use defaults
        memory_type = MemoryType(metadata.get('memory_type', 'EPISODE'))
        domain = metadata.get('domain', 'general')
        plasticity_gain = metadata.get('plasticity_gain', 0.5)
        consolidation_priority = metadata.get('consolidation_priority', 0.5)
        stability = metadata.get('stability', 0.5)
        stress_link = metadata.get('stress_link', 0.0)
        protected = metadata.get('protected', False)
        
        # Generate memory ID
        import uuid
        memory_id = metadata.get('memory_id', str(uuid.uuid4()))
        
        # Create capsule memory
        memory = CapsuleMemory(
            memory_id=memory_id,
            memory_type=memory_type,
            domain=domain,
            content=text,
            plasticity_gain=plasticity_gain,
            consolidation_priority=consolidation_priority,
            stability=stability,
            stress_link=stress_link,
            protected=protected
        )
        
        # Add to store
        self._store.add_memory(memory)
        
        return memory_id
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = 3,
        include_identity: bool = True,
        reranker: Optional[Any] = None,
        query_text: Optional[str] = None,
        emotion_bias: Optional[Dict[str, float]] = None,
        temporal_bias: Optional[Dict[int, float]] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve similar memories
        
        Args:
            query_embedding: Query embedding vector (from default embedder)
            k: Number of memories to retrieve
            include_identity: Whether to include identity memory
            reranker: Optional reranking function (not supported yet)
            query_text: Optional query text (preferred for capsule memory)
            emotion_bias: Optional emotion-based bias (not supported yet)
            temporal_bias: Optional temporal recency bias (not supported yet)
        
        Returns:
            List of (text, similarity_score) tuples
        """
        # Save the query as a memory using the default embedder's embedding
        if query_text:
            # Store query as an EPISODE memory with default cognitive features
            # The query_embedding is from the default embedder (model.get_embedding())
            try:
                self.store(
                    embedding=query_embedding,
                    text=query_text,
                    metadata={
                        'memory_type': 'EPISODE',
                        'domain': 'query',
                        'plasticity_gain': 0.7,  # Moderate learning rate
                        'consolidation_priority': 0.6,  # Moderate priority
                        'stability': 0.5,  # Moderate stability
                        'stress_link': 0.0,
                        'protected': False
                    }
                )
                logger.debug(f"Saved query as memory: {query_text[:50]}...")
            except Exception as e:
                logger.warning(f"Failed to save query as memory: {e}")
        
        # Always include identity first if available
        identity_results = []
        if include_identity and self.identity_text:
            # Create a dummy identity result with high similarity
            identity_results = [(self.identity_text, 1.0)]
            k_adj = k - 1 if k > 1 else k
        else:
            k_adj = k
        
        # Query capsule memory using text (preferred method)
        if query_text:
            results = self._store.query(query_text, k=k_adj)
            # Convert to (text, similarity_score) format
            # Distance is L2, convert to similarity (inverse)
            formatted_results = []
            for memory, distance in results:
                # Convert distance to similarity (normalize)
                similarity = max(0.0, 1.0 / (1.0 + distance))
                formatted_results.append((memory.content, similarity))
            
            # Always prepend identity if included
            if include_identity and self.identity_text:
                return identity_results + formatted_results
            return formatted_results
        
        # If no query text, return identity only if requested
        if include_identity and self.identity_text:
            return identity_results
        return []
    
    def query(self, text: str, k: int = 32) -> List[Tuple[CapsuleMemory, float]]:
        """
        Query capsule memory with text (preferred method)
        
        Args:
            text: Query text
            k: Number of results
        
        Returns:
            List of (CapsuleMemory, distance) tuples
        """
        return self._store.query(text, k=k)
    
    def clear(self) -> None:
        """Clear all memories"""
        self._store.memories = []
        self._store.protected_memories = []
        self._store.rebuild_index()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        stats = self._store.get_stats()
        stats['type'] = 'capsule'
        return stats
    
    def get_identity(self) -> Optional[str]:
        """Get system identity text"""
        return self.identity_text
    
    def store_identity(
        self,
        embedding: np.ndarray,
        identity_text: str
    ) -> None:
        """Store system identity"""
        self.identity_text = identity_text
        
        # Store as protected SELF_STATE memory
        import uuid
        identity_memory = CapsuleMemory(
            memory_id=f"identity_{uuid.uuid4()}",
            memory_type=MemoryType.SELF_STATE,
            domain="identity",
            content=identity_text,
            plasticity_gain=1.0,  # High plasticity for identity
            consolidation_priority=1.0,  # Maximum priority
            stability=1.0,  # Maximum stability
            stress_link=0.0,
            protected=True
        )
        
        self._store.add_memory(identity_memory)
        self.identity_index = len(self._store.memories) - 1
