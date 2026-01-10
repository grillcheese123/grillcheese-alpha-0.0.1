"""
SQLite + FAISS Memory Backend Plugin
Wraps existing MemoryStore as a plugin
"""
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from modules.base import BaseMemoryBackend
from memory_store import MemoryStore
from config import MemoryConfig


class SqliteFaissBackend(BaseMemoryBackend):
    """
    Wrapper around existing MemoryStore for plugin compatibility
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize SQLite + FAISS backend
        
        Args:
            config: Optional configuration dict
        """
        config = config or {}
        db_path = config.get("db_path", MemoryConfig.DB_PATH)
        max_memories = config.get("max_memories", MemoryConfig.MAX_MEMORIES)
        embedding_dim = config.get("embedding_dim", MemoryConfig.EMBEDDING_DIM)
        use_hilbert = config.get("use_hilbert", MemoryConfig.USE_HILBERT_ROUTING)
        
        self._backend = MemoryStore(
            db_path=db_path,
            max_memories=max_memories,
            embedding_dim=embedding_dim,
            use_hilbert=use_hilbert
        )
    
    @property
    def embedding_dim(self) -> int:
        """Dimension of embeddings"""
        return self._backend.embedding_dim
    
    def store(
        self,
        embedding: np.ndarray,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        is_protected: bool = False
    ) -> str:
        """
        Store a memory
        
        Args:
            embedding: Embedding vector
            text: Memory text content
            metadata: Optional metadata dict
            is_protected: If True, memory will never be pruned (passed to underlying MemoryStore)
        
        Returns:
            memory_id: Memory identifier (generated from index)
        """
        # Store memory - pass is_protected to underlying MemoryStore
        self._backend.store(embedding=embedding, text=text, metadata=metadata, is_protected=is_protected)
        
        # Generate ID from current memory count
        memory_id = str(self._backend.num_memories - 1)
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
            query_embedding: Query embedding vector
            k: Number of memories to retrieve
            include_identity: Whether to include identity memory
            reranker: Optional reranking function
            query_text: Optional query text for reranking
            emotion_bias: Optional emotion-based bias
            temporal_bias: Optional temporal recency bias
        
        Returns:
            List of (text, similarity_score) tuples
        """
        # MemoryStore.retrieve() returns List[str], so we need to compute similarities
        # Pass through all parameters to MemoryStore
        texts = self._backend.retrieve(
            query_embedding=query_embedding,
            k=k,
            include_identity=include_identity,
            reranker=reranker,
            query_text=query_text,
            emotion_bias=emotion_bias,
            temporal_bias=temporal_bias
        )
        
        # Compute similarity scores for each retrieved text
        results = []
        for text in texts:
            # Find the text in memory_texts and get its embedding
            try:
                idx = self._backend.memory_texts.index(text)
                if idx < len(self._backend.memory_keys) and idx < self._backend.num_memories:
                    text_embedding = self._backend.memory_keys[idx]
                    # Compute cosine similarity
                    similarity = np.dot(query_embedding, text_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding) + 1e-8
                    )
                    results.append((text, float(similarity)))
                else:
                    # Can't compute similarity, use default
                    results.append((text, 0.5))
            except (ValueError, AttributeError, IndexError):
                results.append((text, 0.5))
        
        return results
    
    def clear(self) -> None:
        """Clear all memories"""
        self._backend.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return self._backend.get_stats()
    
    def get_identity(self) -> Optional[str]:
        """Get system identity text"""
        return self._backend.identity_text
    
    @property
    def identity_index(self) -> int:
        """Get identity index (for compatibility with MemoryStore API)"""
        return getattr(self._backend, 'identity_index', -1)
    
    @property
    def num_memories(self) -> int:
        """Get number of memories (for compatibility)"""
        return getattr(self._backend, 'num_memories', 0)
    
    @property
    def memory_keys(self) -> Optional[np.ndarray]:
        """Get memory keys array (for compatibility)"""
        return getattr(self._backend, 'memory_keys', None)
    
    @property
    def memory_values(self) -> Optional[np.ndarray]:
        """Get memory values array (for compatibility)"""
        return getattr(self._backend, 'memory_values', None)
    
    @property
    def memory_texts(self) -> List[str]:
        """Get memory texts list (for compatibility)"""
        return getattr(self._backend, 'memory_texts', [])
    
    @property
    def gpu(self):
        """Get GPU backend (for compatibility)"""
        return getattr(self._backend, 'gpu', None)
    
    @property
    def db_path(self):
        """Get database path (for compatibility)"""
        return getattr(self._backend, 'db_path', None)
    
    def store_identity(
        self,
        embedding: np.ndarray,
        identity_text: str
    ) -> None:
        """Store system identity"""
        self._backend.store_identity(embedding, identity_text)
