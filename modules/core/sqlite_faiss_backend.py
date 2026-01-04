"""
SQLite + FAISS Memory Backend Plugin

Wraps the existing MemoryStore as a plugin-compatible backend.
"""
import numpy as np
from typing import List, Dict, Any, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.base import BaseMemoryBackend
from memory_store import MemoryStore
from config import MemoryConfig


class SQLiteFAISSBackend(BaseMemoryBackend):
    """
    SQLite + FAISS memory backend plugin.
    
    Wraps the existing MemoryStore implementation.
    """
    
    def __init__(
        self,
        db_path: str = MemoryConfig.DB_PATH,
        max_memories: int = MemoryConfig.MAX_MEMORIES,
        embedding_dim: int = MemoryConfig.EMBEDDING_DIM,
        identity: Optional[str] = None
    ):
        """
        Initialize SQLite + FAISS backend.
        
        Args:
            db_path: Path to SQLite database
            max_memories: Maximum number of memories
            embedding_dim: Embedding dimension
            identity: Optional identity text
        """
        self._backend = MemoryStore(
            db_path=db_path,
            max_memories=max_memories,
            embedding_dim=embedding_dim,
            identity=identity
        )
    
    def store(self, embedding: np.ndarray, text: str) -> None:
        """Store a memory."""
        self._backend.store(embedding, text)
    
    def retrieve(self, embedding: np.ndarray, k: int = 3) -> List[str]:
        """Retrieve k most similar memories."""
        return self._backend.retrieve(embedding, k=k)
    
    def clear(self) -> None:
        """Clear all memories."""
        self._backend.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics."""
        return self._backend.get_stats()
    
    def get_identity(self) -> Optional[str]:
        """Get the stored identity text."""
        return self._backend.get_identity()
    
    @property
    def embedding_dim(self) -> int:
        """Embedding dimension."""
        return self._backend.embedding_dim

