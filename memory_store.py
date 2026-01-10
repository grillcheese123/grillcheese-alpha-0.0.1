"""
GPU-accelerated Memory Store for episodic memory
Uses Vulkan compute shaders for all memory operations
"""
import json
import logging
import sqlite3
import numpy as np
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Callable

from config import MemoryConfig, LogConfig

# Configure logging
logging.basicConfig(level=LogConfig.LEVEL, format=LogConfig.FORMAT)
logger = logging.getLogger(__name__)


class MemoryStore:
    """GPU-accelerated persistent memory store using Vulkan shaders"""
    
    def __init__(
        self,
        db_path: str = MemoryConfig.DB_PATH,
        max_memories: int = MemoryConfig.MAX_MEMORIES,
        embedding_dim: int = MemoryConfig.EMBEDDING_DIM,
        identity: Optional[str] = None,
        use_hilbert: bool = None
    ):
        """
        Initialize memory store
        
        Args:
            db_path: Path to SQLite database for persistence
            max_memories: Maximum number of memories to store
            embedding_dim: Dimension of embeddings
            identity: Optional initial identity text for the AI
            use_hilbert: Enable Hilbert Multiverse Routing (default: from MemoryConfig)
        """
        self.db_path = Path(db_path)
        self.max_memories = max_memories
        self.embedding_dim = embedding_dim
        self.identity_text = identity
        
        # Hilbert routing (optional enhancement)
        if use_hilbert is None:
            use_hilbert = MemoryConfig.USE_HILBERT_ROUTING
        self.use_hilbert = use_hilbert
        self.hilbert_store = None
        
        if self.use_hilbert:
            try:
                from hilbert_routing import HilbertMemoryStore
                self.hilbert_store = HilbertMemoryStore(
                    embedding_dim=self.embedding_dim,
                    universe=MemoryConfig.HILBERT_UNIVERSE
                )
                logger.info(f"{LogConfig.CHECK} Hilbert Multiverse Routing enabled (universe: {MemoryConfig.HILBERT_UNIVERSE})")
            except ImportError as e:
                logger.warning(f"{LogConfig.WARNING} Hilbert routing not available: {e}, using standard similarity")
                self.use_hilbert = False
            except Exception as e:
                logger.warning(f"{LogConfig.WARNING} Hilbert routing init failed: {e}, using standard similarity")
                self.use_hilbert = False
        
        # Initialize GPU backend
        try:
            from vulkan_backend import VulkanCompute
            self.gpu = VulkanCompute()
            self._use_gpu_similarity = True
            logger.info(f"{LogConfig.CHECK} GPU memory store initialized")
        except Exception as e:
            logger.warning(f"{LogConfig.WARNING} GPU init failed: {e}, using CPU fallback")
            self.gpu = None
            self._use_gpu_similarity = False
        
        # Initialize database
        self._init_database()
        
        # Initialize GPU memory buffers
        self.memory_keys: Optional[np.ndarray] = None
        self.memory_values: Optional[np.ndarray] = None
        self.memory_texts: List[str] = []
        self.identity_index: int = -1
        self.next_write_index: int = 0
        self.num_memories: int = 0
        
        # Async stats update queue (batched for performance)
        self._access_stats_queue = deque()
        self._stats_batch_size = 100
        self._stats_flush_interval = 1.0  # seconds
        self._stats_lock = threading.Lock()
        self._stats_thread = None
        self._stats_stop_event = threading.Event()
        self._start_stats_worker()
        
        # Load existing memories
        self._load_memories()
        
        # Initialize Hilbert store with existing memories if enabled
        if self.use_hilbert and self.hilbert_store and self.num_memories > 0:
            self._sync_to_hilbert_store()
        
        # Store identity if provided and not already stored
        if identity and self.identity_index == -1:
            self.set_identity(identity)
    
    def _init_database(self) -> None:
        """Initialize SQLite database for persistent storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB NOT NULL,
                text TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                metadata TEXT,
                is_identity INTEGER DEFAULT 0,
                is_protected INTEGER DEFAULT 0
            )
        """)
        
        # Add columns if they don't exist (for existing databases)
        for column, default in [
            ("is_identity", "INTEGER DEFAULT 0"),
            ("is_protected", "INTEGER DEFAULT 0")
        ]:
            try:
                cursor.execute(f"ALTER TABLE memories ADD COLUMN {column} {default}")
            except sqlite3.OperationalError:
                pass  # Column already exists
        
        # Indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_identity ON memories(is_identity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_protected ON memories(is_protected)")
        
        conn.commit()
        conn.close()
        logger.debug("Database initialized")
    
    def _load_memories(self) -> None:
        """Load memories from database into GPU memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load identity memory first, then others
        cursor.execute("SELECT embedding, text, is_identity FROM memories ORDER BY is_identity DESC, id")
        rows = cursor.fetchall()
        conn.close()
        
        # Initialize GPU buffers
        self.memory_keys = np.zeros((self.max_memories, self.embedding_dim), dtype=np.float32)
        self.memory_values = np.zeros((self.max_memories, self.embedding_dim), dtype=np.float32)
        
        if len(rows) == 0:
            self.num_memories = 0
            self.next_write_index = 0
            self.identity_index = -1
            logger.debug("No existing memories to load")
            return
        
        # Load embeddings from database
        embeddings = []
        texts = []
        self.identity_index = -1
        
        for embedding_blob, text, is_identity in rows:
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            if len(embedding) == self.embedding_dim:
                embeddings.append(embedding)
                texts.append(text)
                if is_identity:
                    self.identity_index = len(embeddings) - 1
                    self.identity_text = text
        
        self.num_memories = len(embeddings)
        
        if self.num_memories > 0:
            # Copy embeddings to GPU buffers
            for i, emb in enumerate(embeddings):
                self.memory_keys[i] = emb
                self.memory_values[i] = emb
            
            self.memory_texts = texts
            self.next_write_index = self.num_memories % self.max_memories
        
        logger.info(f"Loaded {self.num_memories} memories into GPU")
    
    def _sync_to_hilbert_store(self) -> None:
        """Sync existing memories to Hilbert store"""
        if not self.use_hilbert or not self.hilbert_store:
            return
        
        try:
            for i in range(self.num_memories):
                if i < len(self.memory_texts):
                    memory_id = str(i)
                    embedding = self.memory_keys[i]
                    text = self.memory_texts[i]
                    
                    # Get metadata from database if available
                    metadata = None
                    try:
                        conn = sqlite3.connect(self.db_path)
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT metadata FROM memories WHERE id = ?",
                            (i + 1,)  # SQLite IDs are 1-indexed
                        )
                        row = cursor.fetchone()
                        if row and row[0]:
                            metadata = json.loads(row[0])
                        conn.close()
                    except Exception:
                        pass
                    
                    self.hilbert_store.add_memory(
                        memory_id=memory_id,
                        embedding=embedding,
                        text=text,
                        metadata=metadata
                    )
            
            logger.debug(f"Synced {self.num_memories} memories to Hilbert store")
        except Exception as e:
            logger.warning(f"Failed to sync memories to Hilbert store: {e}")
    
    def set_identity(self, identity_text: str, identity_embedding: Optional[np.ndarray] = None) -> None:
        """
        Store system identity - always included in context
        
        Args:
            identity_text: Identity description text
            identity_embedding: Pre-computed embedding (optional)
        """
        self.identity_text = identity_text
        logger.debug(f"Identity text set: {identity_text[:50]}...")
    
    def store_identity(self, embedding: np.ndarray, identity_text: str) -> None:
        """
        Store system identity with embedding - replaces existing identity if any
        
        Args:
            embedding: Identity embedding vector (embedding_dim,)
            identity_text: Identity description text
        """
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
        
        embedding = embedding.astype(np.float32).flatten()
        
        # Initialize GPU buffers if they don't exist
        if self.memory_keys is None:
            self.memory_keys = np.zeros((self.max_memories, self.embedding_dim), dtype=np.float32)
            self.memory_values = np.zeros((self.max_memories, self.embedding_dim), dtype=np.float32)
            self.num_memories = 0
            self.next_write_index = 0
        
        # Store in database with is_identity flag
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete existing identity if any
        cursor.execute("DELETE FROM memories WHERE is_identity = 1")
        
        # Insert new identity memory
        embedding_blob = embedding.tobytes()
        timestamp = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO memories (embedding, text, timestamp, is_identity)
            VALUES (?, ?, ?, 1)
        """, (embedding_blob, identity_text, timestamp))
        
        conn.commit()
        conn.close()
        
        # Determine write index for GPU
        if self.identity_index >= 0 and self.identity_index < self.num_memories:
            write_index = self.identity_index
        else:
            if self.num_memories < self.max_memories:
                write_index = self.num_memories
                self.num_memories += 1
            else:
                write_index = self.next_write_index
                self.next_write_index = (self.next_write_index + 1) % self.max_memories
            
            self.identity_index = write_index
        
        # Write to GPU memory
        if self.gpu is not None:
            updated_keys, updated_values = self.gpu.memory_write(
                embedding, embedding,
                self.memory_keys, self.memory_values,
                write_index, write_mode=0
            )
            self.memory_keys = updated_keys
            self.memory_values = updated_values
        else:
            self.memory_keys[write_index] = embedding
            self.memory_values[write_index] = embedding
        
        # Update text list
        if write_index < len(self.memory_texts):
            self.memory_texts[write_index] = identity_text
        else:
            self.memory_texts.append(identity_text)
        
        self.identity_text = identity_text
        logger.info(f"{LogConfig.CHECK} Identity stored at index {write_index}")
    
    def store(self, embedding: np.ndarray, text: str, metadata: Optional[Dict[str, Any]] = None, is_protected: bool = False) -> None:
        """
        Store a new memory using GPU shaders with atomic operations.
        
        Args:
            embedding: Embedding vector (embedding_dim,)
            text: Associated text content
            metadata: Optional metadata dictionary
            is_protected: If True, memory will never be pruned
        """
        if len(embedding) != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {len(embedding)}")
        
        embedding = embedding.astype(np.float32).flatten()
        
        # Determine write index first (before any operations)
        if self.num_memories < self.max_memories:
            write_index = self.num_memories
            self.num_memories += 1
        else:
            write_index = self.next_write_index
            self.next_write_index = (self.next_write_index + 1) % self.max_memories
        
        # Write to GPU memory first (before database commit)
        try:
            if self.gpu is not None:
                updated_keys, updated_values = self.gpu.memory_write(
                    embedding, embedding,
                    self.memory_keys, self.memory_values,
                    write_index, write_mode=0
                )
                self.memory_keys = updated_keys
                self.memory_values = updated_values
            else:
                self.memory_keys[write_index] = embedding
                self.memory_values[write_index] = embedding
        except Exception as e:
            logger.error(f"GPU memory write failed: {e}")
            # Rollback: restore previous state
            if self.num_memories < self.max_memories:
                self.num_memories -= 1
            else:
                self.next_write_index = (self.next_write_index - 1) % self.max_memories
            raise
        
        # Only commit to database after successful GPU write
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = embedding.tobytes()
        metadata_json = json.dumps(metadata) if metadata else None
        timestamp = datetime.now().isoformat()
        
        try:
            cursor.execute("""
                INSERT INTO memories (embedding, text, timestamp, metadata, is_identity, is_protected)
                VALUES (?, ?, ?, ?, 0, ?)
            """, (embedding_blob, text, timestamp, metadata_json, 1 if is_protected else 0))
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database write failed: {e}")
            # Note: GPU write already succeeded, but database failed
            # This is acceptable - GPU state is primary, DB is for persistence
            raise
        finally:
            conn.close()
        
        # Update text list
        if write_index < len(self.memory_texts):
            self.memory_texts[write_index] = text
        else:
            self.memory_texts.append(text)
        
        # Also add to Hilbert store if enabled
        if self.use_hilbert and self.hilbert_store:
            try:
                memory_id = str(write_index)
                self.hilbert_store.add_memory(
                    memory_id=memory_id,
                    embedding=embedding,
                    text=text,
                    metadata=metadata
                )
            except Exception as e:
                logger.debug(f"Hilbert store update failed (non-critical): {e}")
        
        logger.debug(f"Memory stored at index {write_index}")
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        k: int = MemoryConfig.DEFAULT_K,
        include_identity: bool = True,
        reranker: Optional[Callable[[str, List[str]], List[float]]] = None,
        query_text: Optional[str] = None,
        emotion_bias: Optional[Dict[str, float]] = None,
        temporal_bias: Optional[Dict[int, float]] = None
    ) -> List[str]:
        """
        Retrieve top-K similar memories using GPU similarity search with optional reranking
        
        Args:
            query_embedding: Query embedding vector (embedding_dim,)
            k: Number of memories to retrieve
            include_identity: If True, always include identity memory as first item
            reranker: Optional function(query_text, candidate_texts) -> scores
            query_text: Optional query text for reranking
            emotion_bias: Optional dict mapping memory indices to emotion-based scores
            temporal_bias: Optional dict mapping memory indices to temporal recency scores
        
        Returns:
            List of text strings from retrieved memories (reranked if reranker provided)
        """
        if self.num_memories == 0:
            return []
        
        # Always include identity if it exists
        identity_text = None
        if include_identity and self.identity_index >= 0 and self.identity_index < len(self.memory_texts):
            identity_text = self.memory_texts[self.identity_index]
            k_adj = k - 1 if k > 1 else k
        else:
            k_adj = k
        
        query_embedding = query_embedding.astype(np.float32).flatten()
        
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(f"Query embedding dimension mismatch: expected {self.embedding_dim}, got {len(query_embedding)}")
        
        # Use Hilbert routing if enabled
        if self.use_hilbert and self.hilbert_store:
            try:
                # Retrieve more candidates if reranking
                initial_k = k_adj * 2 if reranker else k_adj
                
                # Use Hilbert similarity search
                hilbert_results = self.hilbert_store.search(query_embedding, k=initial_k)
                
                # Convert to indices
                top_k_indices = []
                for mem_id, sim, data in hilbert_results:
                    try:
                        idx = int(mem_id)
                        if 0 <= idx < self.num_memories:
                            top_k_indices.append(idx)
                    except ValueError:
                        continue
                
                # Fallback to standard if Hilbert didn't return enough results
                if len(top_k_indices) < k_adj:
                    logger.debug("Hilbert search returned insufficient results, falling back to standard")
                    active_keys = self.memory_keys[:self.num_memories]
                    top_k_indices = self._compute_top_k_similarity(query_embedding, active_keys, initial_k)
            except Exception as e:
                logger.warning(f"Hilbert search failed: {e}, falling back to standard similarity")
                active_keys = self.memory_keys[:self.num_memories]
                initial_k = k_adj * 2 if reranker else k_adj
                top_k_indices = self._compute_top_k_similarity(query_embedding, active_keys, initial_k)
        else:
            # Standard similarity search
            active_keys = self.memory_keys[:self.num_memories]
            initial_k = k_adj * 2 if reranker else k_adj
            top_k_indices = self._compute_top_k_similarity(query_embedding, active_keys, initial_k)
        
        # Exclude identity from top-K if we're including it separately
        if include_identity and self.identity_index >= 0:
            top_k_indices = [idx for idx in top_k_indices if idx != self.identity_index]
        
        # Get candidate texts
        candidate_texts = [
            self.memory_texts[idx] 
            for idx in top_k_indices 
            if idx < len(self.memory_texts)
        ]
        candidate_indices = [idx for idx in top_k_indices if idx < len(self.memory_texts)]
        
        # Apply reranking if provided
        if reranker and query_text and candidate_texts:
            try:
                rerank_scores = reranker(query_text, candidate_texts)
                if len(rerank_scores) == len(candidate_texts):
                    # Apply emotion and temporal bias if provided
                    final_scores = []
                    for i, (text, idx) in enumerate(zip(candidate_texts, candidate_indices)):
                        score = rerank_scores[i]
                        # Apply emotion bias
                        if emotion_bias and idx in emotion_bias:
                            score += emotion_bias[idx] * 0.2  # 20% weight
                        # Apply temporal bias
                        if temporal_bias and idx in temporal_bias:
                            score += temporal_bias[idx] * 0.1  # 10% weight
                        final_scores.append((score, text, idx))
                    
                    # Sort by final score and take top k
                    final_scores.sort(reverse=True, key=lambda x: x[0])
                    candidate_texts = [text for _, text, _ in final_scores[:k_adj]]
                    candidate_indices = [idx for _, _, idx in final_scores[:k_adj]]
            except Exception as e:
                logger.warning(f"Reranking failed: {e}, using original ranking")
        
        # Build result list
        retrieved_texts = []
        if include_identity and identity_text:
            retrieved_texts.append(identity_text)
        retrieved_texts.extend(candidate_texts)
        
        # Update access statistics
        self._update_access_stats(candidate_indices)
        
        return retrieved_texts
    
    def _compute_top_k_similarity(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        k: int
    ) -> List[int]:
        """
        Compute top-K most similar memory indices.
        Uses GPU FAISS shaders when available, falls back to CPU.
        """
        if self._use_gpu_similarity and self.gpu is not None:
            try:
                return self._gpu_faiss_topk(query, keys, k)
            except Exception as e:
                logger.warning(f"GPU FAISS failed: {e}, using CPU fallback")
        
        # CPU fallback
        return self._cpu_top_k_similarity(query, keys, k)
    
    def _gpu_faiss_topk(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        k: int
    ) -> List[int]:
        """
        GPU-accelerated top-K using FAISS distance and topk shaders.
        """
        query_batch = query.reshape(1, -1)
        
        distances = self.gpu.faiss_compute_distances(
            query_batch, keys, distance_type='cosine'
        )
        
        topk_indices, _ = self.gpu.faiss_topk(distances, k)
        
        return topk_indices[0].tolist()
    
    def _gpu_top_k_similarity(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        k: int
    ) -> List[int]:
        """
        Optimized vectorized top-K similarity using NumPy.
        Kept as fallback between full GPU and CPU methods.
        """
        query_norm = query / (np.linalg.norm(query) + 1e-8)
        key_norms = np.linalg.norm(keys, axis=1, keepdims=True) + 1e-8
        keys_normalized = keys / key_norms
        
        similarities = keys_normalized @ query_norm
        
        if k >= len(similarities):
            top_k_indices = np.argsort(similarities)[::-1].tolist()
        else:
            top_k_indices = np.argpartition(similarities, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]].tolist()
        
        return top_k_indices
    
    def _cpu_top_k_similarity(
        self,
        query: np.ndarray,
        keys: np.ndarray,
        k: int
    ) -> List[int]:
        """CPU fallback for top-K similarity"""
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []
        
        query_normalized = query / query_norm
        
        # Compute cosine similarities
        similarities = []
        for i in range(len(keys)):
            key = keys[i]
            key_norm = np.linalg.norm(key)
            if key_norm > 0:
                similarity = np.dot(query_normalized, key / key_norm)
                similarities.append((similarity, i))
        
        # Sort and get top-K
        similarities.sort(reverse=True, key=lambda x: x[0])
        return [idx for _, idx in similarities[:k]]
    
    def get_identity(self) -> Optional[str]:
        """Get current system identity text"""
        return self.identity_text
    
    def _start_stats_worker(self):
        """Start background thread for batched stats updates"""
        def worker():
            while not self._stats_stop_event.is_set():
                self._stats_stop_event.wait(self._stats_flush_interval)
                
                if self._stats_stop_event.is_set():
                    break
                
                # Flush batch if ready
                with self._stats_lock:
                    if len(self._access_stats_queue) >= self._stats_batch_size:
                        self._flush_stats_batch()
                    elif len(self._access_stats_queue) > 0:
                        # Also flush if we've been waiting long enough
                        self._flush_stats_batch()
        
        self._stats_thread = threading.Thread(target=worker, daemon=True)
        self._stats_thread.start()
    
    def _flush_stats_batch(self):
        """Flush batched stats updates to database"""
        if not self._access_stats_queue:
            return
        
        batch = []
        while len(batch) < self._stats_batch_size and self._access_stats_queue:
            batch.append(self._access_stats_queue.popleft())
        
        if not batch:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            timestamp = datetime.now().isoformat()
            
            cursor.executemany("""
                UPDATE memories 
                SET access_count = access_count + 1, last_accessed = ?
                WHERE id = ?
            """, [(timestamp, idx + 1) for idx in batch])
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to update access stats: {e}")
    
    def _update_access_stats(self, indices: List[int]) -> None:
        """Queue access statistics updates for batched processing"""
        if not indices:
            return
        
        with self._stats_lock:
            self._access_stats_queue.extend(indices)
            
            # Flush immediately if batch is full
            if len(self._access_stats_queue) >= self._stats_batch_size:
                self._flush_stats_batch()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM memories")
        total_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(access_count) FROM memories")
        total_accesses = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_memories': total_count,
            'gpu_memories': self.num_memories,
            'max_memories': self.max_memories,
            'total_accesses': total_accesses,
            'embedding_dim': self.embedding_dim,
            'gpu_enabled': self._use_gpu_similarity
        }
    
    def __del__(self):
        """Cleanup: stop stats worker thread"""
        if hasattr(self, '_stats_stop_event'):
            self._stats_stop_event.set()
            if hasattr(self, '_stats_thread') and self._stats_thread:
                self._stats_thread.join(timeout=2.0)
            # Flush any remaining stats
            with self._stats_lock:
                self._flush_stats_batch()
    
    def clear(self, include_protected: bool = False) -> None:
        """
        Clear all memories (use with caution!)
        
        Args:
            include_protected: If True, also delete protected memories. Default False.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if include_protected:
            cursor.execute("DELETE FROM memories")
        else:
            # Preserve protected memories and identity
            cursor.execute("DELETE FROM memories WHERE is_protected = 0 AND is_identity = 0")
        
        conn.commit()
        conn.close()
        
        # Reset GPU buffers
        self.memory_keys = np.zeros((self.max_memories, self.embedding_dim), dtype=np.float32)
        self.memory_values = np.zeros((self.max_memories, self.embedding_dim), dtype=np.float32)
        self.memory_texts = []
        self.num_memories = 0
        self.next_write_index = 0
        
        # Clear Hilbert store if enabled
        if self.use_hilbert and self.hilbert_store:
            self.hilbert_store.memories.clear()
            self.hilbert_store.psi_cache.clear()
        
        # Reload protected memories if any
        if not include_protected:
            self._reload_protected_memories()
        else:
            self.identity_index = -1
            self.identity_text = None
        
        logger.info(f"{LogConfig.CHECK} Memory store cleared (protected preserved: {not include_protected})")
    
    def _reload_protected_memories(self) -> None:
        """Reload protected memories from database after clear"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, embedding, text, is_identity 
            FROM memories 
            WHERE is_protected = 1 OR is_identity = 1
            ORDER BY id
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        for row_id, embedding_blob, text, is_identity in rows:
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            
            if is_identity:
                self.identity_text = text
                self.identity_index = self.num_memories
            
            # Restore to GPU
            if self.num_memories < self.max_memories:
                write_index = self.num_memories
                self.memory_keys[write_index] = embedding
                self.memory_values[write_index] = embedding
                self.memory_texts.append(text)
                self.num_memories += 1
    
    def export_memories(self, filepath: str) -> None:
        """Export all memories to JSON file"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT text, timestamp, access_count, metadata FROM memories")
        rows = cursor.fetchall()
        conn.close()
        
        memories = []
        for text, timestamp, access_count, metadata_json in rows:
            memory = {
                'text': text,
                'timestamp': timestamp,
                'access_count': access_count
            }
            if metadata_json:
                memory['metadata'] = json.loads(metadata_json)
            memories.append(memory)
        
        with open(filepath, 'w') as f:
            json.dump(memories, f, indent=2)
        
        logger.info(f"Exported {len(memories)} memories to {filepath}")
