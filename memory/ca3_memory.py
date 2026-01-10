"""
CA3 Autoassociative Memory
Autoassociative memory for pattern completion via FAISS kNN
"""
import numpy as np
import logging
import time
from typing import List, Tuple, Optional, Any

from memory.capsule_memory import CapsuleMemory

logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    # Only warn once, not on every import
    import sys
    if 'faiss' not in sys.modules:
        logger.info("FAISS not available, using numpy-based kNN fallback")
        logger.info("  Install with: pip install faiss-cpu (or faiss-gpu for GPU support)")

# Try to import Vulkan FAISS
try:
    from vulkan_backend.vulkan_compute import VulkanCompute
    from vulkan_backend.base import VULKAN_AVAILABLE
    VULKAN_FAISS_AVAILABLE = VULKAN_AVAILABLE
except ImportError:
    VULKAN_FAISS_AVAILABLE = False


class CA3Memory:
    """
    Autoassociative memory for pattern completion.
    Retrieves full memories from partial cues via FAISS kNN.
    """
    
    def __init__(self, dim: int = 128, nlist: int = 100, use_vulkan: bool = False):
        """
        Initialize CA3 memory
        
        Args:
            dim: Dimension of DG-expanded vectors (default: 128)
            nlist: Number of clusters for IVF index (default: 100)
            use_vulkan: Whether to use Vulkan FAISS if available (default: False - disabled for efficiency)
                       Vulkan FAISS is only efficient for large datasets (>1000 memories)
        """
        self.dim = dim
        self.nlist = nlist
        self.index: Optional[Any] = None  # faiss.Index when FAISS is available
        self.memory_store: List[CapsuleMemory] = []
        self._use_faiss = FAISS_AVAILABLE
        self._use_vulkan_faiss = False
        self._vulkan_compute: Optional[Any] = None  # VulkanCompute when available
        
        # Vulkan FAISS disabled by default - only efficient for large datasets
        # Regular FAISS index is faster for typical use cases
        if use_vulkan and VULKAN_FAISS_AVAILABLE:
            try:
                self._vulkan_compute = VulkanCompute()
                self._use_vulkan_faiss = True
                logger.info("Vulkan FAISS available for CA3 memory (will use for large datasets >1000)")
            except Exception as e:
                logger.debug(f"Vulkan FAISS not available: {e}")
                self._use_vulkan_faiss = False
        
        if not self._use_vulkan_faiss and not FAISS_AVAILABLE:
            # Only log at debug level - FAISS is optional
            logger.debug("FAISS not available, using numpy-based kNN (slower)")
    
    def build_index(self, memories: List[CapsuleMemory]):
        """Build FAISS index from DG-expanded vectors."""
        if not memories:
            self.index = None
            self.memory_store = []
            return
        
        # Extract DG vectors
        vectors = []
        valid_memories = []
        for m in memories:
            if m.dg_vector is not None:
                vectors.append(m.dg_vector)
                valid_memories.append(m)
        
        if not vectors:
            logger.warning("No valid DG vectors found, cannot build index")
            self.index = None
            self.memory_store = []
            return
        
        vectors = np.array(vectors).astype('float32')
        
        if self._use_faiss:
            try:
                # IVF-PQ for large-scale retrieval
                if len(valid_memories) > 10000:
                    quantizer = faiss.IndexFlatL2(self.dim)
                    self.index = faiss.IndexIVFPQ(
                        quantizer, self.dim,
                        nlist=self.nlist,
                        m=16,  # 128/8 = 16 subquantizers
                        nbits=8
                    )
                    self.index.train(vectors)
                else:
                    # Flat index for smaller datasets
                    self.index = faiss.IndexFlatL2(self.dim)
                
                self.index.add(vectors)
                self.memory_store = valid_memories
                logger.info(f"Built FAISS index with {len(valid_memories)} memories")
            except Exception as e:
                logger.error(f"FAISS index build failed: {e}, using numpy fallback")
                self._use_faiss = False
                self.index = None
                self.memory_store = valid_memories
        else:
            # Numpy fallback
            self.index = None
            self.memory_store = valid_memories
    
    def retrieve(self, query_dg: np.ndarray, k: int = 32) -> List[Tuple[CapsuleMemory, float]]:
        """
        Pattern completion via kNN search.
        
        Args:
            query_dg: 128D DG-expanded query vector
            k: Number of neighbors to retrieve
        
        Returns:
            List of (CapsuleMemory, distance) tuples
        """
        if not self.memory_store:
            return []
        
        query_dg = query_dg.astype(np.float32).flatten()
        if len(query_dg) != self.dim:
            raise ValueError(f"Expected {self.dim}D query, got {len(query_dg)}D")
        
        # Use Vulkan FAISS only for large datasets (inefficient for small sets)
        # For small datasets, regular FAISS or numpy is faster
        use_vulkan_for_query = self._use_vulkan_faiss and len(self.memory_store) > 1000
        
        if use_vulkan_for_query and self._vulkan_compute is not None:
            try:
                query_dg = query_dg.reshape(1, -1)
                vectors = np.array([m.dg_vector for m in self.memory_store]).astype(np.float32)
                
                # Compute distances using Vulkan (only efficient for large datasets)
                distances = self._vulkan_compute.faiss_compute_distances(
                    queries=query_dg,
                    database=vectors,
                    distance_type='l2'
                )
                
                # Get top-k using Vulkan
                topk_indices, topk_distances = self._vulkan_compute.faiss_topk(
                    distances=distances,
                    k=min(k, len(self.memory_store))
                )
                
                results = []
                for idx, dist in zip(topk_indices[0], topk_distances[0]):
                    idx = int(idx)
                    if 0 <= idx < len(self.memory_store):
                        memory = self.memory_store[idx]
                        memory.access_count += 1
                        memory.last_access = time.time()
                        results.append((memory, float(dist)))
                
                return results
            except Exception as e:
                logger.warning(f"Vulkan FAISS search failed: {e}, falling back to CPU")
                self._use_vulkan_faiss = False
        
        # Try regular FAISS
        if self._use_faiss and self.index is not None:
            try:
                query_dg = query_dg.reshape(1, -1)
                distances, indices = self.index.search(query_dg, min(k, len(self.memory_store)))
                
                results = []
                for dist, idx in zip(distances[0], indices[0]):
                    if 0 <= idx < len(self.memory_store):
                        memory = self.memory_store[idx]
                        memory.access_count += 1
                        memory.last_access = time.time()
                        results.append((memory, float(dist)))
                
                return results
            except Exception as e:
                logger.warning(f"FAISS search failed: {e}, using numpy fallback")
                self._use_faiss = False
        
        # Numpy fallback: compute L2 distances
        query_dg = query_dg.reshape(1, -1)
        vectors = np.array([m.dg_vector for m in self.memory_store]).astype(np.float32)
        
        # Compute L2 distances
        distances = np.linalg.norm(vectors - query_dg, axis=1)
        
        # Get top-k
        top_k_indices = np.argsort(distances)[:min(k, len(self.memory_store))]
        
        results = []
        for idx in top_k_indices:
            memory = self.memory_store[idx]
            memory.access_count += 1
            memory.last_access = time.time()
            results.append((memory, float(distances[idx])))
        
        return results
    
    def rebuild_index(self):
        """Force index rebuild."""
        self.build_index(self.memory_store)
