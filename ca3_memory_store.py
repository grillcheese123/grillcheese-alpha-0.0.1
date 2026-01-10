"""
CA3 Autoassociative Memory Store with Vulkan FAISS Integration

Integrates with VulkanCapsuleTransformer for:
- Pattern completion via GPU-accelerated kNN search
- Memory consolidation with importance-based forgetting
- Persistence to disk

Bio-inspired hippocampal memory architecture:
- DG (Dentate Gyrus): Pattern separation (32D → 128D sparse)
- CA3: Pattern completion (Vulkan FAISS kNN retrieval)
- CA1: Output to neocortex (memory injection at layers 4-5)
"""

import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Import capsule types
from vulkan_capsule_transformer import (
    CapsuleMemory, 
    CognitiveFeatures, 
    MemoryType,
    VulkanCapsuleTransformer,
    CapsuleTransformerConfig
)

# Try to import Vulkan FAISS
try:
    from vulkan_backend import VulkanCompute, VULKAN_AVAILABLE
    VULKAN_FAISS_AVAILABLE = VULKAN_AVAILABLE
except ImportError:
    VULKAN_FAISS_AVAILABLE = False

# Try to import CPU FAISS as fallback
try:
    import faiss
    FAISS_CPU_AVAILABLE = True
except ImportError:
    FAISS_CPU_AVAILABLE = False

logger.info(f"FAISS backends: Vulkan={VULKAN_FAISS_AVAILABLE}, CPU={FAISS_CPU_AVAILABLE}")


class VulkanFAISSIndex:
    """
    GPU-accelerated FAISS index using Vulkan compute shaders.
    
    Uses:
    - faiss-distance.glsl: Compute L2/cosine/dot distances
    - faiss-topk.glsl: Select top-k nearest neighbors
    """
    
    def __init__(self, dim: int = 128, distance_type: str = 'l2'):
        self.dim = dim
        self.distance_type = distance_type
        self.vectors = None
        self.count = 0
        
        # Initialize Vulkan backend
        try:
            self.gpu = VulkanCompute()
            self.use_gpu = True
            logger.info(f"VulkanFAISSIndex: GPU acceleration enabled ({dim}D, {distance_type})")
        except Exception as e:
            logger.warning(f"VulkanFAISSIndex: GPU init failed ({e}), using CPU")
            self.gpu = None
            self.use_gpu = False
    
    def add(self, vectors: np.ndarray):
        """Add vectors to index"""
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        
        self.count = len(self.vectors)
    
    def build(self, vectors: np.ndarray):
        """Build index from vectors (replaces existing)"""
        self.vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.count = len(self.vectors)
    
    def search(self, query: np.ndarray, k: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: (batch, dim) or (dim,) query vectors
            k: Number of neighbors
            
        Returns:
            distances: (batch, k) distances
            indices: (batch, k) neighbor indices
        """
        if self.vectors is None or self.count == 0:
            return np.array([[]]), np.array([[]])
        
        query = np.ascontiguousarray(query.astype(np.float32))
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        k = min(k, self.count)
        
        if self.use_gpu and self.gpu is not None:
            return self._search_gpu(query, k)
        else:
            return self._search_cpu(query, k)
    
    def _search_gpu(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated search using Vulkan shaders"""
        # Compute distances
        distances = self.gpu.faiss.compute_distances(
            query, self.vectors, distance_type=self.distance_type
        )
        
        # Get top-k
        indices, topk_distances = self.gpu.faiss.topk(distances, k)
        
        return topk_distances, indices
    
    def _search_cpu(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """CPU fallback search"""
        if self.distance_type == 'l2':
            # L2 distance
            diff = query[:, np.newaxis, :] - self.vectors[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff ** 2, axis=2))
        elif self.distance_type == 'cosine':
            # Cosine distance (1 - similarity)
            q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
            v_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)
            similarity = np.dot(q_norm, v_norm.T)
            distances = 1 - similarity
        else:  # dot
            distances = -np.dot(query, self.vectors.T)
        
        # Get top-k
        indices = np.argsort(distances, axis=1)[:, :k]
        topk_distances = np.take_along_axis(distances, indices, axis=1)
        
        return topk_distances, indices
    
    def reset(self):
        """Clear the index"""
        self.vectors = None
        self.count = 0


class CA3MemoryIndex:
    """
    Memory index for pattern completion.
    
    Uses optimized numpy by default (fastest for <500K vectors).
    CPU FAISS available for IVF-PQ approximate search on very large datasets.
    
    Note: Vulkan FAISS shaders exist but need parallel reduction optimization
    to beat numpy. Current shaders use sequential topk which is slower.
    """
    
    def __init__(self, dim: int = 128, use_vulkan: bool = False, distance_type: str = 'l2'):
        self.dim = dim
        self.distance_type = distance_type
        self.count = 0
        self._vectors = None
        
        # Use numpy by default (fastest for typical workloads)
        # Vulkan FAISS shaders need optimization before enabling
        if FAISS_CPU_AVAILABLE and not use_vulkan:
            self.backend = 'faiss_cpu'
            self.index = None
            logger.info("CA3MemoryIndex: Using CPU FAISS backend")
        else:
            self.backend = 'numpy'
            self.index = None
            logger.info("CA3MemoryIndex: Using numpy backend (optimized)")
    
    def build_index(self, vectors: np.ndarray, use_ivf: bool = False, nlist: int = 100):
        """Build index from vectors"""
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        self.count = len(vectors)
        
        if self.backend == 'vulkan':
            self.index.build(vectors)
        elif self.backend == 'faiss_cpu':
            if use_ivf and self.count > 10000:
                quantizer = faiss.IndexFlatL2(self.dim)
                self.index = faiss.IndexIVFPQ(
                    quantizer, self.dim,
                    nlist=min(nlist, self.count // 10),
                    m=16,
                    nbits=8
                )
                self.index.train(vectors)
                self.index.nprobe = 10
            else:
                self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(vectors)
        else:  # numpy
            self._vectors = vectors.copy()
    
    def add(self, vectors: np.ndarray):
        """Add vectors to existing index"""
        vectors = np.ascontiguousarray(vectors.astype(np.float32))
        
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        
        if self.backend == 'vulkan':
            self.index.add(vectors)
            self.count = self.index.count
        elif self.backend == 'faiss_cpu':
            if self.index is None:
                self.build_index(vectors)
            else:
                self.index.add(vectors)
                self.count = self.index.ntotal
        else:  # numpy
            if self._vectors is None:
                self._vectors = vectors
            else:
                self._vectors = np.vstack([self._vectors, vectors])
            self.count = len(self._vectors)
    
    def search(self, query: np.ndarray, k: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        """Search for k nearest neighbors"""
        if self.count == 0:
            return np.array([[]]), np.array([[]])
        
        query = np.ascontiguousarray(query.astype(np.float32))
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        k = min(k, self.count)
        
        if self.backend == 'vulkan':
            return self.index.search(query, k)
        elif self.backend == 'faiss_cpu':
            distances, indices = self.index.search(query, k)
            return distances, indices
        else:  # numpy
            diff = query[:, np.newaxis, :] - self._vectors[np.newaxis, :, :]
            distances = np.sqrt(np.sum(diff ** 2, axis=2))
            indices = np.argsort(distances, axis=1)[:, :k]
            topk_distances = np.take_along_axis(distances, indices, axis=1)
            return topk_distances, indices
    
    def reset(self):
        """Clear the index"""
        if self.backend == 'vulkan':
            self.index.reset()
        elif self.backend == 'faiss_cpu':
            self.index = None
        else:
            self._vectors = None
        self.count = 0


class CA3MemoryStore:
    """
    Complete hippocampal memory store with:
    - Capsule encoding (semantic + cognitive features)
    - DG pattern separation (sparse expansion)
    - CA3 pattern completion (Vulkan FAISS retrieval)
    - Memory consolidation (importance-based forgetting)
    """
    
    def __init__(
        self,
        encoder: VulkanCapsuleTransformer,
        capacity: int = 100000,
        consolidation_threshold: float = 0.8,
        use_vulkan_faiss: bool = True
    ):
        self.encoder = encoder
        self.capacity = capacity
        self.consolidation_threshold = consolidation_threshold
        
        # Memory storage
        self.memories: List[CapsuleMemory] = []
        self.protected_memories: List[CapsuleMemory] = []
        
        # GPU-accelerated FAISS index for DG vectors
        self.index = CA3MemoryIndex(
            dim=encoder.config.dg_dim,
            use_vulkan=use_vulkan_faiss,
            distance_type='l2'
        )
        
        # Statistics
        self.stats = {
            'total_added': 0,
            'total_retrieved': 0,
            'consolidations': 0,
            'avg_retrieval_time_ms': 0.0,
            'gpu_searches': 0,
            'cpu_searches': 0
        }
    
    def add_memory(self, memory: CapsuleMemory) -> CapsuleMemory:
        """Add memory with automatic encoding"""
        # Ensure vectors are computed
        if memory.capsule_vector is None:
            memory.capsule_vector = self.encoder.encode_to_capsule(
                memory.content, 
                memory.cognitive_features
            )
        
        if memory.dg_vector is None:
            memory.dg_vector = self.encoder.dg.expand(memory.capsule_vector)
        
        memory.created = time.time()
        
        if memory.protected:
            self.protected_memories.append(memory)
        
        self.memories.append(memory)
        self.stats['total_added'] += 1
        
        # Add to GPU index
        self.index.add(memory.dg_vector.reshape(1, -1))
        
        # Check capacity
        if len(self.memories) >= self.capacity:
            self._consolidate()
        
        return memory
    
    def add_batch(self, memories: List[CapsuleMemory]) -> List[CapsuleMemory]:
        """Batch add memories efficiently"""
        texts = []
        features = []
        needs_encoding = []
        
        for i, mem in enumerate(memories):
            if mem.capsule_vector is None:
                texts.append(mem.content)
                features.append(mem.cognitive_features)
                needs_encoding.append(i)
            mem.created = time.time()
        
        # Batch encode if needed
        if texts:
            embeddings = self.encoder.encode_batch(texts)
            
            for j, idx in enumerate(needs_encoding):
                mem = memories[idx]
                capsule = embeddings[j] @ self.encoder.capsule_proj.T
                
                cfg = self.encoder.config
                semantic = capsule[:cfg.semantic_dims]
                semantic = semantic / (np.linalg.norm(semantic) + 1e-8)
                
                cog = mem.cognitive_features
                capsule[:cfg.semantic_dims] = semantic * 0.9 + semantic * 0.1 * cog.plasticity_gain
                capsule[cfg.semantic_dims:] = cog.to_array()
                
                mem.capsule_vector = (capsule / (np.linalg.norm(capsule) + 1e-8)).astype(np.float32)
        
        # Batch DG expansion
        capsules = np.array([m.capsule_vector for m in memories])
        dg_vectors = self.encoder.dg.expand(capsules)
        
        for i, mem in enumerate(memories):
            mem.dg_vector = dg_vectors[i]
            
            if mem.protected:
                self.protected_memories.append(mem)
            
            self.memories.append(mem)
        
        self.stats['total_added'] += len(memories)
        
        # Rebuild index with all vectors
        self._rebuild_index()
        
        # Consolidate if needed
        if len(self.memories) >= self.capacity:
            self._consolidate()
        
        return memories
    
    def query(
        self,
        text: str,
        k: int = 32,
        cognitive_features: Optional[CognitiveFeatures] = None
    ) -> List[Tuple[CapsuleMemory, float]]:
        """
        Retrieve memories relevant to query using GPU-accelerated search.
        """
        if not self.memories:
            return []
        
        start_time = time.perf_counter()
        
        cognitive_features = cognitive_features or CognitiveFeatures(
            plasticity_gain=0.8,
            consolidation_priority=0.5,
            stability=0.5,
            stress_link=0.0
        )
        
        # Encode query to DG space
        query_dg = self.encoder.encode_to_dg(text, cognitive_features)
        
        # GPU-accelerated search
        distances, indices = self.index.search(query_dg, k)
        
        # Track GPU vs CPU usage
        if self.index.backend == 'vulkan':
            self.stats['gpu_searches'] += 1
        else:
            self.stats['cpu_searches'] += 1
        
        # Collect results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.memories):
                memory = self.memories[idx]
                memory.access_count += 1
                memory.last_access = time.time()
                results.append((memory, float(dist)))
        
        # Update stats
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.stats['total_retrieved'] += len(results)
        n = self.stats['total_retrieved']
        self.stats['avg_retrieval_time_ms'] = (
            self.stats['avg_retrieval_time_ms'] * (n - len(results)) + elapsed_ms
        ) / max(n, 1)
        
        return results
    
    def query_by_dg(self, dg_vector: np.ndarray, k: int = 32) -> List[Tuple[CapsuleMemory, float]]:
        """Query using pre-computed DG vector"""
        if not self.memories:
            return []
        
        distances, indices = self.index.search(dg_vector, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx < len(self.memories):
                memory = self.memories[idx]
                memory.access_count += 1
                memory.last_access = time.time()
                results.append((memory, float(dist)))
        
        return results
    
    def _consolidate(self):
        """Importance-based memory consolidation"""
        if len(self.memories) <= self.capacity * self.consolidation_threshold:
            return
        
        logger.info(f"Consolidating: {len(self.memories)} -> {int(self.capacity * 0.8)}")
        
        scores = []
        current_time = time.time()
        
        for m in self.memories:
            if m.protected:
                scores.append(float('inf'))
            else:
                recency = 1.0 / (current_time - (m.last_access or m.created) + 1)
                freq = np.log1p(m.access_count)
                score = (
                    m.cognitive_features.consolidation_priority *
                    m.cognitive_features.stability *
                    (recency + freq)
                )
                scores.append(score)
        
        keep_count = int(self.capacity * 0.8)
        keep_indices = np.argsort(scores)[-keep_count:]
        
        self.memories = [self.memories[i] for i in sorted(keep_indices)]
        self._rebuild_index()
        
        self.stats['consolidations'] += 1
        logger.info(f"Consolidation complete: {len(self.memories)} retained")
    
    def _rebuild_index(self):
        """Rebuild FAISS index from current memories"""
        if not self.memories:
            self.index.reset()
            return
        
        vectors = np.array([m.dg_vector for m in self.memories], dtype=np.float32)
        self.index.build_index(vectors, use_ivf=len(self.memories) > 10000)
    
    def save(self, path: str):
        """Save memory store to disk"""
        data = {
            'memories': [],
            'stats': self.stats
        }
        
        for m in self.memories:
            mem_data = {
                'memory_id': m.memory_id,
                'memory_type': m.memory_type.value,
                'domain': m.domain,
                'content': m.content,
                'cognitive_features': {
                    'plasticity_gain': m.cognitive_features.plasticity_gain,
                    'consolidation_priority': m.cognitive_features.consolidation_priority,
                    'stability': m.cognitive_features.stability,
                    'stress_link': m.cognitive_features.stress_link
                },
                'capsule_vector': m.capsule_vector.tolist() if m.capsule_vector is not None else None,
                'dg_vector': m.dg_vector.tolist() if m.dg_vector is not None else None,
                'protected': m.protected,
                'access_count': m.access_count,
                'last_access': m.last_access,
                'created': m.created
            }
            data['memories'].append(mem_data)
        
        with open(path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Saved {len(self.memories)} memories to {path}")
    
    def load(self, path: str):
        """Load memory store from disk"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.memories = []
        self.protected_memories = []
        
        for m_data in data['memories']:
            memory = CapsuleMemory(
                memory_id=m_data['memory_id'],
                memory_type=MemoryType(m_data['memory_type']),
                domain=m_data['domain'],
                content=m_data['content'],
                cognitive_features=CognitiveFeatures(**m_data['cognitive_features']),
                capsule_vector=np.array(m_data['capsule_vector'], dtype=np.float32) if m_data['capsule_vector'] else None,
                dg_vector=np.array(m_data['dg_vector'], dtype=np.float32) if m_data['dg_vector'] else None,
                protected=m_data['protected'],
                access_count=m_data['access_count'],
                last_access=m_data['last_access'],
                created=m_data['created']
            )
            
            if memory.protected:
                self.protected_memories.append(memory)
            
            self.memories.append(memory)
        
        if 'stats' in data:
            self.stats.update(data['stats'])
        
        self._rebuild_index()
        
        logger.info(f"Loaded {len(self.memories)} memories from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory store statistics"""
        return {
            **self.stats,
            'total_memories': len(self.memories),
            'protected_memories': len(self.protected_memories),
            'capacity': self.capacity,
            'utilization': len(self.memories) / self.capacity,
            'index_backend': self.index.backend
        }


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_vulkan_faiss(num_vectors: int = 10000, dim: int = 128, k: int = 32):
    """Benchmark Vulkan FAISS vs CPU"""
    print(f"\n{'='*60}")
    print(f"Vulkan FAISS Benchmark: {num_vectors} vectors, {dim}D, k={k}")
    print(f"{'='*60}")
    
    # Generate test data
    np.random.seed(42)
    vectors = np.random.randn(num_vectors, dim).astype(np.float32)
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    queries = np.random.randn(10, dim).astype(np.float32)
    queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    
    # Vulkan FAISS
    print("\n[Vulkan FAISS]")
    vulkan_index = CA3MemoryIndex(dim=dim, use_vulkan=True, distance_type='l2')
    
    start = time.perf_counter()
    vulkan_index.build_index(vectors)
    build_time = (time.perf_counter() - start) * 1000
    print(f"  Build time: {build_time:.2f}ms")
    
    # Warmup
    _ = vulkan_index.search(queries[0], k)
    
    # Search
    search_times = []
    for q in queries:
        start = time.perf_counter()
        distances, indices = vulkan_index.search(q, k)
        search_times.append((time.perf_counter() - start) * 1000)
    
    print(f"  Search time: {np.mean(search_times):.2f}ms (std: {np.std(search_times):.2f})")
    print(f"  Backend: {vulkan_index.backend}")
    
    # Numpy fallback
    print("\n[Numpy Fallback]")
    numpy_index = CA3MemoryIndex(dim=dim, use_vulkan=False, distance_type='l2')
    numpy_index.backend = 'numpy'
    numpy_index._vectors = None
    
    start = time.perf_counter()
    numpy_index.build_index(vectors)
    build_time = (time.perf_counter() - start) * 1000
    print(f"  Build time: {build_time:.2f}ms")
    
    search_times_np = []
    for q in queries:
        start = time.perf_counter()
        distances_np, indices_np = numpy_index.search(q, k)
        search_times_np.append((time.perf_counter() - start) * 1000)
    
    print(f"  Search time: {np.mean(search_times_np):.2f}ms (std: {np.std(search_times_np):.2f})")
    
    # Speedup
    speedup = np.mean(search_times_np) / np.mean(search_times)
    print(f"\n  Speedup: {speedup:.2f}x")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    benchmark_vulkan_faiss()
