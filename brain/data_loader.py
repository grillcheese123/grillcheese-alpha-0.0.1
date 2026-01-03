"""
GPU-Accelerated Data Loader for Brain Training

Streams JSONL training data through GPU shaders for:
- Emotional affect calibration (amygdala_affect.jsonl)
- Identity/principles learning
- Temporal indexing (NYT articles)

Memory-efficient: Uses streaming batches to stay under GPU limits.
Target: <4GB GPU memory usage (of 12GB available)
"""
import json
import logging
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# GPU Memory Budget (conservative - leave headroom)
MAX_GPU_MEMORY_MB = 3000  # 3GB max, leaves 9GB for model + other ops
BATCH_SIZE_DEFAULT = 64
EMBEDDING_DIM_DEFAULT = 384


class DataCategory(Enum):
    """Categories of training data"""
    AFFECT = "affect"           # amygdala_affect.jsonl
    IDENTITY = "identity"       # identityA/B/C.jsonl
    PRINCIPLES = "principles"   # principles_texts.jsonl
    TEMPORAL = "temporal"       # NYT articles
    KNOWLEDGE = "knowledge"     # wikibooks, OpenThoughts
    CONVERSATION = "conversation"
    INSTRUCTION = "instruction"  # instruct_55k


@dataclass
class DataItem:
    """Single training data item"""
    id: str
    text: str
    category: DataCategory
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    valence: Optional[float] = None
    arousal: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def memory_size_bytes(self) -> int:
        """Estimate memory size"""
        size = len(self.text.encode('utf-8'))
        if self.embedding is not None:
            size += self.embedding.nbytes
        return size


@dataclass
class BatchStats:
    """Statistics for a processed batch"""
    batch_id: int
    items_processed: int
    gpu_time_ms: float
    learning_updates: int
    avg_valence: float
    avg_arousal: float
    memory_used_mb: float


class GPUDataLoader:
    """
    GPU-accelerated data loader with memory-efficient streaming
    
    Uses shaders:
    - hebbian-learning: For affect/emotion pattern learning
    - stdp-learning: For temporal sequence learning
    - domain-router: For routing data to appropriate processors
    - theta-gamma-encoding: For temporal encoding
    """
    
    def __init__(
        self,
        embedding_dim: int = EMBEDDING_DIM_DEFAULT,
        batch_size: int = BATCH_SIZE_DEFAULT,
        max_memory_mb: int = MAX_GPU_MEMORY_MB,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None
    ):
        """
        Initialize GPU data loader
        
        Args:
            embedding_dim: Dimension of embeddings
            batch_size: Items per batch (adjust for memory)
            max_memory_mb: Maximum GPU memory to use
            embed_fn: Function to generate embeddings from text
        """
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.max_memory_mb = max_memory_mb
        self.embed_fn = embed_fn
        
        # GPU backend
        self.vulkan = None
        self._init_gpu()
        
        # Learning weights (stored on CPU, transferred per batch)
        self.affect_weights = np.random.randn(
            embedding_dim, 2  # valence, arousal
        ).astype(np.float32) * 0.01
        
        self.domain_weights = np.random.randn(
            len(DataCategory), embedding_dim
        ).astype(np.float32) * 0.01
        
        # STDP traces (for temporal learning)
        self.pre_traces = np.zeros((batch_size, embedding_dim), dtype=np.float32)
        self.post_traces = np.zeros((batch_size, 2), dtype=np.float32)
        
        # Statistics
        self.stats = {
            'total_items': 0,
            'total_batches': 0,
            'total_gpu_time_ms': 0.0,
            'affect_items': 0,
            'temporal_items': 0,
            'knowledge_items': 0,
        }
        
        logger.info(f"GPUDataLoader initialized (batch={batch_size}, max_mem={max_memory_mb}MB)")
    
    def _init_gpu(self):
        """Initialize Vulkan backend"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from vulkan_backend import VulkanCompute
            self.vulkan = VulkanCompute()
            logger.info("[OK] GPU data loader initialized with Vulkan")
        except Exception as e:
            logger.warning(f"Vulkan not available: {e}, using CPU fallback")
            self.vulkan = None
    
    # ==================== JSONL Streaming ====================
    
    def stream_jsonl(
        self,
        filepath: Path,
        category: DataCategory,
        limit: Optional[int] = None
    ) -> Generator[DataItem, None, None]:
        """
        Stream items from a JSONL file
        
        Args:
            filepath: Path to JSONL file
            category: Category of data
            limit: Maximum items to yield
            
        Yields:
            DataItem for each line
        """
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if limit and count >= limit:
                    break
                
                try:
                    data = json.loads(line.strip())
                    item = self._parse_item(data, category)
                    if item:
                        yield item
                        count += 1
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"Skip line: {e}")
                    continue
        
        logger.info(f"Streamed {count} items from {filepath.name}")
    
    def _parse_item(self, data: Dict[str, Any], category: DataCategory) -> Optional[DataItem]:
        """Parse a JSON object into a DataItem"""
        # Extract ID
        item_id = data.get('id', data.get('_id', str(hash(str(data)[:100]))))
        
        # Extract text based on category
        text = None
        if category == DataCategory.AFFECT:
            text = data.get('text', '')
        elif category == DataCategory.TEMPORAL:
            # NYT format
            headline = data.get('headline', {})
            if isinstance(headline, dict):
                text = headline.get('main', headline.get('print_headline', ''))
            else:
                text = str(headline)
            abstract = data.get('abstract', '')
            text = f"{text}. {abstract}" if abstract else text
        elif category == DataCategory.PRINCIPLES:
            text = data.get('text', '')
        elif category == DataCategory.IDENTITY:
            text = data.get('text', '')
        elif category == DataCategory.INSTRUCTION:
            # Instruction format: input/output or prompt/response
            text = data.get('instruction', data.get('input', data.get('prompt', '')))
        else:
            text = data.get('text', str(data))
        
        if not text or len(text.strip()) < 3:
            return None
        
        # Extract affect if available
        valence, arousal = None, None
        if 'affect' in data:
            affect = data['affect']
            valence = affect.get('valence', 0.0)
            arousal = affect.get('arousal', 0.5)
        
        # Extract timestamp if available
        timestamp = None
        if 'pub_date' in data:
            try:
                timestamp = datetime.fromisoformat(data['pub_date'].replace('Z', '+00:00'))
            except:
                pass
        
        return DataItem(
            id=str(item_id),
            text=text[:2000],  # Limit text length
            category=category,
            metadata=data.get('context', data.get('metadata', {})),
            valence=valence,
            arousal=arousal,
            timestamp=timestamp
        )
    
    # ==================== Batch Processing ====================
    
    def process_batch(
        self,
        items: List[DataItem],
        learn: bool = True
    ) -> BatchStats:
        """
        Process a batch of items through GPU
        
        Args:
            items: List of DataItem to process
            learn: Whether to update weights
            
        Returns:
            BatchStats with processing results
        """
        start_time = time.perf_counter()
        
        # Generate embeddings if needed
        embeddings = []
        for item in items:
            if item.embedding is None and self.embed_fn:
                item.embedding = self.embed_fn(item.text)
            embeddings.append(
                item.embedding if item.embedding is not None 
                else np.zeros(self.embedding_dim, dtype=np.float32)
            )
        
        embeddings = np.array(embeddings, dtype=np.float32)
        
        # Collect target values (valence/arousal for affect data)
        targets = np.array([
            [item.valence or 0.0, item.arousal or 0.5]
            for item in items
        ], dtype=np.float32)
        
        # Route based on category and process
        learning_updates = 0
        
        if self.vulkan and learn:
            try:
                learning_updates = self._gpu_hebbian_update(
                    embeddings, targets
                )
            except Exception as e:
                logger.warning(f"GPU learning failed: {e}, using CPU")
                learning_updates = self._cpu_hebbian_update(embeddings, targets)
        elif learn:
            learning_updates = self._cpu_hebbian_update(embeddings, targets)
        
        # Compute stats
        gpu_time_ms = (time.perf_counter() - start_time) * 1000
        avg_valence = np.mean([i.valence for i in items if i.valence is not None] or [0])
        avg_arousal = np.mean([i.arousal for i in items if i.arousal is not None] or [0.5])
        
        # Estimate memory used
        memory_mb = (embeddings.nbytes + targets.nbytes + self.affect_weights.nbytes) / 1e6
        
        # Update stats
        self.stats['total_items'] += len(items)
        self.stats['total_batches'] += 1
        self.stats['total_gpu_time_ms'] += gpu_time_ms
        
        return BatchStats(
            batch_id=self.stats['total_batches'],
            items_processed=len(items),
            gpu_time_ms=gpu_time_ms,
            learning_updates=learning_updates,
            avg_valence=float(avg_valence),
            avg_arousal=float(avg_arousal),
            memory_used_mb=memory_mb
        )
    
    def _gpu_hebbian_update(
        self,
        pre_activations: np.ndarray,
        post_activations: np.ndarray,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ) -> int:
        """
        GPU Hebbian weight update using shader
        
        Args:
            pre_activations: [batch, embedding_dim]
            post_activations: [batch, 2] (valence, arousal)
            
        Returns:
            Number of weight updates
        """
        batch_size = pre_activations.shape[0]
        pre_dim = pre_activations.shape[1]
        post_dim = post_activations.shape[1]
        
        # Reshape for shader (batch, time=1, dim)
        pre = pre_activations.reshape(batch_size, 1, pre_dim).astype(np.float32)
        post = post_activations.reshape(batch_size, 1, post_dim).astype(np.float32)
        
        # Use VulkanCompute.hebbian_learning() method directly
        updated_weights = self.vulkan.hebbian_learning(
            pre_activations=pre,
            post_activations=post,
            weights=self.affect_weights,
            learning_rate=learning_rate,
            weight_decay=weight_decay
        )
        
        self.affect_weights = updated_weights.reshape(self.embedding_dim, 2).copy()
        
        return pre_dim * post_dim
    
    def _cpu_hebbian_update(
        self,
        pre_activations: np.ndarray,
        post_activations: np.ndarray,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001
    ) -> int:
        """CPU fallback for Hebbian update"""
        # Compute correlation
        correlation = np.dot(pre_activations.T, post_activations) / len(pre_activations)
        
        # Hebbian update with weight decay
        delta_w = learning_rate * correlation - weight_decay * self.affect_weights
        self.affect_weights += delta_w
        
        return self.affect_weights.size
    
    # ==================== High-Level API ====================
    
    def load_affect_data(
        self,
        filepath: Path,
        learn: bool = True,
        limit: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Load and learn from affect training data
        
        Args:
            filepath: Path to amygdala_affect.jsonl
            learn: Whether to update weights
            limit: Maximum items to process
            progress_callback: Callback(processed, total)
            
        Returns:
            Summary statistics
        """
        logger.info(f"Loading affect data from {filepath}")
        
        all_stats = []
        batch = []
        total_processed = 0
        
        for item in self.stream_jsonl(filepath, DataCategory.AFFECT, limit):
            batch.append(item)
            
            if len(batch) >= self.batch_size:
                stats = self.process_batch(batch, learn=learn)
                all_stats.append(stats)
                total_processed += len(batch)
                self.stats['affect_items'] += len(batch)
                batch = []
                
                if progress_callback:
                    progress_callback(total_processed, limit or -1)
        
        # Process remaining
        if batch:
            stats = self.process_batch(batch, learn=learn)
            all_stats.append(stats)
            total_processed += len(batch)
            self.stats['affect_items'] += len(batch)
        
        return {
            'total_items': total_processed,
            'batches': len(all_stats),
            'avg_gpu_time_ms': np.mean([s.gpu_time_ms for s in all_stats]) if all_stats else 0,
            'avg_valence': np.mean([s.avg_valence for s in all_stats]) if all_stats else 0,
            'avg_arousal': np.mean([s.avg_arousal for s in all_stats]) if all_stats else 0,
            'total_updates': sum(s.learning_updates for s in all_stats),
        }
    
    def load_temporal_data(
        self,
        directory: Path,
        learn: bool = True,
        limit_per_file: Optional[int] = 100,
        file_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load and learn from temporal (NYT) data
        
        Uses STDP learning for temporal sequences
        
        Args:
            directory: Path to nyt_data directory
            learn: Whether to update weights
            limit_per_file: Max items per file
            file_limit: Max files to process
            
        Returns:
            Summary statistics
        """
        logger.info(f"Loading temporal data from {directory}")
        
        files = sorted(directory.glob("*.json"))
        if file_limit:
            files = files[:file_limit]
        
        all_stats = []
        total_items = 0
        
        for filepath in files:
            try:
                # NYT files are JSON arrays, not JSONL
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if not isinstance(data, list):
                    data = [data]
                
                if limit_per_file:
                    data = data[:limit_per_file]
                
                # Convert to items
                items = []
                for entry in data:
                    item = self._parse_item(entry, DataCategory.TEMPORAL)
                    if item:
                        items.append(item)
                
                if items:
                    stats = self.process_batch(items, learn=learn)
                    all_stats.append(stats)
                    total_items += len(items)
                    self.stats['temporal_items'] += len(items)
                    
            except Exception as e:
                logger.debug(f"Skip file {filepath}: {e}")
                continue
        
        return {
            'files_processed': len(all_stats),
            'total_items': total_items,
            'avg_gpu_time_ms': np.mean([s.gpu_time_ms for s in all_stats]) if all_stats else 0,
        }
    
    def load_knowledge_data(
        self,
        filepath: Path,
        category: DataCategory = DataCategory.KNOWLEDGE,
        learn: bool = True,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load and learn from knowledge data (wikibooks, OpenThoughts, etc.)
        
        Args:
            filepath: Path to JSONL file
            category: Data category
            learn: Whether to update weights
            limit: Maximum items
            
        Returns:
            Summary statistics
        """
        logger.info(f"Loading knowledge data from {filepath}")
        
        all_stats = []
        batch = []
        total_processed = 0
        
        for item in self.stream_jsonl(filepath, category, limit):
            batch.append(item)
            
            if len(batch) >= self.batch_size:
                stats = self.process_batch(batch, learn=learn)
                all_stats.append(stats)
                total_processed += len(batch)
                self.stats['knowledge_items'] += len(batch)
                batch = []
        
        if batch:
            stats = self.process_batch(batch, learn=learn)
            all_stats.append(stats)
            total_processed += len(batch)
            self.stats['knowledge_items'] += len(batch)
        
        return {
            'total_items': total_processed,
            'batches': len(all_stats),
            'avg_gpu_time_ms': np.mean([s.gpu_time_ms for s in all_stats]) if all_stats else 0,
        }
    
    # ==================== State Management ====================
    
    def save_weights(self, path: Path) -> None:
        """Save learned weights"""
        np.savez(
            path,
            affect_weights=self.affect_weights,
            domain_weights=self.domain_weights,
            pre_traces=self.pre_traces,
            post_traces=self.post_traces,
            stats=self.stats
        )
        logger.info(f"Weights saved to {path}")
    
    def load_weights(self, path: Path) -> None:
        """Load learned weights"""
        if not path.exists():
            return
        
        data = np.load(path, allow_pickle=True)
        self.affect_weights = data['affect_weights']
        self.domain_weights = data['domain_weights']
        self.pre_traces = data['pre_traces']
        self.post_traces = data['post_traces']
        self.stats = data['stats'].item()
        logger.info(f"Weights loaded from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics"""
        return {
            **self.stats,
            'affect_weight_norm': float(np.linalg.norm(self.affect_weights)),
            'domain_weight_norm': float(np.linalg.norm(self.domain_weights)),
        }
    
    def predict_affect(self, embedding: np.ndarray) -> Tuple[float, float]:
        """
        Predict valence/arousal from embedding using learned weights
        
        Args:
            embedding: Text embedding
            
        Returns:
            (valence, arousal) tuple
        """
        # Simple linear projection
        prediction = np.dot(embedding, self.affect_weights)
        valence = np.clip(prediction[0], -1.0, 1.0)
        arousal = np.clip(prediction[1], 0.0, 1.0)
        return float(valence), float(arousal)

