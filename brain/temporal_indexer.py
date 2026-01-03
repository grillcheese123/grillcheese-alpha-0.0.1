"""
GPU-Accelerated Temporal Indexer

Uses time cells and place cells for historical timeline understanding.
Designed for indexing NYT articles (1851-2024) and other temporal data.

Features:
- Time cells: Encode dates as temporal firing patterns
- Place cells: Encode locations from text/keywords
- Theta-gamma encoding: Phase-coupled temporal sequences
- Efficient GPU-accelerated indexing
- Memory-efficient batch processing

GPU Memory Budget: <2GB for indexing operations
"""
import json
import logging
import struct
import time as time_module
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# GPU Memory Budget (conservative)
TEMPORAL_MAX_MEMORY_MB = 2000  # 2GB max

# Temporal constants
YEAR_MIN = 1850  # Start year for normalization
YEAR_MAX = 2030  # End year for normalization
YEAR_RANGE = YEAR_MAX - YEAR_MIN


@dataclass
class TemporalRecord:
    """A temporally-indexed record"""
    id: str
    text: str
    timestamp: datetime
    location: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    
    # Computed embeddings
    text_embedding: Optional[np.ndarray] = None
    temporal_encoding: Optional[np.ndarray] = None
    spatial_encoding: Optional[np.ndarray] = None
    
    # Metadata
    source_file: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def normalized_time(self) -> float:
        """Normalize timestamp to [0, 1] range"""
        year_frac = self.timestamp.year + (self.timestamp.month - 1) / 12.0
        return (year_frac - YEAR_MIN) / YEAR_RANGE


@dataclass
class TemporalQuery:
    """Query for temporal search"""
    text: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    location: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    limit: int = 10


@dataclass
class IndexStats:
    """Statistics for the temporal index"""
    total_records: int = 0
    date_range: Tuple[datetime, datetime] = None
    unique_locations: int = 0
    gpu_indexing_time_ms: float = 0.0
    memory_used_mb: float = 0.0


class GPUTemporalIndexer:
    """
    GPU-accelerated temporal indexer using time cells and place cells
    
    Uses shaders:
    - time-cell: Temporal encoding of dates
    - place-cell: Spatial encoding of locations
    - theta-gamma-encoding: Phase-coupled temporal sequences
    - faiss-distance: Similarity search
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        n_time_cells: int = 128,
        n_place_cells: int = 64,
        temporal_width: float = 0.05,  # ~9 years in normalized space
        spatial_width: float = 1.0,
        max_memory_mb: int = TEMPORAL_MAX_MEMORY_MB,
        use_gpu: bool = True,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None
    ):
        """
        Initialize temporal indexer
        
        Args:
            embedding_dim: Dimension of text embeddings
            n_time_cells: Number of time cells for temporal encoding
            n_place_cells: Number of place cells for spatial encoding
            temporal_width: Width of temporal receptive fields
            spatial_width: Width of spatial receptive fields
            max_memory_mb: Maximum GPU memory to use
            use_gpu: Whether to use GPU acceleration
            embed_fn: Function to generate embeddings from text
        """
        self.embedding_dim = embedding_dim
        self.n_time_cells = n_time_cells
        self.n_place_cells = n_place_cells
        self.temporal_width = temporal_width
        self.spatial_width = spatial_width
        self.max_memory_mb = max_memory_mb
        self.use_gpu = use_gpu
        self.embed_fn = embed_fn
        
        # Initialize time cell preferred times (distributed across normalized range)
        self.time_cell_prefs = np.linspace(0.0, 1.0, n_time_cells).astype(np.float32)
        
        # Initialize place cell centers in 2D semantic space
        # We'll map locations to 2D coordinates
        self.place_cell_centers = self._init_place_cells()
        
        # Location to coordinate mapping (learned/defined)
        self.location_coords: Dict[str, np.ndarray] = {}
        self._init_location_coords()
        
        # Index storage
        self.records: List[TemporalRecord] = []
        self.temporal_matrix: Optional[np.ndarray] = None  # [n_records, n_time_cells]
        self.spatial_matrix: Optional[np.ndarray] = None   # [n_records, n_place_cells]
        self.text_matrix: Optional[np.ndarray] = None      # [n_records, embedding_dim]
        
        # GPU backend
        self.vulkan = None
        if use_gpu:
            self._init_gpu()
        
        # Stats
        self.stats = IndexStats()
        
        logger.info(f"TemporalIndexer initialized (time_cells={n_time_cells}, place_cells={n_place_cells})")
    
    def _init_gpu(self):
        """Initialize Vulkan backend"""
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from vulkan_backend import VulkanCompute
            self.vulkan = VulkanCompute()
            logger.info("[OK] Temporal indexer GPU backend initialized")
        except Exception as e:
            logger.debug(f"GPU not available: {e}")
            self.vulkan = None
    
    def _init_place_cells(self) -> np.ndarray:
        """Initialize place cell centers in 2D space"""
        # Grid of place cells covering 2D semantic space
        n_side = int(np.sqrt(self.n_place_cells))
        x = np.linspace(-5, 5, n_side)
        y = np.linspace(-5, 5, n_side)
        xx, yy = np.meshgrid(x, y)
        centers = np.stack([xx.flatten(), yy.flatten()], axis=1)
        
        # Pad if needed
        if len(centers) < self.n_place_cells:
            extra = np.random.randn(self.n_place_cells - len(centers), 2) * 2
            centers = np.vstack([centers, extra])
        
        return centers[:self.n_place_cells].astype(np.float32)
    
    def _init_location_coords(self):
        """Initialize location to coordinate mapping"""
        # Major geographic regions mapped to 2D semantic space
        # This is a simplified mapping - could be learned from data
        self.location_coords = {
            # Americas
            'usa': np.array([0.0, 0.0]),
            'united states': np.array([0.0, 0.0]),
            'new york': np.array([0.5, 0.5]),
            'new york city': np.array([0.5, 0.5]),
            'washington': np.array([0.3, 0.3]),
            'california': np.array([-0.5, 0.0]),
            'los angeles': np.array([-0.5, -0.1]),
            'chicago': np.array([0.2, 0.2]),
            'boston': np.array([0.6, 0.6]),
            'texas': np.array([-0.2, -0.5]),
            'florida': np.array([0.3, -0.5]),
            'canada': np.array([0.0, 1.5]),
            'mexico': np.array([-0.5, -1.5]),
            'brazil': np.array([0.0, -3.0]),
            
            # Europe
            'europe': np.array([3.0, 2.0]),
            'uk': np.array([2.5, 2.5]),
            'england': np.array([2.5, 2.5]),
            'london': np.array([2.5, 2.5]),
            'france': np.array([3.0, 2.0]),
            'paris': np.array([3.0, 2.0]),
            'germany': np.array([3.5, 2.5]),
            'berlin': np.array([3.5, 2.5]),
            'russia': np.array([4.5, 3.0]),
            'moscow': np.array([4.5, 3.0]),
            'italy': np.array([3.0, 1.5]),
            'rome': np.array([3.0, 1.5]),
            'spain': np.array([2.5, 1.0]),
            
            # Asia
            'asia': np.array([-3.0, 2.0]),
            'china': np.array([-3.5, 2.0]),
            'japan': np.array([-4.0, 1.5]),
            'tokyo': np.array([-4.0, 1.5]),
            'india': np.array([-2.5, 0.5]),
            'korea': np.array([-3.8, 2.0]),
            
            # Middle East
            'middle east': np.array([1.5, 0.0]),
            'israel': np.array([1.5, 0.0]),
            'iran': np.array([2.0, 0.5]),
            'iraq': np.array([1.8, 0.3]),
            
            # Africa
            'africa': np.array([2.0, -2.0]),
            'egypt': np.array([2.0, -0.5]),
            'south africa': np.array([2.0, -3.0]),
            
            # Oceania
            'australia': np.array([-4.0, -2.0]),
            'oceania': np.array([-4.0, -2.0]),
        }
    
    def _get_location_coords(self, location: Optional[str]) -> np.ndarray:
        """Get 2D coordinates for a location"""
        if not location:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        loc_lower = location.lower().strip()
        
        # Direct match
        if loc_lower in self.location_coords:
            return self.location_coords[loc_lower].astype(np.float32)
        
        # Partial match
        for key, coords in self.location_coords.items():
            if key in loc_lower or loc_lower in key:
                return coords.astype(np.float32)
        
        # Unknown location - hash to coordinates
        h = hash(loc_lower) % 1000000
        x = (h % 1000) / 100.0 - 5.0
        y = (h // 1000) / 100.0 - 5.0
        return np.array([x, y], dtype=np.float32)
    
    # ==================== Time Cell Encoding ====================
    
    def encode_time(self, normalized_time: float) -> np.ndarray:
        """
        Encode a normalized time value using time cells
        
        Args:
            normalized_time: Time in [0, 1] range
            
        Returns:
            Time cell activations [n_time_cells]
        """
        if self.vulkan and self.use_gpu:
            try:
                return self._encode_time_gpu(normalized_time)
            except Exception as e:
                logger.debug(f"GPU time encoding failed: {e}")
        
        return self._encode_time_cpu(normalized_time)
    
    def _encode_time_gpu(self, normalized_time: float) -> np.ndarray:
        """GPU time cell encoding"""
        # Create buffers
        time_buf = self.vulkan.create_buffer(
            np.array([normalized_time], dtype=np.float32).tobytes(),
            usage='storage'
        )
        pref_buf = self.vulkan.create_buffer(
            self.time_cell_prefs.tobytes(),
            usage='storage'
        )
        output_buf = self.vulkan.create_buffer(
            self.n_time_cells * 4,
            usage='storage'
        )
        mem_buf = self.vulkan.create_buffer(
            np.zeros(self.n_time_cells, dtype=np.float32).tobytes(),
            usage='storage'
        )
        
        # Create pipeline and dispatch
        pipeline = self.vulkan.create_pipeline('time-cell')
        
        push_data = np.array([
            self.n_time_cells, self.temporal_width, 1.0, 0.0,
            0.01, 0.1, 0, 0
        ], dtype=np.float32)
        
        self.vulkan.dispatch(
            pipeline,
            [time_buf, pref_buf, output_buf, mem_buf],
            push_data.tobytes(),
            groups=((self.n_time_cells + 255) // 256, 1, 1)
        )
        
        result = np.frombuffer(
            self.vulkan.read_buffer(output_buf),
            dtype=np.float32
        )[:self.n_time_cells]
        
        return result
    
    def _encode_time_cpu(self, normalized_time: float) -> np.ndarray:
        """CPU fallback for time cell encoding"""
        t_diff = normalized_time - self.time_cell_prefs
        sigma_sq = self.temporal_width ** 2
        activations = np.exp(-t_diff ** 2 / (2.0 * sigma_sq))
        return activations.astype(np.float32)
    
    # ==================== Place Cell Encoding ====================
    
    def encode_location(self, location: Optional[str]) -> np.ndarray:
        """
        Encode a location using place cells
        
        Args:
            location: Location string
            
        Returns:
            Place cell activations [n_place_cells]
        """
        coords = self._get_location_coords(location)
        
        if self.vulkan and self.use_gpu:
            try:
                return self._encode_location_gpu(coords)
            except Exception as e:
                logger.debug(f"GPU place encoding failed: {e}")
        
        return self._encode_location_cpu(coords)
    
    def _encode_location_gpu(self, coords: np.ndarray) -> np.ndarray:
        """GPU place cell encoding"""
        # Create buffers
        pos_buf = self.vulkan.create_buffer(
            coords.astype(np.float32).tobytes(),
            usage='storage'
        )
        centers_buf = self.vulkan.create_buffer(
            self.place_cell_centers.flatten().tobytes(),
            usage='storage'
        )
        output_buf = self.vulkan.create_buffer(
            self.n_place_cells * 4,
            usage='storage'
        )
        
        # Create pipeline and dispatch
        pipeline = self.vulkan.create_pipeline('place-cell')
        
        push_data = np.array([
            self.n_place_cells, 2,  # n_neurons, spatial_dims
            self.spatial_width, 1.0, 0.0  # field_width, max_rate, baseline
        ], dtype=np.float32)
        
        self.vulkan.dispatch(
            pipeline,
            [pos_buf, centers_buf, output_buf],
            push_data.tobytes(),
            groups=((self.n_place_cells + 255) // 256, 1, 1)
        )
        
        result = np.frombuffer(
            self.vulkan.read_buffer(output_buf),
            dtype=np.float32
        )[:self.n_place_cells]
        
        return result
    
    def _encode_location_cpu(self, coords: np.ndarray) -> np.ndarray:
        """CPU fallback for place cell encoding"""
        diffs = self.place_cell_centers - coords
        dist_sq = np.sum(diffs ** 2, axis=1)
        sigma_sq = self.spatial_width ** 2
        activations = np.exp(-dist_sq / (2.0 * sigma_sq))
        return activations.astype(np.float32)
    
    # ==================== Indexing ====================
    
    def index_nyt_directory(
        self,
        directory: Path,
        batch_size: int = 100,
        file_limit: Optional[int] = None,
        items_per_file: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Index NYT articles from a directory
        
        Args:
            directory: Path to nyt_data directory
            batch_size: Records per batch
            file_limit: Maximum files to process
            items_per_file: Maximum items per file
            
        Returns:
            Indexing statistics
        """
        logger.info(f"Indexing NYT data from {directory}")
        start_time = time_module.perf_counter()
        
        files = sorted(directory.glob("*.json"))
        if file_limit:
            files = files[:file_limit]
        
        records_indexed = 0
        
        for filepath in files:
            batch = list(self._stream_nyt_file(filepath, items_per_file))
            if batch:
                self._index_batch(batch)
                records_indexed += len(batch)
        
        # Build matrices for fast retrieval
        self._build_matrices()
        
        elapsed_ms = (time_module.perf_counter() - start_time) * 1000
        
        # Update stats
        self.stats.total_records = len(self.records)
        self.stats.gpu_indexing_time_ms = elapsed_ms
        
        if self.records:
            dates = [r.timestamp for r in self.records if r.timestamp]
            if dates:
                self.stats.date_range = (min(dates), max(dates))
            
            locations = set(r.location for r in self.records if r.location)
            self.stats.unique_locations = len(locations)
        
        logger.info(f"[OK] Indexed {records_indexed} records in {elapsed_ms:.1f}ms")
        
        return {
            'records_indexed': records_indexed,
            'files_processed': len(files),
            'indexing_time_ms': elapsed_ms,
            'date_range': self.stats.date_range,
            'unique_locations': self.stats.unique_locations
        }
    
    def _stream_nyt_file(
        self,
        filepath: Path,
        limit: Optional[int] = None
    ) -> Generator[TemporalRecord, None, None]:
        """Stream records from a NYT JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                data = [data]
            
            if limit:
                data = data[:limit]
            
            for item in data:
                record = self._parse_nyt_item(item, filepath.name)
                if record:
                    yield record
                    
        except Exception as e:
            logger.debug(f"Error reading {filepath}: {e}")
    
    def _parse_nyt_item(
        self,
        item: Dict[str, Any],
        source_file: str
    ) -> Optional[TemporalRecord]:
        """Parse NYT item into TemporalRecord"""
        try:
            # Extract headline
            headline = item.get('headline', {})
            if isinstance(headline, dict):
                text = headline.get('main', headline.get('print_headline', ''))
            else:
                text = str(headline)
            
            abstract = item.get('abstract', '')
            if abstract:
                text = f"{text}. {abstract}"
            
            if not text or len(text) < 5:
                return None
            
            # Extract timestamp
            pub_date = item.get('pub_date', '')
            timestamp = None
            if pub_date:
                try:
                    timestamp = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                except:
                    # Try to parse from filename (e.g., 1945_5.json)
                    parts = source_file.replace('.json', '').split('_')
                    if len(parts) >= 2:
                        year = int(parts[0])
                        month = int(parts[1])
                        timestamp = datetime(year, month, 1)
            
            if not timestamp:
                return None
            
            # Extract location from keywords
            location = None
            keywords = []
            for kw in item.get('keywords', []):
                if isinstance(kw, dict):
                    if kw.get('name') == 'glocations':
                        location = kw.get('value', '')
                    keywords.append(kw.get('value', ''))
            
            return TemporalRecord(
                id=item.get('_id', str(hash(text[:50]))),
                text=text[:1000],  # Limit text length
                timestamp=timestamp,
                location=location,
                keywords=keywords,
                source_file=source_file,
                metadata={
                    'web_url': item.get('web_url', ''),
                    'section': item.get('section_name', ''),
                }
            )
        except Exception as e:
            logger.debug(f"Error parsing item: {e}")
            return None
    
    def _index_batch(self, batch: List[TemporalRecord]) -> None:
        """Index a batch of records"""
        for record in batch:
            # Encode temporal
            norm_time = record.normalized_time()
            record.temporal_encoding = self.encode_time(norm_time)
            
            # Encode spatial
            record.spatial_encoding = self.encode_location(record.location)
            
            # Encode text (if embedding function available)
            if self.embed_fn:
                record.text_embedding = self.embed_fn(record.text)
            
            self.records.append(record)
    
    def _build_matrices(self) -> None:
        """Build matrices for fast retrieval"""
        if not self.records:
            return
        
        n = len(self.records)
        
        # Temporal matrix
        self.temporal_matrix = np.zeros((n, self.n_time_cells), dtype=np.float32)
        for i, r in enumerate(self.records):
            if r.temporal_encoding is not None:
                self.temporal_matrix[i] = r.temporal_encoding
        
        # Spatial matrix
        self.spatial_matrix = np.zeros((n, self.n_place_cells), dtype=np.float32)
        for i, r in enumerate(self.records):
            if r.spatial_encoding is not None:
                self.spatial_matrix[i] = r.spatial_encoding
        
        # Text matrix (if embeddings available)
        if self.records[0].text_embedding is not None:
            self.text_matrix = np.zeros((n, self.embedding_dim), dtype=np.float32)
            for i, r in enumerate(self.records):
                if r.text_embedding is not None:
                    self.text_matrix[i] = r.text_embedding
        
        # Estimate memory
        mem_mb = (
            self.temporal_matrix.nbytes +
            self.spatial_matrix.nbytes +
            (self.text_matrix.nbytes if self.text_matrix is not None else 0)
        ) / 1e6
        self.stats.memory_used_mb = mem_mb
    
    # ==================== Querying ====================
    
    def query(
        self,
        query: TemporalQuery,
        temporal_weight: float = 0.3,
        spatial_weight: float = 0.2,
        text_weight: float = 0.5
    ) -> List[Tuple[TemporalRecord, float]]:
        """
        Query the temporal index
        
        Args:
            query: Query parameters
            temporal_weight: Weight for temporal similarity
            spatial_weight: Weight for spatial similarity
            text_weight: Weight for text similarity
            
        Returns:
            List of (record, score) tuples
        """
        if not self.records:
            return []
        
        n = len(self.records)
        scores = np.zeros(n, dtype=np.float32)
        
        # Filter by date range first
        valid_mask = np.ones(n, dtype=bool)
        if query.start_date or query.end_date:
            for i, r in enumerate(self.records):
                if r.timestamp is None:
                    continue
                # Normalize to naive datetime for comparison
                record_ts = r.timestamp.replace(tzinfo=None) if r.timestamp.tzinfo else r.timestamp
                
                if query.start_date:
                    start_ts = query.start_date.replace(tzinfo=None) if query.start_date.tzinfo else query.start_date
                    if record_ts < start_ts:
                        valid_mask[i] = False
                
                if query.end_date:
                    end_ts = query.end_date.replace(tzinfo=None) if query.end_date.tzinfo else query.end_date
                    if record_ts > end_ts:
                        valid_mask[i] = False
        
        # Temporal similarity
        if query.start_date:
            mid_date = query.start_date
            if query.end_date:
                mid_date = datetime(
                    (query.start_date.year + query.end_date.year) // 2,
                    (query.start_date.month + query.end_date.month) // 2,
                    1
                )
            
            query_time = (mid_date.year + (mid_date.month - 1) / 12.0 - YEAR_MIN) / YEAR_RANGE
            query_temporal = self.encode_time(query_time)
            
            temporal_sim = np.dot(self.temporal_matrix, query_temporal)
            temporal_sim /= (np.linalg.norm(self.temporal_matrix, axis=1) + 1e-8)
            temporal_sim /= (np.linalg.norm(query_temporal) + 1e-8)
            
            scores += temporal_weight * temporal_sim
        
        # Spatial similarity
        if query.location:
            query_spatial = self.encode_location(query.location)
            
            spatial_sim = np.dot(self.spatial_matrix, query_spatial)
            spatial_sim /= (np.linalg.norm(self.spatial_matrix, axis=1) + 1e-8)
            spatial_sim /= (np.linalg.norm(query_spatial) + 1e-8)
            
            scores += spatial_weight * spatial_sim
        
        # Text similarity
        if query.text and self.text_matrix is not None and self.embed_fn:
            query_emb = self.embed_fn(query.text)
            
            text_sim = np.dot(self.text_matrix, query_emb)
            text_sim /= (np.linalg.norm(self.text_matrix, axis=1) + 1e-8)
            text_sim /= (np.linalg.norm(query_emb) + 1e-8)
            
            scores += text_weight * text_sim
        
        # Apply mask
        scores[~valid_mask] = -float('inf')
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:query.limit]
        
        results = []
        for idx in top_indices:
            if scores[idx] > -float('inf'):
                results.append((self.records[idx], float(scores[idx])))
        
        return results
    
    def query_by_date(
        self,
        year: int,
        month: Optional[int] = None,
        limit: int = 10
    ) -> List[TemporalRecord]:
        """
        Query by specific date
        
        Args:
            year: Year to query
            month: Month to query (optional)
            limit: Maximum results
            
        Returns:
            List of matching records
        """
        if month:
            start = datetime(year, month, 1)
            end_month = month + 1 if month < 12 else 1
            end_year = year if month < 12 else year + 1
            end = datetime(end_year, end_month, 1)
        else:
            start = datetime(year, 1, 1)
            end = datetime(year + 1, 1, 1)
        
        query = TemporalQuery(start_date=start, end_date=end, limit=limit)
        results = self.query(query, temporal_weight=1.0, spatial_weight=0.0, text_weight=0.0)
        
        return [r for r, _ in results]
    
    def query_by_location(
        self,
        location: str,
        limit: int = 10
    ) -> List[TemporalRecord]:
        """
        Query by location
        
        Args:
            location: Location to query
            limit: Maximum results
            
        Returns:
            List of matching records
        """
        query = TemporalQuery(location=location, limit=limit)
        results = self.query(query, temporal_weight=0.0, spatial_weight=1.0, text_weight=0.0)
        
        return [r for r, _ in results]
    
    # ==================== State Management ====================
    
    def save_index(self, path: Path) -> None:
        """Save index to disk"""
        # Save matrices
        np.savez(
            path,
            temporal_matrix=self.temporal_matrix,
            spatial_matrix=self.spatial_matrix,
            text_matrix=self.text_matrix if self.text_matrix is not None else np.array([]),
            time_cell_prefs=self.time_cell_prefs,
            place_cell_centers=self.place_cell_centers
        )
        
        # Save records metadata
        records_path = path.with_suffix('.json')
        records_data = [
            {
                'id': r.id,
                'text': r.text[:200],
                'timestamp': r.timestamp.isoformat() if r.timestamp else None,
                'location': r.location,
                'source_file': r.source_file
            }
            for r in self.records
        ]
        
        with open(records_path, 'w', encoding='utf-8') as f:
            json.dump(records_data, f)
        
        logger.info(f"Index saved to {path}")
    
    def load_index(self, path: Path) -> None:
        """Load index from disk"""
        if not path.exists():
            return
        
        data = np.load(path, allow_pickle=True)
        self.temporal_matrix = data['temporal_matrix']
        self.spatial_matrix = data['spatial_matrix']
        
        text_mat = data['text_matrix']
        self.text_matrix = text_mat if text_mat.size > 0 else None
        
        self.time_cell_prefs = data['time_cell_prefs']
        self.place_cell_centers = data['place_cell_centers']
        
        # Load records metadata
        records_path = path.with_suffix('.json')
        if records_path.exists():
            with open(records_path, 'r', encoding='utf-8') as f:
                records_data = json.load(f)
            
            self.records = []
            for rd in records_data:
                timestamp = None
                if rd.get('timestamp'):
                    timestamp = datetime.fromisoformat(rd['timestamp'])
                
                self.records.append(TemporalRecord(
                    id=rd['id'],
                    text=rd['text'],
                    timestamp=timestamp,
                    location=rd.get('location'),
                    source_file=rd.get('source_file', '')
                ))
        
        self.stats.total_records = len(self.records)
        logger.info(f"Index loaded from {path} ({len(self.records)} records)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get indexer statistics"""
        return {
            'total_records': self.stats.total_records,
            'date_range': (
                self.stats.date_range[0].isoformat() if self.stats.date_range else None,
                self.stats.date_range[1].isoformat() if self.stats.date_range else None
            ) if self.stats.date_range else None,
            'unique_locations': self.stats.unique_locations,
            'gpu_indexing_time_ms': self.stats.gpu_indexing_time_ms,
            'memory_used_mb': self.stats.memory_used_mb,
            'n_time_cells': self.n_time_cells,
            'n_place_cells': self.n_place_cells
        }

