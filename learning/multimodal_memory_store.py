import sqlite3
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class MultimodalMemoryStore:
    """
    Extended memory store supporting multimodal, multilingual memories.
    Compatible with existing MemoryStore interface.
    """
    
    def __init__(self, db_path: Path, embedding_dim: int = 512):
        self.db_path = Path(db_path)
        self.embedding_dim = embedding_dim
        self._init_database()
        
    def _init_database(self):
        """Initialize database with multimodal support"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main memories table (extended)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                modality TEXT DEFAULT 'text',
                language TEXT DEFAULT 'en',
                quality_score REAL DEFAULT 0.5,
                source TEXT DEFAULT 'interaction',
                protected BOOLEAN DEFAULT 0,
                identity BOOLEAN DEFAULT 0,
                access_count INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        ''')
        
        # Multimodal attachments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attachments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id INTEGER NOT NULL,
                attachment_type TEXT NOT NULL,
                file_path TEXT,
                data BLOB,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (memory_id) REFERENCES memories (id) ON DELETE CASCADE
            )
        ''')
        
        # Indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_modality ON memories(modality)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_language ON memories(language)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_quality ON memories(quality_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_protected ON memories(protected)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Multimodal memory database initialized: {self.db_path}")
        
    def add_memory(self,
                   text: str,
                   embedding: np.ndarray = None,
                   modality: str = 'text',
                   language: str = 'en',
                   quality_score: float = 0.5,
                   source: str = 'interaction',
                   protected: bool = False,
                   identity: bool = False,
                   metadata: Dict = None) -> int:
        """
        Add memory with multimodal support.
        
        Returns:
            Memory ID
        """
        if embedding is None:
            raise ValueError("Embedding is required")
            
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embedding.shape[0]}")
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        embedding_blob = embedding.astype(np.float32).tobytes()
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata or {})
        
        cursor.execute('''
            INSERT INTO memories (content, embedding, modality, language, quality_score, 
                                 source, protected, identity, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (text, embedding_blob, modality, language, quality_score,
              source, int(protected), int(identity), timestamp, metadata_json))
        
        memory_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return memory_id
        
    def add_attachment(self,
                      memory_id: int,
                      attachment_type: str,
                      file_path: str = None,
                      data: bytes = None) -> int:
        """
        Add attachment (image, audio) to a memory.
        
        Args:
            memory_id: ID of parent memory
            attachment_type: 'image' or 'audio'
            file_path: Optional path to file
            data: Optional binary data
            
        Returns:
            Attachment ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO attachments (memory_id, attachment_type, file_path, data, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (memory_id, attachment_type, file_path, data, timestamp))
        
        attachment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return attachment_id
        
    def retrieve(self,
                embedding: np.ndarray = None,
                query_text: str = None,
                k: int = 5,
                modality: str = None,
                language: str = None,
                min_quality: float = 0.0) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve similar memories with optional filters.
        
        Args:
            embedding: Query embedding vector
            query_text: Optional text query (for debugging)
            k: Number of results
            modality: Optional modality filter
            language: Optional language filter
            min_quality: Minimum quality threshold
            
        Returns:
            List of (content, similarity_score, metadata)
        """
        if embedding is None:
            raise ValueError("Query embedding required")
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build query with filters
        query = 'SELECT id, content, embedding, modality, language, quality_score, metadata FROM memories WHERE 1=1'
        params = []
        
        if modality:
            query += ' AND modality = ?'
            params.append(modality)
            
        if language:
            query += ' AND language = ?'
            params.append(language)
            
        if min_quality > 0:
            query += ' AND quality_score >= ?'
            params.append(min_quality)
            
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return []
            
        # Compute similarities
        query_vec = embedding.astype(np.float32)
        query_norm = np.linalg.norm(query_vec)
        
        results = []
        for row in rows:
            mem_id, content, emb_blob, mod, lang, qual, meta_json = row
            
            mem_vec = np.frombuffer(emb_blob, dtype=np.float32)
            mem_norm = np.linalg.norm(mem_vec)
            
            # Cosine similarity
            similarity = np.dot(query_vec, mem_vec) / (query_norm * mem_norm + 1e-8)
            
            metadata = json.loads(meta_json)
            metadata.update({
                'id': mem_id,
                'modality': mod,
                'language': lang,
                'quality': qual
            })
            
            results.append((content, float(similarity), metadata))
            
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:k]
        
    def get_attachments(self, memory_id: int) -> List[Dict]:
        """Get all attachments for a memory"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, attachment_type, file_path, data, timestamp
            FROM attachments
            WHERE memory_id = ?
        ''', (memory_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        attachments = []
        for row in rows:
            att_id, att_type, file_path, data, timestamp = row
            attachments.append({
                'id': att_id,
                'type': att_type,
                'file_path': file_path,
                'has_data': data is not None,
                'timestamp': timestamp
            })
            
        return attachments
        
    def get_stats(self) -> Dict:
        """Get memory statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM memories')
        total = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM memories WHERE protected = 1')
        protected = cursor.fetchone()[0]
        
        cursor.execute('SELECT modality, COUNT(*) FROM memories GROUP BY modality')
        by_modality = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute('SELECT language, COUNT(*) FROM memories GROUP BY language')
        by_language = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute('SELECT COUNT(*) FROM attachments')
        attachments = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_memories': total,
            'protected_memories': protected,
            'by_modality': by_modality,
            'by_language': by_language,
            'total_attachments': attachments
        }
        
    def clear(self, preserve_protected: bool = True, preserve_identity: bool = True):
        """Clear memories with optional preservation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = 'DELETE FROM memories WHERE 1=1'
        if preserve_protected:
            query += ' AND protected = 0'
        if preserve_identity:
            query += ' AND identity = 0'
            
        cursor.execute(query)
        
        # Attachments are cascade deleted
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        logger.info(f"Cleared {deleted} memories")
        return deleted
        
    def export_all(self, output_path: Path, include_embeddings: bool = False):
        """Export all memories to JSON"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM memories')
        rows = cursor.fetchall()
        
        memories = []
        for row in rows:
            memory = {
                'id': row[0],
                'content': row[1],
                'modality': row[3],
                'language': row[4],
                'quality_score': row[5],
                'source': row[6],
                'protected': bool(row[7]),
                'identity': bool(row[8]),
                'access_count': row[9],
                'timestamp': row[10],
                'metadata': json.loads(row[11])
            }
            
            if include_embeddings:
                embedding = np.frombuffer(row[2], dtype=np.float32)
                memory['embedding'] = embedding.tolist()
                
            memories.append(memory)
            
        conn.close()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'memories': memories,
                'stats': self.get_stats(),
                'export_timestamp': datetime.now().isoformat()
            }, f, indent=2)
            
        logger.info(f"Exported {len(memories)} memories to {output_path}")
