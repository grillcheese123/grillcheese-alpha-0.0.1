import numpy as np
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json

from .multimodal_encoder import MultimodalEncoder
from .multilingual_utils import MultilingualProcessor

logger = logging.getLogger(__name__)

@dataclass
class DistilledKnowledge:
    """Container for distilled knowledge with metadata"""
    content: str
    modality: str  # 'text', 'image', 'audio', 'multimodal'
    embedding: np.ndarray
    language: Optional[str] = None
    quality_score: float = 0.0
    source: str = 'interaction'  # 'interaction', 'document', 'external'
    timestamp: str = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

class KnowledgeDistillation:
    """
    Multimodal, multilingual knowledge distillation system.
    Extracts high-quality knowledge from interactions and stores in memory.
    """
    
    def __init__(self, 
                 memory_store,
                 encoder: MultimodalEncoder,
                 lang_processor: MultilingualProcessor,
                 quality_threshold: float = 0.7,
                 protected_threshold: float = 0.85):
        self.memory = memory_store
        self.encoder = encoder
        self.lang_processor = lang_processor
        
        self.quality_threshold = quality_threshold
        self.protected_threshold = protected_threshold
        
        # Statistics
        self.stats = {
            'total_distilled': 0,
            'protected_count': 0,
            'by_modality': {},
            'by_language': {},
            'by_quality': {'low': 0, 'medium': 0, 'high': 0}
        }
        
    def distill_interaction(self,
                           user_message: str,
                           assistant_response: str,
                           quality_score: float = None,
                           emotion_context: Dict = None) -> Optional[DistilledKnowledge]:
        """
        Distill a conversation interaction into memory.
        
        Args:
            user_message: User's input
            assistant_response: Assistant's response
            quality_score: Optional quality score (0-1)
            emotion_context: Optional emotional state during interaction
            
        Returns:
            DistilledKnowledge if quality threshold met, None otherwise
        """
        # Auto-compute quality if not provided
        if quality_score is None:
            quality_score = self._compute_quality_score(
                user_message, assistant_response, emotion_context
            )
            
        # Filter low quality
        if quality_score < self.quality_threshold:
            logger.debug(f"Skipping low-quality interaction (score: {quality_score:.3f})")
            return None
            
        # Detect language
        combined_text = f"{user_message}\n{assistant_response}"
        language = self.lang_processor.detect_language(combined_text)
        
        # Normalize text
        normalized = self.lang_processor.normalize_text(combined_text, language)
        
        # Create structured content
        content = self._format_interaction(user_message, assistant_response, language)
        
        # Generate embedding
        embedding = self.encoder.encode_text(content, language)
        
        # Create knowledge object
        knowledge = DistilledKnowledge(
            content=content,
            modality='text',
            embedding=embedding,
            language=language,
            quality_score=quality_score,
            source='interaction',
            metadata={
                'emotion': emotion_context,
                'user_length': len(user_message),
                'response_length': len(assistant_response)
            }
        )
        
        # Store in memory
        protected = quality_score >= self.protected_threshold
        self._store_knowledge(knowledge, protected)
        
        # Update statistics
        self._update_stats(knowledge, protected)
        
        return knowledge
        
    def distill_multimodal(self,
                          text: str = None,
                          image: Union[str, Path] = None,
                          audio: Union[str, Path] = None,
                          quality_score: float = 0.8) -> Optional[DistilledKnowledge]:
        """
        Distill multimodal knowledge.
        
        Args:
            text: Optional text content
            image: Optional image path
            audio: Optional audio path
            quality_score: Quality score (default 0.8)
            
        Returns:
            DistilledKnowledge or None
        """
        if quality_score < self.quality_threshold:
            return None
            
        # Encode all available modalities
        embeddings = self.encoder.encode_multimodal(
            text=text,
            image=image,
            audio=audio
        )
        
        if not embeddings:
            logger.warning("No valid modalities provided")
            return None
            
        # Fuse embeddings
        fused_embedding = self.encoder.fuse_embeddings(embeddings)
        
        # Create content description
        modalities = list(embeddings.keys())
        modality_str = '+'.join(modalities)
        
        content = self._format_multimodal(text, image, audio, modalities)
        
        # Detect language if text present
        language = None
        if text:
            language = self.lang_processor.detect_language(text)
            
        knowledge = DistilledKnowledge(
            content=content,
            modality=modality_str,
            embedding=fused_embedding,
            language=language,
            quality_score=quality_score,
            source='multimodal',
            metadata={
                'modalities': modalities,
                'has_text': text is not None,
                'has_image': image is not None,
                'has_audio': audio is not None
            }
        )
        
        protected = quality_score >= self.protected_threshold
        self._store_knowledge(knowledge, protected)
        self._update_stats(knowledge, protected)
        
        return knowledge
        
    def distill_document(self,
                        text: str,
                        title: str = None,
                        language: str = None,
                        protected: bool = True) -> List[DistilledKnowledge]:
        """
        Distill a document by chunking and storing important parts.
        
        Args:
            text: Document text
            title: Optional document title
            language: Optional language code
            protected: Whether to protect these memories
            
        Returns:
            List of distilled knowledge chunks
        """
        if language is None:
            language = self.lang_processor.detect_language(text)
            
        # Split into chunks (paragraph-based)
        chunks = self._chunk_document(text, language)
        
        distilled = []
        for i, chunk in enumerate(chunks):
            # Score chunk importance
            quality = self._score_chunk_importance(chunk, language)
            
            if quality < self.quality_threshold:
                continue
                
            # Normalize and encode
            normalized = self.lang_processor.normalize_text(chunk, language)
            embedding = self.encoder.encode_text(normalized, language)
            
            # Add context from title
            content = chunk
            if title:
                content = f"[{title}] {chunk}"
                
            knowledge = DistilledKnowledge(
                content=content,
                modality='text',
                embedding=embedding,
                language=language,
                quality_score=quality,
                source='document',
                metadata={
                    'title': title,
                    'chunk_index': i,
                    'chunk_count': len(chunks)
                }
            )
            
            self._store_knowledge(knowledge, protected)
            self._update_stats(knowledge, protected)
            distilled.append(knowledge)
            
        logger.info(f"Distilled {len(distilled)}/{len(chunks)} chunks from document")
        return distilled
        
    def retrieve_knowledge(self,
                          query: str = None,
                          image: Union[str, Path] = None,
                          audio: Union[str, Path] = None,
                          k: int = 5,
                          language: str = None,
                          modality_filter: str = None) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve relevant knowledge using multimodal query.
        
        Args:
            query: Text query
            image: Image query
            audio: Audio query
            k: Number of results
            language: Optional language filter
            modality_filter: Optional modality filter
            
        Returns:
            List of (content, score, metadata) tuples
        """
        # Encode query
        embeddings = self.encoder.encode_multimodal(
            text=query,
            image=image,
            audio=audio
        )
        
        if not embeddings:
            return []
            
        # Fuse query embeddings
        query_embedding = self.encoder.fuse_embeddings(embeddings)
        
        # Retrieve from memory (using existing memory store)
        # This assumes memory_store.retrieve() accepts embedding vector
        results = self.memory.retrieve(
            embedding=query_embedding,
            k=k * 2  # Retrieve more for filtering
        )
        
        # Apply filters
        filtered = []
        for content, score, metadata in results:
            # Language filter
            if language and metadata.get('language') != language:
                continue
                
            # Modality filter
            if modality_filter and metadata.get('modality') != modality_filter:
                continue
                
            filtered.append((content, score, metadata))
            
            if len(filtered) >= k:
                break
                
        return filtered
        
    def _compute_quality_score(self,
                              user_msg: str,
                              assistant_resp: str,
                              emotion_context: Dict = None) -> float:
        """Compute interaction quality score using heuristics"""
        score = 0.5  # Base score
        
        # Length-based quality (balanced responses)
        resp_len = len(assistant_resp)
        if 50 <= resp_len <= 500:
            score += 0.2
        elif resp_len < 20:
            score -= 0.2
            
        # Complexity (unique words)
        unique_ratio = len(set(assistant_resp.split())) / max(len(assistant_resp.split()), 1)
        score += unique_ratio * 0.1
        
        # Emotional positivity (if available)
        if emotion_context:
            valence = emotion_context.get('valence', 0)
            score += valence * 0.1
            
        # Cap at 0-1 range
        return max(0.0, min(1.0, score))
        
    def _score_chunk_importance(self, chunk: str, language: str) -> float:
        """Score document chunk importance"""
        score = 0.6  # Base score
        
        # Length indicator
        words = chunk.split()
        if 30 <= len(words) <= 200:
            score += 0.2
            
        # Information density (unique words)
        unique_ratio = len(set(words)) / max(len(words), 1)
        score += unique_ratio * 0.1
        
        # Keyword presence (simple heuristic)
        keywords = ['important', 'key', 'critical', 'note', 'remember']
        if any(kw in chunk.lower() for kw in keywords):
            score += 0.1
            
        return max(0.0, min(1.0, score))
        
    def _chunk_document(self, text: str, language: str) -> List[str]:
        """Split document into chunks"""
        # Paragraph-based chunking
        paragraphs = text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_length = 0
        max_length = 300  # words
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            words = para.split()
            para_length = len(words)
            
            if current_length + para_length <= max_length:
                current_chunk.append(para)
                current_length += para_length
            else:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [para]
                current_length = para_length
                
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks
        
    def _format_interaction(self, user_msg: str, assistant_resp: str, language: str) -> str:
        """Format interaction for storage"""
        lang_name = self.lang_processor.get_language_name(language)
        return f"[{lang_name}] Q: {user_msg}\nA: {assistant_resp}"
        
    def _format_multimodal(self, text: str, image, audio, modalities: List[str]) -> str:
        """Format multimodal content description"""
        parts = [f"Multimodal: {'+'.join(modalities)}"]
        
        if text:
            parts.append(f"Text: {text[:200]}")
        if image:
            parts.append(f"Image: {Path(image).name if isinstance(image, (str, Path)) else 'provided'}")
        if audio:
            parts.append(f"Audio: {Path(audio).name if isinstance(audio, (str, Path)) else 'provided'}")
            
        return ' | '.join(parts)
        
    def _store_knowledge(self, knowledge: DistilledKnowledge, protected: bool):
        """Store knowledge in memory system"""
        # This integrates with your existing memory_store
        self.memory.add_memory(
            text=knowledge.content,
            embedding=knowledge.embedding,
            protected=protected,
            metadata={
                'modality': knowledge.modality,
                'language': knowledge.language,
                'quality': knowledge.quality_score,
                'source': knowledge.source,
                'timestamp': knowledge.timestamp,
                **knowledge.metadata
            }
        )
        
    def _update_stats(self, knowledge: DistilledKnowledge, protected: bool):
        """Update statistics"""
        self.stats['total_distilled'] += 1
        
        if protected:
            self.stats['protected_count'] += 1
            
        # By modality
        modality = knowledge.modality
        self.stats['by_modality'][modality] = self.stats['by_modality'].get(modality, 0) + 1
        
        # By language
        if knowledge.language:
            lang = knowledge.language
            self.stats['by_language'][lang] = self.stats['by_language'].get(lang, 0) + 1
            
        # By quality
        if knowledge.quality_score >= 0.9:
            self.stats['by_quality']['high'] += 1
        elif knowledge.quality_score >= 0.7:
            self.stats['by_quality']['medium'] += 1
        else:
            self.stats['by_quality']['low'] += 1
            
    def get_stats(self) -> Dict:
        """Get distillation statistics"""
        return self.stats.copy()
        
    def export_knowledge_base(self, output_path: Path):
        """Export all distilled knowledge to file"""
        # This would retrieve all memories and export them
        # Simplified implementation
        output_path = Path(output_path)
        
        with output_path.open('w', encoding='utf-8') as f:
            json.dump({
                'stats': self.stats,
                'timestamp': datetime.now().isoformat(),
                'note': 'Full export requires memory_store.export_all() implementation'
            }, f, indent=2)
            
        logger.info(f"Knowledge base summary exported to {output_path}")
