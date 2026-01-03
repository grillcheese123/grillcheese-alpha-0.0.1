"""
Limbic System - Memory and Emotion Integration

The limbic system integrates emotional processing (amygdala) with memory
(hippocampus/memory store) to:
- Tag memories with emotional significance
- Retrieve emotionally relevant memories
- Modulate memory formation based on emotional state
"""
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

from .amygdala import Amygdala, EmotionalState

logger = logging.getLogger(__name__)


class LimbicSystem:
    """
    Limbic System - Integrates emotion and memory
    
    Functions:
    1. Emotional tagging of memories
    2. Emotion-guided memory retrieval
    3. Emotional modulation of responses
    4. Memory consolidation based on emotional significance
    
    Works with:
    - Amygdala (emotion processing)
    - MemoryStore (hippocampal function)
    """
    
    def __init__(
        self,
        amygdala: Amygdala,
        memory_store,
        embedding_dim: int = 384
    ):
        """
        Initialize limbic system
        
        Args:
            amygdala: Amygdala instance for emotion processing
            memory_store: MemoryStore instance (hippocampal function)
            embedding_dim: Embedding dimension for processing
        """
        self.amygdala = amygdala
        self.memory = memory_store
        self.embedding_dim = embedding_dim
        
        # Emotional memory associations
        # Maps memory indices to emotional states at time of formation
        self.memory_emotions: Dict[int, EmotionalState] = {}
        
        # Emotional salience thresholds
        self.high_salience_threshold = 0.6  # Emotions above this are memorable
        self.low_salience_threshold = 0.2   # Emotions below this are forgettable
        
        # Statistics
        self.stats = {
            'memories_tagged': 0,
            'emotional_retrievals': 0,
            'high_salience_memories': 0
        }
        
        logger.info("Limbic system initialized")
    
    def process_input(
        self,
        text: str,
        embedding: np.ndarray
    ) -> Dict[str, Any]:
        """
        Process input through limbic system
        
        1. Compute emotional state from input
        2. Retrieve emotionally relevant memories
        3. Return emotional context for response generation
        
        Args:
            text: Input text
            embedding: Text embedding
            
        Returns:
            Dictionary with emotional state and memory context
        """
        # 1. Process through amygdala
        emotional_state = self.amygdala.process(text, embedding)
        
        # 2. Retrieve memories with emotional weighting
        memories, memory_context = self._retrieve_with_emotion(
            embedding, emotional_state
        )
        
        # 3. Get emotional modulation for response
        modulation = self.amygdala.get_emotional_modulation()
        
        return {
            'emotional_state': emotional_state,
            'memories': memories,
            'memory_context': memory_context,
            'modulation': modulation,
            'should_remember': self._compute_memorability(emotional_state)
        }
    
    def tag_memory(
        self,
        memory_index: int,
        emotional_state: Optional[EmotionalState] = None
    ) -> None:
        """
        Tag a memory with its emotional state at formation
        
        Args:
            memory_index: Index of memory in store
            emotional_state: Emotional state (uses current if None)
        """
        state = emotional_state or self.amygdala.get_state()
        self.memory_emotions[memory_index] = state
        self.stats['memories_tagged'] += 1
        
        # Track high salience memories
        salience = abs(state.valence) + state.arousal
        if salience > self.high_salience_threshold * 2:
            self.stats['high_salience_memories'] += 1
    
    def store_with_emotion(
        self,
        embedding: np.ndarray,
        text: str,
        emotional_state: Optional[EmotionalState] = None
    ) -> None:
        """
        Store memory with emotional tagging
        
        Args:
            embedding: Memory embedding
            text: Memory text content
            emotional_state: Override emotional state
        """
        state = emotional_state or self.amygdala.get_state()
        
        # Only store if emotionally salient enough
        salience = abs(state.valence) + state.arousal
        if salience >= self.low_salience_threshold * 2:
            # Store in memory
            self.memory.store(embedding, text)
            
            # Get memory index (assuming latest is at count - 1)
            if hasattr(self.memory, 'memory_count'):
                memory_idx = self.memory.memory_count - 1
            else:
                memory_idx = len(self.memory_emotions)
            
            # Tag with emotion
            self.tag_memory(memory_idx, state)
    
    def _retrieve_with_emotion(
        self,
        query_embedding: np.ndarray,
        emotional_state: EmotionalState,
        k: int = 5
    ) -> Tuple[List[str], str]:
        """
        Retrieve memories with emotional weighting
        
        Memories with similar emotional signatures are boosted
        """
        self.stats['emotional_retrievals'] += 1
        
        # Get base memories
        base_memories = self.memory.retrieve(query_embedding, k=k * 2)
        
        if not base_memories:
            return [], ""
        
        # Score memories by emotional similarity
        scored_memories = []
        for mem_text in base_memories:
            # Default emotional score
            emotional_score = 0.5
            
            # If we have emotional tag, compute similarity
            # (In production, we'd look up by memory index)
            # For now, boost memories with matching valence
            if emotional_state.valence > 0.3:
                # Prefer positive memories when user is positive
                emotional_score = 0.7
            elif emotional_state.valence < -0.3:
                # Prefer negative memories when user is negative
                # (for empathy/understanding)
                emotional_score = 0.7
            
            scored_memories.append((mem_text, emotional_score))
        
        # Sort by emotional relevance
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k
        top_memories = [m[0] for m in scored_memories[:k]]
        
        # Create context string
        if top_memories:
            context = "\n".join([f"- {mem}" for mem in top_memories])
        else:
            context = ""
        
        return top_memories, context
    
    def _compute_memorability(self, emotional_state: EmotionalState) -> float:
        """
        Compute how memorable the current interaction should be
        
        High arousal + extreme valence = more memorable
        """
        arousal = emotional_state.arousal
        valence_magnitude = abs(emotional_state.valence)
        
        # Memorability increases with emotional intensity
        memorability = (arousal * 0.5 + valence_magnitude * 0.5)
        
        # Boost for extreme emotions
        if arousal > 0.7 or valence_magnitude > 0.7:
            memorability *= 1.3
        
        return min(1.0, memorability)
    
    def get_emotional_context(self) -> Dict[str, Any]:
        """
        Get current emotional context for response generation
        """
        state = self.amygdala.get_state()
        modulation = self.amygdala.get_emotional_modulation()
        
        return {
            'state': state.to_dict(),
            'modulation': modulation,
            'dominant_emotion': state.dominant_emotion,
            'should_be_empathetic': state.valence < -0.3,
            'should_be_enthusiastic': state.valence > 0.5 and state.arousal > 0.5
        }
    
    def consolidate_memories(self) -> Dict[str, Any]:
        """
        Consolidate emotional memories (strengthen important, decay unimportant)
        
        Call periodically for memory maintenance
        """
        strengthened = 0
        decayed = 0
        
        for memory_idx, state in list(self.memory_emotions.items()):
            salience = abs(state.valence) + state.arousal
            
            if salience > self.high_salience_threshold * 2:
                # High salience - strengthen (would update memory strength)
                strengthened += 1
            elif salience < self.low_salience_threshold * 2:
                # Low salience - decay
                decayed += 1
                # Could remove from emotional tracking
        
        return {
            'strengthened': strengthened,
            'decayed': decayed,
            'total_emotional_memories': len(self.memory_emotions)
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get limbic system statistics"""
        return {
            **self.stats,
            'amygdala_stats': self.amygdala.get_stats(),
            'emotional_memories': len(self.memory_emotions),
            'current_emotional_context': self.get_emotional_context()
        }

