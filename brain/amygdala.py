"""
Amygdala - Emotional Processing Center

The amygdala computes emotional valence and arousal from input features.
This enables GrillCheese to:
- Detect emotional content in user messages
- Modulate responses based on emotional context
- Build emotional associations with memories

Enhanced with GPU-accelerated calibration using:
- hebbian-learning shader for affect weight training
- amygdala_affect.jsonl training data
"""
import json
import logging
import struct
import time as time_module
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# GPU Memory Budget for Amygdala (conservative)
AMYGDALA_MAX_MEMORY_MB = 500  # 500MB max for amygdala operations


@dataclass
class EmotionalState:
    """
    Current emotional state of the system
    
    Attributes:
        arousal: 0.0 (calm) to 1.0 (excited/stressed)
        valence: -1.0 (negative) to 1.0 (positive)
        dominant_emotion: Current primary emotion label
        confidence: Confidence in emotion detection
        cause: What triggered this emotional state (text/context)
        detected_emotions: List of emotion words detected in the input
    """
    arousal: float = 0.5
    valence: float = 0.0
    dominant_emotion: str = "neutral"
    confidence: float = 0.5
    timestamp: float = 0.0
    cause: str = ""
    detected_emotions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'arousal': self.arousal,
            'valence': self.valence,
            'dominant_emotion': self.dominant_emotion,
            'confidence': self.confidence,
            'cause': self.cause,
            'detected_emotions': self.detected_emotions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EmotionalState':
        # Handle missing fields for backward compatibility
        return cls(
            arousal=data.get('arousal', 0.5),
            valence=data.get('valence', 0.0),
            dominant_emotion=data.get('dominant_emotion', 'neutral'),
            confidence=data.get('confidence', 0.5),
            timestamp=data.get('timestamp', 0.0),
            cause=data.get('cause', ''),
            detected_emotions=data.get('detected_emotions', [])
        )


# Emotion words mapped to (valence, arousal) coordinates
# Based on Russell's Circumplex Model of Affect
EMOTION_LEXICON = {
    # High arousal, positive valence
    'excited': (0.8, 0.8), 'happy': (0.7, 0.6), 'elated': (0.9, 0.9),
    'thrilled': (0.8, 0.9), 'enthusiastic': (0.7, 0.7), 'joyful': (0.8, 0.7),
    'delighted': (0.8, 0.6), 'amazing': (0.7, 0.7), 'wonderful': (0.7, 0.5),
    'fantastic': (0.8, 0.7), 'awesome': (0.7, 0.6), 'great': (0.6, 0.5),
    'love': (0.8, 0.6), 'passionate': (0.7, 0.8),
    
    # Low arousal, positive valence
    'calm': (0.4, 0.2), 'relaxed': (0.5, 0.2), 'peaceful': (0.6, 0.2),
    'content': (0.5, 0.3), 'serene': (0.6, 0.2), 'tranquil': (0.5, 0.1),
    'satisfied': (0.5, 0.3), 'comfortable': (0.4, 0.3),
    
    # High arousal, negative valence
    'angry': (-0.7, 0.8), 'furious': (-0.9, 0.9), 'enraged': (-0.9, 1.0),
    'frustrated': (-0.5, 0.7), 'annoyed': (-0.4, 0.6), 'irritated': (-0.4, 0.6),
    'anxious': (-0.4, 0.7), 'stressed': (-0.5, 0.8), 'nervous': (-0.3, 0.7),
    'afraid': (-0.6, 0.8), 'scared': (-0.6, 0.8), 'terrified': (-0.8, 0.9),
    'panic': (-0.7, 0.9), 'worried': (-0.4, 0.6),
    
    # Low arousal, negative valence
    'sad': (-0.6, 0.3), 'depressed': (-0.8, 0.2), 'melancholy': (-0.5, 0.3),
    'gloomy': (-0.5, 0.3), 'hopeless': (-0.7, 0.2), 'lonely': (-0.5, 0.3),
    'bored': (-0.3, 0.2), 'tired': (-0.3, 0.2), 'exhausted': (-0.4, 0.2),
    'disappointed': (-0.5, 0.4),
    
    # Neutral / cognitive
    'curious': (0.3, 0.5), 'interested': (0.3, 0.5), 'thoughtful': (0.2, 0.4),
    'confused': (-0.2, 0.5), 'surprised': (0.1, 0.7), 'shocked': (-0.1, 0.8),
}

# Emotion labels based on circumplex quadrants
EMOTION_QUADRANTS = {
    'excited': (0.5, 1.0, 0.5, 1.0),      # High arousal, positive valence
    'happy': (0.0, 0.5, 0.5, 1.0),        # Medium arousal, positive valence
    'calm': (0.0, 0.5, 0.0, 0.5),         # Low arousal, positive valence
    'sad': (0.0, 0.5, -1.0, 0.0),         # Low arousal, negative valence
    'angry': (0.5, 1.0, -1.0, 0.0),       # High arousal, negative valence
    'anxious': (0.5, 1.0, -0.5, 0.5),     # High arousal, mixed valence
    'neutral': (0.3, 0.7, -0.3, 0.3),     # Medium arousal, neutral valence
}


class Amygdala:
    """
    Amygdala - Emotional processing and valence/arousal computation
    
    Computes emotional state from:
    1. Text content analysis (emotion words)
    2. Embedding patterns (learned associations)
    3. GPU-accelerated neural affect prediction
    4. Context from recent interactions
    
    Provides:
    - Real-time emotional state
    - Emotional modulation for responses
    - Emotion-memory associations
    - GPU-accelerated affect calibration
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dim: int = 128,
        emotion_decay: float = 0.9,
        sensitivity: float = 1.0,
        use_gpu: bool = True,
        dropout_rate: float = 0.1
    ):
        """
        Initialize amygdala with improved 3-layer MLP affect prediction
        
        Architecture: embedding -> hidden1 -> hidden2 -> [valence, arousal]
        Features: LeakyReLU, residual connections, Adam optimizer, dropout
        
        Args:
            embedding_dim: Dimension of input embeddings
            hidden_dim: Hidden layer dimension for MLP (used for both hidden layers)
            emotion_decay: How quickly emotions decay (0-1, higher = slower decay)
            sensitivity: Emotional sensitivity multiplier
            use_gpu: Whether to use GPU acceleration for calibration
            dropout_rate: Dropout rate during training (0.0-0.5)
        """
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.emotion_decay = emotion_decay
        self.sensitivity = sensitivity
        self.use_gpu = use_gpu
        self.dropout_rate = dropout_rate
        self.leaky_slope = 0.01  # LeakyReLU negative slope
        
        # Current emotional state
        self.state = EmotionalState()
        
        # Emotion history for temporal patterns
        self.history: List[EmotionalState] = []
        self.max_history = 100
        
        # Learned emotion associations (embedding patterns)
        self.positive_patterns: List[np.ndarray] = []
        self.negative_patterns: List[np.ndarray] = []
        
        # Learned emotion lexicon (user-defined emotion labels)
        # Maps emotion word -> (valence, arousal)
        self.learned_emotion_lexicon: Dict[str, Tuple[float, float]] = {}
        
        # 3-Layer MLP for affect prediction
        # Layer 1: embedding_dim -> hidden_dim
        # Layer 2: hidden_dim -> hidden_dim (with residual)
        # Layer 3: hidden_dim -> 2 (valence, arousal)
        
        # He initialization (better for ReLU variants)
        self.W1 = (np.random.randn(hidden_dim, embedding_dim).astype(np.float32) 
                   * np.sqrt(2.0 / embedding_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        
        self.W2 = (np.random.randn(hidden_dim, hidden_dim).astype(np.float32)
                   * np.sqrt(2.0 / hidden_dim))
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        
        self.W3 = (np.random.randn(2, hidden_dim).astype(np.float32)
                   * np.sqrt(2.0 / hidden_dim))
        self.b3 = np.array([0.0, 0.5], dtype=np.float32)  # neutral valence, medium arousal
        
        # Adam optimizer state (momentum + RMSprop)
        self.adam_m = {}  # First moment
        self.adam_v = {}  # Second moment
        self.adam_t = 0   # Timestep
        self._init_adam_state()
        
        # Legacy compatibility
        self.affect_weights = self.W3
        self.affect_bias = self.b3
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_samples = 0
        
        # GPU backend (lazy init)
        self.vulkan = None
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'positive_detections': 0,
            'negative_detections': 0,
            'avg_arousal': 0.5,
            'avg_valence': 0.0,
            'calibration_loss': 1.0,
            'calibration_samples': 0
        }
    
    def _init_adam_state(self):
        """Initialize Adam optimizer state for all weights"""
        for name, param in [('W1', self.W1), ('b1', self.b1), 
                            ('W2', self.W2), ('b2', self.b2),
                            ('W3', self.W3), ('b3', self.b3)]:
            self.adam_m[name] = np.zeros_like(param)
            self.adam_v[name] = np.zeros_like(param)
    
    def _init_gpu(self):
        """Lazy initialize GPU backend"""
        if self.vulkan is not None:
            return
        
        if not self.use_gpu:
            return
        
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from vulkan_backend import VulkanCompute
            self.vulkan = VulkanCompute()
            logger.info("[OK] Amygdala GPU backend initialized")
        except Exception as e:
            logger.debug(f"GPU not available for Amygdala: {e}")
            self.vulkan = None
    
    def process(self, text: str, embedding: Optional[np.ndarray] = None) -> EmotionalState:
        """
        Process input and compute emotional state
        
        Args:
            text: Input text to analyze
            embedding: Optional embedding for pattern matching
            
        Returns:
            EmotionalState with computed arousal and valence
        """
        # 1. Analyze text for emotion words
        text_valence, text_arousal, detected_emotions = self._analyze_text(text)
        
        # 2. Analyze embedding patterns (if available)
        emb_valence, emb_arousal = 0.0, 0.5
        if embedding is not None:
            emb_valence, emb_arousal = self._analyze_embedding(embedding)
        
        # 3. Combine signals (text weighted higher for explicit emotions)
        if detected_emotions:
            # Explicit emotion words found - weight text higher
            valence = 0.7 * text_valence + 0.3 * emb_valence
            arousal = 0.7 * text_arousal + 0.3 * emb_arousal
            confidence = 0.8
        else:
            # No explicit emotions - rely more on embedding
            valence = 0.3 * text_valence + 0.7 * emb_valence
            arousal = 0.3 * text_arousal + 0.7 * emb_arousal
            confidence = 0.5
        
        # 4. Apply sensitivity
        valence = np.clip(valence * self.sensitivity, -1.0, 1.0)
        arousal = np.clip(arousal * self.sensitivity, 0.0, 1.0)
        
        # 5. Temporal smoothing with decay
        # Use exponential moving average, but ensure new emotions have sufficient weight
        # If new emotion is significantly different, give it more weight
        old_valence = self.state.valence
        old_arousal = self.state.arousal
        
        # Adaptive decay: if emotion changed significantly, reduce decay to allow faster adaptation
        valence_diff = abs(valence - old_valence)
        arousal_diff = abs(arousal - old_arousal)
        
        # Reduce decay if emotion changed significantly (allows faster adaptation)
        adaptive_decay = self.emotion_decay
        if valence_diff > 0.3 or arousal_diff > 0.3:
            adaptive_decay = max(0.3, self.emotion_decay - 0.2)  # Faster adaptation for strong changes
        
        valence = adaptive_decay * old_valence + (1 - adaptive_decay) * valence
        arousal = adaptive_decay * old_arousal + (1 - adaptive_decay) * arousal
        
        # 6. Determine dominant emotion
        dominant = self._classify_emotion(valence, arousal)
        
        # 6b. Determine cause - what triggered this emotion
        # Use the input text, but summarize if too long
        cause_text = text[:200] if len(text) > 200 else text
        if not cause_text.strip():
            cause_text = "Recent interaction context"
        
        # Extract key emotion words/phrases that contributed
        emotion_causes = []
        if detected_emotions:
            emotion_causes = detected_emotions[:3]  # Top 3 emotion words detected
        
        # 7. Update state
        import time
        new_state = EmotionalState(
            arousal=arousal,
            valence=valence,
            dominant_emotion=dominant,
            confidence=confidence,
            timestamp=time.time(),
            cause=cause_text,
            detected_emotions=emotion_causes
        )
        
        self.state = new_state
        self._update_history(new_state)
        self._update_stats(valence, arousal)
        
        # 8. Learn from embedding if strong emotion detected
        if embedding is not None and abs(valence) > 0.5:
            self._learn_pattern(embedding, valence)
        
        return new_state
    
    def _analyze_text(self, text: str) -> Tuple[float, float, List[str]]:
        """Analyze text for emotion words"""
        text_lower = text.lower()
        words = text_lower.split()
        
        detected = []
        valence_sum = 0.0
        arousal_sum = 0.0
        count = 0
        
        for word in words:
            # Clean punctuation
            word_clean = ''.join(c for c in word if c.isalnum())
            
            if word_clean in EMOTION_LEXICON:
                val, aro = EMOTION_LEXICON[word_clean]
                valence_sum += val
                arousal_sum += aro
                detected.append(word_clean)
                count += 1
        
        if count > 0:
            return valence_sum / count, arousal_sum / count, detected
        
        return 0.0, 0.5, []
    
    def _analyze_embedding(self, embedding: np.ndarray) -> Tuple[float, float]:
        """
        Analyze embedding for emotional content
        
        Uses multiple strategies:
        1. Trained affect weights (if calibrated)
        2. Pattern matching with learned examples
        3. Embedding magnitude for arousal
        """
        # Strategy 1: Use trained neural projection (if calibrated)
        if self.is_calibrated:
            prediction = self._predict_affect_neural(embedding)
            # High confidence in calibrated prediction
            return prediction
        
        valence = 0.0
        arousal = 0.5
        
        # Strategy 2: Compare to learned positive patterns
        positive_contrib = 0.0
        for pattern in self.positive_patterns[-20:]:  # Recent patterns
            similarity = np.dot(embedding, pattern) / (
                np.linalg.norm(embedding) * np.linalg.norm(pattern) + 1e-8
            )
            positive_contrib += similarity * 0.15  # Increased from 0.1 to 0.15
        
        # Compare to learned negative patterns
        negative_contrib = 0.0
        for pattern in self.negative_patterns[-20:]:
            similarity = np.dot(embedding, pattern) / (
                np.linalg.norm(embedding) * np.linalg.norm(pattern) + 1e-8
            )
            negative_contrib += similarity * 0.15  # Increased from 0.1 to 0.15
        
        valence = positive_contrib - negative_contrib
        
        # Strategy 3: Arousal from embedding magnitude and variance
        magnitude = np.linalg.norm(embedding)
        variance = np.var(embedding)  # Higher variance = more complex/arousing content
        arousal = 0.3 + 0.3 * np.tanh(magnitude / 10.0) + 0.2 * np.tanh(variance * 10.0)
        
        return np.clip(valence, -1.0, 1.0), np.clip(arousal, 0.0, 1.0)
    
    def _predict_affect_neural(self, embedding: np.ndarray) -> Tuple[float, float]:
        """
        Predict affect using trained 3-layer MLP with residual connection
        
        Architecture: embedding -> hidden1 (LeakyReLU) -> hidden2 + residual (LeakyReLU) -> output
        
        Args:
            embedding: Text embedding [embedding_dim]
            
        Returns:
            (valence, arousal) tuple
        """
        # Layer 1: Linear + LeakyReLU
        h1_pre = np.dot(self.W1, embedding) + self.b1
        h1 = np.where(h1_pre > 0, h1_pre, self.leaky_slope * h1_pre)
        
        # Layer 2: Linear + Residual + LeakyReLU
        h2_pre = np.dot(self.W2, h1) + self.b2 + h1  # Residual connection
        h2 = np.where(h2_pre > 0, h2_pre, self.leaky_slope * h2_pre)
        
        # Layer 3: Output projection
        output = np.dot(self.W3, h2) + self.b3
        
        # Output activations
        valence = float(np.tanh(output[0]))
        arousal = float(1.0 / (1.0 + np.exp(-np.clip(output[1], -10, 10))))
        
        return valence, arousal
    
    def _forward_mlp(self, embeddings: np.ndarray, apply_output_activation: bool = True, training: bool = False) -> Dict[str, np.ndarray]:
        """
        Forward pass through 3-layer MLP for batch processing
        
        Uses GPU acceleration when available for large batches.
        
        Args:
            embeddings: [batch, embedding_dim]
            apply_output_activation: Whether to apply tanh/sigmoid
            training: Whether to apply dropout
            
        Returns:
            Dict with all intermediate values for backprop
        """
        batch_size = embeddings.shape[0]
        
        # Try GPU forward pass if available and batch is large enough
        if self.use_gpu and hasattr(self, 'vulkan') and self.vulkan is not None and batch_size > 4:
            try:
                predictions, h1, h2 = self.vulkan.affect_mlp_forward(
                    embeddings=embeddings,
                    w1=self.W1.T,  # Shader expects (hidden1_dim, embedding_dim)
                    b1=self.b1,
                    w2=self.W2.T,  # Shader expects (hidden2_dim, hidden1_dim)
                    b2=self.b2,
                    w3=self.W3.T,  # Shader expects (2, hidden2_dim)
                    b3=self.b3,
                    leaky_slope=self.leaky_slope,
                    apply_output_activation=apply_output_activation,
                    dropout_rate=self.dropout_rate if training else 0.0
                )
                
                # Reconstruct intermediate values for backprop compatibility
                # Note: GPU shader doesn't return pre-activations, so we approximate
                h1_pre = h1 / np.where(h1 > 0, 1.0, self.leaky_slope)  # Approximate inverse
                h2_pre = h2 / np.where(h2 > 0, 1.0, self.leaky_slope)
                
                # Recompute output from h2 for consistency
                output = np.dot(h2, self.W3.T) + self.b3
                if not apply_output_activation:
                    predictions = output.copy()
                
                return {
                    'h1_pre': h1_pre,
                    'h1': h1,
                    'h2_pre': h2_pre,
                    'h2': h2,
                    'output': output,
                    'predictions': predictions,
                    'mask1': None,  # GPU handles dropout internally
                    'mask2': None,
                    'gpu_accelerated': True
                }
            except Exception as e:
                logger.debug(f"GPU forward MLP failed: {e}, using CPU")
        
        # CPU implementation (original)
        # Layer 1: Linear + LeakyReLU
        h1_pre = np.dot(embeddings, self.W1.T) + self.b1  # [batch, hidden]
        h1 = np.where(h1_pre > 0, h1_pre, self.leaky_slope * h1_pre)
        
        # Dropout (training only)
        if training and self.dropout_rate > 0:
            mask1 = (np.random.rand(*h1.shape) > self.dropout_rate).astype(np.float32)
            h1 = h1 * mask1 / (1.0 - self.dropout_rate)
        else:
            mask1 = None
        
        # Layer 2: Linear + Residual + LeakyReLU
        h2_pre = np.dot(h1, self.W2.T) + self.b2 + h1  # Residual connection
        h2 = np.where(h2_pre > 0, h2_pre, self.leaky_slope * h2_pre)
        
        # Dropout
        if training and self.dropout_rate > 0:
            mask2 = (np.random.rand(*h2.shape) > self.dropout_rate).astype(np.float32)
            h2 = h2 * mask2 / (1.0 - self.dropout_rate)
        else:
            mask2 = None
        
        # Layer 3: Output projection
        output = np.dot(h2, self.W3.T) + self.b3  # [batch, 2]
        
        if apply_output_activation:
            predictions = np.zeros_like(output)
            predictions[:, 0] = np.tanh(output[:, 0])
            predictions[:, 1] = 1.0 / (1.0 + np.exp(-np.clip(output[:, 1], -10, 10)))
        else:
            predictions = output.copy()
        
        return {
            'h1_pre': h1_pre,
            'h1': h1,
            'h2_pre': h2_pre,
            'h2': h2,
            'output': output,
            'predictions': predictions,
            'mask1': mask1,
            'mask2': mask2
        }
    
    def _classify_emotion(self, valence: float, arousal: float) -> str:
        """Classify emotion based on valence-arousal coordinates"""
        best_emotion = "neutral"
        best_score = float('inf')
        
        # First check learned emotions (user-defined, higher priority)
        for emotion, (learned_val, learned_aro) in self.learned_emotion_lexicon.items():
            # Distance to learned emotion coordinates
            dist = (arousal - learned_aro)**2 + (valence - learned_val)**2
            if dist < best_score:
                best_score = dist
                best_emotion = emotion
        
        # Then check standard emotion quadrants
        for emotion, (aro_min, aro_max, val_min, val_max) in EMOTION_QUADRANTS.items():
            # Skip if already found a learned emotion that's closer
            if best_score < 0.1:  # Very close match in learned emotions
                break
            # Check if in quadrant
            if aro_min <= arousal <= aro_max and val_min <= valence <= val_max:
                # Distance to center of quadrant
                aro_center = (aro_min + aro_max) / 2
                val_center = (val_min + val_max) / 2
                dist = (arousal - aro_center)**2 + (valence - val_center)**2
                if dist < best_score:
                    best_score = dist
                    best_emotion = emotion
        
        return best_emotion
    
    def learn_emotion_label(self, emotion_word: str, valence: float, arousal: float) -> bool:
        """
        Learn a new emotion label from user input
        
        Args:
            emotion_word: The emotion word/label to learn (e.g., "powerful")
            valence: Current valence value to associate with this emotion
            arousal: Current arousal value to associate with this emotion
            
        Returns:
            True if learned successfully
        """
        emotion_word = emotion_word.lower().strip()
        if not emotion_word:
            return False
        
        # Store the learned emotion
        self.learned_emotion_lexicon[emotion_word] = (valence, arousal)
        logger.info(f"Learned new emotion label: '{emotion_word}' -> (valence={valence:.2f}, arousal={arousal:.2f})")
        return True
    
    def _learn_pattern(self, embedding: np.ndarray, valence: float) -> None:
        """Learn emotional association with embedding"""
        if valence > 0.5:
            self.positive_patterns.append(embedding.copy())
            if len(self.positive_patterns) > 100:
                self.positive_patterns.pop(0)
        elif valence < -0.5:
            self.negative_patterns.append(embedding.copy())
            if len(self.negative_patterns) > 100:
                self.negative_patterns.pop(0)
    
    def _update_history(self, state: EmotionalState) -> None:
        """Update emotion history"""
        self.history.append(state)
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def _update_stats(self, valence: float, arousal: float) -> None:
        """Update statistics"""
        self.stats['total_processed'] += 1
        
        if valence > 0.3:
            self.stats['positive_detections'] += 1
        elif valence < -0.3:
            self.stats['negative_detections'] += 1
        
        # Running average
        alpha = 0.95
        self.stats['avg_arousal'] = alpha * self.stats['avg_arousal'] + (1-alpha) * arousal
        self.stats['avg_valence'] = alpha * self.stats['avg_valence'] + (1-alpha) * valence
    
    def get_emotional_modulation(self) -> Dict[str, float]:
        """
        Get modulation factors for response generation
        
        Returns factors that can modulate:
        - response_warmth: How warm/empathetic to be
        - response_energy: Energy level of response
        - response_caution: How careful/measured to be
        """
        valence = self.state.valence
        arousal = self.state.arousal
        
        return {
            'response_warmth': 0.5 + 0.5 * valence,  # More warmth for positive emotions
            'response_energy': arousal,               # Match energy level
            'response_caution': max(0, -valence) * arousal,  # Cautious for negative high-arousal
            'empathy_boost': max(0, -valence),        # More empathetic for negative emotions
        }
    
    def get_state(self) -> EmotionalState:
        """Get current emotional state"""
        return self.state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get amygdala statistics"""
        return {
            **self.stats,
            'current_state': self.state.to_dict(),
            'positive_patterns': len(self.positive_patterns),
            'negative_patterns': len(self.negative_patterns),
            'history_length': len(self.history)
        }
    
    def save_state(self, path: str) -> None:
        """Save amygdala state including 3-layer MLP weights and Adam state"""
        state_dict = {
            'state': self.state.to_dict(),
            'stats': self.stats,
            'positive_patterns': [p.tolist() for p in self.positive_patterns],
            'negative_patterns': [p.tolist() for p in self.negative_patterns],
            'learned_emotion_lexicon': {k: list(v) for k, v in self.learned_emotion_lexicon.items()},
            # 3-layer MLP weights
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist(),
            'W3': self.W3.tolist(),
            'b3': self.b3.tolist(),
            'hidden_dim': self.hidden_dim,
            'leaky_slope': self.leaky_slope,
            'dropout_rate': self.dropout_rate,
            # Adam optimizer state
            'adam_t': self.adam_t,
            'adam_m': {k: v.tolist() for k, v in self.adam_m.items()},
            'adam_v': {k: v.tolist() for k, v in self.adam_v.items()},
            # Calibration state
            'is_calibrated': self.is_calibrated,
            'calibration_samples': self.calibration_samples,
            # Legacy compatibility
            'affect_weights': self.W3.tolist(),
            'affect_bias': self.b3.tolist(),
        }
        with open(path, 'w') as f:
            json.dump(state_dict, f)
        logger.info(f"Amygdala state saved to {path}")
    
    def load_state(self, path: str) -> None:
        """Load amygdala state including 3-layer MLP weights and Adam state"""
        if not Path(path).exists():
            return
        
        with open(path, 'r') as f:
            state_dict = json.load(f)
        
        self.state = EmotionalState.from_dict(state_dict['state'])
        self.stats = state_dict['stats']
        self.positive_patterns = [np.array(p) for p in state_dict.get('positive_patterns', [])]
        self.negative_patterns = [np.array(p) for p in state_dict.get('negative_patterns', [])]
        # Load learned emotion lexicon
        learned_emotions = state_dict.get('learned_emotion_lexicon', {})
        self.learned_emotion_lexicon = {k: tuple(v) for k, v in learned_emotions.items()}
        
        # Load 3-layer MLP weights (new format)
        if 'W3' in state_dict:
            self.W1 = np.array(state_dict['W1'], dtype=np.float32)
            self.b1 = np.array(state_dict['b1'], dtype=np.float32)
            self.W2 = np.array(state_dict['W2'], dtype=np.float32)
            self.b2 = np.array(state_dict['b2'], dtype=np.float32)
            self.W3 = np.array(state_dict['W3'], dtype=np.float32)
            self.b3 = np.array(state_dict['b3'], dtype=np.float32)
            self.hidden_dim = state_dict.get('hidden_dim', self.W1.shape[0])
            self.leaky_slope = state_dict.get('leaky_slope', 0.01)
            self.dropout_rate = state_dict.get('dropout_rate', 0.1)
            
            # Load Adam state if present
            if 'adam_t' in state_dict:
                self.adam_t = state_dict['adam_t']
                self.adam_m = {k: np.array(v, dtype=np.float32) for k, v in state_dict['adam_m'].items()}
                self.adam_v = {k: np.array(v, dtype=np.float32) for k, v in state_dict['adam_v'].items()}
            else:
                self._init_adam_state()
            
            self.affect_weights = self.W3
            self.affect_bias = self.b3
        # 2-layer MLP format
        elif 'W1' in state_dict:
            self.W1 = np.array(state_dict['W1'], dtype=np.float32)
            self.b1 = np.array(state_dict['b1'], dtype=np.float32)
            old_W2 = np.array(state_dict['W2'], dtype=np.float32)
            old_b2 = np.array(state_dict['b2'], dtype=np.float32)
            # Convert 2-layer to 3-layer (W2 becomes identity-like, old W2 becomes W3)
            self.hidden_dim = self.W1.shape[0]
            self.W2 = np.eye(self.hidden_dim, dtype=np.float32) * 0.1
            self.b2 = np.zeros(self.hidden_dim, dtype=np.float32)
            self.W3 = old_W2.T if old_W2.shape[0] != 2 else old_W2
            self.b3 = old_b2
            self._init_adam_state()
        # Legacy linear format
        elif 'affect_weights' in state_dict:
            self.affect_weights = np.array(state_dict['affect_weights'], dtype=np.float32)
            self.affect_bias = np.array(state_dict.get('affect_bias', [0.0, 0.5]), dtype=np.float32)
            self.W3 = self.affect_weights if self.affect_weights.shape[0] == 2 else self.affect_weights.T
            self.b3 = self.affect_bias
            self._init_adam_state()
        
        self.is_calibrated = state_dict.get('is_calibrated', False)
        self.calibration_samples = state_dict.get('calibration_samples', 0)
        
        logger.info(f"Amygdala state loaded from {path}")
    
    # ==================== GPU Calibration ====================
    
    def calibrate_from_jsonl(
        self,
        filepath: Path,
        embed_fn: Callable[[str], np.ndarray],
        batch_size: int = 64,
        epochs: int = 3,
        learning_rate: float = 0.001,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Calibrate affect prediction using amygdala_affect.jsonl training data
        
        Uses GPU-accelerated Hebbian learning for efficient training.
        
        Args:
            filepath: Path to amygdala_affect.jsonl
            embed_fn: Function to generate embeddings from text
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate for weight updates
            limit: Maximum samples to use (None = all)
            
        Returns:
            Training statistics
        """
        logger.info(f"Calibrating Amygdala from {filepath}")
        
        # Initialize GPU if needed
        self._init_gpu()
        
        # Load training data
        data = list(self._stream_affect_jsonl(filepath, limit))
        if not data:
            logger.warning("No training data found")
            return {'error': 'No training data'}
        
        logger.info(f"Loaded {len(data)} training samples")
        
        # Generate embeddings for all samples
        embeddings = []
        targets = []
        
        for item in data:
            emb = embed_fn(item['text'])
            embeddings.append(emb)
            targets.append([item['valence'], item['arousal']])
        
        embeddings = np.array(embeddings, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)
        
        # Training loop
        n_samples = len(embeddings)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        all_losses = []
        
        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(n_samples)
            epoch_loss = 0.0
            
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]
                
                batch_emb = embeddings[batch_indices]
                batch_targets = targets[batch_indices]
                
                # Forward pass through MLP (linear outputs for training)
                fwd = self._forward_mlp(batch_emb, apply_output_activation=False, training=False)
                predictions = fwd['predictions']
                
                # Clamp predictions for loss computation
                clamped = predictions.copy()
                clamped[:, 0] = np.clip(clamped[:, 0], -1.0, 1.0)
                clamped[:, 1] = np.clip(clamped[:, 1], 0.0, 1.0)
                
                # Compute loss (MSE)
                loss = np.mean((clamped - batch_targets) ** 2)
                epoch_loss += loss
                
                # Backward pass with optimizer
                if self.vulkan is not None:
                    self._gpu_calibration_step(batch_emb, batch_targets, learning_rate)
                else:
                    self._cpu_calibration_step(batch_emb, batch_targets, learning_rate)
            
            avg_loss = epoch_loss / n_batches
            all_losses.append(avg_loss)
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Mark as calibrated
        self.is_calibrated = True
        self.calibration_samples = n_samples
        self.stats['calibration_loss'] = float(all_losses[-1])
        self.stats['calibration_samples'] = n_samples
        
        logger.info(f"[OK] Amygdala calibrated on {n_samples} samples")
        
        return {
            'samples': n_samples,
            'epochs': epochs,
            'final_loss': float(all_losses[-1]),
            'loss_history': [float(l) for l in all_losses],
            'is_calibrated': True
        }
    
    def _stream_affect_jsonl(
        self, 
        filepath: Path, 
        limit: Optional[int] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream items from amygdala_affect.jsonl"""
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if limit and count >= limit:
                    break
                
                try:
                    data = json.loads(line.strip())
                    
                    # Extract required fields
                    text = data.get('text', '')
                    affect = data.get('affect', {})
                    valence = affect.get('valence', 0.0)
                    arousal = affect.get('arousal', 0.5)
                    
                    if text and len(text) > 3:
                        yield {
                            'text': text,
                            'valence': valence,
                            'arousal': arousal,
                            'context': data.get('context', {})
                        }
                        count += 1
                except:
                    continue
    
    def _gpu_calibration_step(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        learning_rate: float
    ) -> None:
        """
        GPU-accelerated 3-layer MLP calibration step
        
        Uses GPU Hebbian learning for Layer 1 (largest matrix),
        CPU Adam for layers 2 and 3
        """
        try:
            batch_size = embeddings.shape[0]
            
            # Forward pass
            fwd = self._forward_mlp(embeddings, apply_output_activation=False, training=True)
            
            # Compute loss gradients (MSE)
            d_output = (fwd['output'] - targets) * (2.0 / batch_size)
            
            # ===== Backprop through Layer 3 =====
            dW3 = np.dot(d_output.T, fwd['h2'])
            db3 = np.sum(d_output, axis=0)
            d_h2 = np.dot(d_output, self.W3)
            
            if fwd['mask2'] is not None:
                d_h2 = d_h2 * fwd['mask2'] / (1.0 - self.dropout_rate)
            
            # ===== Backprop through Layer 2 =====
            leaky_mask2 = np.where(fwd['h2_pre'] > 0, 1.0, self.leaky_slope)
            d_h2_pre = d_h2 * leaky_mask2
            
            dW2 = np.dot(d_h2_pre.T, fwd['h1'])
            db2 = np.sum(d_h2_pre, axis=0)
            d_h1 = np.dot(d_h2_pre, self.W2) + d_h2_pre  # Residual
            
            if fwd['mask1'] is not None:
                d_h1 = d_h1 * fwd['mask1'] / (1.0 - self.dropout_rate)
            
            # ===== Backprop through Layer 1 =====
            leaky_mask1 = np.where(fwd['h1_pre'] > 0, 1.0, self.leaky_slope)
            d_h1_pre = d_h1 * leaky_mask1
            
            # GPU Hebbian for Layer 1 (largest: hidden_dim x embedding_dim)
            pre = embeddings.reshape(batch_size, 1, self.embedding_dim).astype(np.float32)
            post = (-d_h1_pre).reshape(batch_size, 1, self.hidden_dim).astype(np.float32)
            
            updated_W1 = self.vulkan.hebbian_learning(
                pre_activations=pre,
                post_activations=post,
                weights=self.W1.T,  # Transpose for shader format
                learning_rate=learning_rate,
                weight_decay=0.0001
            )
            
            self.W1 = updated_W1.reshape(self.embedding_dim, self.hidden_dim).T.copy()
            
            # Adam updates for layers 2, 3 and biases (CPU - smaller)
            self._adam_update('b1', self.b1, np.sum(d_h1_pre, axis=0), learning_rate)
            self._adam_update('W2', self.W2, dW2, learning_rate)
            self._adam_update('b2', self.b2, db2, learning_rate)
            self._adam_update('W3', self.W3, dW3, learning_rate)
            self._adam_update('b3', self.b3, db3, learning_rate)
            
            self.adam_t += 1
            
            # Update aliases
            self.affect_weights = self.W3
            self.affect_bias = self.b3
            
        except Exception as e:
            logger.debug(f"GPU calibration failed: {e}, falling back to CPU")
            self._cpu_calibration_step(embeddings, targets, learning_rate)
    
    def _cpu_calibration_step(
        self,
        embeddings: np.ndarray,
        targets: np.ndarray,
        learning_rate: float
    ) -> None:
        """
        CPU 3-layer MLP calibration step with Adam optimizer
        
        Uses LeakyReLU, residual connections, and Adam for stable training
        """
        batch_size = len(embeddings)
        
        # Forward pass with training=True for dropout
        fwd = self._forward_mlp(embeddings, apply_output_activation=False, training=True)
        
        # Compute loss gradients (MSE)
        d_output = (fwd['output'] - targets) * (2.0 / batch_size)
        
        # ===== Backprop through Layer 3 =====
        dW3 = np.dot(d_output.T, fwd['h2'])
        db3 = np.sum(d_output, axis=0)
        
        # Backprop to h2
        d_h2 = np.dot(d_output, self.W3)
        
        # Apply dropout mask if used
        if fwd['mask2'] is not None:
            d_h2 = d_h2 * fwd['mask2'] / (1.0 - self.dropout_rate)
        
        # ===== Backprop through Layer 2 (LeakyReLU + Residual) =====
        leaky_mask2 = np.where(fwd['h2_pre'] > 0, 1.0, self.leaky_slope)
        d_h2_pre = d_h2 * leaky_mask2
        
        dW2 = np.dot(d_h2_pre.T, fwd['h1'])
        db2 = np.sum(d_h2_pre, axis=0)
        
        # Backprop to h1 (through W2 and residual)
        d_h1 = np.dot(d_h2_pre, self.W2) + d_h2_pre  # Residual gradient
        
        # Apply dropout mask
        if fwd['mask1'] is not None:
            d_h1 = d_h1 * fwd['mask1'] / (1.0 - self.dropout_rate)
        
        # ===== Backprop through Layer 1 (LeakyReLU) =====
        leaky_mask1 = np.where(fwd['h1_pre'] > 0, 1.0, self.leaky_slope)
        d_h1_pre = d_h1 * leaky_mask1
        
        dW1 = np.dot(d_h1_pre.T, embeddings)
        db1 = np.sum(d_h1_pre, axis=0)
        
        # ===== Adam optimizer update =====
        self._adam_update('W1', self.W1, dW1, learning_rate)
        self._adam_update('b1', self.b1, db1, learning_rate)
        self._adam_update('W2', self.W2, dW2, learning_rate)
        self._adam_update('b2', self.b2, db2, learning_rate)
        self._adam_update('W3', self.W3, dW3, learning_rate)
        self._adam_update('b3', self.b3, db3, learning_rate)
        
        self.adam_t += 1
        
        # Update aliases
        self.affect_weights = self.W3
        self.affect_bias = self.b3
    
    def _adam_update(self, name: str, param: np.ndarray, grad: np.ndarray, lr: float,
                     beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
                     weight_decay: float = 0.0001) -> None:
        """
        Adam optimizer update with GPU acceleration when available
        """
        # Try GPU Adam if available and parameter is large enough
        if self.use_gpu and hasattr(self, 'vulkan') and self.vulkan is not None and len(param.flatten()) > 100:
            try:
                # Get or initialize Adam moments
                if not hasattr(self, '_adam_moments'):
                    self._adam_moments = {}
                
                if name not in self._adam_moments:
                    self._adam_moments[name] = {
                        'm': np.zeros_like(param),
                        'v': np.zeros_like(param)
                    }
                
                moments = self._adam_moments[name]
                
                # Use GPU Adam update
                updated_param, updated_m, updated_v = self.vulkan.affect_adam_update(
                    weights=param.flatten(),
                    gradients=grad.flatten(),
                    moment1=moments['m'].flatten(),
                    moment2=moments['v'].flatten(),
                    learning_rate=lr,
                    timestep=self.adam_t,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=eps,
                    weight_decay=weight_decay
                )
                
                # Update parameter and moments
                param[:] = updated_param.reshape(param.shape)
                moments['m'][:] = updated_m.reshape(param.shape)
                moments['v'][:] = updated_v.reshape(param.shape)
                return
            except Exception as e:
                logger.debug(f"GPU Adam failed for {name}: {e}, using CPU")
        
        # CPU implementation (original)
        # Add weight decay
        grad = grad + weight_decay * param
        
        # Initialize moments if needed
        if not hasattr(self, 'adam_m'):
            self.adam_m = {}
            self.adam_v = {}
        
        if name not in self.adam_m:
            self.adam_m[name] = np.zeros_like(param)
            self.adam_v[name] = np.zeros_like(param)
        
        # Update moments
        self.adam_m[name] = beta1 * self.adam_m[name] + (1 - beta1) * grad
        self.adam_v[name] = beta2 * self.adam_v[name] + (1 - beta2) * (grad ** 2)
        
        # Bias correction
        t = self.adam_t + 1
        m_hat = self.adam_m[name] / (1 - beta1 ** t)
        v_hat = self.adam_v[name] / (1 - beta2 ** t)
        
        # Update parameter in-place
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)
        """Apply Adam optimizer update to a parameter"""
        # Add weight decay
        grad = grad + weight_decay * param
        
        # Update moments
        self.adam_m[name] = beta1 * self.adam_m[name] + (1 - beta1) * grad
        self.adam_v[name] = beta2 * self.adam_v[name] + (1 - beta2) * (grad ** 2)
        
        # Bias correction
        t = self.adam_t + 1
        m_hat = self.adam_m[name] / (1 - beta1 ** t)
        v_hat = self.adam_v[name] / (1 - beta2 ** t)
        
        # Update parameter in-place
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)
    
    def evaluate_calibration(
        self,
        filepath: Path,
        embed_fn: Callable[[str], np.ndarray],
        limit: int = 100
    ) -> Dict[str, float]:
        """
        Evaluate calibration accuracy on test data
        
        Args:
            filepath: Path to amygdala_affect.jsonl
            embed_fn: Function to generate embeddings
            limit: Number of samples to evaluate
            
        Returns:
            Evaluation metrics
        """
        data = list(self._stream_affect_jsonl(filepath, limit))
        
        if not data:
            return {'error': 'No data'}
        
        valence_errors = []
        arousal_errors = []
        
        for item in data:
            emb = embed_fn(item['text'])
            pred_v, pred_a = self._predict_affect_neural(emb)
            
            valence_errors.append(abs(pred_v - item['valence']))
            arousal_errors.append(abs(pred_a - item['arousal']))
        
        return {
            'valence_mae': float(np.mean(valence_errors)),
            'arousal_mae': float(np.mean(arousal_errors)),
            'valence_std': float(np.std(valence_errors)),
            'arousal_std': float(np.std(arousal_errors)),
            'samples': len(data),
            'is_calibrated': self.is_calibrated
        }

