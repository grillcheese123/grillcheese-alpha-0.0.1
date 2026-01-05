#!/usr/bin/env python3
"""
GrillCheese CLI - Command-line interface for the AI assistant

Usage:
    python cli.py "Your prompt here"
    python cli.py                    # Interactive mode
    python cli.py --stats            # Show memory statistics
    python cli.py --clear            # Clear all memories
    python cli.py --learning         # Enable continuous learning
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

from grillcheese.backend.identity import DEFAULT_IDENTITY
from grillcheese.backend.model_gguf import find_gguf_model

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from grillcheese.backend.config import LogConfig, MemoryConfig
from grillcheese.backend.memory_store import MemoryStore

# Configure logging
logging.basicConfig(level=LogConfig.LEVEL, format=LogConfig.FORMAT)
logger = logging.getLogger(__name__)

# Try to import model backends
try:
    from model_gguf import Phi3GGUF
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False

try:
    from model import Phi3Model
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Try to import continuous learning
try:
    from learning import ContinuousLearner, LearningConfig
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False

# Try to import brain module
try:
    from brain import UnifiedBrain
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False

from learning.multimodal_encoder import MultimodalEncoder
from learning.multilingual_utils import MultilingualProcessor
from learning.knowledge_distillation import KnowledgeDistillation

class GrillCheeseCLI:
    def __init__(self, embedding_dim: int = MemoryConfig.EMBEDDING_DIM):
        self.learner = None
        self.model = None
        self.embedding_dim = embedding_dim
        self.memory = MemoryStore(
            db_path=MemoryConfig.DB_PATH,
            max_memories=MemoryConfig.MAX_MEMORIES,
            embedding_dim=self.embedding_dim,
            identity=DEFAULT_IDENTITY
        )
        self.brain = UnifiedBrain(
            memory_store=self.memory,
            embedding_dim=self.embedding_dim,
            state_dir="brain_state",
            use_gpu=True
        )

        self.encoder = MultimodalEncoder(models_dir="models")
        self.lang_processor = MultilingualProcessor(primary_language='en')
        self.distillation = KnowledgeDistillation(
            memory_store=self.memory,
            encoder=self.encoder,
            lang_processor=self.lang_processor,
            quality_threshold=0.7
        )

    def _init_model(self):
        """Initialize the language model"""
        print("Loading Phi-3 model...")
        
        # Try GGUF first
        if GGUF_AVAILABLE:
            model_path = find_gguf_model()
            if model_path:
                try:
                    print(f"Using GGUF model: {model_path}")
                    self.model = Phi3GGUF(model_path=model_path, n_gpu_layers=-1)
                    print(f"{LogConfig.CHECK} GGUF model loaded (GPU accelerated)")
                except Exception as e:
                    print(f"{LogConfig.WARNING} Failed to load GGUF model: {e}")

        if not GGUF_AVAILABLE:
            print(f"{LogConfig.CROSS} Failed to load model: No model backend available")
            return None

    def _show_stats(self):
        """Display memory statistics"""
        stats = self.memory.get_stats()
        print("\n=== Memory Statistics ===")
        print(f"Total memories: {stats['total_memories']}")
        print(f"GPU memories: {stats['gpu_memories']}")
        print(f"Max memories: {stats['max_memories']}")
        print(f"Total accesses: {stats['total_accesses']}")
        print(f"Embedding dimension: {stats['embedding_dim']}")
        print(f"GPU similarity: {'enabled' if stats.get('gpu_enabled', False) else 'disabled'}")
        if self.memory.get_identity():
            print(f"\nSystem identity: {self.memory.get_identity()[:100]}...")
        
        if self.learner:
            print("\n=== Learning Statistics ===")
            lstats = self.learner.get_stats()
            print(f"Conversations learned: {lstats['conversations_learned']}")
            print(f"STDP updates: {lstats['stdp_updates']}")
            print(f"Spikes generated: {lstats['spikes_generated']}")

    def run(self, user_input: str):
        """Process user input and generate response"""
        if not user_input.strip():
            return "Please provide a prompt or question."
        
        # Distill user input into knowledge
        knowledge = self.distillation.distill_input(user_input)
        if knowledge:
            print(f"✓ Distilled [{knowledge.language}] quality={knowledge.quality_score:.3f}")
            return knowledge.summary
        
        return "No relevant knowledge found in the memory."
        
    def interactive_mode(self):
        """Interactive mode for natural conversation"""
        print("Welcome to GrillCheese CLI! Type 'exit' to quit")
        print("Type 'help' to show available commands")
        while True:
            user_input = input("> ")
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'help':
                print("Available commands:")
                print("  exit    - Exit the CLI")
                print("  help    - Show this help message")
                print("  stats   - Show memory statistics")
                print("  clear   - Clear all memories")
                print("  learning - Enable continuous learning")
            else:
                response = self.run(user_input)
                print(response)
                
    def show_memory_stats(self):
        """Show memory statistics"""
        print("\nMemory Statistics:")
        print(f"  Total memories: {self.memory.total_memories()}")
        print(f"  Memory usage: {self.memory.memory_usage()}%")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Identity: {self.memory.get_identity()}")
        print(f"  Total interactions: {self.brain.stats['total_interactions']}")
        print(f"  Positive interactions: {self.brain.stats['positive_interactions']}")
        print(f"  Negative interactions: {self.brain.stats['negative_interactions']}")
        print(f"  Empathetic responses: {self.brain.stats['empathetic_responses']}")
        print(f"  Informative responses: {self.brain.stats['informative_responses']}")
        print(f"  GPU operations: {self.brain.stats['gpu_operations']}")
        print(f"  Experiences learned: {self.brain.stats['experiences_learned']}")
        print(f"  Online learning updates: {self.brain.stats['online_learning_updates']}")

    def show_learning_stats(self):
        """Show learning statistics"""
        print("\nLearning Statistics:")
        print(f"  Conversations learned: {self.learner.get_stats()['conversations_learned']}")
        print(f"  STDP updates: {self.learner.get_stats()['stdp_updates']}")
        print(f"  Spikes generated: {self.learner.get_stats()['spikes_generated']}")

    def show_all_stats(self):
        """Show all statistics"""
        self._show_stats()
        self.show_memory_stats()
        self.show_learning_stats()
    
    def clear_memory(self):
        """Clear all memories"""
        self.memory.clear_all()
        print("All memories cleared.")
        