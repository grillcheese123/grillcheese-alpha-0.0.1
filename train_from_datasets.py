"""
Train GrillCheese Brain from Datasets

This script loads the created datasets and trains:
1. ContinuousLearner - STDP learning from conversations
2. UnifiedBrain - Emotional understanding (Amygdala calibration)
3. Memory Store - Store conversation memories
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse

try:
    from tqdm import tqdm as _tqdm
    import tqdm as tqdm_module
    TQDM_AVAILABLE = True
    # Wrap tqdm to use minimal format by default (percentage only)
    def tqdm(iterable=None, desc="", bar_format=None, **kwargs):
        # Use minimal format if bar_format not explicitly provided
        if bar_format is None:
            bar_format = '{desc}: {percentage:3.0f}%'
        return _tqdm(iterable, desc=desc, bar_format=bar_format, **kwargs)
    # Also monkey-patch the module's tqdm to use minimal format by default
    _original_tqdm = tqdm_module.tqdm
    def _patched_tqdm(iterable=None, desc="", bar_format=None, **kwargs):
        if bar_format is None:
            bar_format = '{desc}: {percentage:3.0f}%'
        return _original_tqdm(iterable, desc=desc, bar_format=bar_format, **kwargs)
    tqdm_module.tqdm = _patched_tqdm
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback progress bar - simple class that mimics tqdm interface
    class tqdm:
        def __init__(self, iterable=None, desc="", bar_format=None, **kwargs):
            self.iterable = iterable
            self.desc = desc
            # Use minimal format if bar_format not explicitly provided
            self.bar_format = bar_format if bar_format is not None else '{desc}: {percentage:3.0f}%'
            self.total = len(iterable) if iterable and hasattr(iterable, '__len__') else None
            self.current = 0
            
        def __iter__(self):
            for item in self.iterable:
                self.current += 1
                if self.bar_format and '{percentage' in self.bar_format and self.total:
                    pct = int(self.current / self.total * 100)
                    print(f"\r{self.desc}: {pct}%", end='', flush=True)
                yield item
            if self.bar_format and '{percentage' in self.bar_format:
                print()  # New line after completion

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import MemoryConfig, SNNConfig, ModelConfig, find_gguf_model
from memory_store import MemoryStore
from vulkan_backend import SNNCompute
from learning import ContinuousLearner, LearningConfig
from brain import UnifiedBrain

# Try to import model
try:
    from model_gguf import Phi3GGUF
    GGUF_AVAILABLE = True
except ImportError:
    try:
        from model import Phi3Model as Phi3GGUF
        GGUF_AVAILABLE = True
    except ImportError:
        GGUF_AVAILABLE = False
        print("Warning: No model backend available")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetTrainer:
    """Train GrillCheese brain components from datasets"""
    
    def __init__(
        self,
        memory: MemoryStore,
        snn: SNNCompute,
        learner: ContinuousLearner,
        brain: Optional[UnifiedBrain] = None,
        model = None
    ):
        self.memory = memory
        self.snn = snn
        self.learner = learner
        self.brain = brain
        self.model = model
        self.stats = {
            'conversations_processed': 0,
            'instructions_processed': 0,
            'memories_stored': 0,
            'stdp_updates': 0,
            'brain_experiences': 0,
        }
    
    def train_from_conversations(
        self,
        dataset_path: Path,
        limit: Optional[int] = None,
        store_memories: bool = True
    ) -> Dict[str, Any]:
        """Train from conversations dataset (JSONL)"""
        logger.info(f"Training from conversations: {dataset_path}")
        
        conversations = []
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                conversations.append(json.loads(line))
                if limit and len(conversations) >= limit:
                    break
        
        logger.info(f"Loaded {len(conversations)} conversations")
        
        # Process conversations
        for conv in tqdm(conversations, desc="Training from conversations", bar_format='{desc}: {percentage:3.0f}%'):
            messages = conv.get('messages', [])
            if len(messages) < 2:
                continue
            
            # Process message pairs
            for i in range(len(messages) - 1):
                user_msg = messages[i]
                assistant_msg = messages[i + 1]
                
                if user_msg.get('role') != 'user' or assistant_msg.get('role') != 'assistant':
                    continue
                
                user_text = user_msg.get('content', '')
                assistant_text = assistant_msg.get('content', '')
                
                if not user_text or not assistant_text:
                    continue
                
                # Store in memory
                if store_memories:
                    try:
                        if self.model:
                            user_emb = self.model.get_embedding(user_text)
                            self.memory.store(user_emb, user_text)
                            self.stats['memories_stored'] += 1
                    except Exception as e:
                        logger.warning(f"Memory storage error: {e}")
                
                # Learn from conversation
                try:
                    context = self.memory.retrieve(
                        self.model.get_embedding(user_text) if self.model else None,
                        k=3
                    ) if self.model else []
                    
                    result = self.learner.learn_from_conversation(
                        user_text,
                        assistant_text,
                        context
                    )
                    self.stats['stdp_updates'] += result.get('stdp_updates', 0)
                    self.stats['conversations_processed'] += 1
                except Exception as e:
                    logger.warning(f"Learning error: {e}")
                
                # Train brain if available
                if self.brain and self.model:
                    try:
                        user_emb = self.model.get_embedding(user_text)
                        brain_result = self.brain.process(user_text, user_emb)
                        
                        # Provide feedback for learning
                        self.brain.provide_feedback(
                            quality=0.8,
                            strategy_worked=True
                        )
                        self.stats['brain_experiences'] += 1
                    except Exception as e:
                        logger.warning(f"Brain processing error: {e}")
        
        return self.stats.copy()
    
    def train_from_instructions(
        self,
        dataset_path: Path,
        limit: Optional[int] = None,
        store_memories: bool = True
    ) -> Dict[str, Any]:
        """Train from instructions dataset (JSON)"""
        logger.info(f"Training from instructions: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            instructions = json.load(f)
        
        if limit:
            instructions = instructions[:limit]
        
        logger.info(f"Loaded {len(instructions)} instructions")
        
        # Process instructions
        for inst in tqdm(instructions, desc="Training from instructions", bar_format='{desc}: {percentage:3.0f}%'):
            instruction = inst.get('instruction', '')
            output = inst.get('output', '')
            
            if not instruction or not output:
                continue
            
            # Store in memory
            if store_memories and self.model:
                try:
                    inst_emb = self.model.get_embedding(instruction)
                    self.memory.store(inst_emb, instruction)
                    self.stats['memories_stored'] += 1
                except Exception as e:
                    logger.warning(f"Memory storage error: {e}")
            
            # Learn from instruction-response pair
            try:
                context = self.memory.retrieve(
                    self.model.get_embedding(instruction) if self.model else None,
                    k=3
                ) if self.model else []
                
                result = self.learner.learn_from_conversation(
                    instruction,
                    output,
                    context
                )
                self.stats['stdp_updates'] += result.get('stdp_updates', 0)
                self.stats['instructions_processed'] += 1
            except Exception as e:
                logger.warning(f"Learning error: {e}")
            
            # Train brain if available
            if self.brain and self.model:
                try:
                    inst_emb = self.model.get_embedding(instruction)
                    brain_result = self.brain.process(instruction, inst_emb)
                    
                    self.brain.provide_feedback(
                        quality=0.8,
                        strategy_worked=True
                    )
                    self.stats['brain_experiences'] += 1
                except Exception as e:
                    logger.warning(f"Brain processing error: {e}")
        
        return self.stats.copy()
    
    def train_brain_emotions(
        self,
        conversations_path: Optional[Path] = None,
        affect_data_path: Optional[Path] = None,
        epochs: int = 3,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Train brain's emotional understanding"""
        if not self.brain or not self.model:
            logger.warning("Brain or model not available for emotion training")
            return {}
        
        logger.info(f"Training emotional understanding")
        
        # Use affect data if provided, otherwise extract from conversations
        if affect_data_path and affect_data_path.exists():
            data_path = affect_data_path
        elif conversations_path and conversations_path.exists():
            # Create temporary affect data file from conversations
            logger.info("Extracting emotional training data from conversations...")
            training_data = []
            with open(conversations_path, 'r', encoding='utf-8') as f:
                for line in f:
                    conv = json.loads(line)
                    messages = conv.get('messages', [])
                    
                    for msg in messages:
                        text = msg.get('content', '')
                        if text and len(text) > 10:
                            # Create affect entry (default neutral)
                            training_data.append({
                                'text': text,
                                'valence': 0.0,
                                'arousal': 0.5
                            })
                    
                    if limit and len(training_data) >= limit:
                        break
            
            # Save temporary affect file
            temp_path = Path('temp_affect_training.jsonl')
            with open(temp_path, 'w', encoding='utf-8') as f:
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            data_path = temp_path
            logger.info(f"Created temporary affect data: {len(training_data)} samples")
        else:
            logger.warning("No data source provided for emotion training")
            return {'error': 'No data source'}
        
        # Train amygdala
        try:
            result = self.brain.calibrate_affect(
                embed_fn=self.model.get_embedding,
                data_path=data_path,
                epochs=epochs,
                limit=limit
            )
            logger.info(f"Amygdala calibration complete: {result}")
            
            # Clean up temp file if created
            if data_path.name == 'temp_affect_training.jsonl' and data_path.exists():
                data_path.unlink()
            
            return result
        except Exception as e:
            logger.error(f"Emotion training error: {e}")
            return {'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description='Train GrillCheese brain from datasets')
    parser.add_argument(
        '--conversations',
        type=str,
        default='../../datasets/conversations_dataset_anonymized_cleaned.jsonl',
        help='Path to conversations dataset (JSONL)'
    )
    parser.add_argument(
        '--instructions',
        type=str,
        default='../../datasets/instruct_anonymized_cleaned.json',
        help='Path to instructions dataset (JSON)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of items to process (for testing)'
    )
    parser.add_argument(
        '--no-memories',
        action='store_true',
        help='Skip storing memories'
    )
    parser.add_argument(
        '--train-emotions',
        action='store_true',
        help='Train emotional understanding (Amygdala calibration)'
    )
    parser.add_argument(
        '--emotion-epochs',
        type=int,
        default=3,
        help='Epochs for emotion training'
    )
    parser.add_argument(
        '--db',
        type=str,
        default=None,
        help='Path to memory database'
    )
    
    args = parser.parse_args()
    
    # Initialize components
    logger.info("Initializing GrillCheese components...")
    
    # Memory store
    db_path = args.db or MemoryConfig.DB_PATH
    memory = MemoryStore(db_path=db_path)
    logger.info(f"Memory store initialized: {db_path}")
    
    # SNN compute
    snn = SNNCompute()
    logger.info("SNN compute initialized")
    
    # Model (for embeddings)
    model = None
    if GGUF_AVAILABLE:
        try:
            model_path = find_gguf_model()
            if model_path:
                model = Phi3GGUF(model_path=model_path)
                logger.info(f"Model initialized: {model_path}")
            else:
                logger.warning("No model found, embeddings will be limited")
        except Exception as e:
            logger.warning(f"Model initialization error: {e}")
    else:
        logger.warning("Model backend not available")
    
    # Continuous learner
    learner_config = LearningConfig()
    learner = ContinuousLearner(
        memory_store=memory,
        snn_compute=snn,
        embedder=model,
        config=learner_config
    )
    logger.info("Continuous learner initialized")
    
    # Unified brain (GPU enabled - fixed create_buffer issues)
    brain = None
    # Enable reranking only if model is available
    enable_reranking = model is not None
    if enable_reranking:
        logger.info("Reranking will be enabled (model available)")
    else:
        logger.info("Reranking disabled (no model available)")
    
    try:
        brain = UnifiedBrain(
            memory_store=memory,
            embedding_dim=ModelConfig.EMBEDDING_DIM,
            use_gpu=True,  # GPU now works with fixed create_buffer API
            model=model,  # Pass model for reranking
            enable_reranking=False  # Enable reranking if model available
        )
        rerank_status = "reranking enabled" if enable_reranking else "reranking disabled"
        logger.info(f"Unified brain initialized (GPU mode, {rerank_status})")
    except Exception as e:
        logger.warning(f"Brain initialization error: {e}")
        logger.info("Trying CPU fallback...")
        try:
            brain = UnifiedBrain(
                memory_store=memory,
                embedding_dim=ModelConfig.EMBEDDING_DIM,
                use_gpu=False,
                model=model,  # Pass model for reranking even in CPU mode
                enable_reranking=enable_reranking
            )
            rerank_status = "reranking enabled" if enable_reranking else "reranking disabled"
            logger.info(f"Unified brain initialized (CPU fallback, {rerank_status})")
        except Exception as e2:
            logger.warning(f"Brain initialization failed: {e2}")
            logger.info("Continuing without brain component...")
    
    # Create trainer
    trainer = DatasetTrainer(
        memory=memory,
        snn=snn,
        learner=learner,
        brain=brain,
        model=model
    )
    
    # Train from conversations
    conv_path = Path(args.conversations)
    if conv_path.exists():
        logger.info("\n" + "="*60)
        logger.info("Training from conversations dataset")
        logger.info("="*60)
        stats = trainer.train_from_conversations(
            conv_path,
            limit=args.limit,
            store_memories=not args.no_memories
        )
        logger.info(f"\nConversation training complete:")
        logger.info(f"  Conversations processed: {stats['conversations_processed']}")
        logger.info(f"  Memories stored: {stats['memories_stored']}")
        logger.info(f"  STDP updates: {stats['stdp_updates']}")
        logger.info(f"  Brain experiences: {stats['brain_experiences']}")
    else:
        logger.warning(f"Conversations dataset not found: {conv_path}")
    
    # Train from instructions
    inst_path = Path(args.instructions)
    if inst_path.exists():
        logger.info("\n" + "="*60)
        logger.info("Training from instructions dataset")
        logger.info("="*60)
        stats = trainer.train_from_instructions(
            inst_path,
            limit=args.limit,
            store_memories=not args.no_memories
        )
        logger.info(f"\nInstruction training complete:")
        logger.info(f"  Instructions processed: {stats['instructions_processed']}")
        logger.info(f"  Memories stored: {stats['memories_stored']}")
        logger.info(f"  STDP updates: {stats['stdp_updates']}")
        logger.info(f"  Brain experiences: {stats['brain_experiences']}")
    else:
        logger.warning(f"Instructions dataset not found: {inst_path}")
    
    # Train emotional understanding
    if args.train_emotions:
        logger.info("\n" + "="*60)
        logger.info("Training emotional understanding")
        logger.info("="*60)
        
        # Check for default affect data
        default_affect = Path(__file__).parent.parent.parent / "data_learning" / "jsonl" / "amygdala_affect.jsonl"
        affect_path = default_affect if default_affect.exists() else None
        
        result = trainer.train_brain_emotions(
            conversations_path=conv_path if conv_path.exists() else None,
            affect_data_path=affect_path,
            epochs=args.emotion_epochs,
            limit=args.limit
        )
        logger.info(f"\nEmotion training complete: {result}")
    
    # Save state
    logger.info("\nSaving state...")
    learner._save_state()
    if brain:
        brain.save_state()
    memory.close()
    
    logger.info("\n" + "="*60)
    logger.info("Training complete!")
    logger.info("="*60)
    logger.info(f"\nFinal statistics:")
    logger.info(f"  Total conversations: {trainer.stats['conversations_processed']}")
    logger.info(f"  Total instructions: {trainer.stats['instructions_processed']}")
    logger.info(f"  Total memories: {trainer.stats['memories_stored']}")
    logger.info(f"  Total STDP updates: {trainer.stats['stdp_updates']}")
    logger.info(f"  Brain experiences: {trainer.stats['brain_experiences']}")


if __name__ == '__main__':
    main()
