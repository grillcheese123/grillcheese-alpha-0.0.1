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
import os
import sys
import subprocess
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MemoryConfig, SNNConfig, LogConfig, ModelConfig, ModuleConfig, find_gguf_model
from memory_store import MemoryStore
from identity import DEFAULT_IDENTITY
from vulkan_backend import SNNCompute
from learning.multimodal_encoder import MultimodalEncoder
from learning.multilingual_utils import MultilingualProcessor
from learning.knowledge_distillation import KnowledgeDistillation

# Configure logging first (before any logger usage)
logging.basicConfig(level=LogConfig.LEVEL, format=LogConfig.FORMAT)
logger = logging.getLogger(__name__)

# Module system imports
try:
    from modules.registry import ModuleRegistry
    from modules.tools import ToolExecutor
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    logger.warning("Module system not available, using legacy initialization")

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
LEARNING_AVAILABLE = False
LearningConfig = None
ContinuousLearner = None
try:
    from learning import ContinuousLearner, LearningConfig
    LEARNING_AVAILABLE = True
except ImportError as e:
    pass  # Already set to False/None above

# Try to import brain module
try:
    from brain import UnifiedBrain
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False


def run():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="GrillCheese AI - Local AI assistant with persistent memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "What is machine learning?"
  python cli.py --interactive
  python cli.py --stats
        """
    )
    
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Prompt to send to the AI"
    )
    
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Start interactive mode"
    )
    
    parser.add_argument(
        "-s", "--stats",
        action="store_true",
        help="Show memory statistics and exit"
    )
    
    parser.add_argument(
        "--db",
        type=str,
        default=MemoryConfig.DB_PATH,
        help=f"Path to memory database (default: {MemoryConfig.DB_PATH})"
    )
    
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all memories (use with caution!)"
    )
    
    parser.add_argument(
        "--identity",
        type=str,
        help="Update system identity with custom text"
    )
    
    parser.add_argument(
        "-l", "--learning",
        action="store_true",
        help="Enable continuous STDP learning"
    )
    
    parser.add_argument(
        "--vocab-dir",
        type=str,
        help="Directory for vocabulary ingestion (txt files)"
    )
    
    parser.add_argument(
        "--learning-stats",
        action="store_true",
        help="Show learning statistics"
    )
    
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate the Amygdala affect prediction (trains emotional understanding)"
    )
    
    parser.add_argument(
        "--calibrate-epochs",
        type=int,
        default=20,
        help="Number of epochs for calibration (default: 20)"
    )
    
    parser.add_argument(
        "--teach",
        action="store_true",
        help="Enter protected teaching mode - memories created here will never be deleted"
    )
    
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enter developer mode (password required) - advanced model improvement tools"
    )
    
    parser.add_argument(
        "--train-temporal",
        type=str,
        help="Train on temporal dataset (path to temporal_dataset.jsonl)"
    )
    
    parser.add_argument(
        "--train-temporal-limit",
        type=int,
        default=None,
        help="Limit number of items to process from temporal dataset"
    )
    
    parser.add_argument(
        "--train-conversations",
        type=str,
        help="Train on conversational dataset (path to conversations_dataset.jsonl)"
    )
    
    parser.add_argument(
        "--train-conversations-limit",
        type=int,
        default=None,
        help="Limit number of conversations to process from conversational dataset"
    )
    
    parser.add_argument(
        "--train-conversations-no-memory",
        action="store_true",
        help="Skip storing conversations in memory (only perform STDP learning)"
    )
    
    parser.add_argument(
        "--train-tools",
        type=str,
        metavar="PATH",
        help="Train on tool usage examples from JSONL file (e.g., data/tool_training.jsonl)"
    )
    parser.add_argument(
        "--train-tools-limit",
        type=int,
        metavar="N",
        help="Limit number of tool examples to process"
    )
    parser.add_argument(
        "--train-tools-no-memory",
        action="store_true",
        help="Train on tool examples without storing in memory"
    )
    
    parser.add_argument(
        "--hilbert",
        action="store_true",
        help="Enable Hilbert Multiverse Routing for enhanced semantic similarity (experimental)"
    )
    
    args = parser.parse_args()
    
    # Initialize system
    print("Initializing GrillCheese AI...")
    
    # Initialize module system
    registry = None
    tool_executor = None
    if MODULES_AVAILABLE:
        try:
            print("Loading modules...")
            registry = ModuleRegistry()
            registry.load_all_modules(
                modules_dir=ModuleConfig.MODULES_DIR,
                config_path=ModuleConfig.MODULES_CONFIG_FILE
            )
            
            # Get model and memory from registry
            phi3 = registry.get_active_model_provider()
            memory_backend = registry.get_active_memory_backend()
            
            if memory_backend:
                memory = memory_backend
                embedding_dim = memory.embedding_dim
            else:
                # Fallback to direct initialization
                print("No memory backend found in registry, using legacy initialization")
                phi3 = _init_model()
                if phi3 is None:
                    print(f"{LogConfig.CROSS} No model available")
                    sys.exit(1)
                embedding_dim = ModelConfig.detect_embedding_dim(phi3) if phi3 else MemoryConfig.EMBEDDING_DIM
                use_hilbert = args.hilbert or MemoryConfig.USE_HILBERT_ROUTING
                memory = MemoryStore(
                    db_path=args.db,
                    embedding_dim=embedding_dim,
                    identity=DEFAULT_IDENTITY,
                    use_hilbert=use_hilbert
                )
            
            if not phi3:
                # Fallback to direct initialization (GGUF only)
                print("No model provider found in registry, initializing GGUF model...")
                phi3 = _init_model()
                if phi3 is None:
                    print(f"{LogConfig.CROSS} Failed to initialize GGUF model")
                    print(f"{LogConfig.WARNING} Please ensure GGUF model is available")
                    sys.exit(1)
            
            # Initialize tool executor
            tool_executor = ToolExecutor(registry)
            
            print(f"{LogConfig.CHECK} Module system initialized")
        except Exception as e:
            print(f"{LogConfig.WARNING} Module system initialization failed: {e}")
            print("Falling back to legacy initialization")
            registry = None
            phi3 = _init_model()
            if phi3 is None:
                print(f"{LogConfig.CROSS} No model available")
                sys.exit(1)
            embedding_dim = ModelConfig.detect_embedding_dim(phi3) if phi3 else MemoryConfig.EMBEDDING_DIM
            use_hilbert = args.hilbert or MemoryConfig.USE_HILBERT_ROUTING
            memory = MemoryStore(
                db_path=args.db,
                embedding_dim=embedding_dim,
                identity=DEFAULT_IDENTITY,
                use_hilbert=use_hilbert
            )
    else:
        # Legacy initialization
        phi3 = _init_model()
        if phi3 is None:
            print(f"{LogConfig.CROSS} No model available")
            sys.exit(1)
        
        # Get embedding dimension from model (auto-detected)
        embedding_dim = ModelConfig.detect_embedding_dim(phi3) if phi3 else MemoryConfig.EMBEDDING_DIM
        logger.info(f"Detected embedding dimension: {embedding_dim}")
        
        # Initialize memory store
        print("Initializing memory store...")
        try:
            # Enable Hilbert routing if requested
            use_hilbert = args.hilbert or MemoryConfig.USE_HILBERT_ROUTING
            if use_hilbert:
                print("  [Hilbert Multiverse Routing: ENABLED]")
            
            memory = MemoryStore(
                db_path=args.db,
                embedding_dim=embedding_dim,
                identity=DEFAULT_IDENTITY,
                use_hilbert=use_hilbert
            )
            print(f"{LogConfig.CHECK} Memory store initialized")
        except Exception as e:
            print(f"{LogConfig.CROSS} Failed to initialize memory store: {e}")
            sys.exit(1)
    
    # Initialize SNN
    print("Initializing GPU backend...")
    try:
        snn = SNNCompute(n_neurons=SNNConfig.N_NEURONS, use_vulkan=True)
        print(f"{LogConfig.CHECK} GPU backend ready")
    except Exception as e:
        print(f"{LogConfig.CROSS} Failed to initialize GPU: {e}")
        sys.exit(1)
    
    # Store system identity if not already stored
    # Check if identity exists - handle both MemoryStore and plugin backends
    identity_text = memory.get_identity()
    identity_index = getattr(memory, 'identity_index', -1)
    
    if identity_text and identity_index == -1:
        print("Storing system identity...")
        identity_emb = phi3.get_embedding(DEFAULT_IDENTITY)
        memory.store_identity(identity_emb, DEFAULT_IDENTITY)
        print(f"{LogConfig.CHECK} System identity stored")
    
    # Initialize brain module (emotional intelligence)
    brain = None
    if BRAIN_AVAILABLE:
        print("Initializing brain module...")
        try:
            brain = UnifiedBrain(
                memory_store=memory,
                embedding_dim=embedding_dim,
                state_dir="brain_state",
                use_gpu=True,
                model=phi3,  # Pass model for reranking
                enable_reranking=False  # Disabled for performance (reranking is computationally expensive)
            )
            print(f"{LogConfig.CHECK} Brain module ready (emotional intelligence enabled)")
        except Exception as e:
            print(f"{LogConfig.WARNING} Failed to initialize brain: {e}")
    
    # Initialize continuous learning if enabled
    learner = None
    if args.learning:
        if LEARNING_AVAILABLE:
            print("Initializing continuous learning...")
            try:
                # Import here to avoid scoping issues
                from learning import ContinuousLearner, LearningConfig
                config = LearningConfig(
                    vocab_dir=args.vocab_dir
                )
                learner = ContinuousLearner(
                    memory_store=memory,
                    snn_compute=snn,
                    embedder=phi3,
                    config=config
                )
                print(f"{LogConfig.CHECK} Continuous learning enabled")
            except Exception as e:
                print(f"{LogConfig.WARNING} Failed to initialize learning: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"{LogConfig.WARNING} Continuous learning module not available")
    
    # Handle special commands
    if args.clear:
        response = input("Are you sure you want to clear all memories? (yes/no): ")
        if response.lower() == "yes":
            memory.clear()
            print(f"{LogConfig.CHECK} All memories cleared")
        else:
            print("Cancelled")
        return
    
    if args.identity:
        print("Updating system identity...")
        identity_emb = phi3.get_embedding(args.identity)
        memory.store_identity(identity_emb, args.identity)
        print(f"{LogConfig.CHECK} System identity updated")
        return
    
    if args.stats:
        _show_stats(memory, learner)
        return
    
    if args.learning_stats:
        if learner:
            _show_learning_stats(learner)
        else:
            print("Learning not enabled. Use --learning flag.")
        return
    
    # Handle temporal dataset training
    if args.train_temporal:
        if not LEARNING_AVAILABLE:
            print(f"{LogConfig.CROSS} Continuous learning module not available")
            return
        
        print(f"\n{'='*60}")
        print("Temporal Dataset Training")
        print(f"{'='*60}\n")
        
        # Initialize learner if not already initialized
        if learner is None:
            print("Initializing continuous learner...")
            if not LEARNING_AVAILABLE:
                print(f"{LogConfig.WARNING} Continuous learning module not available")
                return
            from learning.continuous_learner import ContinuousLearner, LearningConfig
            learner_config = LearningConfig()
            learner = ContinuousLearner(
                memory_store=memory,
                snn_compute=snn,
                embedder=phi3,
                config=learner_config
            )
            print(f"{LogConfig.CHECK} Continuous learner initialized")
        
        # Load and train on temporal dataset
        try:
            print(f"Loading temporal dataset from: {args.train_temporal}")
            print(f"Limit: {args.train_temporal_limit or 'unlimited'}\n")
            
            stats = learner.load_temporal_dataset(
                dataset_path=args.train_temporal,
                learn=True,
                limit=args.train_temporal_limit
            )
            
            print(f"\n{'='*60}")
            print("Training Complete")
            print(f"{'='*60}")
            print(f"Events processed: {stats['events_processed']}")
            print(f"Associations processed: {stats['associations_processed']}")
            print(f"STDP updates: {stats['stdp_updates']}")
            print(f"Errors: {stats['errors']}")
            print(f"{'='*60}\n")
            
            # Save learning state
            print("Saving learning state...")
            learner._save_state()
            print(f"{LogConfig.CHECK} Learning state saved")
            
        except FileNotFoundError as e:
            print(f"{LogConfig.CROSS} Dataset file not found: {e}")
        except Exception as e:
            print(f"{LogConfig.CROSS} Training error: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    if args.train_conversations:
        if not LEARNING_AVAILABLE:
            print(f"{LogConfig.CROSS} Continuous learning module not available")
            return
        
        print(f"\n{'='*60}")
        print("Conversational Dataset Training")
        print(f"{'='*60}\n")
        
        # Initialize learner if not already initialized
        if learner is None:
            print("Initializing continuous learner...")
            if not LEARNING_AVAILABLE:
                print(f"{LogConfig.WARNING} Continuous learning module not available")
                return
            from learning.continuous_learner import ContinuousLearner, LearningConfig
            learner_config = LearningConfig()
            learner = ContinuousLearner(
                memory_store=memory,
                snn_compute=snn,
                embedder=phi3,
                config=learner_config
            )
            print(f"{LogConfig.CHECK} Continuous learner initialized")
        
        # Load and train on conversational dataset
        try:
            print(f"Loading conversational dataset from: {args.train_conversations}")
            print(f"Limit: {args.train_conversations_limit or 'unlimited'}")
            print(f"Store memories: {not args.train_conversations_no_memory}\n")
            
            stats = learner.load_conversational_dataset(
                dataset_path=args.train_conversations,
                learn=True,
                store_memories=not args.train_conversations_no_memory,
                limit=args.train_conversations_limit
            )
            
            print(f"\n{'='*60}")
            print("Training Complete")
            print(f"{'='*60}")
            print(f"Conversations processed: {stats['conversations_processed']}")
            print(f"Message pairs processed: {stats['message_pairs_processed']}")
            print(f"Memories stored: {stats['memories_stored']}")
            print(f"STDP updates: {stats['stdp_updates']}")
            print(f"Errors: {stats['errors']}")
            print(f"{'='*60}\n")
            
            # Save learning state
            print("Saving learning state...")
            learner._save_state()
            print(f"{LogConfig.CHECK} Learning state saved")
            
        except FileNotFoundError as e:
            print(f"{LogConfig.CROSS} Dataset file not found: {e}")
        except Exception as e:
            print(f"{LogConfig.CROSS} Training error: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    # Tool training mode
    if args.train_tools:
        if not LEARNING_AVAILABLE:
            print(f"{LogConfig.CROSS} Continuous learning module not available")
            return
        
        print(f"\n{'='*60}")
        print("Tool Usage Training")
        print(f"{'='*60}\n")
        
        # Initialize learner if not already initialized
        if learner is None:
            print("Initializing continuous learner...")
            if not LEARNING_AVAILABLE:
                print(f"{LogConfig.WARNING} Continuous learning module not available")
                return
            from learning.continuous_learner import ContinuousLearner, LearningConfig
            learner_config = LearningConfig()
            learner = ContinuousLearner(
                memory_store=memory,
                snn_compute=snn,
                embedder=phi3,
                config=learner_config
            )
            print(f"{LogConfig.CHECK} Continuous learner initialized")
        
        # Load tool training dataset
        try:
            print(f"Loading tool training dataset from: {args.train_tools}")
            print(f"Limit: {args.train_tools_limit or 'unlimited'}")
            print(f"Store memories: {not args.train_tools_no_memory}\n")
            
            stats = learner.load_tool_training_dataset(
                dataset_path=args.train_tools,
                store_memories=not args.train_tools_no_memory,
                limit=args.train_tools_limit
            )
            
            print(f"\n{'='*60}")
            print("Tool Training Results")
            print(f"{'='*60}")
            print(f"Examples processed: {stats['examples_processed']}")
            print(f"Memories stored: {stats['memories_stored']}")
            print(f"STDP updates: {stats['stdp_updates']}")
            print(f"Errors: {stats['errors']}")
            print(f"{'='*60}\n")
            
            # Save learning state
            print("Saving learning state...")
            learner._save_state()
            print(f"{LogConfig.CHECK} Learning state saved")
            print(f"{LogConfig.CHECK} Tool training completed")
        except FileNotFoundError as e:
            print(f"{LogConfig.CROSS} Dataset file not found: {e}")
        except Exception as e:
            print(f"{LogConfig.CROSS} Tool training error: {e}")
            import traceback
            traceback.print_exc()
        
        return
    
    # Handle calibration
    if args.calibrate:
        if brain is not None:
            _calibrate_brain(brain, phi3, epochs=args.calibrate_epochs)
        else:
            print(f"{LogConfig.CROSS} Brain module not available for calibration")
        return
    
    # Show calibration status
    if brain is not None:
        if brain.is_amygdala_calibrated():
            print(f"{LogConfig.CHECK} Amygdala affect prediction: CALIBRATED")
        else:
            print(f"{LogConfig.WARNING} Amygdala not calibrated. Run with --calibrate for better emotional understanding")
    
    # Get prompt from args or stdin
    prompt = args.prompt
    
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    
    # Developer mode - password protected advanced features
    if args.dev:
        _developer_mode(phi3, memory, snn, brain)
    # Teach mode - protected memory training
    elif args.teach:
        _teach_mode(phi3, memory, snn)
    # Interactive mode or single prompt
    elif args.interactive or (not prompt and not args.stats):
        if learner:
            asyncio.run(_interactive_mode_async(phi3, memory, snn, learner, brain, registry, tool_executor))
        else:
            _interactive_mode(phi3, memory, snn, brain, registry, tool_executor)
    elif prompt:
        _process_prompt(phi3, memory, snn, prompt, learner, brain, registry, tool_executor)
    else:
        print("Error: No prompt provided. Use --interactive for interactive mode.")
        parser.print_help()
        sys.exit(1)


def _init_model():
    """Initialize the language model - GGUF ONLY (no PyTorch fallback)"""
    print("Loading Phi-3 model...")
    
    # Only use GGUF - no PyTorch fallback
    if GGUF_AVAILABLE:
        model_path = find_gguf_model()
        if model_path:
            try:
                print(f"Using GGUF model: {model_path}")
                phi3 = Phi3GGUF(model_path=model_path, n_gpu_layers=-1)
                print(f"{LogConfig.CHECK} GGUF model loaded (GPU accelerated)")
                return phi3
            except Exception as e:
                print(f"{LogConfig.WARNING} Failed to load GGUF model: {e}")
                print(f"{LogConfig.CROSS} GGUF model is required - PyTorch fallback disabled")
                return None
        else:
            print(f"{LogConfig.CROSS} GGUF model not found")
            print(f"{LogConfig.WARNING} Please download Phi-3-mini GGUF model")
            return None
    else:
        print(f"{LogConfig.CROSS} GGUF support not available")
        print(f"{LogConfig.WARNING} Install llama-cpp-python: pip install llama-cpp-python")
        return None


def _show_tiling_info(snn):
    """Display GPU tiling support information"""
    try:
        # Check both 'backend' (SNNCompute) and 'gpu' (other backends) attributes
        vulkan_backend = None
        if hasattr(snn, 'backend') and snn.backend is not None:
            vulkan_backend = snn.backend
        elif hasattr(snn, 'gpu') and snn.gpu is not None:
            vulkan_backend = snn.gpu
        elif hasattr(snn, 'vulkan') and snn.vulkan is not None:
            vulkan_backend = snn.vulkan
        
        if vulkan_backend is not None:
            if hasattr(vulkan_backend, 'get_tiling_info'):
                tiling_info = vulkan_backend.get_tiling_info()
                print("\n=== GPU Tiling Support ===")
                if tiling_info.get('available', False):
                    print(f"Device: {tiling_info.get('device_name', 'Unknown')}")
                    print(f"Vendor: {tiling_info.get('vendor', 'Unknown')} (ID: 0x{tiling_info.get('vendor_id', 0):X})")
                    print(f"\nTiling Features:")
                    print(f"  Sparse Binding: {'✓' if tiling_info.get('sparse_binding') else '✗'}")
                    print(f"  Sparse Residency: {'✓' if tiling_info.get('sparse_residency') else '✗'}")
                    print(f"  Sparse Residency Aliased: {'✓' if tiling_info.get('sparse_residency_aliased') else '✗'}")
                    print(f"  Optimal Tiling: {'✓' if tiling_info.get('optimal_tiling') else '✗'}")
                    print(f"  Shader Image Gather Extended: {'✓' if tiling_info.get('shader_image_gather_extended') else '✗'}")
                    
                    if tiling_info.get('amd_optimized'):
                        print(f"\n  AMD GPU detected - Optimized tiling support available")
                    elif tiling_info.get('nvidia_optimized'):
                        print(f"\n  NVIDIA GPU detected - Optimized tiling support available")
                    elif tiling_info.get('intel_optimized'):
                        print(f"\n  Intel GPU detected - Basic tiling support")
                else:
                    print("Tiling information not available")
                    if 'error' in tiling_info:
                        print(f"Error: {tiling_info['error']}")
            else:
                print("GPU backend does not support tiling info query")
        else:
            # Try to initialize Vulkan directly
            try:
                from vulkan_backend import VulkanCompute
                vulkan = VulkanCompute()
                tiling_info = vulkan.get_tiling_info()
                print("\n=== GPU Tiling Support ===")
                if tiling_info.get('available', False):
                    print(f"Device: {tiling_info.get('device_name', 'Unknown')}")
                    print(f"Vendor: {tiling_info.get('vendor', 'Unknown')} (ID: 0x{tiling_info.get('vendor_id', 0):X})")
                    print(f"\nTiling Features:")
                    print(f"  Sparse Binding: {'✓' if tiling_info.get('sparse_binding') else '✗'}")
                    print(f"  Sparse Residency: {'✓' if tiling_info.get('sparse_residency') else '✗'}")
                    print(f"  Optimal Tiling: {'✓' if tiling_info.get('optimal_tiling') else '✗'}")
                else:
                    print("Tiling information not available")
            except Exception as e:
                print(f"GPU backend not available: {e}")
    except Exception as e:
        print(f"Error checking tiling info: {e}")
        import traceback
        traceback.print_exc()

def _show_stats(memory: MemoryStore, learner=None):
    """Display memory statistics"""
    stats = memory.get_stats()
    print("\n=== Memory Statistics ===")
    print(f"Total memories: {stats['total_memories']}")
    print(f"GPU memories: {stats['gpu_memories']}")
    print(f"Max memories: {stats['max_memories']}")
    print(f"Total accesses: {stats['total_accesses']}")
    print(f"Embedding dimension: {stats['embedding_dim']}")
    print(f"GPU similarity: {'enabled' if stats.get('gpu_enabled', False) else 'disabled'}")
    if memory.get_identity():
        print(f"\nSystem identity: {memory.get_identity()[:100]}...")
    
    if learner:
        print("\n=== Learning Statistics ===")
        lstats = learner.get_stats()
        print(f"Conversations learned: {lstats['conversations_learned']}")
        print(f"STDP updates: {lstats['stdp_updates']}")
        print(f"Spikes generated: {lstats['spikes_generated']}")


def _calibrate_brain(brain, phi3, epochs: int = 20):
    """Calibrate brain's affect prediction"""
    print("\n=== Amygdala Affect Calibration ===")
    print("Training the emotional understanding network...")
    print(f"Architecture: 3-layer MLP with Adam optimizer")
    print(f"Epochs: {epochs}")
    print()
    
    try:
        result = brain.calibrate_affect(
            embed_fn=phi3.get_embedding,
            epochs=epochs,
            learning_rate=0.001,
            batch_size=64,
            limit=1000
        )
        
        if 'error' in result:
            print(f"{LogConfig.CROSS} Calibration failed: {result['error']}")
            return
        
        print(f"\n{LogConfig.CHECK} Calibration complete!")
        print(f"  Final loss: {result['final_loss']:.4f}")
        print(f"  Samples trained: {result.get('samples', 'N/A')}")
        
        losses = result.get('loss_history', [])
        if len(losses) >= 2:
            improvement = (1 - losses[-1] / losses[0]) * 100
            print(f"  Loss reduction: {improvement:.1f}%")
        
        print("\nThe brain will now better understand emotions in your messages.")
        
    except Exception as e:
        print(f"{LogConfig.CROSS} Calibration error: {e}")
        import traceback
        traceback.print_exc()


def _show_learning_stats(learner):
    """Display detailed learning statistics"""
    stats = learner.get_stats()
    stdp_stats = stats.get('stdp_stats', {})
    
    print("\n=== Continuous Learning Statistics ===")
    print(f"Items processed: {stats['items_processed']}")
    print(f"Conversations learned: {stats['conversations_learned']}")
    print(f"Vocab files ingested: {stats['vocab_files_ingested']}")
    print(f"Total STDP updates: {stats['stdp_updates']}")
    print(f"Spikes generated: {stats['spikes_generated']}")
    print(f"Errors: {stats['errors']}")
    
    print("\n=== STDP Learner State ===")
    print(f"Active tokens: {stdp_stats.get('active_tokens', 0)}")
    print(f"Active associations: {stdp_stats.get('active_associations', 0)}")
    print(f"LTP events: {stdp_stats.get('ltp_events', 0)}")
    print(f"LTD events: {stdp_stats.get('ltd_events', 0)}")
    print(f"Total weight: {stdp_stats.get('total_weight', 0):.4f}")


def _process_prompt(phi3, memory: MemoryStore, snn: SNNCompute, prompt: str, learner=None, brain=None, registry=None, tool_executor=None):
    """Process a single prompt and print response"""
    print(f"\nYou: {prompt}\n")
    
    try:
        # Request context for hooks
        context_dict = {
            "user_id": "cli_user",
            "session_id": "cli_session"
        }
        
        # Pre-process hooks
        enhanced_prompt = prompt
        if registry and registry.processing_hooks:
            for hook in registry.processing_hooks:
                try:
                    import asyncio
                    enhanced_prompt = asyncio.run(hook.pre_process(enhanced_prompt, context_dict))
                except Exception as e:
                    logger.warning(f"Hook {hook.name} pre_process failed: {e}")
        
        # Extract embedding
        print("Extracting embedding...", end="\r")
        embedding = phi3.get_embedding(enhanced_prompt)
        print(f"{LogConfig.CHECK} Embedding extracted    ")
        
        # Process through brain module (emotional intelligence)
        brain_result = None
        emotional_context = ""
        if brain is not None:
            print("Processing emotions...", end="\r")
            brain_result = brain.process(enhanced_prompt, embedding)
            emotional_context = brain.get_empathy_prompt()
            print(f"{LogConfig.CHECK} Emotion: {brain_result['emotional_state'].dominant_emotion} (valence: {brain_result['emotional_state'].valence:.2f})")
        
        # Store in memory
        print("Storing in memory...", end="\r")
        memory.store(embedding, enhanced_prompt)
        print(f"{LogConfig.CHECK} Stored in memory      ")
        
        # Retrieve context
        print("Retrieving context...", end="\r")
        retrieved = memory.retrieve(embedding, k=MemoryConfig.DEFAULT_K)
        # Handle both MemoryStore API (returns List[str]) and plugin API (returns List[Tuple[str, float]])
        if retrieved and isinstance(retrieved[0], tuple):
            context = [text for text, score in retrieved]
        else:
            context = retrieved
        print(f"{LogConfig.CHECK} Retrieved {len(context)} context items")
        
        # Build enhanced prompt with emotional context and self-awareness
        if brain is not None:
            self_awareness = brain.get_self_awareness_prompt()
            if emotional_context:
                final_prompt = f"{self_awareness}\n\n{emotional_context}\n\nUser: {enhanced_prompt}"
            else:
                final_prompt = f"{self_awareness}\n\nUser: {enhanced_prompt}"
        elif emotional_context:
            final_prompt = f"{emotional_context}\n\nUser: {enhanced_prompt}"
        else:
            final_prompt = enhanced_prompt
        
        # Generate response (with tools if available)
        device_msg = "GPU" if hasattr(phi3, 'device') and phi3.device != "cpu" else "CPU"
        print(f"Generating response on {device_msg}...")
        
        # Debug: Check tool availability
        print(f"🔍 Debug: registry={bool(registry)}, tool_executor={bool(tool_executor)}, registry.tools={bool(registry.tools if registry else False)}")
        
        if registry and tool_executor and registry.tools:
            tools = registry.get_tools()
            tool_names = [t.name for t in tools]
            print(f"🔧 Available tools: {', '.join(tool_names)}")
            response = phi3.generate_with_tools(
                final_prompt,
                context,
                tools=tools,
                tool_executor=tool_executor
            )
            # Note: Tool calls are logged in model_gguf.py with 🔧 ⚙️ ✅ emojis
        else:
            print(f"⚠️ Tools not available, using regular generate()")
            response = phi3.generate(final_prompt, context)
        print(f"{LogConfig.CHECK} Response generated")
        
        # Provide feedback to brain
        if brain is not None:
            brain.provide_feedback(quality=0.7, strategy_worked=True)
        
        # Compute spike activity
        print("Computing spike activity...", end="\r")
        spike_metrics = snn.process(embedding)
        print(f"{LogConfig.CHECK} Spike activity computed")
        
        # Continuous learning
        if learner:
            print("Learning from conversation...", end="\r")
            learn_result = learner.learn_from_conversation(prompt, response, context)
            print(f"{LogConfig.CHECK} STDP learning: {learn_result.get('stdp_updates', 0)} updates")
        
        # Print response
        print(f"\nGrillCheese: {response}\n")
        
        # Print simplified, meaningful stats
        stats = memory.get_stats()
        
        # Build stats line with useful info
        stats_parts = [f"Memories: {stats['total_memories']}"]
        
        if brain_result is not None:
            emo = brain_result['emotional_state']
            stats_parts.append(f"{emo.dominant_emotion}")
            
            # Add stress if high
            if brain_result['stress_level'] > 0.6:
                stats_parts.append(f"Stress: {brain_result['stress_level']:.1f}")
        
        if learner:
            lstats = learner.get_stats()
            if lstats['stdp_updates'] > 0:
                stats_parts.append(f"Learned: {lstats['stdp_updates']} patterns")
        
        # Show GPU mode if enabled
        if stats.get('gpu_enabled'):
            stats_parts.append("GPU")
        
        print(f"[{' | '.join(stats_parts)}]")
        
    except Exception as e:
        print(f"\n{LogConfig.CROSS} Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def _interactive_mode(phi3, memory: MemoryStore, snn: SNNCompute, brain=None, registry=None, tool_executor=None):
    """Interactive conversation mode with emotional intelligence"""
    print("\n" + "=" * 60)
    print("GrillCheese AI - Interactive Mode")
    if brain is not None:
        print("Emotional intelligence: ENABLED")
    print("Type your messages. Commands: 'quit', 'stats', 'clear', 'emotion', 'tiling'")
    print("=" * 60 + "\n")
    
    # Track recent conversation history
    conversation_history = []
    max_history = 6  # Keep last 3 exchanges (user + assistant pairs)
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if not prompt:
                continue
            
            # Handle commands
            if prompt.lower() in ['quit', 'exit', 'bye', 'q']:
                if brain is not None:
                    brain.consolidate()
                    print("Brain state saved.")
                print("\nGoodbye!")
                break
            
            if prompt.lower() == 'stats':
                _show_stats(memory)
                if brain is not None:
                    brain_stats = brain.get_stats()
                    print("\n=== Brain Statistics ===")
                    print(f"Total interactions: {brain_stats['brain_stats']['total_interactions']}")
                    print(f"Positive interactions: {brain_stats['brain_stats']['positive_interactions']}")
                    print(f"Empathetic responses: {brain_stats['brain_stats']['empathetic_responses']}")
                    print(f"Current strategy: {brain_stats['basal_ganglia']['current_strategy']}")
                    print(f"Consciousness: {brain_stats['cns']['current_state']['consciousness']}")
                    print(f"Stress level: {brain_stats['cns']['current_state']['stress_level']:.2f}")
                print()
                continue
            
            if prompt.lower() == 'tiling' or prompt.lower() == 'gpu-info':
                _show_tiling_info(snn)
                print()
                continue
            
            if prompt.lower() == 'emotion':
                if brain is not None:
                    state = brain.amygdala.get_state()
                    print(f"\n=== Current Emotional State ===")
                    print(f"Valence: {state.valence:.2f} (-1=negative, +1=positive)")
                    print(f"Arousal: {state.arousal:.2f} (0=calm, 1=excited)")
                    print(f"Dominant emotion: {state.dominant_emotion}")
                    print(f"Empathy prompt: {brain.get_empathy_prompt()}")
                    print()
                else:
                    print("Brain module not loaded.\n")
                continue
            
            if prompt.lower() == 'clear':
                response = input("Clear all memories? (yes/no): ")
                if response.lower() == "yes":
                    memory.clear()
                    print(f"{LogConfig.CHECK} Memories cleared\n")
                continue
            
            # Store user prompt in memory FIRST (so it's available for retrieval)
            embedding = phi3.get_embedding(prompt)
            memory.store(embedding, prompt)
            
            # Process through brain module
            brain_result = None
            emotional_context = ""
            context = []
            
            if brain is not None:
                brain_result = brain.process(prompt, embedding)
                emotional_context = brain.get_empathy_prompt()
                # Use brain's memory context (includes identity properly)
                context = brain_result.get('memory_context', [])
                # Extract text if tuples
                if context and isinstance(context[0], tuple):
                    context = [item[0] if isinstance(item, tuple) else item for item in context]
            
            # Fallback to direct memory retrieval if brain not available
            if not context:
                # Retrieve fewer memories and filter by relevance
                retrieved = memory.retrieve(embedding, k=3, include_identity=True)
                # Extract text if tuples and filter out low-relevance items
                if retrieved and isinstance(retrieved[0], tuple):
                    # Filter by similarity score (keep only relevant memories)
                    filtered = []
                    for item in retrieved:
                        if isinstance(item, tuple):
                            text, score = item
                            # Only include if similarity is reasonable (>0.3) or it's identity
                            if score > 0.3 or "GrillCheese" in text or len(text) > 200:
                                filtered.append(text)
                        else:
                            filtered.append(item)
                    context = filtered
                else:
                    context = retrieved
            
            # Enhance context with brain state if available
            # Identity should be first (from memory.retrieve with include_identity=True)
            # Merge self-awareness and emotional context into identity if present
            if brain is not None and context:
                self_awareness = brain.get_self_awareness_prompt()
                
                # Check if first item is identity (contains "GrillCheese" or is long)
                identity_idx = 0
                if context and ("GrillCheese" in context[0] or len(context[0]) > 200):
                    # Merge self-awareness and emotional context into identity
                    identity_parts = [context[0]]  # Start with identity
                    if self_awareness:
                        identity_parts.append(self_awareness)
                    if emotional_context:
                        identity_parts.append(emotional_context)
                    # Combine into single identity string
                    enhanced_identity = "\n\n".join(identity_parts)
                    # Replace first item with enhanced identity
                    context = [enhanced_identity] + context[1:]
                else:
                    # No identity found, prepend it
                    identity_text = memory.get_identity() or ""
                    identity_parts = [identity_text] if identity_text else []
                    if self_awareness:
                        identity_parts.append(self_awareness)
                    if emotional_context:
                        identity_parts.append(emotional_context)
                    if identity_parts:
                        enhanced_identity = "\n\n".join(identity_parts)
                        context = [enhanced_identity] + context
            
            # Add recent conversation history to context
            # PRIORITIZE conversation history over semantic memories for continuity
            if conversation_history:
                # Format recent history
                history_context = []
                for hist_item in conversation_history[-max_history:]:
                    if isinstance(hist_item, dict):
                        if hist_item.get('role') == 'user':
                            history_context.append(f"Previous user: {hist_item.get('content', '')}")
                        elif hist_item.get('role') == 'assistant':
                            history_context.append(f"Previous assistant: {hist_item.get('content', '')}")
                
                if history_context:
                    history_text = "\n".join(history_context)
                    # Insert after identity, BEFORE semantic memories (prioritize conversation flow)
                    if context and ("GrillCheese" in context[0] or len(context[0]) > 200):
                        # Identity is first, add history after it, then semantic memories
                        context = [context[0], history_text] + context[1:]
                    else:
                        # No identity, prepend history
                        context = [history_text] + context
            
            device_msg = "GPU" if hasattr(phi3, 'device') and phi3.device != "cpu" else "CPU"
            print(f"Generating on {device_msg}...")
            
            # Use generate_with_tools if available, otherwise fall back to generate()
            if registry and tool_executor and registry.tools:
                tools = registry.get_tools()
                tool_names = [t.name for t in tools]
                print(f"🔧 Available tools: {', '.join(tool_names)}")
                
                # Debug: Check if method exists
                print(f"🔍 Checking phi3 type: {type(phi3)}")
                print(f"🔍 Has generate_with_tools: {hasattr(phi3, 'generate_with_tools')}")
                
                if hasattr(phi3, 'generate_with_tools'):
                    print(f"🔍 Calling generate_with_tools...")
                    response = phi3.generate_with_tools(
                        prompt,
                        context,
                        tools=tools,
                        tool_executor=tool_executor
                    )
                else:
                    print(f"⚠️ phi3 doesn't have generate_with_tools, using generate()")
                    response = phi3.generate(prompt, context)
            else:
                # Let model.generate() handle proper Phi-3 chat formatting with identity
                response = phi3.generate(prompt, context)
            
            # Store conversation in history
            conversation_history.append({'role': 'user', 'content': prompt})
            conversation_history.append({'role': 'assistant', 'content': response})
            # Keep only recent history
            if len(conversation_history) > max_history:
                conversation_history = conversation_history[-max_history:]
            spike_metrics = snn.process(embedding)
            
            # Provide feedback to brain
            if brain is not None:
                brain.provide_feedback(quality=0.7, strategy_worked=True)
            
            print(f"\nGrillCheese: {response}")
            
            # Show simplified stats
            stats = memory.get_stats()
            stats_parts = [f"Memories: {stats['total_memories']}"]
            
            if brain_result is not None:
                emo = brain_result['emotional_state']
                stats_parts.append(f"{emo.dominant_emotion}")
                if brain_result['stress_level'] > 0.6:
                    stats_parts.append(f"Stress: {brain_result['stress_level']:.1f}")
            
            if stats.get('gpu_enabled'):
                stats_parts.append("GPU")
            
            print(f"[{' | '.join(stats_parts)}]\n")
            
        except KeyboardInterrupt:
            if brain is not None:
                brain.consolidate()
                print("\nBrain state saved.")
            print("\n\nGoodbye!")
            break
        except EOFError:
            if brain is not None:
                brain.consolidate()
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


async def _interactive_mode_async(phi3, memory: MemoryStore, snn: SNNCompute, learner, brain=None, registry=None, tool_executor=None):
    """Interactive conversation mode with continuous learning and emotional intelligence"""
    print("\n" + "=" * 60)
    print("GrillCheese AI - Interactive Mode")
    print("Features: Continuous Learning + Emotional Intelligence")
    print("Commands: 'quit', 'stats', 'learn', 'emotion', 'clear'")
    print("=" * 60 + "\n")
    
    # Start learning background tasks
    await learner.start()
    
    try:
        while True:
            try:
                # Use asyncio for non-blocking input
                prompt = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("You: ").strip()
                )
                
                if not prompt:
                    continue
                
                # Handle commands
                if prompt.lower() in ['quit', 'exit', 'bye', 'q']:
                    print("\nSaving state and exiting...")
                    break
                
                if prompt.lower() == 'stats':
                    _show_stats(memory, learner)
                    if brain is not None:
                        brain_stats = brain.get_stats()
                        print("\n=== Brain Statistics ===")
                        print(f"Total interactions: {brain_stats['brain_stats']['total_interactions']}")
                        print(f"Positive: {brain_stats['brain_stats']['positive_interactions']} | Negative: {brain_stats['brain_stats']['negative_interactions']}")
                        print(f"Empathetic responses: {brain_stats['brain_stats']['empathetic_responses']}")
                        print(f"GPU operations: {brain_stats['brain_stats']['gpu_operations']}")
                        print(f"Consciousness: {brain_stats['cns']['current_state']['consciousness']}")
                        print(f"Stress: {brain_stats['cns']['current_state']['stress_level']:.2f}")
                    print()
                    continue
                
                if prompt.lower() == 'learn':
                    _show_learning_stats(learner)
                    print()
                    continue
                
                if prompt.lower() == 'emotion':
                    if brain is not None:
                        state = brain.amygdala.get_state()
                        print(f"\n=== Emotional State ===")
                        print(f"Valence: {state.valence:.2f} | Arousal: {state.arousal:.2f}")
                        print(f"Dominant: {state.dominant_emotion}")
                        print(f"Response style: {brain.get_response_style()}")
                        print()
                    else:
                        print("Brain module not loaded.\n")
                    continue
                
                if prompt.lower() == 'clear':
                    response = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("Clear all memories? (yes/no): ")
                    )
                    if response.lower() == "yes":
                        memory.clear()
                        print(f"{LogConfig.CHECK} Memories cleared\n")
                    continue
                
                # Process through brain module
                embedding = phi3.get_embedding(prompt)
                
                brain_result = None
                emotional_context = ""
                if brain is not None:
                    brain_result = brain.process(prompt, embedding)
                    emotional_context = brain.get_empathy_prompt()
                
                memory.store(embedding, prompt)
                context = memory.retrieve(embedding, k=MemoryConfig.DEFAULT_K)
                
                # Build enhanced prompt with self-awareness
                if brain is not None:
                    self_awareness = brain.get_self_awareness_prompt()
                    if emotional_context:
                        enhanced_prompt = f"{self_awareness}\n\n{emotional_context}\n\nUser: {prompt}"
                    else:
                        enhanced_prompt = f"{self_awareness}\n\nUser: {prompt}"
                elif emotional_context:
                    enhanced_prompt = f"{emotional_context}\n\nUser: {prompt}"
                else:
                    enhanced_prompt = prompt
                
                device_msg = "GPU" if hasattr(phi3, 'device') and phi3.device != "cpu" else "CPU"
                print(f"Generating on {device_msg}...")
                
                response = phi3.generate(enhanced_prompt, context)
                spike_metrics = snn.process(embedding)
                
                # Provide feedback to brain
                if brain is not None:
                    brain.provide_feedback(quality=0.7, strategy_worked=True)
                
                # Learn from conversation
                learn_result = learner.learn_from_conversation(prompt, response, context)
                
                print(f"\nGrillCheese: {response}")
                
                stats = memory.get_stats()
                lstats = learner.get_stats()
                stats_line = f"[Spikes: {spike_metrics['spike_activity']:.0f} | Memories: {stats['total_memories']} | STDP: {learn_result.get('stdp_updates', 0)}"
                if brain_result is not None:
                    emo = brain_result['emotional_state']
                    stats_line += f" | {emo.dominant_emotion}"
                stats_line += "]"
                print(f"{stats_line}\n")
                
            except KeyboardInterrupt:
                print("\n\nSaving state and exiting...")
                break
            except EOFError:
                print("\n\nSaving state and exiting...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
    finally:
        # Stop learning and save state
        await learner.stop()
        if brain is not None:
            brain.consolidate()
            print("Brain state saved.")
        print("Goodbye!")




def _developer_mode(phi3, memory: MemoryStore, snn: SNNCompute, brain=None):
    """Password-protected developer mode for model improvement"""
    import getpass
    from dev_auth import verify_dev_password
    
    print("\n" + "=" * 60)
    print("DEVELOPER MODE - AUTHENTICATION REQUIRED")
    print("=" * 60)
    
    # Authenticate
    max_attempts = 3
    for attempt in range(max_attempts):
        password = getpass.getpass("Developer password: ")
        
        if verify_dev_password(password):
            print(f"{LogConfig.CHECK} Authentication successful\n")
            break
        else:
            remaining = max_attempts - attempt - 1
            if remaining > 0:
                print(f"{LogConfig.CROSS} Invalid password. {remaining} attempts remaining.\n")
            else:
                print(f"{LogConfig.CROSS} Authentication failed. Access denied.")
                return
    
    # Developer mode interface
    print("=" * 60)
    print("GRILLCHEESE DEVELOPER MODE")
    print("=" * 60)
    print("Advanced model improvement and analysis tools")
    print("\nCommands:")
    print("  export-training     - Export conversation pairs for fine-tuning")
    print("  analyze-memory      - Deep memory analysis and statistics")
    print("  edit-identity       - Edit system identity prompt")
    print("  tune-params         - Adjust model parameters")
    print("  test-retrieval      - Test memory retrieval with queries")
    print("  export-embeddings   - Export embedding space for analysis")
    print("  brain-dump          - Full brain state inspection")
    print("  create-dataset      - Create fine-tuning dataset from sessions")
    print("  stats               - Comprehensive system statistics")
    print("  quit                - Exit developer mode")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("Dev> ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if cmd in ['quit', 'exit', 'q']:
                print("\nExiting developer mode...")
                break
            
            elif cmd == 'export-training':
                _dev_export_training(memory, arg)
            
            elif cmd == 'analyze-memory':
                _dev_analyze_memory(memory)
            
            elif cmd == 'edit-identity':
                _dev_edit_identity(memory, phi3)
            
            elif cmd == 'tune-params':
                _dev_tune_params()
            
            elif cmd == 'test-retrieval':
                _dev_test_retrieval(memory, phi3, arg)
            
            elif cmd == 'export-embeddings':
                _dev_export_embeddings(memory, arg)
            
            elif cmd == 'brain-dump':
                _dev_brain_dump(brain)
            
            elif cmd == 'create-dataset':
                _dev_create_dataset(memory, arg)
            
            elif cmd == 'stats':
                _dev_comprehensive_stats(memory, snn, brain)
            
            else:
                print(f"Unknown command: {cmd}")
                print("Type a command or 'quit' to exit")
        
        except KeyboardInterrupt:
            print("\n\nExiting developer mode...")
            break
        except EOFError:
            print("\n\nExiting developer mode...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


def _dev_export_training(memory: MemoryStore, output_file: str):
    """Export conversation pairs for fine-tuning"""
    import sqlite3
    import json
    
    if not output_file:
        output_file = "training_data.jsonl"
    
    print(f"Exporting training data to {output_file}...")
    
    # Handle both MemoryStore and plugin backends
    db_path = getattr(memory, 'db_path', None)
    if db_path is None and hasattr(memory, '_backend'):
        db_path = getattr(memory._backend, 'db_path', None)
    
    if db_path is None:
        print(f"{LogConfig.CROSS} Cannot access database: no database path available")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all non-identity memories with metadata
    cursor.execute("""
        SELECT text, metadata, timestamp, access_count 
        FROM memories 
        WHERE is_identity = 0
        ORDER BY timestamp DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    # Create training pairs
    pairs = []
    for text, metadata_json, timestamp, access_count in rows:
        metadata = json.loads(metadata_json) if metadata_json else {}
        
        # Extract prompt/response if available
        if 'type' in metadata:
            pairs.append({
                'text': text,
                'metadata': metadata,
                'timestamp': timestamp,
                'access_count': access_count
            })
    
    # Write to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"{LogConfig.CHECK} Exported {len(pairs)} entries to {output_file}")


def _dev_analyze_memory(memory):
    """Deep memory analysis"""
    import sqlite3
    
    # Handle both MemoryStore and plugin backends
    db_path = getattr(memory, 'db_path', None)
    if db_path is None and hasattr(memory, '_backend'):
        db_path = getattr(memory._backend, 'db_path', None)
    
    if db_path is None:
        print(f"{LogConfig.CROSS} Cannot analyze memory: no database path available")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("\n=== Deep Memory Analysis ===\n")
    
    # Total stats
    cursor.execute("SELECT COUNT(*), COUNT(DISTINCT is_protected), COUNT(DISTINCT is_identity) FROM memories")
    total, _, _ = cursor.fetchone()
    
    cursor.execute("SELECT COUNT(*) FROM memories WHERE is_protected = 1")
    protected = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM memories WHERE is_identity = 1")
    identity = cursor.fetchone()[0]
    
    print(f"Total memories: {total}")
    print(f"Protected: {protected} ({protected/total*100:.1f}%)" if total > 0 else "Protected: 0")
    print(f"Identity: {identity}")
    print(f"Regular: {total - protected - identity}")
    
    # Access patterns
    cursor.execute("SELECT AVG(access_count), MAX(access_count) FROM memories")
    avg_access, max_access = cursor.fetchone()
    
    print(f"\nAccess Patterns:")
    print(f"Average access count: {avg_access:.2f}" if avg_access else "No access data")
    print(f"Max access count: {max_access}" if max_access else "No access data")
    
    # Most accessed
    cursor.execute("""
        SELECT text, access_count 
        FROM memories 
        WHERE is_identity = 0
        ORDER BY access_count DESC 
        LIMIT 5
    """)
    
    top_accessed = cursor.fetchall()
    if top_accessed:
        print(f"\nMost Accessed Memories:")
        for i, (text, count) in enumerate(top_accessed, 1):
            print(f"{i}. [{count} accesses] {text[:60]}...")
    
    # Temporal distribution
    cursor.execute("""
        SELECT 
            DATE(timestamp) as date,
            COUNT(*) as count
        FROM memories
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
        LIMIT 7
    """)
    
    daily = cursor.fetchall()
    if daily:
        print(f"\nMemories Created (Last 7 Days):")
        for date, count in daily:
            print(f"{date}: {count} memories")
    
    conn.close()
    print()


def _dev_edit_identity(memory: MemoryStore, phi3):
    """Edit system identity prompt"""
    import sqlite3
    
    print("\n=== Edit System Identity ===\n")
    print("Current identity:")
    print("-" * 60)
    print(memory.identity_text[:500] + "..." if memory.identity_text and len(memory.identity_text) > 500 else memory.identity_text)
    print("-" * 60)
    
    print("\nOptions:")
    print("1. Edit in external editor")
    print("2. Replace with new text")
    print("3. Cancel")
    
    choice = input("\nChoice (1-3): ").strip()
    
    if choice == '1':
        import tempfile
        import subprocess
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(memory.identity_text or "")
            temp_path = f.name
        
        # Open in editor
        editor = os.environ.get('EDITOR', 'notepad' if os.name == 'nt' else 'nano')
        subprocess.call([editor, temp_path])
        
        # Read back
        with open(temp_path, 'r') as f:
            new_identity = f.read().strip()
        
        os.unlink(temp_path)
        
        if new_identity:
            embedding = phi3.get_embedding(new_identity)
            memory.store_identity(new_identity, embedding)
            print(f"{LogConfig.CHECK} Identity updated")
        else:
            print("Cancelled - empty identity")
    
    elif choice == '2':
        print("\nEnter new identity (Ctrl+D or Ctrl+Z when done):")
        lines = []
        try:
            while True:
                line = input()
                lines.append(line)
        except EOFError:
            pass
        
        new_identity = '\n'.join(lines).strip()
        if new_identity:
            embedding = phi3.get_embedding(new_identity)
            memory.store_identity(new_identity, embedding)
            print(f"\n{LogConfig.CHECK} Identity updated")
        else:
            print("\nCancelled - empty identity")
    
    else:
        print("Cancelled")


def _dev_tune_params():
    """Adjust model parameters"""
    print("\n=== Model Parameter Tuning ===\n")
    print("Current parameters:")
    print(f"  Temperature: {ModelConfig.TEMPERATURE}")
    print(f"  Top-P: {ModelConfig.TOP_P}")
    print(f"  Max tokens (GPU): {ModelConfig.MAX_NEW_TOKENS_GPU}")
    print(f"  Max context items: {ModelConfig.MAX_CONTEXT_ITEMS}")
    print("\nNote: Changes are runtime only. Edit config.py for persistence.")
    print("\nPress Enter to continue...")
    input()


def _dev_test_retrieval(memory: MemoryStore, phi3, query: str):
    """Test memory retrieval"""
    if not query:
        query = input("Enter test query: ").strip()
    
    if not query:
        print("No query provided")
        return
    
    print(f"\nTesting retrieval for: '{query}'\n")
    
    embedding = phi3.get_embedding(query)
    results = memory.retrieve(embedding, k=10, include_identity=False)
    
    print(f"Retrieved {len(results)} memories:")
    for i, text in enumerate(results, 1):
        print(f"{i}. {text[:80]}")
    print()


def _dev_export_embeddings(memory: MemoryStore, output_file: str):
    """Export embeddings for analysis"""
    import numpy as np
    
    if not output_file:
        output_file = "embeddings.npz"
    
    print(f"Exporting embeddings to {output_file}...")
    
    # Get memory data - handle both wrapper and direct MemoryStore
    memory_keys = getattr(memory, 'memory_keys', None)
    memory_values = getattr(memory, 'memory_values', None)
    memory_texts = getattr(memory, 'memory_texts', [])
    num_memories = getattr(memory, 'num_memories', 0)
    
    # If wrapper, try to get from underlying backend
    if memory_keys is None and hasattr(memory, '_backend'):
        memory_keys = getattr(memory._backend, 'memory_keys', None)
        memory_values = getattr(memory._backend, 'memory_values', None)
        memory_texts = getattr(memory._backend, 'memory_texts', [])
        num_memories = getattr(memory._backend, 'num_memories', 0)
    
    if memory_keys is not None and memory_values is not None:
        # Export memory keys and values
        np.savez_compressed(
            output_file,
            keys=memory_keys[:num_memories],
            values=memory_values[:num_memories],
            texts=np.array(memory_texts[:num_memories], dtype=object)
        )
        print(f"{LogConfig.CHECK} Exported {num_memories} embeddings")
    else:
        print(f"{LogConfig.WARNING} Cannot export: memory backend doesn't support direct embedding export")


def _dev_brain_dump(brain):
    """Full brain state inspection"""
    if brain is None:
        print("Brain module not loaded")
        return
    
    print("\n=== Brain State Dump ===\n")
    
    stats = brain.get_stats()
    
    print("Amygdala State:")
    emo = stats['amygdala']['current_state']
    print(f"  Valence: {emo['valence']:.3f}")
    print(f"  Arousal: {emo['arousal']:.3f}")
    print(f"  Dominant: {emo['dominant_emotion']}")
    
    print("\nCNS State:")
    cns = stats['cns']['current_state']
    print(f"  Consciousness: {cns['consciousness']}")
    print(f"  Stress: {cns['stress_level']:.3f}")
    
    print("\nBasal Ganglia:")
    bg = stats['basal_ganglia']
    print(f"  Current strategy: {bg.get('current_strategy', 'unknown')}")
    print(f"  Current confidence: {bg.get('current_confidence', 0.0):.3f}")
    print(f"  Go rate: {bg.get('go_rate', 0.0):.3f}")
    if 'region_weights' in bg:
        print(f"  Region weights: {bg['region_weights']}")
    if 'hebbian_weights_shape' in bg:
        print(f"  Hebbian weights shape: {bg['hebbian_weights_shape']}")
    
    print("\nExperience:")
    exp = stats.get('experiential_learning', {})
    if exp and exp.get('total_experiences', 0) > 0:
        print(f"  Total interactions: {exp.get('total_experiences', 0)}")
        avg_quality = exp.get('avg_quality', 0.0)
        if avg_quality > 0:
            print(f"  Average quality: {avg_quality:.3f}")
        best_strategy = exp.get('best_strategy', 'unknown')
        if best_strategy != 'unknown':
            print(f"  Best strategy: {best_strategy}")
        insights = exp.get('insights', [])
        if insights:
            print(f"  Insights: {len(insights)} available")
    else:
        print("  No experience data available")
    print()


def _dev_create_dataset(memory: MemoryStore, output_file: str):
    """Create fine-tuning dataset from memory"""
    import sqlite3
    import json
    
    if not output_file:
        output_file = "finetune_dataset.jsonl"
    
    print(f"Creating fine-tuning dataset: {output_file}")
    print("This will create prompt/completion pairs from memories...")
    
    # This is a placeholder - implement based on your fine-tuning needs
    print("\nDataset creation requires conversation history.")
    print("Current implementation exports raw memories.")
    print("Implement conversation tracking for full training pairs.")


def _dev_comprehensive_stats(memory: MemoryStore, snn: SNNCompute, brain):
    """Comprehensive system statistics"""
    print("\n=== Comprehensive System Statistics ===\n")
    
    # Memory stats
    _show_stats(memory)
    
    # Brain stats
    if brain:
        print("\n=== Brain Statistics ===")
        stats = brain.get_stats()
        print(f"Total interactions: {stats['brain_stats']['total_interactions']}")
        print(f"Current strategy: {stats['basal_ganglia']['current_strategy']}")
        print(f"Stress level: {stats['cns']['current_state']['stress_level']:.2f}")
    
    # GPU stats - handle both wrapper and direct MemoryStore
    gpu = getattr(memory, 'gpu', None)
    if gpu is None and hasattr(memory, '_backend'):
        # Try to get from underlying backend
        gpu = getattr(memory._backend, 'gpu', None)
    
    if gpu:
        print("\n=== GPU Statistics ===")
        print("Vulkan compute: ENABLED")
        print(f"Embedding dimension: {memory.embedding_dim}")
    
    print()


def _teach_mode(phi3, memory: MemoryStore, snn: SNNCompute):
    """Protected teaching mode - create memories that will never be deleted"""
    print("\n" + "=" * 60)
    print("PROTECTED TEACHING MODE")
    print("=" * 60)
    print("All memories created here are PROTECTED and will never be deleted.")
    print("\nCommands:")
    print("  teach <text>        - Store protected memory from text")
    print("  file <path>         - Import training data from file")
    print("  list                - Show all protected memories")
    print("  stats               - Show memory statistics")
    print("  quit                - Exit teach mode")
    print("=" * 60 + "\n")
    
    while True:
        try:
            user_input = input("Teach> ").strip()
            
            if not user_input:
                continue
            
            # Parse command
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if cmd in ['quit', 'exit', 'q']:
                print("\nExiting teach mode...")
                break
            
            elif cmd == 'teach':
                if not arg:
                    print("Usage: teach <text>")
                    continue
                
                # Get embedding
                embedding = phi3.get_embedding(arg)
                
                # Store as protected
                memory.store(embedding, arg, is_protected=True)
                print(f"{LogConfig.CHECK} Protected memory stored: {arg[:60]}...")
            
            elif cmd == 'file':
                if not arg:
                    print("Usage: file <path>")
                    continue
                
                filepath = Path(arg)
                if not filepath.exists():
                    print(f"{LogConfig.CROSS} File not found: {filepath}")
                    continue
                
                # Read file
                try:
                    content = filepath.read_text(encoding='utf-8')
                    lines = [line.strip() for line in content.split('\n') if line.strip()]
                    
                    print(f"Found {len(lines)} lines in {filepath.name}")
                    confirm = input(f"Store all {len(lines)} as protected memories? (yes/no): ")
                    
                    if confirm.lower() != 'yes':
                        print("Cancelled.")
                        continue
                    
                    # Store each line as protected memory
                    for i, line in enumerate(lines, 1):
                        embedding = phi3.get_embedding(line)
                        memory.store(embedding, line, is_protected=True)
                        print(f"\r[{i}/{len(lines)}] Storing...", end='')
                    
                    print(f"\n{LogConfig.CHECK} Imported {len(lines)} protected memories from {filepath.name}")
                
                except Exception as e:
                    print(f"{LogConfig.CROSS} Error reading file: {e}")
            
            elif cmd == 'list':
                # Query database for protected memories
                import sqlite3
                # Handle both MemoryStore and plugin backends
                db_path = getattr(memory, 'db_path', None)
                if db_path is None and hasattr(memory, '_backend'):
                    db_path = getattr(memory._backend, 'db_path', None)
                
                if db_path is None:
                    print(f"{LogConfig.CROSS} Cannot access database: no database path available")
                    continue
                
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT text, timestamp FROM memories WHERE is_protected = 1 ORDER BY timestamp DESC")
                rows = cursor.fetchall()
                conn.close()
                
                if not rows:
                    print("No protected memories found.")
                else:
                    print(f"\n=== Protected Memories ({len(rows)}) ===")
                    for i, (text, timestamp) in enumerate(rows, 1):
                        print(f"{i}. [{timestamp[:19]}] {text[:80]}")
                    print()
            
            elif cmd == 'stats':
                _show_stats(memory)
                
                # Show protected count
                import sqlite3
                # Handle both MemoryStore and plugin backends
                db_path = getattr(memory, 'db_path', None)
                if db_path is None and hasattr(memory, '_backend'):
                    db_path = getattr(memory._backend, 'db_path', None)
                
                if db_path is None:
                    print(f"{LogConfig.CROSS} Cannot access database: no database path available")
                    continue
                
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM memories WHERE is_protected = 1")
                protected_count = cursor.fetchone()[0]
                conn.close()
                
                print(f"\nProtected memories: {protected_count}")
                print()
            
            else:
                print(f"Unknown command: {cmd}")
                print("Use 'teach <text>', 'file <path>', 'list', 'stats', or 'quit'")
        
        except KeyboardInterrupt:
            print("\n\nExiting teach mode...")
            break
        except EOFError:
            print("\n\nExiting teach mode...")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    run()