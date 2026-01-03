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

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import MemoryConfig, SNNConfig, LogConfig, ModelConfig, find_gguf_model
from memory_store import MemoryStore
from identity import DEFAULT_IDENTITY
from vulkan_backend import SNNCompute

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


def main():
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
    
    args = parser.parse_args()
    
    # Initialize system
    print("Initializing GrillCheese AI...")
    
    # Load model
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
        memory = MemoryStore(
            db_path=args.db,
            embedding_dim=embedding_dim,
            identity=DEFAULT_IDENTITY
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
    if memory.get_identity() and memory.identity_index == -1:
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
                use_gpu=True
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
            asyncio.run(_interactive_mode_async(phi3, memory, snn, learner, brain))
        else:
            _interactive_mode(phi3, memory, snn, brain)
    elif prompt:
        _process_prompt(phi3, memory, snn, prompt, learner, brain)
    else:
        print("Error: No prompt provided. Use --interactive for interactive mode.")
        parser.print_help()
        sys.exit(1)


def _init_model():
    """Initialize the language model"""
    print("Loading Phi-3 model...")
    
    # Try GGUF first
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
    
    # Fallback to PyTorch
    if PYTORCH_AVAILABLE:
        try:
            phi3 = Phi3Model()
            print(f"{LogConfig.CHECK} PyTorch model loaded")
            return phi3
        except Exception as e:
            print(f"{LogConfig.CROSS} Failed to load model: {e}")
    
    return None


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


def _process_prompt(phi3, memory: MemoryStore, snn: SNNCompute, prompt: str, learner=None, brain=None):
    """Process a single prompt and print response"""
    print(f"\nYou: {prompt}\n")
    
    try:
        # Extract embedding
        print("Extracting embedding...", end="\r")
        embedding = phi3.get_embedding(prompt)
        print(f"{LogConfig.CHECK} Embedding extracted    ")
        
        # Process through brain module (emotional intelligence)
        brain_result = None
        emotional_context = ""
        if brain is not None:
            print("Processing emotions...", end="\r")
            brain_result = brain.process(prompt, embedding)
            emotional_context = brain.get_empathy_prompt()
            print(f"{LogConfig.CHECK} Emotion: {brain_result['emotional_state'].dominant_emotion} (valence: {brain_result['emotional_state'].valence:.2f})")
        
        # Store in memory
        print("Storing in memory...", end="\r")
        memory.store(embedding, prompt)
        print(f"{LogConfig.CHECK} Stored in memory      ")
        
        # Retrieve context
        print("Retrieving context...", end="\r")
        context = memory.retrieve(embedding, k=MemoryConfig.DEFAULT_K)
        print(f"{LogConfig.CHECK} Retrieved {len(context)} context items")
        
        # Build enhanced prompt with emotional context and self-awareness
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
        
        # Generate response
        device_msg = "GPU" if phi3.device != "cpu" else "CPU"
        print(f"Generating response on {device_msg}...")
        response = phi3.generate(enhanced_prompt, context)
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


def _interactive_mode(phi3, memory: MemoryStore, snn: SNNCompute, brain=None):
    """Interactive conversation mode with emotional intelligence"""
    print("\n" + "=" * 60)
    print("GrillCheese AI - Interactive Mode")
    if brain is not None:
        print("Emotional intelligence: ENABLED")
    print("Type your messages. Commands: 'quit', 'stats', 'clear', 'emotion'")
    print("=" * 60 + "\n")
    
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
            
            device_msg = "GPU" if phi3.device != "cpu" else "CPU"
            print(f"Generating on {device_msg}...")
            
            response = phi3.generate(enhanced_prompt, context)
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


async def _interactive_mode_async(phi3, memory: MemoryStore, snn: SNNCompute, learner, brain=None):
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
                
                device_msg = "GPU" if phi3.device != "cpu" else "CPU"
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
    
    conn = sqlite3.connect(memory.db_path)
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


def _dev_analyze_memory(memory: MemoryStore):
    """Deep memory analysis"""
    import sqlite3
    
    conn = sqlite3.connect(memory.db_path)
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
    
    # Export memory keys and values
    np.savez_compressed(
        output_file,
        keys=memory.memory_keys[:memory.num_memories],
        values=memory.memory_values[:memory.num_memories],
        texts=np.array(memory.memory_texts[:memory.num_memories], dtype=object)
    )
    
    print(f"{LogConfig.CHECK} Exported {memory.num_memories} embeddings")


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
    print(f"  Current strategy: {bg['current_strategy']}")
    print(f"  Hebbian weights shape: {bg['hebbian_weights_shape']}")
    
    print("\nExperience:")
    exp = stats['experience']
    print(f"  Total interactions: {exp['total_experiences']}")
    print(f"  Average quality: {exp['avg_quality']:.3f}")
    print(f"  Best strategy: {exp['best_strategy']}")
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
    
    # GPU stats
    if memory.gpu:
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
                conn = sqlite3.connect(memory.db_path)
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
                conn = sqlite3.connect(memory.db_path)
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
    main()

