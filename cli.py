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
    
    # Interactive mode or single prompt
    if args.interactive or (not prompt and not args.stats):
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
        
        # Print stats
        stats = memory.get_stats()
        firing_rate = spike_metrics.get('firing_rate', 0) * 100
        stats_line = f"[Spikes: {spike_metrics['spike_activity']:.0f} ({firing_rate:.1f}% rate) | Memories: {stats['total_memories']}"
        if learner:
            lstats = learner.get_stats()
            stats_line += f" | STDP: {lstats['stdp_updates']} updates"
        if brain_result is not None:
            stats_line += f" | Strategy: {brain_result['strategy']} | Stress: {brain_result['stress_level']:.2f}"
        stats_line += "]"
        print(stats_line)
        
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
            
            stats = memory.get_stats()
            stats_line = f"[Spikes: {spike_metrics['spike_activity']:.0f} | Memories: {stats['total_memories']}"
            if brain_result is not None:
                emo_state = brain_result['emotional_state']
                stats_line += f" | {emo_state.dominant_emotion} (v:{emo_state.valence:.1f} a:{emo_state.arousal:.1f})"
            stats_line += "]"
            print(f"{stats_line}\n")
            
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


if __name__ == "__main__":
    main()
