"""
GrillCheese AI - FastAPI WebSocket Server
Main entry point for the web demo

Features:
- Persistent memory with GPU acceleration
- Bio-inspired brain module (emotional intelligence)
- SNN-based spike visualization
- Continuous STDP learning
"""
import json
import logging
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import ServerConfig, MemoryConfig, SNNConfig, LogConfig, ModelConfig, ModuleConfig, find_gguf_model
from memory_store import MemoryStore
from identity import DEFAULT_IDENTITY
from vulkan_backend import SNNCompute

# Module system imports
try:
    from modules.registry import ModuleRegistry
    from modules.tools import ToolExecutor
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    logger.warning("Module system not available, using legacy initialization")

# Configure logging
logging.basicConfig(level=LogConfig.LEVEL, format=LogConfig.FORMAT)
logger = logging.getLogger(__name__)

# Try GGUF first (GPU accelerated)
try:
    from model_gguf import Phi3GGUF
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    logger.info("GGUF model not available")

# Fallback to PyTorch
try:
    from model import Phi3Model
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.info("PyTorch model not available")

# Try to import brain module
try:
    from brain import UnifiedBrain
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False
    logger.info("Brain module not available")

# Try to import continuous learning
try:
    from learning import ContinuousLearner, LearningConfig
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False
    logger.info("Continuous learning module not available")

# FastAPI app
app = FastAPI(
    title="GrillCheese AI",
    description="Local AI assistant with persistent memory",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global instances (initialized on startup)
phi3 = None
memory = None
snn = None
brain = None
learner = None
registry = None
tool_executor = None


@app.on_event("startup")
async def startup_event():
    """Initialize model and components on startup"""
    global phi3, memory, snn, brain, learner, registry, tool_executor
    
    logger.info("Starting GrillCheese AI server...")
    
    # Initialize module system
    if MODULES_AVAILABLE:
        try:
            logger.info("Loading modules...")
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
            else:
                # Fallback to direct initialization
                logger.warning("No memory backend found in registry, using legacy initialization")
                phi3 = _init_model()
                embedding_dim = ModelConfig.detect_embedding_dim(phi3) if phi3 else MemoryConfig.EMBEDDING_DIM
                memory = MemoryStore(
                    db_path=MemoryConfig.DB_PATH,
                    max_memories=MemoryConfig.MAX_MEMORIES,
                    embedding_dim=embedding_dim,
                    identity=DEFAULT_IDENTITY
                )
            
            if not phi3:
                # Fallback to direct initialization
                logger.warning("No model provider found in registry, using legacy initialization")
                phi3 = _init_model()
            
            # Initialize tool executor
            tool_executor = ToolExecutor(registry)
            
            # Register API extensions
            for extension in registry.api_extensions:
                extension.register_routes(app)
                extension.register_websockets(app)
            
            logger.info(f"{LogConfig.CHECK} Module system initialized")
        except Exception as e:
            logger.error(f"Module system initialization failed: {e}", exc_info=True)
            logger.info("Falling back to legacy initialization")
            phi3 = _init_model()
            embedding_dim = ModelConfig.detect_embedding_dim(phi3) if phi3 else MemoryConfig.EMBEDDING_DIM
            memory = MemoryStore(
                db_path=MemoryConfig.DB_PATH,
                max_memories=MemoryConfig.MAX_MEMORIES,
                embedding_dim=embedding_dim,
                identity=DEFAULT_IDENTITY
            )
    else:
        # Legacy initialization
        phi3 = _init_model()
        embedding_dim = ModelConfig.detect_embedding_dim(phi3) if phi3 else MemoryConfig.EMBEDDING_DIM
        logger.info(f"Detected embedding dimension: {embedding_dim}")
        logger.info(f"Initializing memory store (embedding_dim={embedding_dim})...")
        memory = MemoryStore(
            db_path=MemoryConfig.DB_PATH,
            max_memories=MemoryConfig.MAX_MEMORIES,
            embedding_dim=embedding_dim,
            identity=DEFAULT_IDENTITY
        )
        logger.info(f"{LogConfig.CHECK} Memory store initialized")
    
    # Initialize SNN
    logger.info("Initializing SNN compute...")
    snn = SNNCompute(n_neurons=SNNConfig.N_NEURONS, use_vulkan=True)
    logger.info(f"{LogConfig.CHECK} SNN compute initialized")
    
    # Initialize Brain module (emotional intelligence)
    if BRAIN_AVAILABLE:
        logger.info("Initializing brain module...")
        brain = UnifiedBrain(
            memory_store=memory,
            embedding_dim=embedding_dim,
            state_dir="brain_state",
            use_gpu=True,
            model=phi3,  # Pass model for reranking
            enable_reranking=False  # Disabled for performance (reranking is computationally expensive)
        )
        
        # Check if affect prediction is calibrated
        if brain.is_amygdala_calibrated():
            logger.info(f"{LogConfig.CHECK} Brain module initialized (affect calibrated, GPU mode)")
        else:
            logger.info(f"{LogConfig.CHECK} Brain module initialized (GPU mode)")
            logger.info(f"{LogConfig.WARNING} Amygdala not calibrated - emotional understanding limited")
            logger.info("Run CLI with --calibrate to train affect prediction")
    else:
        logger.info("Brain module not available")
    
    # Initialize continuous learning
    if LEARNING_AVAILABLE and brain is not None:
        logger.info("Initializing continuous learning...")
        learner = ContinuousLearner(
            memory_store=memory,
            snn_compute=snn,
            embedder=phi3,
            config=LearningConfig()
        )
        await learner.start()
        logger.info(f"{LogConfig.CHECK} Continuous learning started")
    
    # Store system identity if not already stored
    # Check if identity exists - handle both MemoryStore and plugin backends
    identity_text = memory.get_identity()
    identity_index = getattr(memory, 'identity_index', -1)
    
    if identity_text and identity_index == -1:
        logger.info("Storing system identity...")
        identity_emb = phi3.get_embedding(DEFAULT_IDENTITY)
        memory.store_identity(identity_emb, DEFAULT_IDENTITY)
        logger.info(f"{LogConfig.CHECK} System identity stored")
    else:
        logger.info(f"{LogConfig.CHECK} System identity already stored")


def _init_model():
    """Initialize the language model (GGUF preferred, PyTorch fallback)"""
    logger.info("Loading language model...")
    
    # Try GGUF first
    if GGUF_AVAILABLE:
        model_path = find_gguf_model()
        if model_path:
            try:
                logger.info(f"Using GGUF model: {model_path}")
                model = Phi3GGUF(model_path=model_path, n_gpu_layers=-1)
                logger.info(f"{LogConfig.CHECK} GGUF model loaded (GPU accelerated)")
                return model
            except Exception as e:
                logger.warning(f"Failed to load GGUF model: {e}")
        else:
            logger.info("GGUF model file not found")
    
    # Fallback to PyTorch
    if PYTORCH_AVAILABLE:
        try:
            model = Phi3Model()
            logger.info(f"{LogConfig.CHECK} PyTorch model loaded")
            return model
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {e}")
            raise
    
    raise RuntimeError("No model available. Install llama-cpp-python or transformers")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for chat interface"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            prompt_preview = msg.get('prompt', '')[:50]
            logger.info(f"Received message: {prompt_preview}...")
            
            if phi3 is None:
                await websocket.send_text(json.dumps({
                    "error": "Model not loaded yet. Please wait..."
                }))
                continue
            
            user_prompt = msg.get('prompt', '')
            if not user_prompt:
                await websocket.send_text(json.dumps({"error": "No prompt provided"}))
                continue
            
            # Process the prompt
            response_data = await _process_prompt(user_prompt)
            await websocket.send_text(json.dumps(response_data))
            logger.info(f"Sent response with {response_data.get('spike_activity', 0):.0f} spikes")
            
    except WebSocketDisconnect:
        logger.info("WebSocket connection closed by client")
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint: {e}")
        try:
            await websocket.close()
        except:
            pass


async def _process_prompt(prompt: str) -> dict:
    """Process a user prompt and return response data"""
    try:
        # Request context for hooks
        context_dict = {
            "user_id": "default",
            "session_id": "default"
        }
        
        # Pre-process hooks
        enhanced_prompt = prompt
        if registry and registry.processing_hooks:
            for hook in registry.processing_hooks:
                try:
                    enhanced_prompt = await hook.pre_process(enhanced_prompt, context_dict)
                except Exception as e:
                    logger.warning(f"Hook {hook.name} pre_process failed: {e}")
        
        # Extract embedding
        embedding = phi3.get_embedding(enhanced_prompt)
        
        # Process through brain module (emotional intelligence)
        brain_result = None
        emotional_context = ""
        if brain is not None:
            brain_result = brain.process(enhanced_prompt, embedding)
            
            # Get empathy-aware prompt prefix
            emotional_context = brain.get_empathy_prompt()
            
            # Get response style recommendations
            response_style = brain.get_response_style()
        
        # Store in memory
        memory.store(embedding, enhanced_prompt)
        
        # Retrieve similar memories (identity automatically included)
        # Handle both MemoryStore API (returns List[str]) and plugin API (returns List[Tuple[str, float]])
        retrieved = memory.retrieve(embedding, k=MemoryConfig.DEFAULT_K)
        if retrieved and isinstance(retrieved[0], tuple):
            # Plugin API: extract texts from tuples
            context = [text for text, score in retrieved]
        else:
            # Legacy API: already List[str]
            context = retrieved
        
        # Build enhanced prompt with emotional context
        if emotional_context:
            final_prompt = f"{emotional_context}\n\nUser: {enhanced_prompt}"
        else:
            final_prompt = enhanced_prompt
        
        # Generate response with context (and tools if available)
        if registry and tool_executor and registry.tools:
            tools = registry.get_tools()
            response_text = phi3.generate_with_tools(
                final_prompt,
                context,
                tools=tools,
                tool_executor=tool_executor
            )
        else:
            response_text = phi3.generate(final_prompt, context)
        
        # Provide feedback to brain for learning
        if brain is not None and brain_result is not None:
            brain.provide_feedback(quality=0.7, strategy_worked=True)
        
        # Continuous learning
        if learner is not None:
            learner.learn_from_conversation(prompt, response_text, context)
        
        # Compute spike activity for visualization
        spike_metrics = snn.process(embedding)
        
        # Build response
        response_data = {
            "response": response_text,
            "memories": context,
            "spike_activity": spike_metrics['spike_activity'],
            "firing_rate": spike_metrics.get('firing_rate', 0),
            "memory_stats": memory.get_stats()
        }
        
        # Add brain state if available
        if brain_result is not None:
            response_data["brain_state"] = {
                "emotional_state": {
                    "valence": brain_result['emotional_state'].valence,
                    "arousal": brain_result['emotional_state'].arousal,
                    "dominant_emotion": brain_result['emotional_state'].dominant_emotion
                },
                "strategy": brain_result['strategy'],
                "confidence": brain_result['confidence'],
                "consciousness_level": brain_result['consciousness_level'],
                "stress_level": brain_result['stress_level'],
                "spatial_context": brain_result.get('spatial_context', {}),
                "modulation": brain_result['modulation']
            }
        
        # Post-process hooks
        if registry and registry.processing_hooks:
            for hook in registry.processing_hooks:
                try:
                    response_data = await hook.post_process(response_data, context_dict)
                except Exception as e:
                    logger.warning(f"Hook {hook.name} post_process failed: {e}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing prompt: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "response": f"[Error: {str(e)}]",
            "spike_activity": 0
        }


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global learner, brain
    
    logger.info("Shutting down GrillCheese AI server...")
    
    # Stop continuous learning
    if learner is not None:
        await learner.stop()
        logger.info("Continuous learning stopped")
    
    # Save brain state
    if brain is not None:
        brain.consolidate()
        logger.info("Brain state saved")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": phi3 is not None,
        "memory_initialized": memory is not None,
        "snn_initialized": snn is not None,
        "brain_initialized": brain is not None,
        "learning_active": learner is not None and learner.is_running
    }


@app.get("/stats")
async def get_stats():
    """Get comprehensive system statistics"""
    stats = {}
    
    if memory is not None:
        stats["memory"] = memory.get_stats()
    
    if brain is not None:
        brain_stats = brain.get_stats()
        stats["brain"] = {
            "total_interactions": brain_stats['brain_stats']['total_interactions'],
            "positive_interactions": brain_stats['brain_stats']['positive_interactions'],
            "negative_interactions": brain_stats['brain_stats']['negative_interactions'],
            "empathetic_responses": brain_stats['brain_stats']['empathetic_responses'],
            "gpu_operations": brain_stats['brain_stats']['gpu_operations'],
            "consciousness_level": brain_stats['cns']['current_state']['consciousness'],
            "stress_level": brain_stats['cns']['current_state']['stress_level'],
            "current_emotion": brain_stats['amygdala']['current_state'],
            "hormone_levels": brain_stats['endocrine']['levels']
        }
    
    if learner is not None:
        stats["learning"] = learner.get_stats()
    
    return stats


@app.get("/brain/emotional-state")
async def get_emotional_state():
    """Get current emotional state of the brain"""
    if brain is None:
        return {"error": "Brain module not initialized"}
    
    state = brain.amygdala.get_state()
    modulation = brain.amygdala.get_emotional_modulation()
    
    return {
        "emotional_state": state.to_dict(),
        "modulation": modulation,
        "empathy_prompt": brain.get_empathy_prompt(),
        "response_style": brain.get_response_style()
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting GrillCheese server...")
    print(f"Server will be available at http://{ServerConfig.HOST}:{ServerConfig.PORT}")
    uvicorn.run(app, host=ServerConfig.HOST, port=ServerConfig.PORT)
