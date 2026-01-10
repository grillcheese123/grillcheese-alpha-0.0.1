"""
Integration Example: Multimodal, Multilingual Knowledge Distillation

This shows how to integrate the knowledge distillation system with 
your existing GrillCheese architecture.
"""

from pathlib import Path
import numpy as np
from learning.multimodal_encoder import MultimodalEncoder
from learning.multilingual_utils import MultilingualProcessor
from learning.knowledge_distillation import KnowledgeDistillation
from learning.multimodal_memory_store import MultimodalMemoryStore

class GrillCheeseMultimodal:
    """Enhanced GrillCheese with multimodal distillation"""
    
    def __init__(self, models_dir: Path, db_path: Path):
        # Initialize components
        self.encoder = MultimodalEncoder(models_dir, device="auto")
        self.lang_processor = MultilingualProcessor(primary_language='en')
        self.memory = MultimodalMemoryStore(db_path, embedding_dim=384)
        
        # Initialize knowledge distillation
        self.distillation = KnowledgeDistillation(
            memory_store=self.memory,
            encoder=self.encoder,
            lang_processor=self.lang_processor,
            quality_threshold=0.7,
            protected_threshold=0.85
        )
        
        # Load encoders
        self._load_encoders()
        
    def _load_encoders(self):
        """Load available encoders"""
        # Try loading each encoder
        text_loaded = self.encoder.load_text_encoder()
        image_loaded = self.encoder.load_image_encoder()
        # audio_loaded = self.encoder.load_audio_encoder()
        
        print(f"Encoders loaded - Text: {text_loaded}, Image: {image_loaded}")
        
    def process_interaction(self, user_msg: str, assistant_resp: str, 
                          emotion_context: dict = None) -> bool:
        """Process a conversation interaction"""
        knowledge = self.distillation.distill_interaction(
            user_message=user_msg,
            assistant_response=assistant_resp,
            emotion_context=emotion_context
        )
        
        if knowledge:
            print(f"✓ Distilled [{knowledge.language}] quality={knowledge.quality_score:.3f}")
            return True
        return False
        
    def process_multimodal(self, text: str = None, image_path: str = None, 
                          audio_path: str = None):
        """Process multimodal input"""
        knowledge = self.distillation.distill_multimodal(
            text=text,
            image=image_path,
            audio=audio_path,
            quality_score=0.8
        )
        
        if knowledge:
            print(f"✓ Stored multimodal: {knowledge.modality}")
            
    def process_document(self, text: str, title: str = None, protected: bool = True):
        """Process and chunk a document"""
        chunks = self.distillation.distill_document(
            text=text,
            title=title,
            protected=protected
        )
        
        print(f"✓ Distilled {len(chunks)} chunks from document")
        return chunks
        
    def search(self, query: str, k: int = 5, language: str = None):
        """Search knowledge base"""
        results = self.distillation.retrieve_knowledge(
            query=query,
            k=k,
            language=language
        )
        
        print(f"\nFound {len(results)} results:")
        for i, (content, score, metadata) in enumerate(results, 1):
            lang = metadata.get('language', 'unknown')
            mod = metadata.get('modality', 'text')
            print(f"{i}. [{lang}|{mod}] {score:.3f} - {content[:100]}...")
            
        return results
        
    def get_statistics(self):
        """Get comprehensive statistics"""
        distill_stats = self.distillation.get_stats()
        memory_stats = self.memory.get_stats()
        
        print("\n=== Knowledge Distillation Stats ===")
        print(f"Total distilled: {distill_stats['total_distilled']}")
        print(f"Protected: {distill_stats['protected_count']}")
        print(f"By modality: {distill_stats['by_modality']}")
        print(f"By language: {distill_stats['by_language']}")
        
        print("\n=== Memory Store Stats ===")
        print(f"Total memories: {memory_stats['total_memories']}")
        print(f"Protected: {memory_stats['protected_memories']}")
        print(f"Attachments: {memory_stats['total_attachments']}")
        
        return distill_stats, memory_stats

# Example usage
def example_usage():
    """Example demonstrating the system"""
    
    # Initialize system
    system = GrillCheeseMultimodal(
        models_dir=Path("./models"),
        db_path=Path("./memories_multimodal.db")
    )
    
    # Example 1: Text interaction (English)
    print("\n1. Processing English interaction...")
    system.process_interaction(
        user_msg="How does photosynthesis work?",
        assistant_resp="Photosynthesis converts light energy into chemical energy in plants...",
        emotion_context={'valence': 0.5, 'arousal': 0.3}
    )
    
    # Example 2: Multilingual interaction (Spanish)
    print("\n2. Processing Spanish interaction...")
    system.process_interaction(
        user_msg="¿Cómo funciona la fotosíntesis?",
        assistant_resp="La fotosíntesis convierte la energía luminosa en energía química...",
        emotion_context={'valence': 0.6, 'arousal': 0.4}
    )
    
    # Example 3: Multimodal (text + image)
    print("\n3. Processing multimodal input...")
    system.process_multimodal(
        text="This diagram shows the photosynthesis process",
        image_path="./images/photosynthesis_diagram.png"
    )
    
    # Example 4: Document processing
    print("\n4. Processing document...")
    document = """
    Photosynthesis is the process by which plants use sunlight, water, and 
    carbon dioxide to produce oxygen and energy in the form of sugar.
    
    The process occurs in chloroplasts, specifically in structures called thylakoids.
    Light-dependent reactions capture energy from sunlight.
    """
    system.process_document(
        text=document,
        title="Photosynthesis Overview",
        protected=True
    )
    
    # Example 5: Search
    print("\n5. Searching knowledge base...")
    system.search("photosynthesis plants", k=3)
    
    # Example 6: Multilingual search
    print("\n6. Searching in Spanish...")
    system.search("fotosíntesis", k=2, language='es')
    
    # Example 7: Statistics
    print("\n7. Getting statistics...")
    system.get_statistics()

# Advanced integration example with existing GrillCheese
def integrate_with_existing_system():
    """
    Example showing integration with your existing system
    """
    
    # Your existing components
    from model_gguf import Phi3GGUF  # Your existing model
    from brain.unified_brain import UnifiedBrain  # Your existing brain
    from memory_store import MemoryStore

    memory = MemoryStore(db_path="./memories.db", embedding_dim=384)
    # Initialize your existing system
    phi3 = Phi3GGUF(model_path="./models/Phi-3-mini-4k-instruct-q4.gguf")
    brain = UnifiedBrain(
        memory_store=memory, 
        embedding_dim=384, 
        state_dir="./brain_state",
        model=phi3,  # Pass model for reranking
        enable_reranking=True  # Enable reranking for better memory retrieval
    )
    
    # Add multimodal distillation
    multimodal = GrillCheeseMultimodal(
        models_dir=Path("./models"),
        db_path=Path("./memories_multimodal.db")
    )
    
    # Enhanced conversation loop
    def enhanced_conversation(user_input: str):
        # Detect language
        language = multimodal.lang_processor.detect_language(user_input)
        print(f"Detected language: {language}")
        
        # Retrieve relevant context (multimodal)
        context = multimodal.distillation.retrieve_knowledge(
            query=user_input,
            k=5,
            language=language
        )
        
        # Get emotional state from brain
        emotion = brain.amygdala.get_state()
        
        # Generate response with Phi-3
        response = phi3.generate(user_input, context=context)
        embedding = phi3.get_embedding(response)
        
        # Update brain state
        brain.process(user_input, embedding, context=context)
        
        # Distill knowledge (auto quality scoring)
        multimodal.process_interaction(
            user_msg=user_input,
            assistant_resp=response,
            emotion_context=emotion.to_dict()
        )
        
        return response
    
    # Test the integrated system
    response = enhanced_conversation("Explique-moi la photosynthèse")
    print(f"Response: {response}")

if __name__ == "__main__":
    # Run basic example
    #example_usage()
    
    # Uncomment to test integration
    integrate_with_existing_system()
