"""
Continuous Learning Orchestrator for GrillCheese

Coordinates background learning from:
- User conversations (primary)
- Local text files (vocabulary expansion)
- RSS feeds (optional, for knowledge updates)

Integrates with:
- Memory Store (persistence)
- SNN Compute (spike-based processing)
- STDP Learner (temporal associations)
"""
import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .events import EventBus
from .stdp_learner import STDPLearner

logger = logging.getLogger(__name__)


class ContentCategory(Enum):
    """Categories for learning content"""
    CONVERSATION = "conversation"
    LOCAL_FILE = "local_file"
    RSS_FEED = "rss_feed"
    SYSTEM = "system"


class ProcessingPriority(Enum):
    """Processing priority levels"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class LearningConfig:
    """Configuration for continuous learning"""
    # Learning rates
    stdp_lr_plus: float = 0.01
    stdp_lr_minus: float = 0.012
    stdp_time_window: int = 5
    
    # Processing settings
    queue_size: int = 1000
    batch_size: int = 10
    process_interval_sec: float = 1.0
    
    # Local vocab settings
    vocab_dir: Optional[str] = None
    vocab_scan_interval_sec: float = 30.0
    max_file_size: int = 8000
    
    # Persistence
    state_dir: str = "learning_state"
    save_interval_sec: float = 300.0  # 5 minutes
    
    # RSS (optional)
    rss_enabled: bool = False
    rss_feeds: List[str] = field(default_factory=list)


@dataclass
class ContentItem:
    """Item to be processed for learning"""
    text: str
    category: ContentCategory
    priority: ProcessingPriority
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    content_hash: str = ""
    processed: bool = False
    learning_result: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.content_hash:
            key = f"{self.text[:100]}:{self.source}"
            self.content_hash = hashlib.sha256(key.encode()).hexdigest()[:16]


class ContinuousLearner:
    """
    Continuous Learning Orchestrator for GrillCheese
    
    Features:
    - Background STDP learning from conversations
    - Local vocabulary ingestion from text files
    - Optional RSS feed processing
    - Persistent learning state
    
    Usage:
        learner = ContinuousLearner(memory_store, snn_compute, embedder)
        await learner.start()  # Start background processing
        
        # Process user conversation
        learner.learn_from_conversation(user_text, response_text)
        
        await learner.stop()  # Stop and save state
    """
    
    def __init__(
        self,
        memory_store,
        snn_compute,
        embedder,
        config: Optional[LearningConfig] = None
    ):
        """
        Initialize continuous learner
        
        Args:
            memory_store: MemoryStore instance for persistence
            snn_compute: SNNCompute instance for spike processing
            embedder: Model with get_embedding() method
            config: Learning configuration
        """
        self.memory = memory_store
        self.snn = snn_compute
        self.embedder = embedder
        self.config = config or LearningConfig()
        
        # Event bus for decoupled communication
        self.event_bus = EventBus()
        
        # STDP learner
        self.stdp = STDPLearner(
            learning_rate_plus=self.config.stdp_lr_plus,
            learning_rate_minus=self.config.stdp_lr_minus,
            time_window=self.config.stdp_time_window
        )
        
        # Content queue
        self.content_queue: asyncio.Queue[ContentItem] = asyncio.Queue(
            maxsize=self.config.queue_size
        )
        
        # Processed content cache (avoid reprocessing)
        self.processed_hashes: set = set()
        
        # State
        self.is_running = False
        self._tasks: List[asyncio.Task] = []
        self._seen_vocab_files: Dict[str, float] = {}
        
        # Statistics
        self.stats = {
            'items_processed': 0,
            'conversations_learned': 0,
            'vocab_files_ingested': 0,
            'stdp_updates': 0,
            'spikes_generated': 0,
            'errors': 0,
            'start_time': None,
            'last_save': None
        }
        
        # Subscribe to events
        self.event_bus.subscribe('memory_stored', self._on_memory_stored)
        self.event_bus.subscribe('spike_generated', self._on_spike_generated)
        
        # Load existing state if available
        self._load_state()
    
    # ==================== Public API ====================
    
    async def start(self) -> None:
        """Start background learning loops"""
        if self.is_running:
            return
        
        self.is_running = True
        self.stats['start_time'] = datetime.now().isoformat()
        
        logger.info("Starting continuous learning...")
        
        # Create background tasks
        self._tasks = [
            asyncio.create_task(self._process_queue_loop()),
            asyncio.create_task(self._save_state_loop()),
        ]
        
        # Optional: vocab directory scanning
        if self.config.vocab_dir and os.path.isdir(self.config.vocab_dir):
            self._tasks.append(asyncio.create_task(self._vocab_scan_loop()))
            logger.info(f"Vocab scanning enabled: {self.config.vocab_dir}")
        
        # Optional: RSS feed processing
        if self.config.rss_enabled and self.config.rss_feeds:
            self._tasks.append(asyncio.create_task(self._rss_loop()))
            logger.info(f"RSS processing enabled: {len(self.config.rss_feeds)} feeds")
        
        logger.info("[OK] Continuous learning started")
    
    async def stop(self) -> None:
        """Stop background learning and save state"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("Stopping continuous learning...")
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        # Save final state
        self._save_state()
        logger.info("[OK] Continuous learning stopped")
    
    def learn_from_conversation(
        self,
        user_text: str,
        response_text: str,
        context: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Learn from a user conversation (synchronous)
        
        This is called after each interaction to:
        1. Extract embeddings
        2. Update STDP associations
        3. Process through SNN
        4. Emit learning events
        
        Args:
            user_text: User's input text
            response_text: AI's response
            context: Retrieved memory context (if any)
        
        Returns:
            Learning statistics
        """
        try:
            # Get embeddings
            user_emb = self.embedder.get_embedding(user_text)
            response_emb = self.embedder.get_embedding(response_text)
            
            # Hash for token indices (simplified - use embedding indices)
            user_indices = self._embedding_to_indices(user_emb)
            response_indices = self._embedding_to_indices(response_emb)
            
            # STDP: Learn temporal association between user input and response
            stdp_result = self.stdp.process_sequence(user_indices + response_indices)
            
            # Learn association between query and response
            assoc_result = self.stdp.process_embedding_pair(
                user_indices, response_indices, relevance=0.8
            )
            
            # Process through SNN for spike patterns
            combined_emb = (user_emb + response_emb) / 2
            spike_result = self.snn.process(combined_emb)
            
            # Emit event
            self.event_bus.emit('learning_update', {
                'type': 'conversation',
                'stdp_updates': stdp_result.get('updates', 0),
                'spike_activity': spike_result.get('spike_activity', 0)
            })
            
            # Update stats
            self.stats['conversations_learned'] += 1
            self.stats['stdp_updates'] += stdp_result.get('updates', 0)
            self.stats['spikes_generated'] += spike_result.get('spike_activity', 0)
            
            return {
                'success': True,
                'stdp_updates': stdp_result.get('updates', 0),
                'association_updates': assoc_result.get('updates', 0),
                'spike_activity': spike_result.get('spike_activity', 0)
            }
            
        except Exception as e:
            logger.error(f"Conversation learning error: {e}")
            self.stats['errors'] += 1
            return {'success': False, 'error': str(e)}
    
    def queue_content(self, item: ContentItem) -> bool:
        """Add content to processing queue"""
        if item.content_hash in self.processed_hashes:
            return False
        
        try:
            self.content_queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            logger.warning("Content queue full, dropping item")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            **self.stats,
            'queue_size': self.content_queue.qsize(),
            'stdp_stats': self.stdp.get_stats(),
            'event_stats': self.event_bus.get_stats()
        }
    
    def load_conversational_dataset(
        self,
        dataset_path: str,
        learn: bool = True,
        store_memories: bool = True,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load and train on conversational dataset (JSONL format)
        
        Processes conversations in the format:
        {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }
        
        Args:
            dataset_path: Path to conversations_dataset.jsonl file
            learn: Whether to perform STDP learning
            store_memories: Whether to store conversations in memory
            limit: Maximum number of conversations to process
            
        Returns:
            Training statistics
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Conversational dataset not found: {dataset_path}")
        
        logger.info(f"Loading conversational dataset from: {dataset_path}")
        
        conversations_processed = 0
        message_pairs_processed = 0
        stdp_updates_total = 0
        memories_stored = 0
        errors = 0
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if limit and conversations_processed >= limit:
                    break
                
                try:
                    conv = json.loads(line.strip())
                    messages = conv.get('messages', [])
                    
                    if len(messages) < 2:
                        continue
                    
                    conversations_processed += 1
                    
                    # Process message pairs (user -> assistant)
                    for i in range(len(messages) - 1):
                        user_msg = messages[i]
                        assistant_msg = messages[i + 1]
                        
                        # Skip if not user->assistant pair
                        if user_msg.get('role') != 'user' or assistant_msg.get('role') != 'assistant':
                            continue
                        
                        user_text = user_msg.get('content', '').strip()
                        assistant_text = assistant_msg.get('content', '').strip()
                        
                        if not user_text or not assistant_text:
                            continue
                        
                        message_pairs_processed += 1
                        
                        # Store in memory if requested
                        if store_memories:
                            try:
                                user_emb = self.embedder.get_embedding(user_text)
                                self.memory.store(user_emb, user_text)
                                memories_stored += 1
                            except Exception as e:
                                logger.warning(f"Memory storage error: {e}")
                                errors += 1
                        
                        # Learn from conversation pair
                        if learn:
                            try:
                                result = self.learn_from_conversation(user_text, assistant_text)
                                if result.get('success'):
                                    stdp_updates_total += result.get('stdp_updates', 0)
                            except Exception as e:
                                logger.warning(f"Learning error: {e}")
                                errors += 1
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                    errors += 1
                except Exception as e:
                    logger.error(f"Error processing conversation at line {line_num}: {e}")
                    errors += 1
        
        logger.info(
            f"Conversational dataset loaded: {conversations_processed} conversations, "
            f"{message_pairs_processed} message pairs, {memories_stored} memories stored, "
            f"{stdp_updates_total} STDP updates"
        )
        
        return {
            'conversations_processed': conversations_processed,
            'message_pairs_processed': message_pairs_processed,
            'memories_stored': memories_stored,
            'stdp_updates': stdp_updates_total,
            'errors': errors
        }
    
    def load_tool_training_dataset(
        self,
        dataset_path: str,
        store_memories: bool = True,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load and train on tool usage examples
        
        This teaches the model when and how to use tools by:
        1. Storing tool usage examples as memories
        2. Learning associations between user queries and tool calls via STDP
        3. Building patterns for tool selection
        
        Args:
            dataset_path: Path to tool_training.jsonl file
            store_memories: Whether to store examples in memory
            limit: Maximum number of examples to process
            
        Returns:
            Training statistics
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Tool training dataset not found: {dataset_path}")
        
        logger.info(f"Loading tool training dataset from: {dataset_path}")
        
        examples_processed = 0
        memories_stored = 0
        stdp_updates_total = 0
        errors = 0
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if limit and examples_processed >= limit:
                    break
                
                try:
                    data = json.loads(line.strip())
                    instruction = data.get('instruction', '')
                    tool_calls = data.get('tool_calls', [])
                    tool_results = data.get('tool_results', [])
                    final_response = data.get('final_response', '')
                    
                    if not instruction:
                        continue
                    
                    # Create training example text that shows tool usage pattern
                    example_text = f"User: {instruction}\n"
                    
                    if tool_calls:
                        example_text += "Assistant: "
                        for i, call in enumerate(tool_calls):
                            tool_name = call.get('tool', '')
                            params = call.get('parameters', {})
                            import json as json_module
                            example_text += f'{{"tool": "{tool_name}", "parameters": {json_module.dumps(params)}}}'
                            if i < len(tool_calls) - 1:
                                example_text += "\n"
                    
                    if tool_results:
                        example_text += "\nTool Results:\n"
                        for result in tool_results:
                            tool_name = result.get('tool', '')
                            result_data = result.get('result', {})
                            import json as json_module
                            example_text += f"{tool_name}: {json_module.dumps(result_data)}\n"
                    
                    if final_response:
                        example_text += f"Final Response: {final_response}"
                    
                    # Store as memory if enabled
                    if store_memories:
                        try:
                            embedding = self.embedder.get_embedding(example_text)
                            
                            # Handle dimension mismatch
                            if len(embedding) != self.memory.embedding_dim:
                                embedding = self._resize_embedding(embedding, self.memory.embedding_dim)
                            
                            metadata = {
                                'type': 'tool_training_example',
                                'instruction': instruction,
                                'tool_calls': tool_calls,
                                'has_tools': len(tool_calls) > 0
                            }
                            
                            # Check if memory backend supports is_protected parameter
                            import inspect
                            try:
                                store_sig = inspect.signature(self.memory.store)
                                supports_protected = 'is_protected' in store_sig.parameters
                            except:
                                supports_protected = False
                            
                            if supports_protected:
                                # MemoryStore supports is_protected
                                self.memory.store(
                                    embedding=embedding,
                                    text=example_text,
                                    metadata=metadata,
                                    is_protected=True
                                )
                            else:
                                # Plugin backend - store without is_protected, mark in metadata
                                metadata['is_protected'] = True
                                self.memory.store(
                                    embedding=embedding,
                                    text=example_text,
                                    metadata=metadata
                                )
                                # Try to access underlying MemoryStore if available
                                if hasattr(self.memory, '_backend'):
                                    try:
                                        backend_store_sig = inspect.signature(self.memory._backend.store)
                                        if 'is_protected' in backend_store_sig.parameters:
                                            # Re-store with protection flag
                                            self.memory._backend.store(
                                                embedding=embedding,
                                                text=example_text,
                                                metadata=metadata,
                                                is_protected=True
                                            )
                                    except:
                                        pass
                            memories_stored += 1
                        except Exception as e:
                            logger.warning(f"Failed to store tool example: {e}")
                    
                    # STDP learning: learn association between instruction and tool usage
                    if tool_calls:
                        try:
                            instruction_emb = self.embedder.get_embedding(instruction)
                            if len(instruction_emb) != self.memory.embedding_dim:
                                instruction_emb = self._resize_embedding(instruction_emb, self.memory.embedding_dim)
                            
                            # Create tool call pattern embedding
                            tool_pattern = f"Use tool: {tool_calls[0].get('tool', '')}"
                            tool_emb = self.embedder.get_embedding(tool_pattern)
                            if len(tool_emb) != self.memory.embedding_dim:
                                tool_emb = self._resize_embedding(tool_emb, self.memory.embedding_dim)
                            
                            # Learn association
                            instruction_indices = self._embedding_to_indices(instruction_emb)
                            tool_indices = self._embedding_to_indices(tool_emb)
                            
                            # STDP: instruction -> tool call pattern
                            stdp_result = self.stdp.process_sequence(instruction_indices + tool_indices)
                            stdp_updates_total += stdp_result.get('updates', 0)
                            
                            # Also learn bidirectional association
                            assoc_result = self.stdp.process_embedding_pair(
                                instruction_indices, tool_indices, relevance=0.9
                            )
                            stdp_updates_total += assoc_result.get('updates', 0)
                        except Exception as e:
                            logger.warning(f"STDP learning error: {e}")
                    
                    examples_processed += 1
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                    errors += 1
                except Exception as e:
                    logger.error(f"Error processing tool example at line {line_num}: {e}")
                    errors += 1
        
        logger.info(
            f"Tool training dataset loaded: {examples_processed} examples, "
            f"{memories_stored} memories stored, {stdp_updates_total} STDP updates"
        )
        
        return {
            'examples_processed': examples_processed,
            'memories_stored': memories_stored,
            'stdp_updates': stdp_updates_total,
            'errors': errors
        }
    
    def load_temporal_dataset(
        self,
        dataset_path: str,
        learn: bool = True,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load and train on temporal dataset (events and associations)
        
        Args:
            dataset_path: Path to temporal_dataset.jsonl file
            learn: Whether to perform STDP learning
            limit: Maximum number of items to process
            
        Returns:
            Training statistics
        """
        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Temporal dataset not found: {dataset_path}")
        
        logger.info(f"Loading temporal dataset from: {dataset_path}")
        
        events_processed = 0
        associations_processed = 0
        stdp_updates_total = 0
        errors = 0
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if limit and (events_processed + associations_processed) >= limit:
                    break
                
                try:
                    data = json.loads(line.strip())
                    item_type = data.get('type')
                    item_data = data.get('data', {})
                    
                    if item_type == 'event':
                        result = self._process_temporal_event(item_data, learn=learn)
                        events_processed += 1
                        if result.get('stdp_updates'):
                            stdp_updates_total += result['stdp_updates']
                    elif item_type == 'association':
                        result = self._process_temporal_association(item_data, learn=learn)
                        associations_processed += 1
                        if result.get('stdp_updates'):
                            stdp_updates_total += result['stdp_updates']
                    else:
                        logger.warning(f"Unknown item type: {item_type}")
                        
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                    errors += 1
                except Exception as e:
                    logger.error(f"Error processing line {line_num}: {e}")
                    errors += 1
        
        logger.info(
            f"Temporal dataset loaded: {events_processed} events, "
            f"{associations_processed} associations, {stdp_updates_total} STDP updates"
        )
        
        return {
            'events_processed': events_processed,
            'associations_processed': associations_processed,
            'stdp_updates': stdp_updates_total,
            'errors': errors
        }
    
    def _process_temporal_event(
        self,
        event_data: Dict[str, Any],
        learn: bool = True
    ) -> Dict[str, Any]:
        """Process a temporal event for learning"""
        try:
            text = event_data.get('text', '')
            if not text:
                return {'error': 'No text in event'}
            
            # Get embedding
            embedding = self.embedder.get_embedding(text)
            
            # Handle dimension mismatch
            if len(embedding) != self.memory.embedding_dim:
                embedding = self._resize_embedding(embedding, self.memory.embedding_dim)
            
            # Get indices for STDP
            indices = self._embedding_to_indices(embedding)
            
            # STDP learning on event sequence
            stdp_result = self.stdp.process_sequence(indices) if learn else {'updates': 0}
            
            # SNN processing
            spike_result = self.snn.process(embedding)
            
            # Store in memory with temporal metadata
            if spike_result.get('spike_activity', 0) > 5:
                try:
                    metadata = {
                        'timestamp': event_data.get('timestamp'),
                        'date': event_data.get('date'),
                        'location': event_data.get('location'),
                        'keywords': event_data.get('keywords', []),
                        'section': event_data.get('section'),
                        'type': 'temporal_event'
                    }
                    self.memory.store(
                        embedding=embedding,
                        text=text[:500],
                        metadata=metadata
                    )
                except Exception as e:
                    logger.warning(f"Failed to store event: {e}")
            
            return {
                'stdp_updates': stdp_result.get('updates', 0),
                'spike_activity': spike_result.get('spike_activity', 0)
            }
            
        except Exception as e:
            logger.error(f"Error processing temporal event: {e}")
            return {'error': str(e)}
    
    def _process_temporal_association(
        self,
        assoc_data: Dict[str, Any],
        learn: bool = True
    ) -> Dict[str, Any]:
        """Process a temporal association for STDP learning"""
        try:
            # Get embeddings for both events
            event1_text = assoc_data.get('event1_text', '')
            event2_text = assoc_data.get('event2_text', '')
            
            if not event1_text or not event2_text:
                return {'error': 'Missing event text in association'}
            
            emb1 = self.embedder.get_embedding(event1_text)
            emb2 = self.embedder.get_embedding(event2_text)
            
            # Handle dimension mismatch
            if len(emb1) != self.memory.embedding_dim:
                emb1 = self._resize_embedding(emb1, self.memory.embedding_dim)
            if len(emb2) != self.memory.embedding_dim:
                emb2 = self._resize_embedding(emb2, self.memory.embedding_dim)
            
            # Get indices
            indices1 = self._embedding_to_indices(emb1)
            indices2 = self._embedding_to_indices(emb2)
            
            # STDP learning: learn association between events
            # Strength based on association type and temporal proximity
            strength = assoc_data.get('strength', 0.5)
            temporal_order = assoc_data.get('temporal_order', 'before')
            
            if learn:
                # Process as sequence if temporal_order is 'before'
                if temporal_order == 'before':
                    stdp_result = self.stdp.process_sequence(indices1 + indices2)
                else:
                    stdp_result = self.stdp.process_sequence(indices2 + indices1)
                
                # Also learn bidirectional association
                assoc_result = self.stdp.process_embedding_pair(
                    indices1, indices2, relevance=strength
                )
                total_updates = stdp_result.get('updates', 0) + assoc_result.get('updates', 0)
            else:
                total_updates = 0
            
            return {
                'stdp_updates': total_updates,
                'association_type': assoc_data.get('type'),
                'strength': strength
            }
            
        except Exception as e:
            logger.error(f"Error processing temporal association: {e}")
            return {'error': str(e)}
    
    # ==================== Background Loops ====================
    
    async def _process_queue_loop(self) -> None:
        """Background loop to process queued content"""
        logger.info("Starting content processing loop")
        
        while self.is_running:
            try:
                # Get item with timeout
                try:
                    item = await asyncio.wait_for(
                        self.content_queue.get(),
                        timeout=self.config.process_interval_sec
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process item
                result = await self._process_content_item(item)
                item.processed = True
                item.learning_result = result
                
                self.processed_hashes.add(item.content_hash)
                self.stats['items_processed'] += 1
                
                # Emit event
                self.event_bus.emit('content_processed', {
                    'hash': item.content_hash,
                    'category': item.category.value,
                    'result': result
                })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Process loop error: {e}")
                self.stats['errors'] += 1
    
    async def _vocab_scan_loop(self) -> None:
        """Background loop to scan vocabulary directory"""
        logger.info("Starting vocabulary scan loop")
        
        while self.is_running and self.config.vocab_dir:
            try:
                queued = 0
                
                for root, _, files in os.walk(self.config.vocab_dir):
                    for name in files:
                        if not name.lower().endswith('.txt'):
                            continue
                        
                        fpath = os.path.join(root, name)
                        
                        try:
                            mtime = os.path.getmtime(fpath)
                            
                            # Skip already processed
                            if fpath in self._seen_vocab_files:
                                if self._seen_vocab_files[fpath] >= mtime:
                                    continue
                            
                            # Read file
                            with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                                text = f.read(self.config.max_file_size)
                            
                            # Create content item
                            title = os.path.splitext(os.path.basename(fpath))[0]
                            item = ContentItem(
                                text=f"{title}. {text}",
                                category=ContentCategory.LOCAL_FILE,
                                priority=ProcessingPriority.MEDIUM,
                                source=fpath
                            )
                            
                            if self.queue_content(item):
                                self._seen_vocab_files[fpath] = mtime
                                queued += 1
                                self.stats['vocab_files_ingested'] += 1
                            
                            if queued >= 50:  # Batch limit
                                break
                                
                        except Exception as e:
                            logger.warning(f"Vocab file error {fpath}: {e}")
                    
                    if queued >= 50:
                        break
                
                if queued > 0:
                    logger.info(f"Queued {queued} vocab files for learning")
                
                await asyncio.sleep(self.config.vocab_scan_interval_sec)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Vocab scan error: {e}")
                await asyncio.sleep(60)
    
    async def _rss_loop(self) -> None:
        """Background loop for RSS feed processing (optional)"""
        logger.info("Starting RSS processing loop")
        
        try:
            import aiohttp
            import feedparser
        except ImportError:
            logger.warning("RSS requires: pip install aiohttp feedparser")
            return
        
        while self.is_running and self.config.rss_enabled:
            try:
                for feed_url in self.config.rss_feeds:
                    await self._fetch_rss_feed(feed_url)
                
                await asyncio.sleep(1800)  # 30 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"RSS loop error: {e}")
                await asyncio.sleep(300)
    
    async def _save_state_loop(self) -> None:
        """Background loop to periodically save state"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.save_interval_sec)
                self._save_state()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Save state error: {e}")
    
    # ==================== Processing ====================
    
    async def _process_content_item(self, item: ContentItem) -> Dict[str, Any]:
        """Process a single content item for learning"""
        try:
            # Get embedding
            embedding = self.embedder.get_embedding(item.text)
            
            # Handle embedding dimension mismatch
            if len(embedding) != self.memory.embedding_dim:
                logger.warning(
                    f"Embedding dimension mismatch: expected {self.memory.embedding_dim}, "
                    f"got {len(embedding)}. Resizing embedding..."
                )
                # Resize embedding to match memory store dimension
                embedding = self._resize_embedding(embedding, self.memory.embedding_dim)
            
            # Get token indices for STDP
            indices = self._embedding_to_indices(embedding)
            
            # STDP learning
            stdp_result = self.stdp.process_sequence(indices)
            
            # SNN processing
            spike_result = self.snn.process(embedding)
            
            # Store in memory if significant
            if spike_result.get('spike_activity', 0) > 10:
                try:
                    self.memory.store(embedding, item.text[:500])
                except ValueError as e:
                    # If dimension still doesn't match after resize, skip storage
                    logger.warning(f"Skipping memory storage due to dimension mismatch: {e}")
            
            # Update stats
            self.stats['stdp_updates'] += stdp_result.get('updates', 0)
            self.stats['spikes_generated'] += spike_result.get('spike_activity', 0)
            
            return {
                'stdp_updates': stdp_result.get('updates', 0),
                'spike_activity': spike_result.get('spike_activity', 0),
                'processing_time': time.time()
            }
            
        except Exception as e:
            logger.error(f"Content processing error: {e}")
            return {'error': str(e)}
    
    async def _fetch_rss_feed(self, feed_url: str) -> None:
        """Fetch and process RSS feed"""
        try:
            import aiohttp
            import feedparser
            import re
            
            async with aiohttp.ClientSession() as session:
                async with session.get(feed_url, timeout=30) as resp:
                    if resp.status != 200:
                        return
                    text = await resp.text()
            
            parsed = feedparser.parse(text)
            
            for entry in parsed.entries[:10]:
                title = entry.get('title', '')
                content = entry.get('summary', '') or entry.get('description', '')
                
                # Strip HTML
                content = re.sub('<[^<]+?>', '', content)
                
                item = ContentItem(
                    text=f"{title}. {content}",
                    category=ContentCategory.RSS_FEED,
                    priority=ProcessingPriority.LOW,
                    source=feed_url
                )
                
                self.queue_content(item)
                
        except Exception as e:
            logger.warning(f"RSS fetch error {feed_url}: {e}")
    
    # ==================== Helpers ====================
    
    def _embedding_to_indices(self, embedding: np.ndarray, top_k: int = 100) -> List[int]:
        """Convert embedding to token indices for STDP"""
        # Use top-k absolute values as "active" indices
        abs_emb = np.abs(embedding)
        indices = np.argsort(abs_emb)[-top_k:]
        return indices.tolist()
    
    def _resize_embedding(self, embedding: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Resize embedding to target dimension
        
        If embedding is larger, truncate or average pool.
        If embedding is smaller, pad with zeros or repeat.
        """
        current_dim = len(embedding)
        
        if current_dim == target_dim:
            return embedding
        
        if current_dim > target_dim:
            # Truncate or average pool
            # Use PCA-like approach: take first target_dim dimensions
            # (in practice, embeddings are usually normalized, so this is reasonable)
            return embedding[:target_dim]
        else:
            # Pad with zeros (or repeat pattern)
            padding = np.zeros(target_dim - current_dim, dtype=embedding.dtype)
            return np.concatenate([embedding, padding])
    
    def _save_state(self) -> None:
        """Save learning state to disk"""
        try:
            state_dir = Path(self.config.state_dir)
            state_dir.mkdir(exist_ok=True)
            
            # Save STDP state
            self.stdp.save_state(str(state_dir / "stdp_state.json"))
            
            # Save learner stats
            with open(state_dir / "learner_stats.json", 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
            
            # Save processed hashes (limit to recent)
            recent_hashes = list(self.processed_hashes)[-10000:]
            with open(state_dir / "processed_hashes.json", 'w') as f:
                json.dump(recent_hashes, f)
            
            self.stats['last_save'] = datetime.now().isoformat()
            logger.debug("Learning state saved")
            
        except Exception as e:
            logger.error(f"Save state error: {e}")
    
    def _load_state(self) -> None:
        """Load learning state from disk"""
        try:
            state_dir = Path(self.config.state_dir)
            
            # Load STDP state
            stdp_path = state_dir / "stdp_state.json"
            if stdp_path.exists():
                self.stdp.load_state(str(stdp_path))
            
            # Load processed hashes
            hashes_path = state_dir / "processed_hashes.json"
            if hashes_path.exists():
                with open(hashes_path, 'r') as f:
                    self.processed_hashes = set(json.load(f))
            
            logger.info(f"Loaded learning state: {len(self.processed_hashes)} processed items")
            
        except Exception as e:
            logger.warning(f"Load state error: {e}")
    
    # ==================== Event Handlers ====================
    
    def _on_memory_stored(self, event) -> None:
        """Handle memory storage event"""
        pass  # Future: track memory patterns
    
    def _on_spike_generated(self, event) -> None:
        """Handle spike generation event"""
        pass  # Future: analyze spike patterns

