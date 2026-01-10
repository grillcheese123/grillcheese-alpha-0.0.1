"""
Identity Dataset Loader
Loads identity capsules from JSONL format
"""
import json
import logging
from pathlib import Path
from typing import List, Optional

from memory.capsule_memory import CapsuleMemory, MemoryType

logger = logging.getLogger(__name__)


def load_identity_dataset(path: str) -> List[CapsuleMemory]:
    """
    Load identity capsule dataset from JSONL file
    
    Args:
        path: Path to JSONL file
    
    Returns:
        List of CapsuleMemory objects
    """
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning(f"Identity dataset not found: {path}")
        return []
    
    memories = []
    
    try:
        with open(path_obj, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines (JSON objects are separated by blank lines)
        # Handle multi-line JSON objects
        json_blocks = []
        current_block = []
        brace_count = 0
        
        for line in content.split('\n'):
            stripped = line.strip()
            
            # Skip markdown headers/comments
            if stripped.startswith('##') or (stripped.startswith('#') and not stripped.startswith('{')):
                if current_block and brace_count == 0:
                    json_blocks.append('\n'.join(current_block))
                    current_block = []
                continue
            
            # Count braces to track JSON object boundaries
            brace_count += line.count('{') - line.count('}')
            
            if stripped:
                current_block.append(line)
            
            # Complete JSON object when braces are balanced and we hit empty line or new object starts
            if brace_count == 0 and current_block:
                if not stripped or (stripped.startswith('{') and len(current_block) > 1):
                    # End of current object
                    block_text = '\n'.join(current_block)
                    if block_text.strip():
                        json_blocks.append(block_text)
                    current_block = []
                    brace_count = 0
        
        # Add last block if exists
        if current_block:
            block_text = '\n'.join(current_block)
            if block_text.strip():
                json_blocks.append(block_text)
        
        # Parse each JSON block
        for block_num, block in enumerate(json_blocks, 1):
            if not block.strip():
                continue
            
            try:
                data = json.loads(block)
                
                # Extract required fields
                memory_id = data.get('memory_id', f"identity_{block_num}")
                memory_type_str = data.get('memory_type', 'SELF_STATE')
                domain = data.get('domain', 'identity')
                content = data.get('content', '')
                
                if not content:
                    continue
                
                # Extract cognitive features
                cognitive_features = data.get('cognitive_features', {})
                plasticity_gain = cognitive_features.get('plasticity_gain', 1.0)
                consolidation_priority = cognitive_features.get('consolidation_priority', 1.0)
                stability = cognitive_features.get('stability', 1.0)
                stress_link = cognitive_features.get('stress_link', 0.0)
                
                # Create memory
                try:
                    memory_type = MemoryType(memory_type_str)
                except ValueError:
                    memory_type = MemoryType.SELF_STATE
                
                memory = CapsuleMemory(
                    memory_id=memory_id,
                    memory_type=memory_type,
                    domain=domain,
                    content=content,
                    plasticity_gain=plasticity_gain,
                    consolidation_priority=consolidation_priority,
                    stability=stability,
                    stress_link=stress_link,
                    protected=True  # Identity memories are always protected
                )
                
                memories.append(memory)
                
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in block {block_num}: {e}")
                logger.debug(f"Block content: {block[:100]}...")
                continue
            except Exception as e:
                logger.warning(f"Error parsing block {block_num}: {e}")
                continue
        
        logger.info(f"Loaded {len(memories)} identity memories from {path}")
        return memories
    
    except Exception as e:
        logger.error(f"Failed to load identity dataset: {e}")
        return []


def load_identity_from_json(path: str) -> List[CapsuleMemory]:
    """
    Load identity from JSON file (alternative format)
    
    Args:
        path: Path to JSON file
    
    Returns:
        List of CapsuleMemory objects
    """
    path_obj = Path(path)
    if not path_obj.exists():
        logger.warning(f"Identity JSON not found: {path}")
        return []
    
    memories = []
    
    try:
        with open(path_obj, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and 'memories' in data:
            items = data['memories']
        elif isinstance(data, dict):
            items = [data]
        else:
            logger.warning(f"Unexpected JSON structure in {path}")
            return []
        
        for item in items:
            try:
                memory_id = item.get('memory_id', f"identity_{len(memories)}")
                memory_type_str = item.get('memory_type', 'SELF_STATE')
                domain = item.get('domain', 'identity')
                content = item.get('content', '')
                
                if not content:
                    continue
                
                cognitive_features = item.get('cognitive_features', {})
                
                memory = CapsuleMemory(
                    memory_id=memory_id,
                    memory_type=MemoryType(memory_type_str),
                    domain=domain,
                    content=content,
                    plasticity_gain=cognitive_features.get('plasticity_gain', 1.0),
                    consolidation_priority=cognitive_features.get('consolidation_priority', 1.0),
                    stability=cognitive_features.get('stability', 1.0),
                    stress_link=cognitive_features.get('stress_link', 0.0),
                    protected=True
                )
                
                memories.append(memory)
            except Exception as e:
                logger.warning(f"Error parsing memory item: {e}")
                continue
        
        logger.info(f"Loaded {len(memories)} identity memories from {path}")
        return memories
    
    except Exception as e:
        logger.error(f"Failed to load identity JSON: {e}")
        return []
