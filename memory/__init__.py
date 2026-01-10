"""
Capsule Memory System
Bio-inspired 32-dimensional capsule memory with hippocampal architecture
"""

from memory.capsule_memory import CapsuleMemory, MemoryType
from memory.capsule_encoder import CapsuleEncoder
from memory.dentate_gyrus import DentateGyrus
from memory.ca3_memory import CA3Memory
from memory.capsule_store import CapsuleMemoryStore

__all__ = [
    'CapsuleMemory',
    'MemoryType',
    'CapsuleEncoder',
    'DentateGyrus',
    'CA3Memory',
    'CapsuleMemoryStore'
]
