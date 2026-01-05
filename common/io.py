from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class GrillCheeseInput:
    id: str
    text: str
    modality: str
    embedding: np.ndarray
    context: List[str]
    timestamp: int
    metadata: Dict[str, Any]
    private: bool
    learnable: bool
    cost: float
    duration: float
    tokens: int
    
    def to_dict(self):
        return {
            "id": self.id,
            "text": self.text,
            "modality": self.modality,
            "embedding": self.embedding.tolist(),
            "context": self.context,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "private": self.private,
            "learnable": self.learnable,
            "cost": self.cost,
            "duration": self.duration,
            "tokens": self.tokens
        }


@dataclass
class GrillCheeseOutput:
    id: str
    input_id: str
    text: str
    response: str
    timestamp: int
    duration: float
    tokens: int
    cost: float
    code: int
    error: str
    def to_dict(self):
        return {
            "id": self.id,
            "input_id": self.input_id,
            "text": self.text,
            "response": self.response,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "tokens": self.tokens,
            "cost": self.cost,
            "code": self.code,
            "error": self.error
        }