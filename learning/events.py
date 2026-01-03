"""
Event Bus for GrillCheese
Simple pub/sub system for decoupled event handling
"""
import time
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """Event with type, payload, and timestamp"""
    event_type: str
    data: Dict[str, Any]
    timestamp: float
    
    def __repr__(self) -> str:
        return f"Event({self.event_type}, ts={self.timestamp:.2f})"


class EventBus:
    """
    Simple synchronous event bus for GrillCheese
    
    Events:
        - memory_stored: New memory added
        - memory_retrieved: Memory accessed
        - spike_generated: SNN produced spikes
        - learning_update: STDP weight update occurred
        - content_processed: Content item fully processed
    """
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}
        self._event_count = 0
    
    def subscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """Register a handler for an event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        if handler not in self._subscribers[event_type]:
            self._subscribers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: Callable[[Event], None]) -> None:
        """Remove a handler"""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
            except ValueError:
                pass
    
    def emit(self, event_type: str, data: Dict[str, Any]) -> Event:
        """Create and publish an event"""
        event = Event(
            event_type=event_type,
            data=data,
            timestamp=time.time()
        )
        self._event_count += 1
        
        for handler in self._subscribers.get(event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.warning(f"Event handler error for {event_type}: {e}")
        
        return event
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            'total_events': self._event_count,
            'subscribers': {k: len(v) for k, v in self._subscribers.items()}
        }

