"""Event bus implementation for asset events."""

from typing import Dict, List, Callable, Any
import asyncio
import logging
from datetime import datetime


logger = logging.getLogger(__name__)


class EventBus:
    """Central event bus for asset events."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_history: List[Dict[str, Any]] = []
    
    async def publish(self, event_type: str, data: Dict[str, Any]) -> None:
        """Publish an event to all subscribers."""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': datetime.utcnow(),
            'id': f"{event_type}_{datetime.utcnow().timestamp()}"
        }
        
        # Store event in history
        self._event_history.append(event)
        
        # Notify subscribers
        if event_type in self._subscribers:
            logger.info(f"Publishing event {event_type} to {len(self._subscribers[event_type])} subscribers")
            
            # Run all handlers concurrently
            tasks = []
            for handler in self._subscribers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        tasks.append(handler(event))
                    else:
                        # Wrap sync function in async
                        tasks.append(asyncio.create_task(
                            self._run_sync_handler(handler, event)
                        ))
                except Exception as e:
                    logger.error(f"Error preparing handler for {event_type}: {e}")
            
            # Wait for all handlers to complete
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_sync_handler(self, handler: Callable, event: Dict[str, Any]) -> None:
        """Run synchronous handler in async context."""
        try:
            handler(event)
        except Exception as e:
            logger.error(f"Error in sync handler: {e}")
    
    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to events of a specific type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(handler)
        logger.info(f"Subscribed handler to {event_type}")
    
    def unsubscribe(self, event_type: str, handler: Callable) -> None:
        """Unsubscribe from events of a specific type."""
        if event_type in self._subscribers:
            try:
                self._subscribers[event_type].remove(handler)
                logger.info(f"Unsubscribed handler from {event_type}")
            except ValueError:
                logger.warning(f"Handler not found for {event_type}")
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get event history, optionally filtered by type."""
        if event_type:
            filtered_events = [e for e in self._event_history if e['type'] == event_type]
            return filtered_events[-limit:]
        return self._event_history[-limit:]
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history.clear()
        logger.info("Event history cleared")