"""
Observer Pattern for Pipeline Events.

This module implements the Observer Pattern to add comprehensive monitoring
and event handling to the pipeline execution process.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of pipeline events."""
    # Pipeline lifecycle
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_FAILED = "pipeline_failed"
    
    # Stage events
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    STAGE_FAILED = "stage_failed"
    STAGE_PROGRESS = "stage_progress"
    
    # Data events
    DATA_LOADED = "data_loaded"
    DATA_PROCESSED = "data_processed"
    DATA_SAVED = "data_saved"
    
    # Model events
    MODEL_TRAINING_STARTED = "model_training_started"
    MODEL_TRAINING_COMPLETED = "model_training_completed"
    MODEL_TRAINING_FAILED = "model_training_failed"
    MODEL_SAVED = "model_saved"
    
    # Error events
    WARNING = "warning"
    ERROR = "error"
    CRITICAL_ERROR = "critical_error"
    
    # Progress events
    PROGRESS_UPDATE = "progress_update"
    METRICS_UPDATE = "metrics_update"


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class PipelineEvent:
    """Event data structure for pipeline events."""
    
    event_type: EventType
    timestamp: datetime
    source: str  # Component that generated the event
    data: Dict[str, Any] = field(default_factory=dict)
    priority: EventPriority = EventPriority.NORMAL
    stage: Optional[str] = None
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'data': self.data,
            'priority': self.priority.value,
            'stage': self.stage,
            'correlation_id': self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineEvent':
        """Create event from dictionary."""
        return cls(
            event_type=EventType(data['event_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            source=data['source'],
            data=data.get('data', {}),
            priority=EventPriority(data.get('priority', EventPriority.NORMAL.value)),
            stage=data.get('stage'),
            correlation_id=data.get('correlation_id')
        )


class EventObserver(ABC):
    """Abstract base class for event observers."""
    
    @abstractmethod
    def handle_event(self, event: PipelineEvent) -> None:
        """Handle a pipeline event."""
        pass
    
    def get_interested_events(self) -> List[EventType]:
        """Return list of event types this observer is interested in."""
        return list(EventType)  # Default: interested in all events
    
    def get_name(self) -> str:
        """Get observer name."""
        return self.__class__.__name__


class EventBus:
    """Central event bus for pipeline events."""
    
    def __init__(self):
        self.observers: List[EventObserver] = []
        self.event_history: List[PipelineEvent] = []
        self.max_history_size = 1000
        self._enabled = True
    
    def subscribe(self, observer: EventObserver) -> None:
        """Subscribe an observer to events."""
        if observer not in self.observers:
            self.observers.append(observer)
            logger.debug(f"Subscribed observer: {observer.get_name()}")
    
    def unsubscribe(self, observer: EventObserver) -> None:
        """Unsubscribe an observer from events."""
        if observer in self.observers:
            self.observers.remove(observer)
            logger.debug(f"Unsubscribed observer: {observer.get_name()}")
    
    def publish(self, event: PipelineEvent) -> None:
        """Publish an event to all interested observers."""
        if not self._enabled:
            return
        
        # Add to history
        self.event_history.append(event)
        
        # Maintain history size limit
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size:]
        
        # Notify observers
        for observer in self.observers:
            try:
                if event.event_type in observer.get_interested_events():
                    observer.handle_event(event)
            except Exception as e:
                logger.error(f"Observer {observer.get_name()} failed to handle event: {str(e)}")
    
    def publish_event(self, 
                     event_type: EventType,
                     source: str,
                     data: Dict[str, Any] = None,
                     priority: EventPriority = EventPriority.NORMAL,
                     stage: str = None,
                     correlation_id: str = None) -> None:
        """Convenience method to publish an event."""
        event = PipelineEvent(
            event_type=event_type,
            timestamp=datetime.now(),
            source=source,
            data=data or {},
            priority=priority,
            stage=stage,
            correlation_id=correlation_id
        )
        self.publish(event)
    
    def get_events(self, 
                  event_type: EventType = None,
                  source: str = None,
                  since: datetime = None) -> List[PipelineEvent]:
        """Get events matching criteria."""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if source:
            events = [e for e in events if e.source == source]
        
        if since:
            events = [e for e in events if e.timestamp >= since]
        
        return events
    
    def clear_history(self) -> None:
        """Clear event history."""
        self.event_history.clear()
    
    def enable(self) -> None:
        """Enable event processing."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable event processing."""
        self._enabled = False


class LoggingObserver(EventObserver):
    """Observer that logs events to the standard logger."""
    
    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level
        self.logger = logging.getLogger(f"{__name__}.LoggingObserver")
    
    def handle_event(self, event: PipelineEvent) -> None:
        """Log the event."""
        message = f"[{event.source}] {event.event_type.value}"
        
        if event.stage:
            message += f" ({event.stage})"
        
        # Add relevant data to message
        if event.data:
            if 'message' in event.data:
                message += f": {event.data['message']}"
            elif 'assets_count' in event.data:
                message += f": {event.data['assets_count']} assets"
            elif 'execution_time' in event.data:
                message += f": completed in {event.data['execution_time']:.2f}s"
        
        # Log at appropriate level based on event type and priority
        if event.event_type in [EventType.ERROR, EventType.CRITICAL_ERROR, EventType.PIPELINE_FAILED, EventType.STAGE_FAILED]:
            self.logger.error(message)
        elif event.event_type == EventType.WARNING:
            self.logger.warning(message)
        elif event.priority in [EventPriority.HIGH, EventPriority.CRITICAL]:
            self.logger.warning(message)
        else:
            self.logger.info(message)


class ProgressObserver(EventObserver):
    """Observer that tracks pipeline progress."""
    
    def __init__(self):
        self.progress_data: Dict[str, Any] = {}
        self.start_times: Dict[str, datetime] = {}
    
    def handle_event(self, event: PipelineEvent) -> None:
        """Track progress from events."""
        if event.event_type == EventType.PIPELINE_STARTED:
            self.progress_data['status'] = 'running'
            self.progress_data['started_at'] = event.timestamp
            self.progress_data['stages'] = {}
            self.start_times['pipeline'] = event.timestamp
        
        elif event.event_type == EventType.STAGE_STARTED:
            stage = event.stage or 'unknown'
            self.progress_data['stages'][stage] = {
                'status': 'running',
                'started_at': event.timestamp
            }
            self.start_times[stage] = event.timestamp
        
        elif event.event_type == EventType.STAGE_COMPLETED:
            stage = event.stage or 'unknown'
            if stage in self.progress_data.get('stages', {}):
                duration = (event.timestamp - self.start_times.get(stage, event.timestamp)).total_seconds()
                self.progress_data['stages'][stage].update({
                    'status': 'completed',
                    'completed_at': event.timestamp,
                    'duration_seconds': duration
                })
        
        elif event.event_type == EventType.STAGE_FAILED:
            stage = event.stage or 'unknown'
            if stage in self.progress_data.get('stages', {}):
                self.progress_data['stages'][stage].update({
                    'status': 'failed',
                    'failed_at': event.timestamp,
                    'error': event.data.get('error', 'Unknown error')
                })
        
        elif event.event_type == EventType.PIPELINE_COMPLETED:
            duration = (event.timestamp - self.start_times.get('pipeline', event.timestamp)).total_seconds()
            self.progress_data.update({
                'status': 'completed',
                'completed_at': event.timestamp,
                'total_duration_seconds': duration
            })
        
        elif event.event_type == EventType.PIPELINE_FAILED:
            self.progress_data.update({
                'status': 'failed',
                'failed_at': event.timestamp,
                'error': event.data.get('error', 'Unknown error')
            })
        
        elif event.event_type == EventType.PROGRESS_UPDATE:
            # Update progress percentage if provided
            if 'progress_percent' in event.data:
                self.progress_data['progress_percent'] = event.data['progress_percent']
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress data."""
        return self.progress_data.copy()
    
    def get_interested_events(self) -> List[EventType]:
        """Return events this observer cares about."""
        return [
            EventType.PIPELINE_STARTED,
            EventType.PIPELINE_COMPLETED,
            EventType.PIPELINE_FAILED,
            EventType.STAGE_STARTED,
            EventType.STAGE_COMPLETED,
            EventType.STAGE_FAILED,
            EventType.PROGRESS_UPDATE
        ]


class MetricsObserver(EventObserver):
    """Observer that collects performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {
            'stage_durations': {},
            'error_counts': {},
            'success_counts': {},
            'asset_counts': {},
            'total_execution_time': 0
        }
    
    def handle_event(self, event: PipelineEvent) -> None:
        """Collect metrics from events."""
        if event.event_type == EventType.STAGE_COMPLETED and 'execution_time' in event.data:
            stage = event.stage or 'unknown'
            self.metrics['stage_durations'][stage] = event.data['execution_time']
            
            # Track success
            if stage not in self.metrics['success_counts']:
                self.metrics['success_counts'][stage] = 0
            self.metrics['success_counts'][stage] += 1
        
        elif event.event_type == EventType.STAGE_FAILED:
            stage = event.stage or 'unknown'
            if stage not in self.metrics['error_counts']:
                self.metrics['error_counts'][stage] = 0
            self.metrics['error_counts'][stage] += 1
        
        elif 'assets_count' in event.data:
            stage = event.stage or event.source
            self.metrics['asset_counts'][stage] = event.data['assets_count']
        
        elif event.event_type == EventType.PIPELINE_COMPLETED and 'total_execution_time' in event.data:
            self.metrics['total_execution_time'] = event.data['total_execution_time']
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return self.metrics.copy()
    
    def get_interested_events(self) -> List[EventType]:
        """Return events this observer cares about."""
        return [
            EventType.STAGE_COMPLETED,
            EventType.STAGE_FAILED,
            EventType.DATA_LOADED,
            EventType.DATA_PROCESSED,
            EventType.PIPELINE_COMPLETED
        ]


class FileLogObserver(EventObserver):
    """Observer that logs events to a file."""
    
    def __init__(self, log_file: Path):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    def handle_event(self, event: PipelineEvent) -> None:
        """Log event to file."""
        try:
            with open(self.log_file, 'a') as f:
                json.dump(event.to_dict(), f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to write event to file {self.log_file}: {str(e)}")
    
    def get_name(self) -> str:
        """Get observer name."""
        return f"FileLogObserver({self.log_file.name})"


class EventEmitter:
    """Mixin class for components that emit events."""
    
    def __init__(self, event_bus: EventBus, source_name: str):
        self.event_bus = event_bus
        self.source_name = source_name
        self.correlation_id = None
    
    def emit_event(self, 
                  event_type: EventType,
                  data: Dict[str, Any] = None,
                  priority: EventPriority = EventPriority.NORMAL,
                  stage: str = None) -> None:
        """Emit an event."""
        self.event_bus.publish_event(
            event_type=event_type,
            source=self.source_name,
            data=data or {},
            priority=priority,
            stage=stage,
            correlation_id=self.correlation_id
        )
    
    def set_correlation_id(self, correlation_id: str) -> None:
        """Set correlation ID for tracking related events."""
        self.correlation_id = correlation_id


def create_default_event_system(log_file: Path = None) -> EventBus:
    """Create an event bus with default observers."""
    event_bus = EventBus()
    
    # Add logging observer
    event_bus.subscribe(LoggingObserver())
    
    # Add progress observer
    event_bus.subscribe(ProgressObserver())
    
    # Add metrics observer
    event_bus.subscribe(MetricsObserver())
    
    # Add file log observer if specified
    if log_file:
        event_bus.subscribe(FileLogObserver(log_file))
    
    return event_bus


def get_observer_by_type(event_bus: EventBus, observer_type: type) -> Optional[EventObserver]:
    """Get an observer of a specific type from the event bus."""
    for observer in event_bus.observers:
        if isinstance(observer, observer_type):
            return observer
    return None