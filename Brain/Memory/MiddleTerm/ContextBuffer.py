"""
ContextBuffer.py
Maintains context across multiple cycles
"""

from typing import Dict, List, Any, Optional, Deque
from collections import deque
from dataclasses import dataclass, field
import time


@dataclass
class ContextFrame:
    """A frame of context information"""
    cycle_number: int
    timestamp: float
    state: Dict[str, Any] = field(default_factory=dict)
    events: List[str] = field(default_factory=list)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    importance: float = 0.5


class ContextBuffer:
    """Maintains context across multiple processing cycles."""
    
    def __init__(self, max_frames: int = 100, retention_time_seconds: int = 300):
        """Initialize context buffer.
        
        Parameters
        ----------
        max_frames : int
            Maximum frames to keep in buffer
        retention_time_seconds : int
            How long to retain frames
        """
        self.max_frames = max_frames
        self.retention_time = retention_time_seconds
        self.frames: Deque[ContextFrame] = deque(maxlen=max_frames)
        self.current_frame: Optional[ContextFrame] = None
        self.cycle_counter = 0
    
    def start_new_frame(self) -> ContextFrame:
        """Start a new context frame.
        
        Returns
        -------
        ContextFrame
            New context frame
        """
        self.cycle_counter += 1
        self.current_frame = ContextFrame(
            cycle_number=self.cycle_counter,
            timestamp=time.time()
        )
        self.frames.append(self.current_frame)
        return self.current_frame
    
    def add_event(self, event: str) -> bool:
        """Add an event to current frame.
        
        Parameters
        ----------
        event : str
            Event description
            
        Returns
        -------
        bool
            True if added
        """
        if self.current_frame:
            self.current_frame.events.append(event)
            return True
        return False
    
    def add_decision(self, decision: Dict[str, Any]) -> bool:
        """Add a decision to current frame.
        
        Parameters
        ----------
        decision : Dict[str, Any]
            Decision information
            
        Returns
        -------
        bool
            True if added
        """
        if self.current_frame:
            self.current_frame.decisions.append(decision)
            return True
        return False
    
    def get_recent_context(self, num_frames: int = 10) -> List[ContextFrame]:
        """Get recent context frames.
        
        Parameters
        ----------
        num_frames : int
            Number of frames to retrieve
            
        Returns
        -------
        List[ContextFrame]
            Recent frames
        """
        return list(self.frames)[-num_frames:]
    
    def get_context_trend(self) -> Dict[str, Any]:
        """Get trends in context over recent frames.
        
        Returns
        -------
        Dict[str, Any]
            Context trends
        """
        recent = self.get_recent_context(10)
        
        if not recent:
            return {}
        
        return {
            "total_events": sum(len(f.events) for f in recent),
            "total_decisions": sum(len(f.decisions) for f in recent),
            "average_importance": sum(f.importance for f in recent) / len(recent),
            "timespan_seconds": recent[-1].timestamp - recent[0].timestamp if len(recent) > 1 else 0
        }
    
    def prune_old_frames(self) -> int:
        """Remove frames older than retention time.
        
        Returns
        -------
        int
            Number of frames removed
        """
        current_time = time.time()
        removed = 0
        
        while self.frames and (current_time - self.frames[0].timestamp) > self.retention_time:
            self.frames.popleft()
            removed += 1
        
        return removed
    
    def get_status(self) -> Dict[str, Any]:
        """Get buffer status.
        
        Returns
        -------
        Dict[str, Any]
            Status information
        """
        return {
            "max_frames": self.max_frames,
            "current_frames": len(self.frames),
            "utilization_percent": (len(self.frames) / self.max_frames * 100) if self.max_frames > 0 else 0,
            "cycle_counter": self.cycle_counter,
            "retention_time_seconds": self.retention_time
        }
