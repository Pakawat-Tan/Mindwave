"""
TopicState.py
Tracks current topic state and context.
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import time


@dataclass
class TopicStateData:
    """Represents topic state"""
    topic_id: str
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0


class TopicState:
    """Tracks the current topic state."""
    
    def __init__(self):
        """Initialize topic state."""
        self.current_topic: Optional[TopicStateData] = None
        self.previous_topics: list = []
        self.max_history = 10
    
    def set_topic(self, topic_id: str, confidence: float = 1.0, 
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Set the current topic.
        
        Parameters
        ----------
        topic_id : str
            Topic ID
        confidence : float
            Confidence in topic (0.0-1.0)
        metadata : Optional[Dict[str, Any]]
            Topic metadata
            
        Returns
        -------
        bool
            Success
        """
        confidence = max(0.0, min(1.0, confidence))
        
        # Save previous topic
        if self.current_topic:
            self.previous_topics.append(self.current_topic)
            if len(self.previous_topics) > self.max_history:
                self.previous_topics.pop(0)
        
        self.current_topic = TopicStateData(
            topic_id=topic_id,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        return True
    
    def get_current_topic(self) -> Optional[str]:
        """Get current topic ID.
        
        Returns
        -------
        Optional[str]
            Topic ID or None
        """
        return self.current_topic.topic_id if self.current_topic else None
    
    def get_topic_confidence(self) -> float:
        """Get confidence in current topic.
        
        Returns
        -------
        float
            Confidence score
        """
        return self.current_topic.confidence if self.current_topic else 0.0
    
    def get_topic_metadata(self) -> Dict[str, Any]:
        """Get current topic metadata.
        
        Returns
        -------
        Dict[str, Any]
            Metadata
        """
        return self.current_topic.metadata if self.current_topic else {}
    
    def get_previous_topics(self) -> list:
        """Get previous topics.
        
        Returns
        -------
        list
            Previous topic states
        """
        return self.previous_topics.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Get topic state status.
        
        Returns
        -------
        Dict[str, Any]
            Status information
        """
        return {
            "current_topic": self.get_current_topic(),
            "confidence": self.get_topic_confidence(),
            "history_length": len(self.previous_topics),
        }
        """Get current topic information."""
        pass
    
    def update_topic_confidence(self, confidence):
        """Update confidence in current topic."""
        pass
    
    def clear_topic(self):
        """Clear current topic."""
        pass
