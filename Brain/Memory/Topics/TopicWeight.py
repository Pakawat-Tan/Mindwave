"""
TopicWeight.py
Manages importance weights for topics.
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass, field
import time


@dataclass
class TopicWeightData:
    """Topic weight data"""
    topic_id: str
    weight: float = 0.5
    timestamp: float = field(default_factory=time.time)
    priority: int = 1  # 1-10
    access_count: int = 0


class TopicWeight:
    """Manages importance weights for topics."""
    
    def __init__(self):
        """Initialize topic weights."""
        self.weights: Dict[str, TopicWeightData] = {}
        self.priorities: Dict[str, int] = {}
    
    def set_weight(self, topic_id: str, weight: float, priority: int = 1) -> bool:
        """Set importance weight for a topic.
        
        Parameters
        ----------
        topic_id : str
            Topic ID
        weight : float
            Weight (0.0-1.0)
        priority : int
            Priority level (1-10)
            
        Returns
        -------
        bool
            Success
        """
        weight = max(0.0, min(1.0, weight))
        priority = max(1, min(10, priority))
        
        if topic_id in self.weights:
            self.weights[topic_id].weight = weight
            self.weights[topic_id].priority = priority
        else:
            self.weights[topic_id] = TopicWeightData(topic_id, weight, priority=priority)
        
        self.priorities[topic_id] = priority
        
        return True
    
    def get_weight(self, topic_id: str) -> Optional[float]:
        """Get weight for a topic.
        
        Parameters
        ----------
        topic_id : str
            Topic ID
            
        Returns
        -------
        Optional[float]
            Weight or None
        """
        if topic_id in self.weights:
            self.weights[topic_id].access_count += 1
            return self.weights[topic_id].weight
        return None
    
    def get_priority(self, topic_id: str) -> Optional[int]:
        """Get priority for a topic.
        
        Parameters
        ----------
        topic_id : str
            Topic ID
            
        Returns
        -------
        Optional[int]
            Priority or None
        """
        return self.priorities.get(topic_id)
    
    def get_weighted_topics(self, count: int = 5) -> list:
        """Get topics sorted by weight.
        
        Parameters
        ----------
        count : int
            Number of topics
            
        Returns
        -------
        list
            Sorted topics
        """
        sorted_topics = sorted(self.weights.items(), key=lambda x: x[1].weight, reverse=True)
        return [topic_id for topic_id, _ in sorted_topics[:count]]
    
    def get_status(self) -> Dict[str, Any]:
        """Get status.
        
        Returns
        -------
        Dict[str, Any]
            Status information
        """
        return {
            "total_weighted_topics": len(self.weights),
            "average_weight": sum(w.weight for w in self.weights.values()) / len(self.weights) if self.weights else 0.0,
        }
        pass
    
    def get_ranked_topics(self):
        """Get topics ranked by weight."""
        pass
    
    def normalize_weights(self):
        """Normalize all weights to sum to 1."""
        pass
    
    def adjust_weight(self, topic_id, adjustment):
        """Adjust weight of a topic."""
        pass
