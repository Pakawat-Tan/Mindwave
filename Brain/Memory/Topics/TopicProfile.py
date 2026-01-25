"""
TopicProfile.py
Profiles expertise and interest in topics.
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import time


@dataclass
class TopicExpertise:
    """Expertise data for a topic"""
    topic_name: str
    proficiency: float  # 0.0 to 1.0
    interest: float  # 0.0 to 1.0
    experience_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    expertise_data: Dict[str, Any] = field(default_factory=dict)


class TopicProfile:
    """Stores proficiency and interest profiles for topics."""
    
    def __init__(self):
        """Initialize topic profile."""
        self.proficiency: Dict[str, float] = {}
        self.interest: Dict[str, float] = {}
        self.expertise: Dict[str, TopicExpertise] = {}
    
    def set_proficiency(self, topic_id: str, level: float) -> bool:
        """Set proficiency level for a topic.
        
        Parameters
        ----------
        topic_id : str
            Topic ID
        level : float
            Proficiency level (0.0-1.0)
            
        Returns
        -------
        bool
            Success
        """
        level = max(0.0, min(1.0, level))
        self.proficiency[topic_id] = level
        
        if topic_id not in self.expertise:
            self.expertise[topic_id] = TopicExpertise(topic_id, level, self.interest.get(topic_id, 0.5))
        else:
            self.expertise[topic_id].proficiency = level
        
        return True
    
    def set_interest(self, topic_id: str, level: float) -> bool:
        """Set interest level for a topic.
        
        Parameters
        ----------
        topic_id : str
            Topic ID
        level : float
            Interest level (0.0-1.0)
            
        Returns
        -------
        bool
            Success
        """
        level = max(0.0, min(1.0, level))
        self.interest[topic_id] = level
        
        if topic_id not in self.expertise:
            self.expertise[topic_id] = TopicExpertise(topic_id, self.proficiency.get(topic_id, 0.5), level)
        else:
            self.expertise[topic_id].interest = level
        
        return True
    
    def get_proficiency(self, topic_id: str) -> Optional[float]:
        """Get proficiency level.
        
        Parameters
        ----------
        topic_id : str
            Topic ID
            
        Returns
        -------
        Optional[float]
            Proficiency or None
        """
        return self.proficiency.get(topic_id)
    
    def get_interest(self, topic_id: str) -> Optional[float]:
        """Get interest level.
        
        Parameters
        ----------
        topic_id : str
            Topic ID
            
        Returns
        -------
        Optional[float]
            Interest or None
        """
        return self.interest.get(topic_id)
    
    def get_best_topics(self, count: int = 5, metric: str = "proficiency") -> List[Tuple[str, float]]:
        """Get top topics by metric.
        
        Parameters
        ----------
        count : int
            Number of topics
        metric : str
            Metric to sort by (proficiency or interest)
            
        Returns
        -------
        List[Tuple[str, float]]
            Top topics with scores
        """
        if metric == "proficiency":
            items = sorted(self.proficiency.items(), key=lambda x: x[1], reverse=True)
        else:  # interest
            items = sorted(self.interest.items(), key=lambda x: x[1], reverse=True)
        
        return items[:count]
    
    def get_status(self) -> Dict[str, Any]:
        """Get profile status.
        
        Returns
        -------
        Dict[str, Any]
            Status information
        """
        return {
            "total_topics": len(self.expertise),
            "average_proficiency": sum(self.proficiency.values()) / len(self.proficiency) if self.proficiency else 0.0,
            "average_interest": sum(self.interest.values()) / len(self.interest) if self.interest else 0.0,
        }
        pass
    
    def get_interested_topics(self, count=5):
        """Get top topics by interest."""
        pass
