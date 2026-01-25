"""
TopicContext.py
Maintains context specific to different topics
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
import time


@dataclass
class TopicContextData:
    """Context data for a specific topic"""
    topic: str
    created_time: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    relevance: float = 0.5
    related_topics: Set[str] = field(default_factory=set)
    context_items: List[Dict[str, Any]] = field(default_factory=list)
    active: bool = True


class TopicContext:
    """Maintains topic-specific context across cycles."""
    
    def __init__(self, max_topics: int = 50):
        """Initialize topic context.
        
        Parameters
        ----------
        max_topics : int
            Maximum concurrent topics
        """
        self.max_topics = max_topics
        self.topics: Dict[str, TopicContextData] = {}
        self.active_topics: Set[str] = set()
        self.topic_history: List[str] = []
    
    def create_topic(self, topic: str) -> bool:
        """Create a new topic context.
        
        Parameters
        ----------
        topic : str
            Topic name
            
        Returns
        -------
        bool
            True if created
        """
        if len(self.topics) >= self.max_topics:
            self._evict_topic()
        
        self.topics[topic] = TopicContextData(topic=topic)
        self.active_topics.add(topic)
        self.topic_history.append(topic)
        
        return True
    
    def add_context_item(self, topic: str, item: Dict[str, Any]) -> bool:
        """Add a context item to a topic.
        
        Parameters
        ----------
        topic : str
            Topic name
        item : Dict[str, Any]
            Context item
            
        Returns
        -------
        bool
            True if added
        """
        if topic not in self.topics:
            self.create_topic(topic)
        
        self.topics[topic].context_items.append(item)
        self.topics[topic].last_updated = time.time()
        
        return True
    
    def relate_topics(self, topic1: str, topic2: str) -> bool:
        """Create a relationship between topics.
        
        Parameters
        ----------
        topic1 : str
            First topic
        topic2 : str
            Second topic
            
        Returns
        -------
        bool
            True if related
        """
        if topic1 not in self.topics:
            self.create_topic(topic1)
        if topic2 not in self.topics:
            self.create_topic(topic2)
        
        self.topics[topic1].related_topics.add(topic2)
        self.topics[topic2].related_topics.add(topic1)
        
        return True
    
    def update_relevance(self, topic: str, relevance: float) -> bool:
        """Update topic relevance.
        
        Parameters
        ----------
        topic : str
            Topic name
        relevance : float
            Relevance score (0.0-1.0)
            
        Returns
        -------
        bool
            True if updated
        """
        if topic in self.topics:
            self.topics[topic].relevance = max(0.0, min(1.0, relevance))
            return True
        return False
    
    def get_context(self, topic: str) -> Optional[TopicContextData]:
        """Get context for a topic.
        
        Parameters
        ----------
        topic : str
            Topic name
            
        Returns
        -------
        Optional[TopicContextData]
            Topic context or None
        """
        return self.topics.get(topic)
    
    def get_related_topics(self, topic: str) -> Set[str]:
        """Get topics related to a given topic.
        
        Parameters
        ----------
        topic : str
            Topic name
            
        Returns
        -------
        Set[str]
            Related topics
        """
        if topic in self.topics:
            return self.topics[topic].related_topics.copy()
        return set()
    
    def _evict_topic(self) -> Optional[str]:
        """Evict lowest relevance topic.
        
        Returns
        -------
        Optional[str]
            Evicted topic
        """
        if not self.topics:
            return None
        
        evict_topic = min(
            self.topics.items(),
            key=lambda x: (x[1].active, x[1].relevance)
        )
        
        evicted = evict_topic[0]
        del self.topics[evicted]
        self.active_topics.discard(evicted)
        
        return evicted
    
    def get_status(self) -> Dict[str, Any]:
        """Get topic context status.
        
        Returns
        -------
        Dict[str, Any]
            Status information
        """
        return {
            "max_topics": self.max_topics,
            "active_topics": len(self.active_topics),
            "total_topics": len(self.topics),
            "utilization_percent": (len(self.topics) / self.max_topics * 100) if self.max_topics > 0 else 0,
            "average_relevance": sum(t.relevance for t in self.topics.values()) / len(self.topics) if self.topics else 0.0
        }
        pass
    
    def switch_topic(self, new_topic_id):
        """Switch focus to a different topic."""
        pass
