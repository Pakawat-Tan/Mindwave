"""
WorkingMemory.py
Short-term working memory for immediate processing
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time


class MemoryPriority(Enum):
    """Priority levels for working memory items"""
    CRITICAL = 4
    HIGH = 3
    NORMAL = 2
    LOW = 1


@dataclass
class WorkingMemoryItem:
    """Item stored in working memory"""
    id: str
    content: Any
    timestamp: float
    priority: MemoryPriority = MemoryPriority.NORMAL
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    relevance_score: float = 0.5
    confidence: float = 1.0


class WorkingMemory:
    """Manages short-term working memory for immediate processing."""
    
    def __init__(self, max_capacity: int = 50, decay_rate: float = 0.01):
        """Initialize working memory.
        
        Parameters
        ----------
        max_capacity : int
            Maximum items in working memory
        decay_rate : float
            Decay rate per minute for items
        """
        self.max_capacity = max_capacity
        self.decay_rate = decay_rate
        self.items: Dict[str, WorkingMemoryItem] = {}
        self.access_history: List[Tuple[str, float]] = []
        self.consolidation_threshold = 0.6
    
    def store(self, item_id: str, content: Any, priority: MemoryPriority = MemoryPriority.NORMAL,
              relevance: float = 0.5) -> bool:
        """Store an item in working memory.
        
        Parameters
        ----------
        item_id : str
            Unique identifier for the item
        content : Any
            Content to store
        priority : MemoryPriority
            Priority level
        relevance : float
            Relevance score (0.0-1.0)
            
        Returns
        -------
        bool
            True if item was stored
        """
        # Check capacity
        if len(self.items) >= self.max_capacity:
            # Remove lowest priority/oldest item
            self._evict_item()
        
        self.items[item_id] = WorkingMemoryItem(
            id=item_id,
            content=content,
            timestamp=time.time(),
            priority=priority,
            relevance_score=relevance
        )
        
        return True
    
    def retrieve(self, item_id: str) -> Optional[Any]:
        """Retrieve an item from working memory.
        
        Parameters
        ----------
        item_id : str
            ID of the item to retrieve
            
        Returns
        -------
        Optional[Any]
            The item content, or None if not found
        """
        if item_id not in self.items:
            return None
        
        item = self.items[item_id]
        item.access_count += 1
        item.last_accessed = time.time()
        
        # Record access
        self.access_history.append((item_id, item.last_accessed))
        
        return item.content
    
    def update_relevance(self, item_id: str, relevance: float) -> bool:
        """Update relevance score of an item.
        
        Parameters
        ----------
        item_id : str
            ID of the item
        relevance : float
            New relevance score (0.0-1.0)
            
        Returns
        -------
        bool
            True if updated
        """
        if item_id in self.items:
            self.items[item_id].relevance_score = max(0.0, min(1.0, relevance))
            return True
        return False
    
    def apply_decay(self) -> Dict[str, float]:
        """Apply decay to all items based on age.
        
        Returns
        -------
        Dict[str, float]
            Updated relevance scores after decay
        """
        current_time = time.time()
        decay_result = {}
        
        for item_id, item in self.items.items():
            # Calculate age in minutes
            age_minutes = (current_time - item.timestamp) / 60.0
            
            # Apply exponential decay
            decay_factor = (1.0 - self.decay_rate) ** age_minutes
            item.relevance_score *= decay_factor
            
            # Mark for removal if below threshold
            if item.relevance_score < 0.1:
                decay_result[item_id] = item.relevance_score
        
        return decay_result
    
    def _evict_item(self) -> Optional[str]:
        """Evict lowest priority item from memory.
        
        Returns
        -------
        Optional[str]
            ID of evicted item
        """
        if not self.items:
            return None
        
        # Sort by priority (ascending) then by relevance (ascending)
        evict_item = min(
            self.items.items(),
            key=lambda x: (x[1].priority.value, x[1].relevance_score)
        )
        
        evicted_id = evict_item[0]
        del self.items[evicted_id]
        
        return evicted_id
    
    def get_consolidation_candidates(self) -> List[WorkingMemoryItem]:
        """Get items ready for consolidation to medium-term memory.
        
        Returns
        -------
        List[WorkingMemoryItem]
            Items above consolidation threshold
        """
        candidates = [
            item for item in self.items.values()
            if item.relevance_score >= self.consolidation_threshold
        ]
        
        # Sort by relevance (descending)
        return sorted(candidates, key=lambda x: x.relevance_score, reverse=True)
    
    def get_status(self) -> Dict[str, Any]:
        """Get working memory status.
        
        Returns
        -------
        Dict[str, Any]
            Current status
        """
        return {
            "capacity": self.max_capacity,
            "current_size": len(self.items),
            "utilization_percent": (len(self.items) / self.max_capacity * 100) if self.max_capacity > 0 else 0,
            "total_accesses": len(self.access_history),
            "consolidation_candidates": len(self.get_consolidation_candidates()),
            "decay_rate": self.decay_rate,
            "average_relevance": sum(item.relevance_score for item in self.items.values()) / len(self.items) if self.items else 0.0
        }
    
    def clear(self):
        """Clear all working memory."""
        self.items.clear()
        self.access_history.clear()
