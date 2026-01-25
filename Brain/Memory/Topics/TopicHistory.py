"""
TopicHistory.py
History of topic changes and focus over time.
"""

from typing import Dict, List, Optional, Tuple, Any
from collections import deque
import time
from collections import Counter


class TopicHistory:
    """Tracks history of topic changes."""
    
    def __init__(self, max_history: int = 500):
        """Initialize topic history.
        
        Parameters
        ----------
        max_history : int
            Maximum history entries
        """
        self.history: deque = deque(maxlen=max_history)
        self.max_history = max_history
        self.timestamps: deque = deque(maxlen=max_history)
    
    def record_topic_change(self, new_topic_id: str, previous_topic_id: Optional[str] = None) -> None:
        """Record a topic change.
        
        Parameters
        ----------
        new_topic_id : str
            New topic ID
        previous_topic_id : Optional[str]
            Previous topic ID
        """
        timestamp = time.time()
        
        self.history.append({
            "new_topic": new_topic_id,
            "previous_topic": previous_topic_id,
            "timestamp": timestamp
        })
        self.timestamps.append(timestamp)
    
    def get_topic_sequence(self, window_size: int = 10) -> List[str]:
        """Get recent sequence of topics.
        
        Parameters
        ----------
        window_size : int
            Number of recent topics
            
        Returns
        -------
        List[str]
            Topic sequence
        """
        recent = list(self.history)[-window_size:]
        return [entry["new_topic"] for entry in recent]
    
    def get_most_frequent_topics(self, count: int = 5) -> List[Tuple[str, int]]:
        """Get most frequently visited topics.
        
        Parameters
        ----------
        count : int
            Number of topics
            
        Returns
        -------
        List[Tuple[str, int]]
            Topic frequency pairs
        """
        topics = [entry["new_topic"] for entry in self.history]
        counter = Counter(topics)
        return counter.most_common(count)
    
    def get_topic_transitions(self) -> Dict[str, Dict[str, int]]:
        """Get transition frequency between topics.
        
        Returns
        -------
        Dict[str, Dict[str, int]]
            Transition counts
        """
        transitions = {}
        
        for entry in self.history:
            if entry["previous_topic"]:
                from_topic = entry["previous_topic"]
                to_topic = entry["new_topic"]
                
                if from_topic not in transitions:
                    transitions[from_topic] = {}
                
                if to_topic not in transitions[from_topic]:
                    transitions[from_topic][to_topic] = 0
                
                transitions[from_topic][to_topic] += 1
        
        return transitions
    
    def get_topic_dwell_time(self) -> Dict[str, float]:
        """Get average dwell time for each topic.
        
        Returns
        -------
        Dict[str, float]
            Dwell times in seconds
        """
        dwell_times = {}
        topic_starts = {}
        
        for entry in self.history:
            timestamp = entry["timestamp"]
            topic = entry["new_topic"]
            
            # Record start of this topic
            topic_starts[topic] = timestamp
            
            # If there was a previous topic, calculate its dwell time
            if entry["previous_topic"]:
                prev = entry["previous_topic"]
                if prev in topic_starts:
                    dwell = timestamp - topic_starts[prev]
                    
                    if prev not in dwell_times:
                        dwell_times[prev] = []
                    dwell_times[prev].append(dwell)
        
        # Average the dwell times
        avg_dwell = {}
        for topic, times in dwell_times.items():
            avg_dwell[topic] = sum(times) / len(times) if times else 0.0
        
        return avg_dwell
    
    def get_status(self) -> Dict[str, Any]:
        """Get history status.
        
        Returns
        -------
        Dict[str, Any]
            Status information
        """
        return {
            "max_history": self.max_history,
            "current_entries": len(self.history),
            "utilization_percent": (len(self.history) / self.max_history * 100) if self.max_history > 0 else 0,
            "unique_topics": len(set(entry["new_topic"] for entry in self.history)),
        }
        """Get sequence of recent topics."""
        pass
    
    def get_topic_dwell_time(self, topic_id):
        """Get average time spent on a topic."""
        pass
    
    def analyze_topic_patterns(self):
        """Analyze patterns in topic switching."""
        pass
