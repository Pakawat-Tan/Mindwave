"""
ExperienceStore.py
Stores episodic experiences and their outcomes
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time
import json


@dataclass
class Experience:
    """Represents a stored experience"""
    experience_id: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    outcome: Optional[Dict[str, Any]] = None
    reward: float = 0.0
    success: bool = False
    importance: float = 0.5
    retrievals: int = 0
    last_accessed: float = field(default_factory=time.time)


class ExperienceStore:
    """Stores episodic experiences and outcomes for learning."""
    
    def __init__(self, max_experiences: int = 1000):
        """Initialize experience store.
        
        Parameters
        ----------
        max_experiences : int
            Maximum number of experiences to store
        """
        self.max_experiences = max_experiences
        self.experiences: Dict[str, Experience] = {}
        self.experience_index: List[str] = []
        self.outcome_patterns: Dict[str, int] = defaultdict(int)
        self.success_rate = 0.0
    
    def store_experience(self, experience_id: str, context: Dict[str, Any], 
                        actions: List[Dict[str, Any]], importance: float = 0.5) -> bool:
        """Store a new experience.
        
        Parameters
        ----------
        experience_id : str
            Unique experience ID
        context : Dict[str, Any]
            Context of the experience
        actions : List[Dict[str, Any]]
            List of actions taken
        importance : float
            Importance score (0.0-1.0)
            
        Returns
        -------
        bool
            Success
        """
        if len(self.experiences) >= self.max_experiences:
            self._evict_experience()
        
        exp = Experience(
            experience_id=experience_id,
            context=context,
            actions=actions,
            importance=importance
        )
        
        self.experiences[experience_id] = exp
        self.experience_index.append(experience_id)
        
        return True
    
    def record_outcome(self, experience_id: str, outcome: Dict[str, Any], 
                      reward: float = 0.0, success: bool = False) -> bool:
        """Record the outcome of an experience.
        
        Parameters
        ----------
        experience_id : str
            Experience ID
        outcome : Dict[str, Any]
            Outcome description
        reward : float
            Reward signal
        success : bool
            Whether experience was successful
            
        Returns
        -------
        bool
            Success
        """
        if experience_id not in self.experiences:
            return False
        
        exp = self.experiences[experience_id]
        exp.outcome = outcome
        exp.reward = reward
        exp.success = success
        
        # Track outcome patterns
        outcome_key = json.dumps(outcome, sort_keys=True, default=str)
        self.outcome_patterns[outcome_key] += 1
        
        # Update success rate
        self._update_success_rate()
        
        return True
    
    def retrieve_similar_experiences(self, context: Dict[str, Any], 
                                   top_k: int = 5) -> List[Experience]:
        """Retrieve experiences similar to given context.
        
        Parameters
        ----------
        context : Dict[str, Any]
            Query context
        top_k : int
            Number of similar experiences to retrieve
            
        Returns
        -------
        List[Experience]
            Similar experiences
        """
        similar = []
        
        for exp_id, exp in self.experiences.items():
            similarity = self._calculate_similarity(context, exp.context)
            similar.append((similarity, exp))
        
        # Sort by similarity and return top-k
        similar.sort(key=lambda x: x[0], reverse=True)
        
        results = [exp for _, exp in similar[:top_k]]
        
        # Update access tracking
        for exp in results:
            exp.retrievals += 1
            exp.last_accessed = time.time()
        
        return results
    
    def _calculate_similarity(self, context1: Dict[str, Any], 
                            context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts.
        
        Parameters
        ----------
        context1 : Dict[str, Any]
            First context
        context2 : Dict[str, Any]
            Second context
            
        Returns
        -------
        float
            Similarity score (0.0-1.0)
        """
        if not context1 or not context2:
            return 0.0
        
        keys1 = set(context1.keys())
        keys2 = set(context2.keys())
        
        # Jaccard similarity for keys
        intersection = len(keys1 & keys2)
        union = len(keys1 | keys2)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def get_successful_experiences(self) -> List[Experience]:
        """Get all successful experiences.
        
        Returns
        -------
        List[Experience]
            Successful experiences
        """
        return [exp for exp in self.experiences.values() if exp.success]
    
    def get_failed_experiences(self) -> List[Experience]:
        """Get all failed experiences.
        
        Returns
        -------
        List[Experience]
            Failed experiences
        """
        return [exp for exp in self.experiences.values() if not exp.success]
    
    def learn_from_experiences(self) -> Dict[str, Any]:
        """Extract learning patterns from stored experiences.
        
        Returns
        -------
        Dict[str, Any]
            Learning insights
        """
        successful = self.get_successful_experiences()
        failed = self.get_failed_experiences()
        
        learning = {
            "total_experiences": len(self.experiences),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": self.success_rate,
            "top_outcomes": self._get_top_outcomes(5),
            "average_reward": self._calculate_average_reward(),
            "most_important": self._get_most_important_experiences(5),
        }
        
        return learning
    
    def _get_top_outcomes(self, top_k: int = 5) -> List[Tuple[str, int]]:
        """Get most common outcomes.
        
        Parameters
        ----------
        top_k : int
            Number of top outcomes
            
        Returns
        -------
        List[Tuple[str, int]]
            Top outcomes with counts
        """
        sorted_outcomes = sorted(self.outcome_patterns.items(), 
                                key=lambda x: x[1], reverse=True)
        return sorted_outcomes[:top_k]
    
    def _calculate_average_reward(self) -> float:
        """Calculate average reward.
        
        Returns
        -------
        float
            Average reward
        """
        if not self.experiences:
            return 0.0
        
        total_reward = sum(exp.reward for exp in self.experiences.values())
        return total_reward / len(self.experiences)
    
    def _get_most_important_experiences(self, top_k: int = 5) -> List[str]:
        """Get most important experiences.
        
        Parameters
        ----------
        top_k : int
            Number of experiences
            
        Returns
        -------
        List[str]
            Most important experience IDs
        """
        sorted_exp = sorted(self.experiences.items(), 
                          key=lambda x: x[1].importance, reverse=True)
        return [exp_id for exp_id, _ in sorted_exp[:top_k]]
    
    def _update_success_rate(self) -> None:
        """Update success rate metric."""
        if not self.experiences:
            self.success_rate = 0.0
            return
        
        successful = sum(1 for exp in self.experiences.values() if exp.success)
        self.success_rate = successful / len(self.experiences)
    
    def _evict_experience(self) -> Optional[str]:
        """Evict lowest priority experience.
        
        Returns
        -------
        Optional[str]
            Evicted experience ID
        """
        if not self.experiences:
            return None
        
        # Evict based on importance and recency
        evict_exp = min(
            self.experiences.items(),
            key=lambda x: (x[1].importance, x[1].retrievals, x[1].last_accessed)
        )
        
        evicted_id = evict_exp[0]
        del self.experiences[evicted_id]
        if evicted_id in self.experience_index:
            self.experience_index.remove(evicted_id)
        
        return evicted_id
    
    def get_status(self) -> Dict[str, Any]:
        """Get experience store status.
        
        Returns
        -------
        Dict[str, Any]
            Status information
        """
        return {
            "max_experiences": self.max_experiences,
            "current_count": len(self.experiences),
            "utilization_percent": (len(self.experiences) / self.max_experiences * 100) if self.max_experiences > 0 else 0,
            "success_rate": self.success_rate,
            "total_retrievals": sum(exp.retrievals for exp in self.experiences.values()),
            "average_importance": sum(exp.importance for exp in self.experiences.values()) / len(self.experiences) if self.experiences else 0.0,
        }
