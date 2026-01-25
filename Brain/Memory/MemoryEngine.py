"""
MemoryEngine.py
Core memory management engine coordinating all memory tiers
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time
import json

from .ShortTerm.WorkingMemory import WorkingMemory, MemoryPriority
from .ShortTerm.AttentionMap import AttentionMap
from .MiddleTerm.ContextBuffer import ContextBuffer
from .MiddleTerm.TopicContext import TopicContext


@dataclass
class ConsolidationConfig:
    """Configuration for memory consolidation"""
    # Consolidation thresholds
    working_to_middle_threshold: float = 0.6
    middle_to_long_threshold: float = 0.7
    
    # Consolidation interval
    consolidation_interval_cycles: int = 10
    
    # Priority escalation
    priority_escalation_factor: float = 1.2


class MemoryEngine:
    """Coordinates hierarchical memory system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize memory engine.
        
        Parameters
        ----------
        config : Optional[Dict[str, Any]]
            Configuration dictionary
        """
        self.config = ConsolidationConfig()
        if config:
            self._apply_config(config)
        
        # Memory tiers
        self.working_memory = WorkingMemory(max_items=50)
        self.attention_map = AttentionMap()
        self.context_buffer = ContextBuffer(max_frames=100, retention_time_seconds=300)
        self.topic_context = TopicContext(max_topics=50)
        
        # Consolidation tracking
        self.consolidation_cycle = 0
        self.consolidation_history: List[Dict[str, Any]] = []
        
        # Memory statistics
        self.total_stored = 0
        self.total_consolidated = 0
        self.total_retrieved = 0
    
    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply configuration."""
        if "working_to_middle_threshold" in config:
            self.config.working_to_middle_threshold = config["working_to_middle_threshold"]
        if "middle_to_long_threshold" in config:
            self.config.middle_to_long_threshold = config["middle_to_long_threshold"]
        if "consolidation_interval_cycles" in config:
            self.config.consolidation_interval_cycles = config["consolidation_interval_cycles"]
        if "priority_escalation_factor" in config:
            self.config.priority_escalation_factor = config["priority_escalation_factor"]
    
    def store_memory(self, content: Any, priority: str = "NORMAL", context: Optional[Dict] = None) -> str:
        """Store memory in working memory.
        
        Parameters
        ----------
        content : Any
            Memory content
        priority : str
            Priority level (CRITICAL, HIGH, NORMAL, LOW)
        context : Optional[Dict]
            Additional context
            
        Returns
        -------
        str
            Memory ID
        """
        priority_enum = MemoryPriority[priority]
        memory_id = self.working_memory.store(content, priority_enum)
        self.total_stored += 1
        
        # Add context if provided
        if context:
            if "events" in context:
                self.context_buffer.add_event(context["events"])
        
        return memory_id
    
    def retrieve_memory(self, memory_id: str) -> Optional[Any]:
        """Retrieve memory from working memory.
        
        Parameters
        ----------
        memory_id : str
            Memory ID
            
        Returns
        -------
        Optional[Any]
            Memory content or None
        """
        self.total_retrieved += 1
        return self.working_memory.retrieve(memory_id)
    
    def add_context_event(self, event: Dict[str, Any]) -> None:
        """Add event to context buffer.
        
        Parameters
        ----------
        event : Dict[str, Any]
            Event description
        """
        self.context_buffer.add_event(event)
    
    def add_context_decision(self, decision: Dict[str, Any]) -> None:
        """Add decision to context buffer.
        
        Parameters
        ----------
        decision : Dict[str, Any]
            Decision description
        """
        self.context_buffer.add_decision(decision)
    
    def set_attention(self, attention_type: str, target: str, intensity: float = 0.8) -> None:
        """Set attention focus.
        
        Parameters
        ----------
        attention_type : str
            Type of attention (VISUAL, AUDITORY, etc.)
        target : str
            Attention target
        intensity : float
            Attention intensity (0.0-1.0)
        """
        self.attention_map.set_focus(attention_type, target, intensity)
    
    def add_topic(self, topic: str) -> None:
        """Add a topic to context.
        
        Parameters
        ----------
        topic : str
            Topic name
        """
        self.topic_context.create_topic(topic)
    
    def relate_topics(self, topic1: str, topic2: str) -> None:
        """Create relationship between topics.
        
        Parameters
        ----------
        topic1 : str
            First topic
        topic2 : str
            Second topic
        """
        self.topic_context.relate_topics(topic1, topic2)
    
    def update_topic_relevance(self, topic: str, relevance: float) -> None:
        """Update topic relevance.
        
        Parameters
        ----------
        topic : str
            Topic name
        relevance : float
            Relevance score (0.0-1.0)
        """
        self.topic_context.update_relevance(topic, relevance)
    
    def consolidate_memory(self) -> Dict[str, Any]:
        """Perform memory consolidation between tiers.
        
        Returns
        -------
        Dict[str, Any]
            Consolidation results
        """
        self.consolidation_cycle += 1
        
        consolidation_result = {
            "cycle": self.consolidation_cycle,
            "timestamp": time.time(),
            "promoted_items": 0,
            "pruned_items": 0,
        }
        
        # Get consolidation candidates from working memory
        candidates = self.working_memory.get_consolidation_candidates()
        
        # Promote high-relevance items to context buffer
        for candidate_id in candidates:
            memory_item = self.working_memory.retrieve(candidate_id)
            if memory_item and memory_item.relevance_score >= self.config.working_to_middle_threshold:
                # Add to context buffer
                self.context_buffer.add_decision({
                    "type": "memory_promotion",
                    "source_memory_id": candidate_id,
                    "relevance": memory_item.relevance_score,
                    "priority": memory_item.priority.name
                })
                consolidation_result["promoted_items"] += 1
        
        # Prune old context
        pruned = self.context_buffer.prune_old_frames()
        consolidation_result["pruned_items"] = pruned
        
        # Record consolidation
        self.consolidation_history.append(consolidation_result)
        self.total_consolidated += len(candidates)
        
        return consolidation_result
    
    def get_working_memory_status(self) -> Dict[str, Any]:
        """Get working memory status.
        
        Returns
        -------
        Dict[str, Any]
            Working memory status
        """
        return self.working_memory.get_status()
    
    def get_attention_status(self) -> Dict[str, Any]:
        """Get attention status.
        
        Returns
        -------
        Dict[str, Any]
            Attention status
        """
        return self.attention_map.get_status()
    
    def get_context_status(self) -> Dict[str, Any]:
        """Get context buffer status.
        
        Returns
        -------
        Dict[str, Any]
            Context buffer status
        """
        return self.context_buffer.get_status()
    
    def get_topic_status(self) -> Dict[str, Any]:
        """Get topic context status.
        
        Returns
        -------
        Dict[str, Any]
            Topic context status
        """
        return self.topic_context.get_status()
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get overall engine status.
        
        Returns
        -------
        Dict[str, Any]
            Engine status
        """
        return {
            "consolidation_cycle": self.consolidation_cycle,
            "total_stored": self.total_stored,
            "total_consolidated": self.total_consolidated,
            "total_retrieved": self.total_retrieved,
            "consolidation_history_length": len(self.consolidation_history),
            "working_memory": self.get_working_memory_status(),
            "attention": self.get_attention_status(),
            "context": self.get_context_status(),
            "topics": self.get_topic_status(),
        }
    
    def save_state(self, filepath: str) -> bool:
        """Save memory engine state.
        
        Parameters
        ----------
        filepath : str
            Path to save state
            
        Returns
        -------
        bool
            Success
        """
        try:
            state = {
                "consolidation_cycle": self.consolidation_cycle,
                "total_stored": self.total_stored,
                "total_consolidated": self.total_consolidated,
                "total_retrieved": self.total_retrieved,
                "consolidation_history": self.consolidation_history,
                "timestamp": time.time()
            }
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving state: {e}")
            return False
    
    def load_state(self, filepath: str) -> bool:
        """Load memory engine state.
        
        Parameters
        ----------
        filepath : str
            Path to load state from
            
        Returns
        -------
        bool
            Success
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.consolidation_cycle = state.get("consolidation_cycle", 0)
            self.total_stored = state.get("total_stored", 0)
            self.total_consolidated = state.get("total_consolidated", 0)
            self.total_retrieved = state.get("total_retrieved", 0)
            self.consolidation_history = state.get("consolidation_history", [])
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False
