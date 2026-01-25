"""
AcquisitionRule.py
Controls memory storage acquisition strategy
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path

# Add parent directory to path
rules_dir = Path(__file__).parent.parent
if str(rules_dir) not in sys.path:
    sys.path.insert(0, str(rules_dir))
from ConfigLoader import ConfigLoader


class AcquisitionTrigger(Enum):
    """Triggers for memory acquisition"""
    NOVELTY = "novelty"
    EMOTION = "emotion"
    IMPORTANCE = "importance"
    FREQUENCY = "frequency"
    EXPLICIT = "explicit"


@dataclass
class AcquisitionTriggerConfig:
    """Configuration for memory acquisition triggers"""
    novelty_threshold: float = 0.6
    emotion_strength_threshold: float = 0.5
    importance_threshold: float = 0.6
    frequency_threshold: int = 3  # Acquire after seen 3 times


@dataclass
class MemoryCapacity:
    """Memory storage capacity limits"""
    short_term_max: int = 50
    medium_term_max: int = 500
    long_term_max: int = 10000
    total_max: int = 11000


@dataclass
class FilteringConfig:
    """Filtering policies for memory acquisition"""
    filter_duplicate_content: bool = True
    filter_low_quality: bool = True
    quality_threshold: float = 0.3
    filter_personal_info: bool = False
    filter_sensitive_content: bool = True


class AcquisitionRule:
    """Manages memory storage acquisition and filtering."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize acquisition rule.
        
        Parameters
        ----------
        config_path : str, optional
            Path to AcquisitionRule.json configuration file
        """
        self.triggers = AcquisitionTriggerConfig()
        self.capacity = MemoryCapacity()
        self.filtering = FilteringConfig()
        self.config_loader = ConfigLoader()
        self.memory_store: Dict[str, List[Dict[str, Any]]] = {
            "short_term": [],
            "medium_term": [],
            "long_term": []
        }
        self.acquisition_history: List[Dict[str, Any]] = []
        
        if config_path:
            self.load_from_json(config_path)
        else:
            self.load_from_json()
    
    def load_from_json(self, config_path: Optional[str] = None) -> bool:
        """Load acquisition rules from JSON configuration.
        
        Returns
        -------
        bool
            True if successfully loaded
        """
        try:
            config = self.config_loader.load_config("AcquisitionRule", "Memory")
            
            if not config:
                print("Warning: AcquisitionRule configuration not found")
                return False
            
            # Load acquisition triggers
            triggers = config.get("acquisition_triggers", {})
            self.triggers.novelty_threshold = triggers.get("novelty_threshold", 0.6)
            self.triggers.emotion_strength_threshold = triggers.get("emotion_strength_threshold", 0.5)
            self.triggers.importance_threshold = triggers.get("importance_threshold", 0.6)
            self.triggers.frequency_threshold = triggers.get("frequency_threshold", 3)
            
            # Load memory capacity
            capacity = config.get("memory_capacity", {})
            self.capacity.short_term_max = capacity.get("short_term_max", 50)
            self.capacity.medium_term_max = capacity.get("medium_term_max", 500)
            self.capacity.long_term_max = capacity.get("long_term_max", 10000)
            self.capacity.total_max = (
                self.capacity.short_term_max + 
                self.capacity.medium_term_max + 
                self.capacity.long_term_max
            )
            
            # Load filtering config
            filters = config.get("filtering", {})
            self.filtering.filter_duplicate_content = filters.get("filter_duplicate_content", True)
            self.filtering.filter_low_quality = filters.get("filter_low_quality", True)
            self.filtering.quality_threshold = filters.get("quality_threshold", 0.3)
            self.filtering.filter_personal_info = filters.get("filter_personal_info", False)
            self.filtering.filter_sensitive_content = filters.get("filter_sensitive_content", True)
            
            return True
        except Exception as e:
            print(f"Error loading AcquisitionRule from JSON: {e}")
            return False
    
    def should_acquire_memory(self, memory: Dict[str, Any]) -> Tuple[bool, str]:
        """Determine if a memory should be acquired.
        
        Parameters
        ----------
        memory : Dict[str, Any]
            The memory to evaluate
            
        Returns
        -------
        Tuple[bool, str]
            (should_acquire, reason)
        """
        # Check filtering
        if not self._passes_filters(memory):
            return False, "filtered"
        
        # Check acquisition triggers
        triggers_met = []
        
        # Novelty trigger
        if memory.get("novelty", 0) >= self.triggers.novelty_threshold:
            triggers_met.append(AcquisitionTrigger.NOVELTY)
        
        # Emotion trigger
        if memory.get("emotional_level", 0) >= self.triggers.emotion_strength_threshold:
            triggers_met.append(AcquisitionTrigger.EMOTION)
        
        # Importance trigger
        if memory.get("importance", 0) >= self.triggers.importance_threshold:
            triggers_met.append(AcquisitionTrigger.IMPORTANCE)
        
        # Frequency trigger
        if memory.get("frequency_count", 0) >= self.triggers.frequency_threshold:
            triggers_met.append(AcquisitionTrigger.FREQUENCY)
        
        # Explicit trigger
        if memory.get("explicit_store", False):
            triggers_met.append(AcquisitionTrigger.EXPLICIT)
        
        if triggers_met:
            return True, f"triggers_met: {[t.value for t in triggers_met]}"
        
        return False, "no_triggers_met"
    
    def _passes_filters(self, memory: Dict[str, Any]) -> bool:
        """Check if memory passes all filters.
        
        Parameters
        ----------
        memory : Dict[str, Any]
            The memory to check
            
        Returns
        -------
        bool
            True if memory passes filters
        """
        # Check quality
        if self.filtering.filter_low_quality:
            if memory.get("quality", 0.5) < self.filtering.quality_threshold:
                return False
        
        # Check for duplicates
        if self.filtering.filter_duplicate_content:
            if self._is_duplicate(memory):
                return False
        
        # Check for sensitive content
        if self.filtering.filter_sensitive_content:
            if memory.get("is_sensitive", False):
                return False
        
        # Check for personal info (if filtering enabled)
        if self.filtering.filter_personal_info:
            if memory.get("contains_personal_info", False):
                return False
        
        return True
    
    def _is_duplicate(self, memory: Dict[str, Any]) -> bool:
        """Check if memory is a duplicate.
        
        Parameters
        ----------
        memory : Dict[str, Any]
            The memory to check
            
        Returns
        -------
        bool
            True if it's a duplicate
        """
        memory_content = memory.get("content", "")
        
        for store_type, memories in self.memory_store.items():
            for existing in memories:
                if existing.get("content", "") == memory_content:
                    return True
        
        return False
    
    def acquire_memory(self, memory: Dict[str, Any], priority: str = "medium") -> Dict[str, Any]:
        """Acquire a memory into storage.
        
        Parameters
        ----------
        memory : Dict[str, Any]
            The memory to acquire
        priority : str
            Priority level (low, medium, high)
            
        Returns
        -------
        Dict[str, Any]
            Result of acquisition
        """
        should_acquire, reason = self.should_acquire_memory(memory)
        
        if not should_acquire:
            return {
                "acquired": False,
                "reason": reason,
                "memory_id": memory.get("id", "unknown")
            }
        
        # Determine storage location based on priority
        if priority == "high":
            storage = "long_term"
        elif priority == "medium":
            storage = "medium_term"
        else:
            storage = "short_term"
        
        # Check capacity
        if len(self.memory_store[storage]) >= getattr(self.capacity, f"{storage}_max"):
            storage = self._find_available_storage()
            if not storage:
                return {
                    "acquired": False,
                    "reason": "no_available_storage",
                    "memory_id": memory.get("id", "unknown")
                }
        
        # Store memory
        self.memory_store[storage].append(memory)
        
        # Log acquisition
        self.acquisition_history.append({
            "memory_id": memory.get("id", "unknown"),
            "storage": storage,
            "timestamp": __import__("time").time()
        })
        
        return {
            "acquired": True,
            "reason": reason,
            "storage": storage,
            "memory_id": memory.get("id", "unknown")
        }
    
    def _find_available_storage(self) -> Optional[str]:
        """Find available storage location with space.
        
        Returns
        -------
        Optional[str]
            Available storage type, or None if all full
        """
        for storage_type in ["short_term", "medium_term", "long_term"]:
            capacity = getattr(self.capacity, f"{storage_type}_max")
            if len(self.memory_store[storage_type]) < capacity:
                return storage_type
        
        return None
    
    def get_storage_status(self) -> Dict[str, Any]:
        """Get current storage status.
        
        Returns
        -------
        Dict[str, Any]
            Storage usage and statistics
        """
        status = {}
        
        for storage_type, memories in self.memory_store.items():
            capacity = getattr(self.capacity, f"{storage_type}_max")
            status[storage_type] = {
                "current_size": len(memories),
                "max_size": capacity,
                "utilization_percent": (len(memories) / capacity * 100) if capacity > 0 else 0
            }
        
        total_stored = sum(len(m) for m in self.memory_store.values())
        status["total"] = {
            "current_size": total_stored,
            "max_size": self.capacity.total_max,
            "utilization_percent": (total_stored / self.capacity.total_max * 100) if self.capacity.total_max > 0 else 0
        }
        
        return status
    
    def get_acquisition_rate(self, time_window_seconds: int = 60) -> float:
        """Get acquisition rate over time window.
        
        Parameters
        ----------
        time_window_seconds : int
            Time window in seconds
            
        Returns
        -------
        float
            Memories acquired per second
        """
        import time
        
        current_time = time.time()
        cutoff_time = current_time - time_window_seconds
        
        recent_acquisitions = sum(
            1 for entry in self.acquisition_history
            if entry.get("timestamp", 0) >= cutoff_time
        )
        
        return recent_acquisitions / time_window_seconds if time_window_seconds > 0 else 0
