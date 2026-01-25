"""
MemoryRule.py
Manages memory consolidation, retrieval, and decay policies
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import sys
from pathlib import Path

# Add parent directory to path
rules_dir = Path(__file__).parent.parent
if str(rules_dir) not in sys.path:
    sys.path.insert(0, str(rules_dir))
from ConfigLoader import ConfigLoader


class ConsolidationType(Enum):
    """Types of memory consolidation"""
    NONE = "none"
    SHORT_TO_MEDIUM = "short_to_medium"
    MEDIUM_TO_LONG = "medium_to_long"
    FULL_CONSOLIDATION = "full_consolidation"


@dataclass
class MemoryConsolidationConfig:
    """Configuration for memory consolidation"""
    enable_consolidation: bool = True
    consolidation_interval_minutes: int = 30
    consolidation_threshold: float = 0.5
    consolidation_types: List[ConsolidationType] = field(default_factory=lambda: [ConsolidationType.FULL_CONSOLIDATION])


@dataclass
class MemoryDecayConfig:
    """Configuration for memory decay/forgetting"""
    enable_forgetting: bool = True
    base_decay_rate: float = 0.05  # 5% per unit time
    decay_by_recency: bool = True
    emotional_protection: bool = True
    emotional_protection_factor: float = 0.5  # Reduce decay by 50% for emotional memories


class MemoryRule:
    """Manages memory consolidation and decay policies."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize memory rule.
        
        Parameters
        ----------
        config_path : str, optional
            Path to MemoryRule.json configuration file
        """
        self.consolidation_config = MemoryConsolidationConfig()
        self.decay_config = MemoryDecayConfig()
        self.config_loader = ConfigLoader()
        self.last_consolidation_time: Dict[str, float] = {}
        self.memory_access_history: Dict[str, List[float]] = {}
        
        if config_path:
            self.load_from_json(config_path)
        else:
            self.load_from_json()
    
    def load_from_json(self, config_path: Optional[str] = None) -> bool:
        """Load memory rules from JSON configuration.
        
        Returns
        -------
        bool
            True if successfully loaded
        """
        try:
            config = self.config_loader.load_config("MemoryRule", "Memory")
            
            if not config:
                print("Warning: MemoryRule configuration not found")
                return False
            
            # Load consolidation config
            consolidation = config.get("memory_consolidation", {})
            self.consolidation_config.enable_consolidation = consolidation.get("enable_consolidation", True)
            self.consolidation_config.consolidation_interval_minutes = consolidation.get("consolidation_interval_minutes", 30)
            self.consolidation_config.consolidation_threshold = consolidation.get("consolidation_threshold", 0.5)
            
            consolidation_types = consolidation.get("consolidation_types", ["full_consolidation"])
            self.consolidation_config.consolidation_types = [
                ConsolidationType(ct) for ct in consolidation_types
            ]
            
            # Load decay config
            decay = config.get("memory_decay", {})
            self.decay_config.enable_forgetting = decay.get("enable_forgetting", True)
            self.decay_config.base_decay_rate = decay.get("base_decay_rate", 0.05)
            self.decay_config.decay_by_recency = decay.get("decay_by_recency", True)
            self.decay_config.emotional_protection = decay.get("emotional_protection", True)
            self.decay_config.emotional_protection_factor = decay.get("emotional_protection_factor", 0.5)
            
            return True
        except Exception as e:
            print(f"Error loading MemoryRule from JSON: {e}")
            return False
    
    def should_consolidate(self, memory_type: str) -> bool:
        """Check if consolidation should occur for a memory type.
        
        Parameters
        ----------
        memory_type : str
            Type of memory (e.g., 'short_term', 'medium_term')
            
        Returns
        -------
        bool
            True if consolidation should occur
        """
        if not self.consolidation_config.enable_consolidation:
            return False
        
        last_time = self.last_consolidation_time.get(memory_type, 0)
        current_time = time.time()
        elapsed_minutes = (current_time - last_time) / 60
        
        return elapsed_minutes >= self.consolidation_config.consolidation_interval_minutes
    
    def consolidate_memory(self, memory_type: str, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolidate memories.
        
        Parameters
        ----------
        memory_type : str
            Type of memory being consolidated
        memories : List[Dict[str, Any]]
            List of memories to consolidate
            
        Returns
        -------
        Dict[str, Any]
            Consolidation result with processed memories
        """
        if not self.consolidation_config.enable_consolidation:
            return {"consolidated": False}
        
        # Filter memories above threshold
        strong_memories = [
            m for m in memories 
            if m.get("strength", 0) >= self.consolidation_config.consolidation_threshold
        ]
        
        # Update last consolidation time
        self.last_consolidation_time[memory_type] = time.time()
        
        return {
            "consolidated": True,
            "memory_type": memory_type,
            "original_count": len(memories),
            "consolidated_count": len(strong_memories),
            "consolidation_type": self.consolidation_config.consolidation_types[0].value if self.consolidation_config.consolidation_types else "none"
        }
    
    def calculate_memory_decay(self, memory: Dict[str, Any], age_minutes: float) -> float:
        """Calculate decay factor for a memory.
        
        Parameters
        ----------
        memory : Dict[str, Any]
            The memory to calculate decay for
        age_minutes : float
            Age of memory in minutes
            
        Returns
        -------
        float
            Decay factor (0.0 to 1.0, where 1.0 = no decay)
        """
        if not self.decay_config.enable_forgetting:
            return 1.0
        
        # Base exponential decay
        decay_factor = 1.0 - (self.decay_config.base_decay_rate * age_minutes)
        decay_factor = max(0.0, min(1.0, decay_factor))
        
        # Apply emotional protection if applicable
        if self.decay_config.emotional_protection:
            is_emotional = memory.get("is_emotional", False)
            if is_emotional:
                emotional_level = memory.get("emotional_level", 0.5)
                # Reduce decay for emotional memories
                protection_factor = self.decay_config.emotional_protection_factor * emotional_level
                decay_factor = decay_factor + (1.0 - decay_factor) * protection_factor
        
        return decay_factor
    
    def apply_decay_to_memory(self, memory: Dict[str, Any], age_minutes: float) -> Dict[str, Any]:
        """Apply decay to memory strength.
        
        Parameters
        ----------
        memory : Dict[str, Any]
            The memory to apply decay to
        age_minutes : float
            Age of memory in minutes
            
        Returns
        -------
        Dict[str, Any]
            Memory with updated strength
        """
        decay_factor = self.calculate_memory_decay(memory, age_minutes)
        original_strength = memory.get("strength", 0.5)
        new_strength = original_strength * decay_factor
        
        memory["strength"] = new_strength
        memory["decay_factor"] = decay_factor
        
        return memory
    
    def retrieve_memory(self, memory_id: str, memory: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve a memory and update access history.
        
        Parameters
        ----------
        memory_id : str
            ID of the memory
        memory : Dict[str, Any]
            The memory being retrieved
            
        Returns
        -------
        Dict[str, Any]
            Retrieved memory with updated metadata
        """
        current_time = time.time()
        
        # Update access history
        if memory_id not in self.memory_access_history:
            self.memory_access_history[memory_id] = []
        
        self.memory_access_history[memory_id].append(current_time)
        
        # Update retrieval count
        memory["retrieval_count"] = memory.get("retrieval_count", 0) + 1
        memory["last_retrieved"] = current_time
        
        return memory
    
    def get_memory_statistics(self, memory_type: str) -> Dict[str, Any]:
        """Get statistics about memories of a type.
        
        Parameters
        ----------
        memory_type : str
            Type of memory
            
        Returns
        -------
        Dict[str, Any]
            Memory statistics
        """
        return {
            "memory_type": memory_type,
            "enable_consolidation": self.consolidation_config.enable_consolidation,
            "consolidation_interval_minutes": self.consolidation_config.consolidation_interval_minutes,
            "consolidation_threshold": self.consolidation_config.consolidation_threshold,
            "enable_forgetting": self.decay_config.enable_forgetting,
            "base_decay_rate": self.decay_config.base_decay_rate,
            "emotional_protection": self.decay_config.emotional_protection,
            "last_consolidation_minutes_ago": (time.time() - self.last_consolidation_time.get(memory_type, 0)) / 60
        }
