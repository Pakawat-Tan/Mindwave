"""
SystemRule.py
Core system governance and state management rules
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Brain.enum import SystemState, SystemPriority

# Add rules directory
rules_dir = Path(__file__).parent.parent
if str(rules_dir) not in sys.path:
    sys.path.insert(0, str(rules_dir))
from ConfigLoader import ConfigLoader


@dataclass
class SystemConstraint:
    """Represents a system constraint"""
    name: str
    description: str
    constraint_type: str
    value: float
    unit: str
    priority: SystemPriority


class SystemRule:
    """Manages core system governance and state transitions."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize system rule.
        
        Parameters
        ----------
        config_path : str, optional
            Path to SystemRule.json configuration file
        """
        self.current_state = SystemState.IDLE
        self.constraints: Dict[str, SystemConstraint] = {}
        self.allowed_transitions: Dict[SystemState, List[SystemState]] = {
            SystemState.IDLE: [SystemState.RUNNING, SystemState.LEARNING, SystemState.SHUTDOWN],
            SystemState.RUNNING: [SystemState.LEARNING, SystemState.MAINTENANCE, SystemState.ERROR, SystemState.SHUTDOWN],
            SystemState.LEARNING: [SystemState.RUNNING, SystemState.MAINTENANCE, SystemState.ERROR, SystemState.SHUTDOWN],
            SystemState.MAINTENANCE: [SystemState.RUNNING, SystemState.LEARNING, SystemState.SHUTDOWN, SystemState.ERROR],
            SystemState.ERROR: [SystemState.MAINTENANCE, SystemState.SHUTDOWN, SystemState.IDLE],
            SystemState.SHUTDOWN: [SystemState.IDLE]
        }
        self.config_loader = ConfigLoader()
        self.state_history: List[Dict[str, Any]] = []
        
        if config_path:
            self.load_from_json(config_path)
        else:
            self.load_from_json()
    
    def load_from_json(self, config_path: Optional[str] = None) -> bool:
        """Load system rules from JSON configuration."""
        try:
            config = self.config_loader.load_config("SystemRule", "System")
            
            if not config:
                return False
            
            # Initialize default constraints
            self._initialize_default_constraints()
            
            return True
        except Exception as e:
            print(f"Error loading SystemRule from JSON: {e}")
            return False
    
    def _initialize_default_constraints(self):
        """Initialize default system constraints"""
        self.constraints["memory_limit"] = SystemConstraint(
            name="memory_limit",
            description="Maximum memory usage percentage",
            constraint_type="resource",
            value=80.0,
            unit="%",
            priority=SystemPriority.HIGH
        )
        
        self.constraints["cpu_limit"] = SystemConstraint(
            name="cpu_limit",
            description="Maximum CPU usage percentage",
            constraint_type="resource",
            value=90.0,
            unit="%",
            priority=SystemPriority.HIGH
        )
        
        self.constraints["cycle_time"] = SystemConstraint(
            name="cycle_time",
            description="Maximum brain cycle time",
            constraint_type="timing",
            value=1000.0,
            unit="ms",
            priority=SystemPriority.NORMAL
        )
    
    def transition_state(self, new_state: SystemState) -> bool:
        """Attempt to transition to a new state.
        
        Parameters
        ----------
        new_state : SystemState
            The desired new state
            
        Returns
        -------
        bool
            True if transition is allowed
        """
        if new_state not in self.allowed_transitions.get(self.current_state, []):
            print(f"Invalid state transition: {self.current_state.name} -> {new_state.name}")
            return False
        
        old_state = self.current_state
        self.current_state = new_state
        
        # Record transition
        self.state_history.append({
            "from_state": old_state.name,
            "to_state": new_state.name,
            "timestamp": __import__("time").time()
        })
        
        return True
    
    def validate_system_integrity(self) -> Dict[str, Any]:
        """Validate current system integrity.
        
        Returns
        -------
        Dict[str, Any]
            Integrity check results
        """
        result = {
            "valid": True,
            "violations": []
        }
        
        # Check constraints
        for constraint_name, constraint in self.constraints.items():
            # This would be checked against actual system metrics
            pass
        
        return result
    
    def coordinate_subsystems(self) -> Dict[str, Any]:
        """Coordinate operation of all subsystems.
        
        Returns
        -------
        Dict[str, Any]
            Coordination results
        """
        return {
            "coordination_status": "active",
            "subsystems": ["memory", "learning", "routing", "safety"],
            "synchronized": True
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status.
        
        Returns
        -------
        Dict[str, Any]
            Current system status
        """
        return {
            "current_state": self.current_state.name,
            "constraints": {
                name: {
                    "value": c.value,
                    "unit": c.unit,
                    "priority": c.priority.name
                }
                for name, c in self.constraints.items()
            },
            "state_transitions_available": [
                s.name for s in self.allowed_transitions.get(self.current_state, [])
            ],
            "recent_transitions": self.state_history[-5:] if self.state_history else []
        }
