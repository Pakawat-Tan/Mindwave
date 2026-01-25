"""
SafetyRule.py
Safety constraints and action validation rules
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add dependencies
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from Brain.enum import SafetyLevel

rules_dir = Path(__file__).parent.parent
if str(rules_dir) not in sys.path:
    sys.path.insert(0, str(rules_dir))
from ConfigLoader import ConfigLoader


@dataclass
class MotionConstraint:
    """Motion safety constraints"""
    enabled: bool = True
    max_velocity: float = 0.5
    max_force_newtons: float = 100.0
    require_emergency_stop: bool = True


@dataclass
class OutputSafetyConstraint:
    """Output safety constraints"""
    filter_harmful_content: bool = True
    max_risk_level: float = 0.7
    require_review_threshold: float = 0.5


class SafetyRule:
    """Manages safety constraints and action validation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize safety rule."""
        self.motion_constraint = MotionConstraint()
        self.output_constraint = OutputSafetyConstraint()
        self.safety_level = SafetyLevel.NORMAL
        self.config_loader = ConfigLoader()
        self.emergency_stop_active = False
        self.safety_events: List[Dict[str, Any]] = []
        
        if config_path:
            self.load_from_json(config_path)
        else:
            self.load_from_json()
    
    def load_from_json(self, config_path: Optional[str] = None) -> bool:
        """Load safety rules from JSON configuration."""
        try:
            config = self.config_loader.load_config("SafetyPolicy", "Safety")
            
            if not config:
                return False
            
            # Load motion safety
            motion = config.get("motion_safety", {})
            self.motion_constraint.enabled = motion.get("enabled", True)
            self.motion_constraint.max_velocity = motion.get("max_movement_velocity", 0.5)
            self.motion_constraint.max_force_newtons = motion.get("max_force_newtons", 100.0)
            
            # Load output safety
            output = config.get("output_safety", {})
            self.output_constraint.filter_harmful_content = output.get("filter_harmful_content", True)
            self.output_constraint.max_risk_level = output.get("max_response_risk_level", 0.7)
            
            # Load safety level
            safety_level_name = config.get("default_safety_level", "NORMAL")
            try:
                self.safety_level = SafetyLevel[safety_level_name]
            except KeyError:
                self.safety_level = SafetyLevel.NORMAL
            
            return True
        except Exception as e:
            print(f"Error loading SafetyRule from JSON: {e}")
            return False
    
    def check_action_safety(self, action_type: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """Check if an action is safe to execute.
        
        Parameters
        ----------
        action_type : str
            Type of action (motion, output, decision, etc.)
        action : Dict[str, Any]
            The action to check
            
        Returns
        -------
        Dict[str, Any]
            Safety check result
        """
        result = {
            "safe": True,
            "action_type": action_type,
            "violations": [],
            "safety_level": self.safety_level.name
        }
        
        if action_type == "motion":
            if not self._check_motion_safety(action):
                result["safe"] = False
                result["violations"].append("Motion exceeds safety constraints")
        
        elif action_type == "output":
            if not self._check_output_safety(action):
                result["safe"] = False
                result["violations"].append("Output exceeds safety constraints")
        
        return result
    
    def _check_motion_safety(self, motion: Dict[str, Any]) -> bool:
        """Check motion safety constraints."""
        if not self.motion_constraint.enabled:
            return True
        
        velocity = motion.get("velocity", 0.0)
        force = motion.get("force", 0.0)
        
        if velocity > self.motion_constraint.max_velocity:
            return False
        
        if force > self.motion_constraint.max_force_newtons:
            return False
        
        return True
    
    def _check_output_safety(self, output: Dict[str, Any]) -> bool:
        """Check output safety constraints."""
        if not self.output_constraint.filter_harmful_content:
            return True
        
        risk_level = output.get("risk_level", 0.0)
        
        if risk_level > self.output_constraint.max_risk_level:
            return False
        
        return True
    
    def enforce_safety_constraints(self) -> Dict[str, Any]:
        """Enforce all active safety constraints."""
        return {
            "motion_enabled": self.motion_constraint.enabled,
            "output_filtering": self.output_constraint.filter_harmful_content,
            "emergency_stop_active": self.emergency_stop_active,
            "safety_level": self.safety_level.name
        }
    
    def trigger_emergency_stop(self) -> bool:
        """Trigger emergency stop."""
        self.emergency_stop_active = True
        self.safety_events.append({
            "event": "emergency_stop",
            "timestamp": __import__("time").time()
        })
        return True
    
    def release_emergency_stop(self) -> bool:
        """Release emergency stop."""
        self.emergency_stop_active = False
        return True
    
    def request_override(self, override_type: str) -> Dict[str, Any]:
        """Request a safety override."""
        return {
            "override_type": override_type,
            "approved": False,
            "reason": "Safety override requires authorization"
        }
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get comprehensive safety status."""
        return {
            "safety_level": self.safety_level.name,
            "emergency_stop": self.emergency_stop_active,
            "motion_constraints": {
                "enabled": self.motion_constraint.enabled,
                "max_velocity": self.motion_constraint.max_velocity,
                "max_force": self.motion_constraint.max_force_newtons
            },
            "output_constraints": {
                "filter_harmful": self.output_constraint.filter_harmful_content,
                "max_risk_level": self.output_constraint.max_risk_level
            },
            "recent_events": self.safety_events[-5:] if self.safety_events else []
        }
