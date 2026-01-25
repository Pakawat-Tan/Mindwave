"""
AdaptionRule.py
Manages structural, behavioral, and emotional adaptation rules
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# Add parent directory to path
rules_dir = Path(__file__).parent.parent
if str(rules_dir) not in sys.path:
    sys.path.insert(0, str(rules_dir))
from ConfigLoader import ConfigLoader


class AdaptationType(Enum):
    """Types of system adaptation"""
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    EMOTIONAL = "emotional"


@dataclass
class AdaptationTrigger:
    """Represents an adaptation trigger"""
    metric_name: str
    threshold: float
    comparison: str  # "greater_than", "less_than", "equal"
    adaptation_type: AdaptationType


class AdaptionRule:
    """Manages system adaptation rules."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize adaption rule.
        
        Parameters
        ----------
        config_path : str, optional
            Path to AdaptionRule.json configuration file
        """
        self.structural_adaptation_enabled = True
        self.behavioral_adaptation_enabled = True
        self.emotional_adaptation_enabled = True
        self.config_loader = ConfigLoader()
        
        self.structural_settings: Dict[str, Any] = {}
        self.behavioral_settings: Dict[str, Any] = {}
        self.emotional_settings: Dict[str, Any] = {}
        
        self.adaptation_history: List[Dict[str, Any]] = []
        
        if config_path:
            self.load_from_json(config_path)
        else:
            self.load_from_json()
    
    def load_from_json(self, config_path: Optional[str] = None) -> bool:
        """Load adaptation rules from JSON configuration.
        
        Returns
        -------
        bool
            True if successfully loaded
        """
        try:
            config = self.config_loader.load_config("AdaptionRule", "Adaption")
            
            if not config:
                print("Warning: AdaptionRule configuration not found")
                return False
            
            # Load structural adaptation
            structural = config.get("structural_adaptation", {})
            self.structural_adaptation_enabled = structural.get("enabled", True)
            self.structural_settings = {
                "allow_neuron_addition": structural.get("allow_neuron_addition", True),
                "allow_neuron_removal": structural.get("allow_neuron_removal", False),
                "allow_connection_modification": structural.get("allow_connection_modification", True),
                "max_structural_change_rate": structural.get("max_structural_change_rate", 0.05),
                "require_approval": structural.get("require_advisor_approval", True)
            }
            
            # Load behavioral adaptation
            behavioral = config.get("behavioral_adaptation", {})
            self.behavioral_adaptation_enabled = behavioral.get("enabled", True)
            self.behavioral_settings = {
                "allow_strategy_change": behavioral.get("allow_strategy_change", True),
                "allow_parameter_tuning": behavioral.get("allow_parameter_tuning", True),
                "max_behavioral_change_rate": behavioral.get("max_behavioral_change_rate", 0.1),
                "monitoring_enabled": behavioral.get("enable_behavior_monitoring", True)
            }
            
            # Load emotional adaptation
            emotional = config.get("emotional_adaptation", {})
            self.emotional_adaptation_enabled = emotional.get("enabled", True)
            self.emotional_settings = {
                "allow_emotion_profile_update": emotional.get("allow_emotion_profile_update", True),
                "emotion_learning_rate": emotional.get("emotion_learning_rate", 0.05),
                "enable_emotional_response": emotional.get("enable_emotional_response", True),
                "emotional_dampening": emotional.get("emotional_dampening_factor", 0.9)
            }
            
            return True
        except Exception as e:
            print(f"Error loading AdaptionRule from JSON: {e}")
            return False
    
    def should_adapt_structure(self, metrics: Dict[str, Any]) -> bool:
        """Check if structural adaptation should occur.
        
        Parameters
        ----------
        metrics : Dict[str, Any]
            System metrics to evaluate
            
        Returns
        -------
        bool
            True if structural adaptation should occur
        """
        if not self.structural_adaptation_enabled:
            return False
        
        # Check if performance metrics indicate need for adaptation
        performance = metrics.get("performance", 0.5)
        adaptation_frequency = metrics.get("adaptation_frequency", 0.0)
        
        # Adapt if performance is low and we haven't adapted too frequently
        if performance < 0.5 and adaptation_frequency < self.structural_settings["max_structural_change_rate"]:
            return True
        
        return False
    
    def adapt_structure(self, adaptation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply structural adaptation.
        
        Parameters
        ----------
        adaptation_params : Dict[str, Any]
            Parameters for structural adaptation
            
        Returns
        -------
        Dict[str, Any]
            Result of adaptation
        """
        if not self.structural_adaptation_enabled:
            return {"adapted": False, "reason": "structural_adaptation_disabled"}
        
        change_type = adaptation_params.get("change_type", "unknown")
        requires_approval = self.structural_settings["require_approval"]
        
        result = {
            "adapted": True,
            "adaptation_type": "structural",
            "change_type": change_type,
            "requires_approval": requires_approval
        }
        
        self.adaptation_history.append(result)
        
        return result
    
    def should_adapt_behavior(self, metrics: Dict[str, Any]) -> bool:
        """Check if behavioral adaptation should occur.
        
        Parameters
        ----------
        metrics : Dict[str, Any]
            System metrics to evaluate
            
        Returns
        -------
        bool
            True if behavioral adaptation should occur
        """
        if not self.behavioral_adaptation_enabled:
            return False
        
        # Check behavioral metrics
        success_rate = metrics.get("success_rate", 0.5)
        efficiency = metrics.get("efficiency", 0.5)
        
        # Adapt if success rate is low
        if success_rate < 0.6 or efficiency < 0.5:
            return True
        
        return False
    
    def adapt_behavior(self, adaptation_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply behavioral adaptation.
        
        Parameters
        ----------
        adaptation_params : Dict[str, Any]
            Parameters for behavioral adaptation
            
        Returns
        -------
        Dict[str, Any]
            Result of adaptation
        """
        if not self.behavioral_adaptation_enabled:
            return {"adapted": False, "reason": "behavioral_adaptation_disabled"}
        
        strategy = adaptation_params.get("strategy", "default")
        
        result = {
            "adapted": True,
            "adaptation_type": "behavioral",
            "strategy": strategy
        }
        
        self.adaptation_history.append(result)
        
        return result
    
    def should_adapt_emotion(self, emotion_metrics: Dict[str, Any]) -> bool:
        """Check if emotional adaptation should occur.
        
        Parameters
        ----------
        emotion_metrics : Dict[str, Any]
            Emotional metrics to evaluate
            
        Returns
        -------
        bool
            True if emotional adaptation should occur
        """
        if not self.emotional_adaptation_enabled:
            return False
        
        # Check emotional metrics
        emotion_stability = emotion_metrics.get("emotional_stability", 0.5)
        emotion_volatility = emotion_metrics.get("emotional_volatility", 0.5)
        
        # Adapt if emotions are unstable or too volatile
        if emotion_stability < 0.4 or emotion_volatility > 0.8:
            return True
        
        return False
    
    def adapt_emotion(self, emotion_params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply emotional adaptation.
        
        Parameters
        ----------
        emotion_params : Dict[str, Any]
            Parameters for emotional adaptation
            
        Returns
        -------
        Dict[str, Any]
            Result of adaptation
        """
        if not self.emotional_adaptation_enabled:
            return {"adapted": False, "reason": "emotional_adaptation_disabled"}
        
        emotion_profile_update = emotion_params.get("profile_update", {})
        
        # Apply dampening to emotion changes
        dampening_factor = self.emotional_settings["emotional_dampening"]
        dampened_update = {
            key: value * dampening_factor 
            for key, value in emotion_profile_update.items()
        }
        
        result = {
            "adapted": True,
            "adaptation_type": "emotional",
            "update_applied": dampened_update,
            "dampening_factor": dampening_factor
        }
        
        self.adaptation_history.append(result)
        
        return result
    
    def evaluate_adaptation_need(self, system_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate overall adaptation needs.
        
        Parameters
        ----------
        system_metrics : Dict[str, Any]
            Overall system metrics
            
        Returns
        -------
        Dict[str, Any]
            Adaptation recommendations
        """
        recommendations = {
            "structural_adaptation_needed": self.should_adapt_structure(system_metrics),
            "behavioral_adaptation_needed": self.should_adapt_behavior(system_metrics),
            "emotional_adaptation_needed": self.should_adapt_emotion(system_metrics.get("emotion_metrics", {}))
        }
        
        return recommendations
    
    def get_adaptation_status(self) -> Dict[str, Any]:
        """Get current adaptation status.
        
        Returns
        -------
        Dict[str, Any]
            Status of all adaptation systems
        """
        return {
            "structural_enabled": self.structural_adaptation_enabled,
            "behavioral_enabled": self.behavioral_adaptation_enabled,
            "emotional_enabled": self.emotional_adaptation_enabled,
            "recent_adaptations": self.adaptation_history[-10:] if self.adaptation_history else [],
            "total_adaptations": len(self.adaptation_history)
        }
