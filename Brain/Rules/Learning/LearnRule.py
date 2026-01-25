"""
LearnRule.py
Implements learning rules and policies
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path

# Add parent directory to path
rules_dir = Path(__file__).parent.parent
if str(rules_dir) not in sys.path:
    sys.path.insert(0, str(rules_dir))
from ConfigLoader import ConfigLoader


class LearningMode(Enum):
    """Available learning modes"""
    GRADIENT = "gradient"
    ADVISOR = "advisor"
    EVOLUTION = "evolution"
    REPLAY = "replay"
    SELF = "self"
    REINFORCEMENT = "reinforcement"


@dataclass
class LearningConstraints:
    """Constraints for learning"""
    max_learning_rate: float = 0.1
    min_learning_rate: float = 0.0001
    gradient_clip_value: float = 1.0
    momentum: float = 0.9
    enable_dropout: bool = True
    dropout_rate: float = 0.5
    enable_regularization: bool = True
    regularization_strength: float = 0.0001


class LearnRule:
    """Manages learning rules and policies."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize learning rule.
        
        Parameters
        ----------
        config_path : str, optional
            Path to LearningRule.json configuration file
        """
        self.active_modes: List[LearningMode] = [LearningMode.GRADIENT]
        self.constraints = LearningConstraints()
        self.config_loader = ConfigLoader()
        self.learning_stats: Dict[str, Any] = {
            mode.value: {"samples": 0, "loss": 0.0}
            for mode in LearningMode
        }
        self.learning_enabled = True
        
        if config_path:
            self.load_from_json(config_path)
        else:
            self.load_from_json()
    
    def load_from_json(self, config_path: Optional[str] = None) -> bool:
        """Load learning rules from JSON configuration.
        
        Returns
        -------
        bool
            True if successfully loaded
        """
        try:
            config = self.config_loader.load_config("LearningRule", "Learning")
            
            if not config:
                print("Warning: LearningRule configuration not found")
                return False
            
            # Load active modes
            enabled_modes = config.get("enabled_modes", ["gradient"])
            self.active_modes = [
                LearningMode(mode) for mode in enabled_modes if mode in [m.value for m in LearningMode]
            ]
            
            # Load gradient learning settings
            gradient = config.get("gradient_learning", {})
            self.constraints.max_learning_rate = gradient.get("learning_rate", 0.01)
            self.constraints.momentum = gradient.get("momentum", 0.9)
            self.constraints.gradient_clip_value = gradient.get("gradient_clipping", 1.0)
            
            # Load regularization
            regularization = config.get("regularization", {})
            self.constraints.enable_regularization = regularization.get("enable_regularization", True)
            self.constraints.regularization_strength = regularization.get("regularization_strength", 0.0001)
            
            # Load dropout settings
            dropout = config.get("dropout", {})
            self.constraints.enable_dropout = dropout.get("enable_dropout", True)
            self.constraints.dropout_rate = dropout.get("dropout_rate", 0.5)
            
            return True
        except Exception as e:
            print(f"Error loading LearnRule from JSON: {e}")
            return False
    
    def is_mode_enabled(self, mode: LearningMode) -> bool:
        """Check if a learning mode is enabled.
        
        Parameters
        ----------
        mode : LearningMode
            The mode to check
            
        Returns
        -------
        bool
            True if mode is enabled
        """
        return mode in self.active_modes and self.learning_enabled
    
    def enable_learning_mode(self, mode: LearningMode) -> bool:
        """Enable a learning mode.
        
        Parameters
        ----------
        mode : LearningMode
            The mode to enable
            
        Returns
        -------
        bool
            True if successfully enabled
        """
        if mode not in self.active_modes:
            self.active_modes.append(mode)
        return True
    
    def disable_learning_mode(self, mode: LearningMode) -> bool:
        """Disable a learning mode.
        
        Parameters
        ----------
        mode : LearningMode
            The mode to disable
            
        Returns
        -------
        bool
            True if successfully disabled
        """
        if mode in self.active_modes:
            self.active_modes.remove(mode)
        return True
    
    def set_learning_rate(self, rate: float) -> bool:
        """Set the learning rate.
        
        Parameters
        ----------
        rate : float
            Learning rate value
            
        Returns
        -------
        bool
            True if successfully set (within constraints)
        """
        if self.constraints.min_learning_rate <= rate <= self.constraints.max_learning_rate:
            self.constraints.max_learning_rate = rate
            return True
        return False
    
    def get_learning_settings(self, mode: LearningMode) -> Dict[str, Any]:
        """Get learning settings for a specific mode.
        
        Parameters
        ----------
        mode : LearningMode
            The learning mode
            
        Returns
        -------
        Dict[str, Any]
            Settings for the mode
        """
        settings = {
            "mode": mode.value,
            "enabled": self.is_mode_enabled(mode),
            "learning_rate": self.constraints.max_learning_rate,
            "gradient_clipping": self.constraints.gradient_clip_value
        }
        
        if mode == LearningMode.GRADIENT:
            settings.update({
                "momentum": self.constraints.momentum,
                "dropout": self.constraints.enable_dropout,
                "dropout_rate": self.constraints.dropout_rate,
                "regularization": self.constraints.enable_regularization,
                "regularization_strength": self.constraints.regularization_strength
            })
        
        return settings
    
    def validate_learning_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate learning parameters.
        
        Parameters
        ----------
        params : Dict[str, Any]
            Parameters to validate
            
        Returns
        -------
        bool
            True if parameters are valid
        """
        # Check learning rate
        if "learning_rate" in params:
            lr = params["learning_rate"]
            if not (self.constraints.min_learning_rate <= lr <= self.constraints.max_learning_rate):
                return False
        
        # Check momentum
        if "momentum" in params:
            if not (0.0 <= params["momentum"] <= 1.0):
                return False
        
        # Check dropout rate
        if "dropout_rate" in params:
            if not (0.0 <= params["dropout_rate"] <= 1.0):
                return False
        
        return True
    
    def record_learning_step(self, mode: LearningMode, loss: float, samples: int = 1):
        """Record a learning step.
        
        Parameters
        ----------
        mode : LearningMode
            The learning mode used
        loss : float
            Loss value from this step
        samples : int
            Number of samples trained on
        """
        if mode.value in self.learning_stats:
            self.learning_stats[mode.value]["samples"] += samples
            self.learning_stats[mode.value]["loss"] = loss
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics.
        
        Returns
        -------
        Dict[str, Any]
            Statistics for all learning modes
        """
        return {
            "learning_enabled": self.learning_enabled,
            "active_modes": [mode.value for mode in self.active_modes],
            "statistics": self.learning_stats,
            "constraints": {
                "max_learning_rate": self.constraints.max_learning_rate,
                "gradient_clipping": self.constraints.gradient_clip_value,
                "regularization_enabled": self.constraints.enable_regularization,
                "dropout_enabled": self.constraints.enable_dropout
            }
        }
    
    def enable_learning(self) -> bool:
        """Enable all learning."""
        self.learning_enabled = True
        return True
    
    def disable_learning(self) -> bool:
        """Disable all learning."""
        self.learning_enabled = False
        return True
