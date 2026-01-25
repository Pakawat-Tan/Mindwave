"""
GradientRule.py
Implements gradient descent-based learning rules
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to path
rules_dir = Path(__file__).parent.parent
if str(rules_dir) not in sys.path:
    sys.path.insert(0, str(rules_dir))
from ConfigLoader import ConfigLoader


@dataclass
class GradientConfig:
    """Configuration for gradient learning"""
    learning_rate: float = 0.01
    momentum: float = 0.9
    gradient_clip_value: float = 1.0
    adaptive_learning: bool = True
    adaptive_decay: float = 0.999


class GradientRule:
    """Manages gradient descent-based learning."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize gradient rule."""
        self.config = GradientConfig()
        self.config_loader = ConfigLoader()
        self.velocity: Dict[str, float] = {}
        self.step_count = 0
        
        if config_path:
            self.load_from_json(config_path)
        else:
            self.load_from_json()
    
    def load_from_json(self, config_path: Optional[str] = None) -> bool:
        """Load gradient rules from JSON configuration."""
        try:
            config = self.config_loader.load_config("LearningRule", "Learning")
            
            if not config:
                return False
            
            gradient = config.get("gradient_learning", {})
            self.config.learning_rate = gradient.get("learning_rate", 0.01)
            self.config.momentum = gradient.get("momentum", 0.9)
            self.config.gradient_clip_value = gradient.get("gradient_clipping", 1.0)
            
            return True
        except Exception as e:
            print(f"Error loading GradientRule from JSON: {e}")
            return False
    
    def compute_gradient_step(self, gradients: Dict[str, float]) -> Dict[str, float]:
        """Compute gradient descent step.
        
        Parameters
        ----------
        gradients : Dict[str, float]
            Gradients for each parameter
            
        Returns
        -------
        Dict[str, float]
            Weight updates
        """
        updates = {}
        self.step_count += 1
        
        for param_name, gradient in gradients.items():
            # Clip gradient
            clipped_gradient = max(-self.config.gradient_clip_value, 
                                  min(self.config.gradient_clip_value, gradient))
            
            # Update velocity with momentum
            if param_name not in self.velocity:
                self.velocity[param_name] = 0.0
            
            self.velocity[param_name] = (
                self.config.momentum * self.velocity[param_name] +
                (1 - self.config.momentum) * clipped_gradient
            )
            
            # Adaptive learning rate
            if self.config.adaptive_learning:
                adaptive_lr = self.config.learning_rate / (1.0 - self.config.adaptive_decay ** self.step_count)
            else:
                adaptive_lr = self.config.learning_rate
            
            updates[param_name] = adaptive_lr * self.velocity[param_name]
        
        return updates
