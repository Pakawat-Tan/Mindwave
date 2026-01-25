"""
EmotionDrivenRule.py
Implements emotion-driven decision rules
"""

from typing import Dict, List, Any, Optional
from enum import Enum
import sys
from pathlib import Path

# Add parent directory to path
rules_dir = Path(__file__).parent.parent
if str(rules_dir) not in sys.path:
    sys.path.insert(0, str(rules_dir))
from ConfigLoader import ConfigLoader


class EmotionLevel(Enum):
    """Emotion intensity levels"""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    INTENSE = "intense"


class EmotionDrivenRule:
    """Manages emotion-driven decision rules."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize emotion-driven rule."""
        self.config_loader = ConfigLoader()
        self.emotion_thresholds: Dict[str, float] = {}
        self.emotion_effects: Dict[str, Dict[str, float]] = {}
        
        if config_path:
            self.load_from_json(config_path)
        else:
            self.load_from_json()
    
    def load_from_json(self, config_path: Optional[str] = None) -> bool:
        """Load emotion-driven rules from JSON configuration."""
        try:
            config = self.config_loader.load_config("AdaptionRule", "Adaption")
            
            if not config:
                return False
            
            emotional = config.get("emotional_adaptation", {})
            
            # Set default thresholds
            self.emotion_thresholds = {
                "curiosity": 0.5,
                "caution": 0.6,
                "confidence": 0.7,
                "uncertainty": 0.4
            }
            
            return True
        except Exception as e:
            print(f"Error loading EmotionDrivenRule from JSON: {e}")
            return False
    
    def get_emotion_level(self, emotion_score: float) -> EmotionLevel:
        """Get emotion level from score."""
        if emotion_score < 0.2:
            return EmotionLevel.MINIMAL
        elif emotion_score < 0.4:
            return EmotionLevel.LOW
        elif emotion_score < 0.6:
            return EmotionLevel.MODERATE
        elif emotion_score < 0.8:
            return EmotionLevel.HIGH
        else:
            return EmotionLevel.INTENSE
    
    def apply_emotion_to_decision(self, emotion_type: str, emotion_score: float, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Apply emotion to decision-making."""
        level = self.get_emotion_level(emotion_score)
        
        modified_decision = decision.copy()
        modified_decision["emotional_influence"] = {
            "emotion_type": emotion_type,
            "emotion_score": emotion_score,
            "emotion_level": level.value
        }
        
        return modified_decision
