"""
EmotionRule.py
Implements emotion-driven decision rules
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


class EmotionalState(Enum):
    """Emotional states"""
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ALERT = "alert"
    CURIOUS = "curious"


@dataclass
class EmotionThresholds:
    """Thresholds for emotion-driven decisions"""
    positive_threshold: float = 0.6
    negative_threshold: float = -0.6
    alert_threshold: float = 0.8
    curiosity_threshold: float = 0.5
    emotional_influence_strength: float = 0.5


class EmotionRule:
    """Manages emotion-driven decision rules."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize emotion rule."""
        self.thresholds = EmotionThresholds()
        self.config_loader = ConfigLoader()
        self.current_emotion: Dict[str, float] = {}
        self.emotion_history: List[Dict[str, Any]] = []
        
        if config_path:
            self.load_from_json(config_path)
        else:
            self.load_from_json()
    
    def load_from_json(self, config_path: Optional[str] = None) -> bool:
        """Load emotion rules from JSON configuration."""
        try:
            config = self.config_loader.load_config("EmotionRule", "Memory")
            
            if not config:
                return False
            
            emotion_config = config.get("emotional_memory", {})
            self.thresholds.positive_threshold = emotion_config.get("positive_threshold", 0.6)
            self.thresholds.negative_threshold = emotion_config.get("negative_threshold", -0.6)
            self.thresholds.emotional_influence_strength = emotion_config.get("emotional_influence_strength", 0.5)
            
            return True
        except Exception as e:
            print(f"Error loading EmotionRule from JSON: {e}")
            return False
    
    def evaluate_emotional_influence(self, emotion_score: float, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate how emotions influence decisions.
        
        Parameters
        ----------
        emotion_score : float
            Current emotion score
        decision : Dict[str, Any]
            The decision to evaluate
            
        Returns
        -------
        Dict[str, Any]
            Modified decision with emotional influence
        """
        influenced_decision = decision.copy()
        
        if emotion_score >= self.thresholds.positive_threshold:
            # Positive emotions increase confidence
            influenced_decision["confidence"] = decision.get("confidence", 0.5) + self.thresholds.emotional_influence_strength
        elif emotion_score <= self.thresholds.negative_threshold:
            # Negative emotions decrease confidence
            influenced_decision["confidence"] = max(0.0, decision.get("confidence", 0.5) - self.thresholds.emotional_influence_strength)
        
        return influenced_decision
    
    def get_emotional_state(self) -> EmotionalState:
        """Get current emotional state."""
        if not self.current_emotion:
            return EmotionalState.NEUTRAL
        
        avg_emotion = sum(self.current_emotion.values()) / len(self.current_emotion)
        
        if avg_emotion >= self.thresholds.alert_threshold:
            return EmotionalState.ALERT
        elif avg_emotion >= self.thresholds.positive_threshold:
            return EmotionalState.POSITIVE
        elif avg_emotion <= self.thresholds.negative_threshold:
            return EmotionalState.NEGATIVE
        
        return EmotionalState.NEUTRAL
