"""
EmotionWeight.py
Emotional biases that affect decision-making and confidence.
"""

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class BiasProfile:
    """Represents emotion-based decision bias"""
    emotion: str
    bias_direction: float  # -1 to 1, negative = risk-averse, positive = risk-seeking
    magnitude: float  # 0 to 1, strength of bias
    confidence_multiplier: float = 1.0  # how emotion affects confidence
    decision_modifier: float = 0.0  # -1 to 1, shift in decision value
    timestamp: float = field(default_factory=time.time)


class EmotionWeight:
    """Manages emotional biases that affect decision-making."""
    
    def __init__(self):
        """Initialize emotion weights."""
        self.bias_map: Dict[str, BiasProfile] = self._initialize_bias_map()
        self.amplification_factors: Dict[str, float] = {}
        self.bias_history: List[Tuple[float, str, float]] = []
    
    def _initialize_bias_map(self) -> Dict[str, BiasProfile]:
        """Initialize default bias profiles.
        
        Returns
        -------
        Dict[str, BiasProfile]
            Default biases for each emotion
        """
        return {
            "joy": BiasProfile(
                emotion="joy",
                bias_direction=0.7,  # risk-seeking
                magnitude=0.6,
                confidence_multiplier=1.2,
                decision_modifier=0.3
            ),
            "sadness": BiasProfile(
                emotion="sadness",
                bias_direction=-0.6,  # risk-averse
                magnitude=0.5,
                confidence_multiplier=0.8,
                decision_modifier=-0.2
            ),
            "anger": BiasProfile(
                emotion="anger",
                bias_direction=0.8,  # aggressive/risk-seeking
                magnitude=0.7,
                confidence_multiplier=1.1,
                decision_modifier=0.4
            ),
            "fear": BiasProfile(
                emotion="fear",
                bias_direction=-0.9,  # very risk-averse
                magnitude=0.8,
                confidence_multiplier=0.7,
                decision_modifier=-0.5
            ),
            "trust": BiasProfile(
                emotion="trust",
                bias_direction=0.5,  # slightly risk-seeking
                magnitude=0.4,
                confidence_multiplier=1.15,
                decision_modifier=0.2
            ),
            "disgust": BiasProfile(
                emotion="disgust",
                bias_direction=-0.7,  # risk-averse
                magnitude=0.6,
                confidence_multiplier=0.9,
                decision_modifier=-0.3
            ),
            "surprise": BiasProfile(
                emotion="surprise",
                bias_direction=0.3,  # neutral-ish
                magnitude=0.5,
                confidence_multiplier=0.95,
                decision_modifier=0.1
            ),
            "anticipation": BiasProfile(
                emotion="anticipation",
                bias_direction=0.6,  # slightly risk-seeking
                magnitude=0.5,
                confidence_multiplier=1.1,
                decision_modifier=0.25
            ),
        }
    
    def apply_emotional_bias(self, decision_value: float, emotional_state: Dict[str, float]) -> float:
        """Apply emotional bias to a decision.
        
        Parameters
        ----------
        decision_value : float
            Original decision value
        emotional_state : Dict[str, float]
            Current emotional state
            
        Returns
        -------
        float
            Decision value with emotional bias applied
        """
        biased_value = decision_value
        
        for emotion, intensity in emotional_state.items():
            if emotion in self.bias_map:
                bias_profile = self.bias_map[emotion]
                
                # Calculate emotional influence
                emotional_influence = (
                    bias_profile.decision_modifier * 
                    intensity * 
                    bias_profile.magnitude
                )
                
                biased_value += emotional_influence
                
                # Record in history
                self.bias_history.append((time.time(), emotion, emotional_influence))
        
        # Clamp to reasonable range
        return max(-1.0, min(1.0, biased_value))
    
    def get_bias_magnitude(self, emotion_type: str, intensity: float = 1.0) -> float:
        """Get the magnitude of emotional bias.
        
        Parameters
        ----------
        emotion_type : str
            Type of emotion
        intensity : float
            Emotion intensity (0.0-1.0)
            
        Returns
        -------
        float
            Bias magnitude
        """
        if emotion_type in self.bias_map:
            bias_profile = self.bias_map[emotion_type]
            return bias_profile.magnitude * intensity
        return 0.0
    
    def modulate_confidence(self, confidence: float, emotional_state: Dict[str, float]) -> float:
        """Modulate confidence based on emotional state.
        
        Parameters
        ----------
        confidence : float
            Base confidence (0.0-1.0)
        emotional_state : Dict[str, float]
            Current emotional state
            
        Returns
        -------
        float
            Modulated confidence
        """
        confidence_multiplier = 1.0
        
        for emotion, intensity in emotional_state.items():
            if emotion in self.bias_map:
                bias_profile = self.bias_map[emotion]
                
                # Emotions modulate confidence
                multiplier_delta = (bias_profile.confidence_multiplier - 1.0) * intensity
                confidence_multiplier += multiplier_delta * 0.5  # soften the effect
        
        modulated = confidence * confidence_multiplier
        
        # Clamp to valid range
        return max(0.0, min(1.0, modulated))
    
    def get_risk_preference(self, emotional_state: Dict[str, float]) -> float:
        """Get overall risk preference based on emotions.
        
        Parameters
        ----------
        emotional_state : Dict[str, float]
            Current emotional state
            
        Returns
        -------
        float
            Risk preference (-1 = risk-averse, 1 = risk-seeking)
        """
        weighted_preference = 0.0
        total_weight = 0.0
        
        for emotion, intensity in emotional_state.items():
            if emotion in self.bias_map:
                bias_profile = self.bias_map[emotion]
                weighted_preference += bias_profile.bias_direction * intensity
                total_weight += intensity
        
        if total_weight > 0:
            return weighted_preference / total_weight
        return 0.0
    
    def update_bias(self, emotion: str, bias_direction: Optional[float] = None,
                   magnitude: Optional[float] = None,
                   confidence_multiplier: Optional[float] = None) -> bool:
        """Update a bias profile.
        
        Parameters
        ----------
        emotion : str
            Emotion name
        bias_direction : Optional[float]
            New bias direction
        magnitude : Optional[float]
            New magnitude
        confidence_multiplier : Optional[float]
            New confidence multiplier
            
        Returns
        -------
        bool
            Success
        """
        if emotion not in self.bias_map:
            return False
        
        bias_profile = self.bias_map[emotion]
        
        if bias_direction is not None:
            bias_profile.bias_direction = max(-1.0, min(1.0, bias_direction))
        if magnitude is not None:
            bias_profile.magnitude = max(0.0, min(1.0, magnitude))
        if confidence_multiplier is not None:
            bias_profile.confidence_multiplier = max(0.5, min(1.5, confidence_multiplier))
        
        bias_profile.timestamp = time.time()
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get emotion weight status.
        
        Returns
        -------
        Dict[str, Any]
            Status information
        """
        return {
            "bias_profiles": len(self.bias_map),
            "bias_history_length": len(self.bias_history),
            "bias_magnitudes": {
                emotion: profile.magnitude 
                for emotion, profile in self.bias_map.items()
            },
            "confidence_multipliers": {
                emotion: profile.confidence_multiplier 
                for emotion, profile in self.bias_map.items()
            },
        }
    
    def compute_risk_aversion(self, current_emotions):
        """Compute risk aversion based on emotions."""
        pass
