"""
EmotionProfile.py
Emotional personality profile and characteristics.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import time


@dataclass
class EmotionalTrait:
    """Represents an emotional personality trait"""
    trait_name: str
    value: float  # 0.0 to 1.0
    baseline: float = 0.5
    variability: float = 0.2
    updated_at: float = field(default_factory=time.time)


class EmotionProfile:
    """Manages emotional personality profile."""
    
    def __init__(self):
        """Initialize emotion profile."""
        self.personality_traits: Dict[str, EmotionalTrait] = {}
        self.emotional_baseline: Dict[str, float] = self._initialize_baselines()
        self.emotional_range: Dict[str, tuple] = self._initialize_ranges()
        self.profile_created_at = time.time()
    
    def _initialize_baselines(self) -> Dict[str, float]:
        """Initialize baseline emotional states.
        
        Returns
        -------
        Dict[str, float]
            Baseline values for emotions
        """
        return {
            "joy": 0.5,
            "sadness": 0.3,
            "anger": 0.2,
            "fear": 0.3,
            "trust": 0.6,
            "disgust": 0.2,
            "surprise": 0.4,
            "anticipation": 0.5,
        }
    
    def _initialize_ranges(self) -> Dict[str, tuple]:
        """Initialize emotional ranges.
        
        Returns
        -------
        Dict[str, tuple]
            (min, max) ranges for emotions
        """
        return {
            "joy": (0.0, 1.0),
            "sadness": (0.0, 1.0),
            "anger": (0.0, 1.0),
            "fear": (0.0, 1.0),
            "trust": (0.0, 1.0),
            "disgust": (0.0, 1.0),
            "surprise": (0.0, 1.0),
            "anticipation": (0.0, 1.0),
        }
    
    def set_trait(self, trait_name: str, value: float, 
                 baseline: Optional[float] = None, variability: Optional[float] = None) -> bool:
        """Set an emotional personality trait.
        
        Parameters
        ----------
        trait_name : str
            Name of the trait
        value : float
            Trait value (0.0-1.0)
        baseline : Optional[float]
            New baseline for this trait
        variability : Optional[float]
            Trait variability
            
        Returns
        -------
        bool
            Success
        """
        value = max(0.0, min(1.0, value))
        
        if trait_name in self.personality_traits:
            trait = self.personality_traits[trait_name]
            trait.value = value
            if baseline is not None:
                trait.baseline = max(0.0, min(1.0, baseline))
            if variability is not None:
                trait.variability = max(0.0, min(1.0, variability))
            trait.updated_at = time.time()
        else:
            trait = EmotionalTrait(
                trait_name=trait_name,
                value=value,
                baseline=baseline or 0.5,
                variability=variability or 0.2
            )
            self.personality_traits[trait_name] = trait
        
        return True
    
    def get_trait(self, trait_name: str) -> Optional[float]:
        """Get a trait value.
        
        Parameters
        ----------
        trait_name : str
            Name of the trait
            
        Returns
        -------
        Optional[float]
            Trait value or None
        """
        if trait_name in self.personality_traits:
            return self.personality_traits[trait_name].value
        return None
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get summary of emotional profile.
        
        Returns
        -------
        Dict[str, Any]
            Profile summary
        """
        return {
            "profile_created_at": self.profile_created_at,
            "num_traits": len(self.personality_traits),
            "traits": {
                name: {
                    "value": trait.value,
                    "baseline": trait.baseline,
                    "variability": trait.variability
                }
                for name, trait in self.personality_traits.items()
            },
            "emotional_baselines": self.emotional_baseline,
        }
    
    def predict_emotional_response(self, stimulus_type: str, 
                                  stimulus_intensity: float) -> Dict[str, float]:
        """Predict emotional response to stimulus.
        
        Parameters
        ----------
        stimulus_type : str
            Type of stimulus (e.g., "threat", "reward", "loss")
        stimulus_intensity : float
            Intensity of stimulus (0.0-1.0)
            
        Returns
        -------
        Dict[str, float]
            Predicted emotional responses
        """
        responses = {}
        
        # Base responses depend on stimulus type
        if stimulus_type == "threat":
            responses = {
                "fear": 0.8 * stimulus_intensity,
                "anger": 0.5 * stimulus_intensity,
                "anticipation": 0.3 * stimulus_intensity,
                "trust": -0.3 * stimulus_intensity,
            }
        elif stimulus_type == "reward":
            responses = {
                "joy": 0.8 * stimulus_intensity,
                "trust": 0.6 * stimulus_intensity,
                "anticipation": 0.5 * stimulus_intensity,
            }
        elif stimulus_type == "loss":
            responses = {
                "sadness": 0.7 * stimulus_intensity,
                "anger": 0.5 * stimulus_intensity,
                "fear": 0.4 * stimulus_intensity,
                "trust": -0.4 * stimulus_intensity,
            }
        elif stimulus_type == "novel":
            responses = {
                "surprise": 0.7 * stimulus_intensity,
                "anticipation": 0.6 * stimulus_intensity,
                "fear": 0.2 * stimulus_intensity,
            }
        
        # Modulate by personality traits
        for emotion, response in responses.items():
            trait_name = emotion
            if trait_name in self.personality_traits:
                trait = self.personality_traits[trait_name]
                # Traits modulate the intensity
                responses[emotion] = response * (trait.value / 0.5)  # normalize at 0.5
        
        # Clamp to valid range
        return {
            emotion: max(0.0, min(1.0, response + self.emotional_baseline.get(emotion, 0.5)))
            for emotion, response in responses.items()
        }
    
    def evolve_profile(self, experiences: List[Dict[str, Any]]) -> None:
        """Evolve profile based on experiences.
        
        Parameters
        ----------
        experiences : List[Dict[str, Any]]
            List of experiences with emotional outcomes
        """
        if not experiences:
            return
        
        # Calculate experience-weighted trait changes
        for experience in experiences:
            if "outcome" in experience and "emotions" in experience:
                outcome = experience["outcome"]
                emotions = experience["emotions"]
                weight = experience.get("weight", 1.0)
                
                for emotion, intensity in emotions.items():
                    if emotion in self.personality_traits:
                        trait = self.personality_traits[emotion]
                        # Shift baseline towards experienced emotions
                        baseline_shift = (intensity - trait.baseline) * weight * 0.05
                        trait.baseline = max(0.0, min(1.0, trait.baseline + baseline_shift))
    
    def get_status(self) -> Dict[str, Any]:
        """Get profile status.
        
        Returns
        -------
        Dict[str, Any]
            Status information
        """
        return {
            "total_traits": len(self.personality_traits),
            "profile_age_seconds": time.time() - self.profile_created_at,
            "trait_summary": self.get_profile_summary(),
            "emotional_baselines": self.emotional_baseline,
        }
