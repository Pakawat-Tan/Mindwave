"""
EmotionState.py
Represents current emotional state of the system.
"""

from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
import time


@dataclass
class EmotionIntensity:
    """Represents an emotion's current state"""
    emotion_name: str
    intensity: float  # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    delta_rate: float = 0.0  # change per second
    source: str = ""  # what triggered this emotion


class EmotionState:
    """Manages the system's current emotional state."""
    
    def __init__(self):
        """Initialize emotion state."""
        self.emotions: Dict[str, EmotionIntensity] = self._initialize_emotions()
        self.valence = 0.0  # positive to negative (-1 to 1)
        self.arousal = 0.0  # calm to excited (0 to 1)
        self.state_updated_at = time.time()
        self.emotion_history: List[Tuple[float, Dict[str, float]]] = []
    
    def _initialize_emotions(self) -> Dict[str, EmotionIntensity]:
        """Initialize basic emotions.
        
        Returns
        -------
        Dict[str, EmotionIntensity]
            Initial emotions
        """
        emotions = {
            "joy": EmotionIntensity("joy", 0.5),
            "sadness": EmotionIntensity("sadness", 0.3),
            "anger": EmotionIntensity("anger", 0.2),
            "fear": EmotionIntensity("fear", 0.3),
            "trust": EmotionIntensity("trust", 0.6),
            "disgust": EmotionIntensity("disgust", 0.2),
            "surprise": EmotionIntensity("surprise", 0.4),
            "anticipation": EmotionIntensity("anticipation", 0.5),
        }
        return emotions
    
    def update_emotion(self, emotion_name: str, intensity: float, 
                      source: str = "") -> bool:
        """Update an emotion's intensity.
        
        Parameters
        ----------
        emotion_name : str
            Name of the emotion
        intensity : float
            New intensity (0.0-1.0)
        source : str
            What triggered this emotion
            
        Returns
        -------
        bool
            Success
        """
        intensity = max(0.0, min(1.0, intensity))
        
        if emotion_name in self.emotions:
            old_emotion = self.emotions[emotion_name]
            old_intensity = old_emotion.intensity
            
            # Calculate delta rate
            time_diff = time.time() - old_emotion.timestamp
            if time_diff > 0:
                delta_rate = (intensity - old_intensity) / time_diff
            else:
                delta_rate = 0.0
            
            self.emotions[emotion_name] = EmotionIntensity(
                emotion_name=emotion_name,
                intensity=intensity,
                delta_rate=delta_rate,
                source=source
            )
            
            # Record in history
            self.emotion_history.append((time.time(), {
                name: emo.intensity for name, emo in self.emotions.items()
            }))
            
            # Update valence and arousal
            self._update_valence_arousal()
            self.state_updated_at = time.time()
            
            return True
        
        return False
    
    def _update_valence_arousal(self) -> None:
        """Update valence and arousal based on current emotions."""
        # Valence: positive emotions increase it, negative emotions decrease it
        positive_emotions = ["joy", "trust"]
        negative_emotions = ["sadness", "anger", "fear", "disgust"]
        
        positive_val = sum(self.emotions.get(e, EmotionIntensity(e, 0.0)).intensity 
                          for e in positive_emotions) / len(positive_emotions)
        negative_val = sum(self.emotions.get(e, EmotionIntensity(e, 0.0)).intensity 
                          for e in negative_emotions) / len(negative_emotions)
        
        self.valence = positive_val - negative_val  # Range -1 to 1
        
        # Arousal: high emotion intensity = high arousal
        high_arousal_emotions = ["joy", "anger", "fear", "surprise"]
        self.arousal = sum(self.emotions.get(e, EmotionIntensity(e, 0.0)).intensity 
                          for e in high_arousal_emotions) / len(high_arousal_emotions)
    
    def get_emotion(self, emotion_name: str) -> Optional[float]:
        """Get an emotion's intensity.
        
        Parameters
        ----------
        emotion_name : str
            Name of the emotion
            
        Returns
        -------
        Optional[float]
            Emotion intensity or None
        """
        if emotion_name in self.emotions:
            return self.emotions[emotion_name].intensity
        return None
    
    def get_all_emotions(self) -> Dict[str, float]:
        """Get all emotions.
        
        Returns
        -------
        Dict[str, float]
            All emotions and intensities
        """
        return {
            name: emo.intensity 
            for name, emo in self.emotions.items()
        }
    
    def get_dominant_emotion(self) -> Optional[Tuple[str, float]]:
        """Get the dominant emotion.
        
        Returns
        -------
        Optional[Tuple[str, float]]
            (emotion_name, intensity) or None
        """
        if not self.emotions:
            return None
        
        dominant = max(self.emotions.items(), 
                      key=lambda x: x[1].intensity)
        return (dominant[0], dominant[1].intensity)
    
    def get_valence_arousal(self) -> Tuple[float, float]:
        """Get valence and arousal coordinates.
        
        Returns
        -------
        Tuple[float, float]
            (valence, arousal) where valence is -1 to 1, arousal is 0 to 1
        """
        return (self.valence, self.arousal)
    
    def get_emotional_momentum(self, emotion_name: str) -> float:
        """Get the emotional momentum (change rate).
        
        Parameters
        ----------
        emotion_name : str
            Emotion name
            
        Returns
        -------
        float
            Change rate of emotion
        """
        if emotion_name in self.emotions:
            return self.emotions[emotion_name].delta_rate
        return 0.0
    
    def apply_emotion_decay(self, decay_rate: float = 0.01) -> None:
        """Apply decay to emotions over time.
        
        Parameters
        ----------
        decay_rate : float
            Decay rate per update (0.0-1.0)
        """
        for emotion_name in self.emotions:
            current = self.emotions[emotion_name].intensity
            # Decay towards baseline
            baseline = 0.5  # neutral baseline
            decayed = current + (baseline - current) * decay_rate
            self.emotions[emotion_name].intensity = max(0.0, min(1.0, decayed))
        
        self._update_valence_arousal()
    
    def reset_emotions(self, reset_to_baseline: bool = True) -> None:
        """Reset emotional state.
        
        Parameters
        ----------
        reset_to_baseline : bool
            Reset to baseline (0.5) or neutral (0.0)
        """
        baseline = 0.5 if reset_to_baseline else 0.3
        
        for emotion_name in self.emotions:
            self.emotions[emotion_name] = EmotionIntensity(
                emotion_name=emotion_name,
                intensity=baseline,
                delta_rate=0.0
            )
        
        self._update_valence_arousal()
        self.state_updated_at = time.time()
    
    def get_status(self) -> Dict[str, any]:
        """Get emotion state status.
        
        Returns
        -------
        Dict[str, Any]
            Status information
        """
        return {
            "state_updated_at": self.state_updated_at,
            "emotions": self.get_all_emotions(),
            "dominant_emotion": self.get_dominant_emotion(),
            "valence": self.valence,
            "arousal": self.arousal,
            "history_length": len(self.emotion_history),
        }
