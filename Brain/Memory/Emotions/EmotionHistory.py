"""
EmotionHistory.py
History of emotional states over time for analysis and trends.
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import time
from statistics import mean, stdev


class EmotionHistory:
    """Tracks emotional history over time."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize emotion history.
        
        Parameters
        ----------
        max_history : int
            Maximum history entries to keep
        """
        self.history: deque = deque(maxlen=max_history)
        self.max_history = max_history
        self.timestamps: deque = deque(maxlen=max_history)
        self.first_recorded_at = None
    
    def record_emotion_state(self, emotion_state: Dict[str, float]) -> None:
        """Record current emotional state.
        
        Parameters
        ----------
        emotion_state : Dict[str, float]
            Current emotions and intensities
        """
        timestamp = time.time()
        
        if self.first_recorded_at is None:
            self.first_recorded_at = timestamp
        
        self.history.append(emotion_state.copy())
        self.timestamps.append(timestamp)
    
    def get_emotional_trend(self, emotion: str, window_size: int = 10) -> Dict[str, Any]:
        """Get trend for a specific emotion.
        
        Parameters
        ----------
        emotion : str
            Emotion name
        window_size : int
            Number of recent samples to analyze
            
        Returns
        -------
        Dict[str, Any]
            Trend information
        """
        if not self.history:
            return {"trend": "none", "current": 0.0, "history_length": 0}
        
        recent = list(self.history)[-window_size:]
        values = [state.get(emotion, 0.0) for state in recent]
        
        if len(values) < 2:
            return {
                "trend": "stable",
                "current": values[0] if values else 0.0,
                "history_length": len(values)
            }
        
        # Calculate trend
        current_mean = mean(values[-5:]) if len(values) >= 5 else mean(values)
        previous_mean = mean(values[:5]) if len(values) >= 10 else mean(values[:-1])
        
        delta = current_mean - previous_mean
        
        if abs(delta) < 0.05:
            trend = "stable"
        elif delta > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        return {
            "emotion": emotion,
            "trend": trend,
            "current": values[-1],
            "mean": mean(values),
            "stdev": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "delta": delta,
            "history_length": len(values)
        }
    
    def get_all_trends(self, window_size: int = 10) -> Dict[str, Dict[str, Any]]:
        """Get trends for all emotions.
        
        Parameters
        ----------
        window_size : int
            Number of recent samples
            
        Returns
        -------
        Dict[str, Dict[str, Any]]
            Trends for all emotions
        """
        if not self.history:
            return {}
        
        # Get all unique emotions
        all_emotions = set()
        for state in self.history:
            all_emotions.update(state.keys())
        
        trends = {}
        for emotion in all_emotions:
            trends[emotion] = self.get_emotional_trend(emotion, window_size)
        
        return trends
    
    def get_dominant_emotion_history(self, window_size: int = 10) -> List[Tuple[str, float]]:
        """Get sequence of dominant emotions.
        
        Parameters
        ----------
        window_size : int
            Number of recent samples
            
        Returns
        -------
        List[Tuple[str, float]]
            Sequence of (emotion, intensity)
        """
        if not self.history:
            return []
        
        recent = list(self.history)[-window_size:]
        dominant_sequence = []
        
        for state in recent:
            if state:
                dominant = max(state.items(), key=lambda x: x[1])
                dominant_sequence.append(dominant)
        
        return dominant_sequence
    
    def get_emotional_stability(self) -> float:
        """Get measure of emotional stability.
        
        Returns
        -------
        float
            Stability score (0.0-1.0, higher = more stable)
        """
        if not self.history or len(self.history) < 2:
            return 0.5
        
        # Get all emotions and their standard deviations
        all_emotions = set()
        for state in self.history:
            all_emotions.update(state.keys())
        
        if not all_emotions:
            return 0.5
        
        # Calculate average variability
        variabilities = []
        for emotion in all_emotions:
            values = [state.get(emotion, 0.0) for state in self.history]
            if len(values) > 1:
                variabilities.append(stdev(values))
        
        if not variabilities:
            return 0.5
        
        avg_variability = mean(variabilities)
        
        # Stability is inverse of variability
        stability = 1.0 - min(1.0, avg_variability)
        
        return stability
    
    def get_emotional_intensity_average(self) -> float:
        """Get average emotional intensity.
        
        Returns
        -------
        float
            Average intensity (0.0-1.0)
        """
        if not self.history:
            return 0.0
        
        all_values = []
        for state in self.history:
            all_values.extend(state.values())
        
        if not all_values:
            return 0.0
        
        return mean(all_values)
    
    def get_status(self) -> Dict[str, Any]:
        """Get history status.
        
        Returns
        -------
        Dict[str, Any]
            Status information
        """
        history_duration = 0.0
        if self.timestamps:
            history_duration = self.timestamps[-1] - self.timestamps[0]
        
        return {
            "max_history": self.max_history,
            "current_entries": len(self.history),
            "utilization_percent": (len(self.history) / self.max_history * 100) if self.max_history > 0 else 0,
            "history_duration_seconds": history_duration,
            "emotional_stability": self.get_emotional_stability(),
            "average_intensity": self.get_emotional_intensity_average(),
            "recording_started_at": self.first_recorded_at,
        }
        """Get trend in emotions over recent history."""
        pass
    
    def get_emotion_at_time(self, timestamp):
        """Get emotional state at a specific time."""
        pass
    
    def analyze_emotional_patterns(self):
        """Analyze patterns in emotional history."""
        pass
