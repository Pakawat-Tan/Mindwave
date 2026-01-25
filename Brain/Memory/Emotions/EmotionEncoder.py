"""
EmotionEncoder.py
Encodes emotions as vectors for processing and similarity computation.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class EmotionVector:
    """Represents an encoded emotion"""
    vector: np.ndarray
    emotion_state: Dict[str, float]
    norm: float = 0.0
    timestamp: float = 0.0


class EmotionEncoder:
    """Encodes emotions as vectors for neural processing."""
    
    def __init__(self, vector_dim: int = 32, emotion_names: Optional[List[str]] = None):
        """Initialize emotion encoder.
        
        Parameters
        ----------
        vector_dim : int
            Dimension of emotion vectors
        emotion_names : Optional[List[str]]
            Names of emotions (default: standard set)
        """
        self.vector_dim = vector_dim
        self.emotion_names = emotion_names or [
            "joy", "sadness", "anger", "fear",
            "trust", "disgust", "surprise", "anticipation"
        ]
        self.num_emotions = len(self.emotion_names)
        self.emotion_vectors: Dict[str, np.ndarray] = {}
        self._initialize_emotion_vectors()
    
    def _initialize_emotion_vectors(self) -> None:
        """Initialize basis vectors for each emotion."""
        np.random.seed(42)  # For reproducibility
        
        for emotion in self.emotion_names:
            # Create a random basis vector for each emotion
            vector = np.random.randn(self.vector_dim) * 0.5
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm
            self.emotion_vectors[emotion] = vector
    
    def encode_emotion(self, emotion_state: Dict[str, float]) -> np.ndarray:
        """Encode emotional state as a vector.
        
        Parameters
        ----------
        emotion_state : Dict[str, float]
            Emotional state with intensities
            
        Returns
        -------
        np.ndarray
            Encoded emotion vector
        """
        vector = np.zeros(self.vector_dim)
        
        for emotion, intensity in emotion_state.items():
            if emotion in self.emotion_vectors:
                vector += self.emotion_vectors[emotion] * intensity
        
        # Also add multi-dimensional representations
        # Valence dimension
        positive_emotions = ["joy", "trust", "anticipation"]
        negative_emotions = ["sadness", "anger", "fear", "disgust"]
        
        positive_intensity = sum(emotion_state.get(e, 0.0) for e in positive_emotions) / len(positive_emotions)
        negative_intensity = sum(emotion_state.get(e, 0.0) for e in negative_emotions) / len(negative_emotions)
        
        valence_vector = positive_intensity - negative_intensity
        
        # Arousal dimension
        high_arousal_emotions = ["joy", "anger", "fear", "surprise"]
        arousal_intensity = sum(emotion_state.get(e, 0.0) for e in high_arousal_emotions) / len(high_arousal_emotions)
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        
        return vector
    
    def encode_valence_arousal(self, valence: float, arousal: float) -> np.ndarray:
        """Encode valence-arousal space as vector.
        
        Parameters
        ----------
        valence : float
            Valence (-1 to 1)
        arousal : float
            Arousal (0 to 1)
            
        Returns
        -------
        np.ndarray
            Encoded vector
        """
        vector = np.zeros(self.vector_dim)
        
        # Map 2D valence-arousal to vector space
        positive_emotions = ["joy", "trust", "anticipation"]
        negative_emotions = ["sadness", "anger", "fear", "disgust"]
        high_arousal_emotions = ["joy", "anger", "fear", "surprise"]
        low_arousal_emotions = ["sadness", "trust"]
        
        intensity_map = {}
        
        # Quadrant 1: positive-high arousal (joy, excitement)
        if valence > 0 and arousal > 0.5:
            intensity_map["joy"] = valence * arousal
            intensity_map["anticipation"] = valence * arousal * 0.5
        
        # Quadrant 2: negative-high arousal (anger, fear)
        if valence < 0 and arousal > 0.5:
            intensity_map["anger"] = -valence * arousal * 0.7
            intensity_map["fear"] = -valence * arousal * 0.3
        
        # Quadrant 3: negative-low arousal (sadness)
        if valence < 0 and arousal <= 0.5:
            intensity_map["sadness"] = -valence * (1 - arousal)
        
        # Quadrant 4: positive-low arousal (trust, contentment)
        if valence > 0 and arousal <= 0.5:
            intensity_map["trust"] = valence * (1 - arousal)
        
        # Encode
        for emotion, intensity in intensity_map.items():
            if emotion in self.emotion_vectors:
                vector += self.emotion_vectors[emotion] * max(0.0, min(1.0, intensity))
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        
        return vector
    
    def decode_vector(self, vector: np.ndarray) -> Dict[str, float]:
        """Decode a vector back to emotional state.
        
        Parameters
        ----------
        vector : np.ndarray
            Encoded emotion vector
            
        Returns
        -------
        Dict[str, float]
            Reconstructed emotion state
        """
        emotion_state = {}
        
        for emotion in self.emotion_names:
            if emotion in self.emotion_vectors:
                # Compute dot product as intensity
                intensity = np.dot(vector, self.emotion_vectors[emotion])
                # Clamp to 0-1 range
                emotion_state[emotion] = max(0.0, min(1.0, intensity))
        
        # Normalize so they sum to 1
        total = sum(emotion_state.values())
        if total > 0:
            emotion_state = {e: v / total for e, v in emotion_state.items()}
        
        return emotion_state
    
    def compute_similarity(self, emotion1: Dict[str, float], 
                          emotion2: Dict[str, float]) -> float:
        """Compute similarity between two emotional states.
        
        Parameters
        ----------
        emotion1 : Dict[str, float]
            First emotion state
        emotion2 : Dict[str, float]
            Second emotion state
            
        Returns
        -------
        float
            Similarity (0.0-1.0)
        """
        vector1 = self.encode_emotion(emotion1)
        vector2 = self.encode_emotion(emotion2)
        
        # Cosine similarity
        similarity = np.dot(vector1, vector2)
        
        # Clamp to valid range
        return max(0.0, min(1.0, similarity))
    
    def compute_vector_similarity(self, vector1: np.ndarray, 
                                 vector2: np.ndarray) -> float:
        """Compute similarity between emotion vectors.
        
        Parameters
        ----------
        vector1 : np.ndarray
            First vector
        vector2 : np.ndarray
            Second vector
            
        Returns
        -------
        float
            Similarity (0.0-1.0)
        """
        similarity = np.dot(vector1, vector2)
        return max(0.0, min(1.0, similarity))
    
    def find_nearest_emotions(self, emotion_state: Dict[str, float], 
                            top_k: int = 3) -> List[Tuple[str, float]]:
        """Find emotionally similar states.
        
        Parameters
        ----------
        emotion_state : Dict[str, float]
            Query emotion state
        top_k : int
            Number of similar emotions to return
            
        Returns
        -------
        List[Tuple[str, float]]
            (emotion_name, similarity) pairs
        """
        query_vector = self.encode_emotion(emotion_state)
        similarities = []
        
        for emotion in self.emotion_names:
            emotion_vec = self.emotion_vectors.get(emotion)
            if emotion_vec is not None:
                sim = np.dot(query_vector, emotion_vec)
                similarities.append((emotion, max(0.0, min(1.0, sim))))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_emotion_distance_matrix(self) -> Dict[Tuple[str, str], float]:
        """Get pairwise distances between basic emotions.
        
        Returns
        -------
        Dict[Tuple[str, str], float]
            Pairwise distances
        """
        distances = {}
        
        for i, emotion1 in enumerate(self.emotion_names):
            for emotion2 in self.emotion_names[i+1:]:
                vec1 = self.emotion_vectors.get(emotion1)
                vec2 = self.emotion_vectors.get(emotion2)
                
                if vec1 is not None and vec2 is not None:
                    # Euclidean distance
                    distance = np.linalg.norm(vec1 - vec2)
                    distances[(emotion1, emotion2)] = distance
        
        return distances
    
    def get_status(self) -> Dict[str, Any]:
        """Get encoder status.
        
        Returns
        -------
        Dict[str, Any]
            Status information
        """
        return {
            "vector_dim": self.vector_dim,
            "num_emotions": self.num_emotions,
            "emotion_names": self.emotion_names,
            "vector_magnitude": float(np.mean([np.linalg.norm(v) for v in self.emotion_vectors.values()])),
        }
        """Compute similarity between two emotions."""
        pass
    
    def blend_emotions(self, emotions, weights):
        """Blend multiple emotions with weights."""
        pass
