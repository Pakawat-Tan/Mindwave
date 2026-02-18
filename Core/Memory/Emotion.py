"""
Emotion Data Module

Emotion representation based on the VAD (Valence-Arousal-Dominance) model.
Values are produced by an unsupervised model that analyzes sentence context
and clusters emotional characteristics — not manually assigned.

Dimensions:
    valence   : -1.0 (very negative) → +1.0 (very positive)
    arousal   : 0.0 (calm) → 1.0 (highly activated)
    dominance : 0.0 (submissive) → 1.0 (dominant/in-control)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import json
import math


# ============================================================================
# TENDENCY REGIONS — derived from VAD space, not manually assigned
# ============================================================================

# Each entry: (label, v_min, v_max, a_min, a_max, d_min, d_max)
# None means "no constraint on that axis"

_TENDENCY_REGIONS = [
    ("excited",   0.3,  1.0,  0.6, 1.0, None, None),
    ("relaxed",   0.3,  1.0,  0.0, 0.3, None, None),
    ("happy",     0.3,  1.0,  0.3, 0.6, None, None),
    ("content",   0.1,  1.0,  0.0, 0.4, None, None),
    ("angry",    -1.0, -0.3,  0.6, 1.0,  0.5,  1.0),
    ("stressed", -1.0, -0.3,  0.6, 1.0,  0.0,  0.5),
    ("anxious",  -1.0, -0.3,  0.4, 0.6, None, None),
    ("sad",      -1.0, -0.3,  0.0, 0.4, None, None),
    ("bored",    -0.3,  0.3,  0.0, 0.3, None, None),
    ("neutral",  -0.3,  0.3,  0.3, 0.7, None, None),
]


def _in_range(value: float, lo: Optional[float], hi: Optional[float]) -> bool:
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


# ============================================================================
# EMOTION DATA
# ============================================================================

@dataclass
class EmotionData:
    """
    Emotion as a point in 3D VAD space.

    Produced by unsupervised model — never manually assigned.
    The `tendency` label is derived automatically from coordinates.

    Attributes:
        valence    : Pleasantness  (-1.0 = very negative, +1.0 = very positive)
        arousal    : Activation    (0.0 = calm, 1.0 = highly excited)
        dominance  : Control level (0.0 = submissive, 1.0 = dominant)
        confidence : Model confidence (0.0–1.0)
        context    : Raw context the model analyzed
    """
    valence:    float
    arousal:    float
    dominance:  float
    confidence: float = 0.5
    context:    str   = ""

    def __post_init__(self) -> None:
        self.valence    = max(-1.0, min(1.0, self.valence))
        self.arousal    = max( 0.0, min(1.0, self.arousal))
        self.dominance  = max( 0.0, min(1.0, self.dominance))
        self.confidence = max( 0.0, min(1.0, self.confidence))

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def tendency(self) -> str:
        """Label derived from VAD coordinates. Returns 'undefined' if no region matches."""
        for label, v_min, v_max, a_min, a_max, d_min, d_max in _TENDENCY_REGIONS:
            if (
                _in_range(self.valence,   v_min, v_max) and
                _in_range(self.arousal,   a_min, a_max) and
                _in_range(self.dominance, d_min, d_max)
            ):
                return label
        return "undefined"

    @property
    def is_positive(self) -> bool:
        return self.valence > 0.1

    @property
    def is_negative(self) -> bool:
        return self.valence < -0.1

    @property
    def is_neutral(self) -> bool:
        return abs(self.valence) <= 0.1

    @property
    def intensity(self) -> float:
        """Euclidean distance from neutral origin, normalized to [0, 1]."""
        raw = math.sqrt(self.valence**2 + self.arousal**2 + self.dominance**2)
        return min(1.0, raw / math.sqrt(3.0))

    @property
    def vad(self) -> Tuple[float, float, float]:
        return (self.valence, self.arousal, self.dominance)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valence":    self.valence,
            "arousal":    self.arousal,
            "dominance":  self.dominance,
            "confidence": self.confidence,
            "context":    self.context,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmotionData":
        return cls(
            valence    = data["valence"],
            arousal    = data["arousal"],
            dominance  = data["dominance"],
            confidence = data.get("confidence", 0.5),
            context    = data.get("context", ""),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "EmotionData":
        return cls.from_dict(json.loads(json_str))

    def __str__(self) -> str:
        return (
            f"EmotionData("
            f"V={self.valence:+.2f}, "
            f"A={self.arousal:.2f}, "
            f"D={self.dominance:.2f}, "
            f"tendency='{self.tendency}', "
            f"conf={self.confidence:.2f})"
        )


# ============================================================================
# FACTORY
# ============================================================================

def create_emotion(
    valence:    float = 0.0,
    arousal:    float = 0.0,
    dominance:  float = 0.5,
    confidence: float = 0.5,
    context:    str   = "",
) -> EmotionData:
    """Factory — in production, values come from unsupervised model output."""
    return EmotionData(
        valence=valence, arousal=arousal, dominance=dominance,
        confidence=confidence, context=context,
    )


NEUTRAL_EMOTION = EmotionData(valence=0.0, arousal=0.0, dominance=0.5, confidence=0.0)