"""
AdvisorLearner.py
เรียนจากคำแนะนำภายนอก
Learning from external feedback and advice.
"""

from datetime import datetime
import numpy as np


class AdvisorLearner:
    """Learning from external advisors and feedback."""

    def __init__(self):
        """Initialize advisor learner."""
        self.advisors = {}               # advisor_id -> advisor info
        self.feedback_history = []       # chronological feedback log
        self.advice_quality = {}         # advisor_id -> reliability score (0-1)

    # =====================================================
    # Advisor management
    # =====================================================
    def register_advisor(self, advisor_id, advisor=None):
        """Register an external advisor."""
        self.advisors[advisor_id] = advisor
        self.advice_quality.setdefault(advisor_id, 0.5)  # neutral trust
        return True

    # =====================================================
    # Feedback intake
    # =====================================================
    def receive_feedback(self, advisor_id, feedback, target):
        """
        Receive feedback from advisor.

        feedback example:
        {
            "type": "correction" | "reward" | "penalty" | "suggestion",
            "value": float or dict,
            "confidence": float (0-1)
        }
        """
        entry = {
            "advisor_id": advisor_id,
            "feedback": feedback,
            "target": target,
            "timestamp": datetime.utcnow().isoformat()
        }

        self.feedback_history.append(entry)

        # Update advisor quality incrementally
        self._update_advisor_quality(advisor_id, feedback)

        return entry

    # =====================================================
    # Advisor evaluation
    # =====================================================
    def evaluate_advisor_quality(self, advisor_id):
        """Evaluate reliability of an advisor."""
        return self.advice_quality.get(advisor_id, 0.0)

    def _update_advisor_quality(self, advisor_id, feedback):
        """Internal trust update logic."""
        confidence = feedback.get("confidence", 0.5)
        ftype = feedback.get("type")

        delta = 0.0
        if ftype in ("reward", "correction"):
            delta = +0.05 * confidence
        elif ftype == "penalty":
            delta = -0.05 * confidence

        self.advice_quality[advisor_id] = float(
            np.clip(self.advice_quality.get(advisor_id, 0.5) + delta, 0.0, 1.0)
        )

    # =====================================================
    # Learning application
    # =====================================================
    def apply_advice_to_weights(self, advice, confidence=1.0):
        """
        Convert advice into a weight update signal.

        advice example:
        {
            "weight_name": "layer1.W",
            "direction": +1 | -1,
            "magnitude": float
        }
        """
        scaled_update = {
            "weight": advice.get("weight_name"),
            "delta": advice.get("direction", 0)
                     * advice.get("magnitude", 0.0)
                     * confidence
        }

        return scaled_update

    def learn_from_corrections(self, correction):
        """
        Learn from explicit corrections.

        correction example:
        {
            "expected": any,
            "actual": any,
            "error": float,
            "confidence": float
        }
        """
        learning_signal = {
            "error_signal": correction.get("error", 0.0),
            "confidence": correction.get("confidence", 1.0),
            "timestamp": datetime.utcnow().isoformat()
        }

        self.feedback_history.append({
            "advisor_id": "correction",
            "feedback": learning_signal,
            "target": correction.get("target")
        })

        return learning_signal
