"""
LearningEngine.py
ตัวประสาน learner ทั้งหมด
Central engine coordinating all learning mechanisms.
"""

from typing import Dict, Any


class LearningEngine:
    """Central coordinator for all learning mechanisms."""

    def __init__(self):
        """Initialize learning engine."""
        self.learners: Dict[str, Any] = {}
        self.learning_schedule = {}
        self.performance_metrics = {
            "loss": [],
            "accuracy": [],
            "confidence": [],
        }
        self.current_mode = None

    # =====================================================
    # Registration
    # =====================================================
    def register_learner(self, learner_name, learner_instance):
        """
        Register a learning mechanism.

        Example:
        register_learner("gradient", GradientLearner())
        """
        self.learners[learner_name] = learner_instance

    # =====================================================
    # Mode selection
    # =====================================================
    def select_learning_mode(self, context):
        """
        Select appropriate learning mode based on context.

        context examples:
        - high_loss
        - advisor_feedback
        - stagnation
        - exploration
        """
        if context.get("external_feedback"):
            self.current_mode = "advisor"

        elif context.get("loss_trend", 0) > 0:
            self.current_mode = "gradient"

        elif context.get("stagnant"):
            self.current_mode = "evolution"

        else:
            self.current_mode = "gradient"

        return self.current_mode

    # =====================================================
    # Learning coordination
    # =====================================================
    def coordinate_learning(self, training_data):
        """
        Coordinate learning across all mechanisms.

        training_data structure is intentionally abstract.
        """
        if self.current_mode is None:
            raise RuntimeError("Learning mode not selected")

        learner = self.learners.get(self.current_mode)

        if learner is None:
            raise RuntimeError(f"Learner '{self.current_mode}' not registered")

        # Dispatch learning
        if hasattr(learner, "learn"):
            result = learner.learn(training_data)
        elif hasattr(learner, "update_weights"):
            result = learner.update_weights(
                training_data["weights"],
                training_data["gradients"],
            )
        else:
            result = None

        return result

    # =====================================================
    # Progress & metrics
    # =====================================================
    def get_learning_progress(self):
        """Get overall learning progress."""
        return {
            "mode": self.current_mode,
            "metrics": self.performance_metrics,
            "registered_learners": list(self.learners.keys()),
        }

    def update_performance(self, *, loss=None, accuracy=None, confidence=None):
        """Update performance metrics."""
        if loss is not None:
            self.performance_metrics["loss"].append(loss)
        if accuracy is not None:
            self.performance_metrics["accuracy"].append(accuracy)
        if confidence is not None:
            self.performance_metrics["confidence"].append(confidence)

    # =====================================================
    # Adaptive learning
    # =====================================================
    def adapt_learning_rate(self):
        """
        Adapt learning rates based on recent performance.
        """
        if not self.performance_metrics["loss"]:
            return

        recent_loss = self.performance_metrics["loss"][-5:]

        if len(recent_loss) < 2:
            return

        loss_trend = recent_loss[-1] - recent_loss[0]

        for learner in self.learners.values():
            if hasattr(learner, "learning_rate"):
                if loss_trend > 0:
                    learner.learning_rate *= 0.7
                else:
                    learner.learning_rate *= 1.05
