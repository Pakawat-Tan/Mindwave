"""
SelfLearner.py
เรียนรู้จาก introspection
Learning from self-reflection and introspection.
"""

from typing import Any, Dict, List


class SelfLearner:
    """Learning through self-reflection and analysis."""

    def __init__(self):
        """Initialize self-learner."""
        self.reflections: List[Dict[str, Any]] = []
        self.insights: Dict[str, Any] = {}

    # =====================================================
    # Reflection
    # =====================================================
    def reflect_on_decision(self, decision_id):
        """
        Reflect on a past decision.
        """
        reflection = {
            "decision_id": decision_id,
            "observations": [],
            "issues": [],
            "successes": [],
        }
        self.reflections.append(reflection)
        return reflection

    # =====================================================
    # Insight extraction
    # =====================================================
    def extract_insight(self, reflection: Dict[str, Any]):
        """
        Extract learning insights from reflection.
        """
        insight = {
            "decision_id": reflection.get("decision_id"),
            "strengths": reflection.get("successes", []),
            "weaknesses": reflection.get("issues", []),
            "recommendations": [],
        }

        # simple heuristic
        for issue in insight["weaknesses"]:
            insight["recommendations"].append(
                f"Improve handling of {issue}"
            )

        self.insights[reflection["decision_id"]] = insight
        return insight

    # =====================================================
    # Self-model update
    # =====================================================
    def update_self_model(self, insights: Dict[str, Any]):
        """
        Update self-model based on insights.
        """
        updates = {
            "new_limitations": [],
            "new_capabilities": [],
        }

        for weakness in insights.get("weaknesses", []):
            updates["new_limitations"].append(weakness)

        for strength in insights.get("strengths", []):
            updates["new_capabilities"].append(strength)

        return updates

    # =====================================================
    # Improvement discovery
    # =====================================================
    def identify_improvement_areas(self):
        """
        Identify areas for self-improvement.
        """
        improvement_areas = {}

        for reflection in self.reflections:
            for issue in reflection.get("issues", []):
                improvement_areas[issue] = improvement_areas.get(issue, 0) + 1

        return {
            "priority_areas": sorted(
                improvement_areas.items(),
                key=lambda x: x[1],
                reverse=True,
            )
        }

    # =====================================================
    # Learning from mistakes
    # =====================================================
    def learn_from_mistakes(self, mistake_analysis: Dict[str, Any]):
        """
        Learn from identified mistakes.
        """
        lessons = []

        for mistake, frequency in mistake_analysis.get("priority_areas", []):
            lessons.append({
                "mistake": mistake,
                "action": f"Adjust strategy to reduce {mistake}",
                "priority": frequency,
            })

        return lessons
