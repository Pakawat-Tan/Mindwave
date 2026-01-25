"""
Introspection.py
มองย้อนการตัดสินใจ วิเคราะห์ตนเอง
Analyzes past decisions and performs self-reflection.
"""

from datetime import datetime
from typing import Dict, List
import uuid


class Introspection:
    """Performs introspection and self-analysis."""

    def __init__(self):
        """Initialize introspection module."""
        self.decision_history: List[dict] = []
        self.analysis: Dict[str, dict] = {}

    def record_decision(
        self,
        decision_type: str,
        context: dict,
        action: str,
        outcome: dict | None = None,
    ) -> str:
        """
        Record a decision for later introspection.
        """
        decision_id = str(uuid.uuid4())

        record = {
            "id": decision_id,
            "type": decision_type,
            "context": context,
            "action": action,
            "outcome": outcome,
            "timestamp": datetime.utcnow(),
        }

        self.decision_history.append(record)
        return decision_id

    def analyze_decision(self, decision_id: str) -> dict:
        """
        Analyze a specific past decision.
        """
        decision = next(
            (d for d in self.decision_history if d["id"] == decision_id), None
        )

        if not decision:
            raise ValueError(f"Decision ID {decision_id} not found")

        score = 0.0
        if decision["outcome"]:
            score = decision["outcome"].get("success_score", 0.0)

        analysis = {
            "decision_id": decision_id,
            "effectiveness": score,
            "context_complexity": len(decision["context"]),
            "has_outcome": decision["outcome"] is not None,
            "analyzed_at": datetime.utcnow(),
        }

        self.analysis[decision_id] = analysis
        return analysis

    def reflect_on_outcomes(self) -> dict:
        """
        Reflect on outcomes of past decisions.
        """
        total = len(self.decision_history)
        with_outcome = [
            d for d in self.decision_history if d["outcome"] is not None
        ]

        success_scores = [
            d["outcome"].get("success_score", 0.0)
            for d in with_outcome
        ]

        reflection = {
            "total_decisions": total,
            "decisions_with_outcomes": len(with_outcome),
            "average_success":
                sum(success_scores) / len(success_scores)
                if success_scores else 0.0,
            "reflection_time": datetime.utcnow(),
        }

        return reflection

    def identify_patterns(self) -> dict:
        """
        Identify patterns in decision-making.
        """
        pattern = {}

        for d in self.decision_history:
            key = d["type"]
            pattern.setdefault(key, {"count": 0, "success": []})
            pattern[key]["count"] += 1

            if d["outcome"]:
                pattern[key]["success"].append(
                    d["outcome"].get("success_score", 0.0)
                )

        summary = {}
        for k, v in pattern.items():
            avg = (
                sum(v["success"]) / len(v["success"])
                if v["success"] else 0.0
            )
            summary[k] = {
                "frequency": v["count"],
                "average_success": avg,
            }

        return summary
