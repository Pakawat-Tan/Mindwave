"""
ConfidenceScorer.py
ประเมินความมั่นใจของ output
Scores confidence in outputs.
"""

from typing import Dict, Any, List
import time
import math


class ConfidenceScorer:
    """Scores confidence in system outputs."""

    def __init__(self):
        self.confidence_scores: Dict[str, float] = {}
        self.scoring_history: List[Dict[str, Any]] = []

        # default weights (ปรับได้ภายหลัง)
        self.factor_weights = {
            "model": 0.4,
            "rule": 0.2,
            "memory": 0.2,
            "consensus": 0.2
        }

    # -------------------------------------------------

    def compute_confidence(self, output: Dict[str, Any], evidence: Dict[str, Any]) -> float:
        """
        evidence example:
        {
            "model_confidence": 0.85,
            "rule_support": 1.0,
            "memory_support": 0.6,
            "consensus": 0.75,
            "uncertainty": 0.2
        }
        """

        score = 0.0

        model_c = evidence.get("model_confidence", 0.0)
        rule_c = evidence.get("rule_support", 0.0)
        memory_c = evidence.get("memory_support", 0.0)
        consensus_c = evidence.get("consensus", 0.0)
        uncertainty = evidence.get("uncertainty", 0.0)

        score += model_c * self.factor_weights["model"]
        score += rule_c * self.factor_weights["rule"]
        score += memory_c * self.factor_weights["memory"]
        score += consensus_c * self.factor_weights["consensus"]

        # uncertainty penalty
        score *= (1.0 - min(max(uncertainty, 0.0), 1.0))

        # clamp
        score = max(0.0, min(1.0, score))

        output_id = output.get("id", f"out_{len(self.scoring_history)}")
        self.confidence_scores[output_id] = score

        self.scoring_history.append({
            "output_id": output_id,
            "score": score,
            "evidence": evidence,
            "timestamp": time.time()
        })

        return score

    # -------------------------------------------------

    def get_confidence_interval(self, output: Dict[str, Any]) -> Dict[str, float]:
        """
        Simple confidence interval estimation
        """
        output_id = output.get("id")
        base = self.confidence_scores.get(output_id, 0.0)

        # naive interval (can be Bayesian later)
        margin = 0.1 + (0.2 * (1 - base))

        return {
            "low": max(0.0, base - margin),
            "high": min(1.0, base + margin)
        }

    # -------------------------------------------------

    def analyze_confidence_factors(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze factors affecting confidence."""

        output_id = output.get("id")
        records = [
            r for r in self.scoring_history
            if r["output_id"] == output_id
        ]

        if not records:
            return {"error": "no_confidence_data"}

        last = records[-1]
        evidence = last["evidence"]

        impact = {}
        for k, v in evidence.items():
            impact[k] = {
                "value": v,
                "impact": v * self.factor_weights.get(
                    k.replace("_support", "").replace("_confidence", ""), 0.1
                )
            }

        return {
            "output_id": output_id,
            "confidence": last["score"],
            "factors": impact
        }

    # -------------------------------------------------

    def calibrate_confidence(self):
        """
        Adjust weights based on historical accuracy.
        (placeholder: simple normalization)
        """

        total = sum(self.factor_weights.values())
        if total == 0:
            return self.factor_weights

        for k in self.factor_weights:
            self.factor_weights[k] /= total

        return self.factor_weights

    # -------------------------------------------------

    def flag_low_confidence(self, threshold: float = 0.5) -> List[str]:
        """Return output IDs with confidence below threshold."""
        return [
            oid for oid, score in self.confidence_scores.items()
            if score < threshold
        ]
