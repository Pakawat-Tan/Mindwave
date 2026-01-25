"""
ApproveEngine.py
ตัดสินว่าผลลัพธ์ผ่านหรือไม่
Determines if outputs pass approval criteria.
"""

from typing import Dict, Any, List
import time


class ApproveEngine:
    """Evaluates and approves outputs."""

    def __init__(self):
        self.approval_criteria: Dict[str, Any] = {}
        self.approval_history: List[Dict[str, Any]] = []
        self.confidence_threshold: float = 0.7

    # -------------------------------------------------

    def set_approval_criteria(self, criteria: Dict[str, Any]):
        """
        criteria example:
        {
            "min_confidence": 0.75,
            "require_safe": True,
            "max_uncertainty": 0.3,
            "allowed_actions": ["respond", "store", "plan"]
        }
        """
        self.approval_criteria = criteria or {}

    # -------------------------------------------------

    def evaluate_output(self, output: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate if output passes approval.

        output example:
        {
            "id": "out_123",
            "content": "...",
            "confidence": 0.82,
            "uncertainty": 0.15,
            "action_type": "respond",
            "flags": ["safe"]
        }
        """

        decision = {
            "output_id": output.get("id"),
            "approved": False,
            "reason": None,
            "timestamp": time.time()
        }

        confidence = output.get("confidence", 0.0)
        uncertainty = output.get("uncertainty", 1.0)
        flags = output.get("flags", [])
        action_type = output.get("action_type")

        # 1. Confidence check
        min_conf = self.approval_criteria.get(
            "min_confidence", self.confidence_threshold
        )
        if confidence < min_conf:
            decision["reason"] = "confidence_too_low"
            self._record(decision)
            return decision

        # 2. Uncertainty check
        max_uncertainty = self.approval_criteria.get("max_uncertainty")
        if max_uncertainty is not None and uncertainty > max_uncertainty:
            decision["reason"] = "uncertainty_too_high"
            self._record(decision)
            return decision

        # 3. Safety / rule flags
        if self.approval_criteria.get("require_safe", True):
            if "unsafe" in flags:
                decision["reason"] = "unsafe_flag_detected"
                self._record(decision)
                return decision

        # 4. Action whitelist
        allowed = self.approval_criteria.get("allowed_actions")
        if allowed and action_type not in allowed:
            decision["reason"] = "action_not_allowed"
            self._record(decision)
            return decision

        # 5. Context-based veto (optional hook)
        if context.get("force_reject"):
            decision["reason"] = "context_forced_reject"
            self._record(decision)
            return decision

        # PASSED
        decision["approved"] = True
        decision["reason"] = "approved"
        self._record(decision)
        return decision

    # -------------------------------------------------

    def approve_output(self, output_id: str):
        """Mark output as approved explicitly."""
        self.approval_history.append({
            "output_id": output_id,
            "approved": True,
            "reason": "manual_approve",
            "timestamp": time.time()
        })

    def reject_output(self, output_id: str, reason: str):
        """Reject output with reason."""
        self.approval_history.append({
            "output_id": output_id,
            "approved": False,
            "reason": reason,
            "timestamp": time.time()
        })

    # -------------------------------------------------

    def get_approval_rate(self) -> Dict[str, float]:
        """Get approval rate statistics."""
        if not self.approval_history:
            return {"approval_rate": 0.0}

        total = len(self.approval_history)
        approved = sum(1 for h in self.approval_history if h["approved"])
        return {
            "total": total,
            "approved": approved,
            "approval_rate": approved / total
        }

    # -------------------------------------------------

    def _record(self, decision: Dict[str, Any]):
        """Internal helper to store decisions."""
        self.approval_history.append(decision)
