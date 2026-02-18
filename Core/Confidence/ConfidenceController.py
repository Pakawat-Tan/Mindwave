"""
Gate ว่าระบบ "ควร" ตอบหรือไม่ — Confidence = Permission to Commit

Execution Priority:
  Rule → Confidence → Skill → Personality → Emotion

evaluate() รับ factors แล้วคืน ConfidenceResult
  Phase 3: weighted average (placeholder)
  Phase 4: formula จริง

Hard conflicts ตรวจก่อน scoring เสมอ:
  identity_conflict → REJECT
  system_error      → REJECT
  rule_blocked      → SILENCE
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Core.Condition.ConditionController import ConditionController

import logging
from typing import Optional, Dict

import sys, os
from Core.Confidence.ConfidenceData import (
    ConfidenceResult, ConfidenceLevel, ConfidenceOutcome, ConflictType,
    score_to_level, level_to_outcome,
    IDENTITY_CONFLICT_RESULT, SYSTEM_ERROR_RESULT, RULE_CONFLICT_RESULT,
    make_reject, make_silence,
)


# Default weights สำหรับ Phase 3 placeholder formula
DEFAULT_WEIGHTS: Dict[str, float] = {
    "rule_score":      0.35,  # Rule compliance
    "context_score":   0.25,  # Context relevance
    "skill_score":     0.20,  # Skill relevance
    "identity_score":  0.20,  # Identity safety
}


class ConfidenceController:

    def __init__(self, weights: Optional[Dict[str, float]] = None, condition=None):
        """
        Args:
            weights: custom weights สำหรับ formula
                     (default ใช้ DEFAULT_WEIGHTS)
        """
        self._weights = weights or dict(DEFAULT_WEIGHTS)
        self._logger  = logging.getLogger("mindwave.confidence")
        self._history: list[ConfidenceResult] = []
        self._condition = condition

    # ─────────────────────────────────────────────────────────────
    # Hard conflict checks (ตรวจก่อนเสมอ)
    # ─────────────────────────────────────────────────────────────

    def reject_identity_conflict(self, reason: str = "") -> ConfidenceResult:
        """Identity conflict → REJECT ทันที"""
        result = make_reject(
            ConflictType.IDENTITY_CONFLICT,
            reason or "identity conflict detected",
        )
        self._log_and_store(result)
        return result

    def reject_system_error(self, reason: str = "") -> ConfidenceResult:
        """System error → REJECT ทันที"""
        result = make_reject(
            ConflictType.SYSTEM_ERROR,
            reason or "system error detected",
        )
        self._log_and_store(result)
        return result

    def silence_rule_conflict(self, reason: str = "") -> ConfidenceResult:
        """Rule conflict → SILENCE ทันที"""
        result = make_silence(reason or "rule conflict — silence")
        self._log_and_store(result)
        return result

    # ─────────────────────────────────────────────────────────────
    # Main evaluate
    # ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        rule_score:       float = 1.0,
        context_score:    float = 0.8,
        skill_score:      float = 0.5,
        identity_score:   float = 1.0,
        identity_conflict: bool = False,
        system_error:      bool = False,
        rule_blocked:      bool = False,
    ) -> ConfidenceResult:
        """
        ประเมิน confidence และคืน ConfidenceResult

        Phase 3: weighted average formula (placeholder)
        Phase 4: formula จริงจะถูก operationalize

        Hard conflict checks ก่อนเสมอ:
          identity_conflict=True → REJECT
          system_error=True      → REJECT
          rule_blocked=True      → SILENCE

        Args:
            rule_score      : 0–1 compliance กับ Rule (0 = violated)
            context_score   : 0–1 ความเหมาะสมของ context
            skill_score     : 0–1 ความเกี่ยวข้องของ skill
            identity_score  : 0–1 ความปลอดภัยของ identity
            identity_conflict: hard conflict flag
            system_error    : hard error flag
            rule_blocked    : rule blocked flag

        Returns:
            ConfidenceResult
        """
        # ── Condition Gate ─────────────────────────────────────
        if self._condition is not None:
            _allowed, _reason = self._condition.is_confidence_allowed()
            if not _allowed:
                self._logger.warning(
                    f"[ConfidenceController] BLOCKED reason={_reason}"
                )
                return make_reject(ConflictType.RULE_CONFLICT, f"condition blocked: {_reason}")

        # ── Hard conflicts ─────────────────────────────────────────
        if identity_conflict or identity_score <= 0.0:
            return self.reject_identity_conflict(
                f"identity_score={identity_score:.3f}"
            )
        if system_error:
            return self.reject_system_error()
        if rule_blocked or rule_score <= 0.0:
            return self.silence_rule_conflict(
                f"rule_score={rule_score:.3f}"
            )

        # ── Score calculation (Phase 3 placeholder) ───────────────
        factors = {
            "rule_score":     max(0.0, min(1.0, rule_score)),
            "context_score":  max(0.0, min(1.0, context_score)),
            "skill_score":    max(0.0, min(1.0, skill_score)),
            "identity_score": max(0.0, min(1.0, identity_score)),
        }

        score = sum(
            factors[k] * self._weights.get(k, 0.0)
            for k in factors
        )
        score = round(max(0.0, min(1.0, score)), 4)

        # ── Level & outcome ────────────────────────────────────────
        level   = score_to_level(score)
        conflict = (
            ConflictType.LOW_CONFIDENCE
            if level in (ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW)
            else ConflictType.NONE
        )
        outcome = level_to_outcome(level, conflict)

        reason = self._build_reason(score, level, outcome, factors)
        result = ConfidenceResult(
            score    = score,
            level    = level,
            outcome  = outcome,
            conflict = conflict,
            reason   = reason,
            factors  = factors,
        )
        self._log_and_store(result)
        return result

    # ─────────────────────────────────────────────────────────────
    # Weights (ปรับได้จาก PolicyController)
    # ─────────────────────────────────────────────────────────────

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        ปรับ formula weights — เรียกจาก PolicyController

        weights ต้องรวมกัน = 1.0 (±0.001)
        """
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"[ConfidenceController] weights must sum to 1.0, "
                f"got {total:.4f}"
            )
        self._weights = dict(weights)
        self._logger.info(
            f"[ConfidenceController] WEIGHTS_UPDATED {weights}"
        )

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)

    # ─────────────────────────────────────────────────────────────
    # History
    # ─────────────────────────────────────────────────────────────

    @property
    def last_result(self) -> Optional[ConfidenceResult]:
        return self._history[-1] if self._history else None

    def history(self, n: int = 10) -> list:
        """คืน n รายการล่าสุด"""
        return list(self._history[-n:])

    def clear_history(self) -> None:
        self._history.clear()

    # ─────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        if not self._history:
            return {
                "total_evaluations": 0,
                "commit_rate":       0.0,
                "reject_rate":       0.0,
                "silence_rate":      0.0,
                "avg_score":         0.0,
            }
        total   = len(self._history)
        commits = sum(1 for r in self._history if r.can_commit)
        rejects = sum(1 for r in self._history if r.should_reject)
        silence = sum(1 for r in self._history if r.should_silence)
        avg     = sum(r.score for r in self._history) / total
        return {
            "total_evaluations": total,
            "commit_rate":       round(commits / total, 4),
            "reject_rate":       round(rejects / total, 4),
            "silence_rate":      round(silence / total, 4),
            "avg_score":         round(avg, 4),
        }

    # ─────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────

    def _log_and_store(self, result: ConfidenceResult) -> None:
        self._history.append(result)
        log = self._logger.warning if result.should_reject else self._logger.debug
        log(f"[ConfidenceController] EVAL {result}")

    @staticmethod
    def _build_reason(
        score:   float,
        level:   ConfidenceLevel,
        outcome: ConfidenceOutcome,
        factors: Dict[str, float],
    ) -> str:
        low_factors = [k for k, v in factors.items() if v < 0.5]
        base = f"score={score:.3f} level={level} → {outcome}"
        if low_factors:
            base += f" | low: {', '.join(low_factors)}"
        return base