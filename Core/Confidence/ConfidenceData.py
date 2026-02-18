"""
ConfidenceData — Permission to Commit

Spec (Phase 3):
  Confidence = gate ว่าระบบ "ควร" ตอบหรือไม่
  ไม่ใช่ความมั่นใจในข้อมูล แต่คือ permission

Levels:
  HIGH      ≥ 0.75  → Commit
  MEDIUM    ≥ 0.50  → Conditional Response
  LOW       ≥ 0.25  → Ask Clarification
  VERY_LOW  < 0.25  → Silence

Execution Priority:
  Rule → Confidence → Skill → Personality → Emotion
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
import time
import uuid


# ============================================================================
# ENUMS
# ============================================================================

class ConfidenceLevel(Enum):
    """ระดับ confidence"""
    HIGH      = "high"       # ≥ 0.75
    MEDIUM    = "medium"     # ≥ 0.50
    LOW       = "low"        # ≥ 0.25
    VERY_LOW  = "very_low"   # < 0.25

    def __str__(self): return self.value.upper()


class ConfidenceOutcome(Enum):
    """
    ผลลัพธ์จาก Confidence gate

    Execution Priority ใน Skill Contract:
      Commit → Conditional → Ask → Silence → Reject
    """
    COMMIT      = "commit"       # ผ่านทุก check → ตอบได้เลย
    CONDITIONAL = "conditional"  # medium → ตอบแบบมีเงื่อนไข
    ASK         = "ask"          # low confidence → ขอ clarification
    SILENCE     = "silence"      # rule conflict / very low → ไม่ตอบ
    REJECT      = "reject"       # identity conflict / system error → ปฏิเสธ

    def __str__(self): return self.value.upper()


class ConflictType(Enum):
    """ประเภทของ conflict ที่ตรวจพบ"""
    NONE             = "none"
    RULE_CONFLICT    = "rule_conflict"      # → Silence
    IDENTITY_CONFLICT = "identity_conflict"  # → Reject
    SYSTEM_ERROR     = "system_error"       # → Reject
    LOW_CONFIDENCE   = "low_confidence"     # → Ask

    def __str__(self): return self.value.upper()


# ── Level thresholds ──────────────────────────────────────────────────────

LEVEL_THRESHOLDS = {
    ConfidenceLevel.HIGH:     0.75,
    ConfidenceLevel.MEDIUM:   0.50,
    ConfidenceLevel.LOW:      0.25,
    ConfidenceLevel.VERY_LOW: 0.0,
}


def score_to_level(score: float) -> ConfidenceLevel:
    """แปลง score [0.0, 1.0] → ConfidenceLevel"""
    if score >= LEVEL_THRESHOLDS[ConfidenceLevel.HIGH]:
        return ConfidenceLevel.HIGH
    if score >= LEVEL_THRESHOLDS[ConfidenceLevel.MEDIUM]:
        return ConfidenceLevel.MEDIUM
    if score >= LEVEL_THRESHOLDS[ConfidenceLevel.LOW]:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.VERY_LOW


def level_to_outcome(
    level:    ConfidenceLevel,
    conflict: ConflictType = ConflictType.NONE,
) -> ConfidenceOutcome:
    """
    แปลง level + conflict → ConfidenceOutcome

    Conflict override ก่อน level:
      IDENTITY_CONFLICT / SYSTEM_ERROR → REJECT
      RULE_CONFLICT                    → SILENCE
      LOW_CONFIDENCE                   → ASK (same as LOW level)
    """
    # Conflict overrides
    if conflict in (ConflictType.IDENTITY_CONFLICT, ConflictType.SYSTEM_ERROR):
        return ConfidenceOutcome.REJECT
    if conflict == ConflictType.RULE_CONFLICT:
        return ConfidenceOutcome.SILENCE

    # Level-based
    if level == ConfidenceLevel.HIGH:
        return ConfidenceOutcome.COMMIT
    if level == ConfidenceLevel.MEDIUM:
        return ConfidenceOutcome.CONDITIONAL
    if level == ConfidenceLevel.LOW:
        return ConfidenceOutcome.ASK
    return ConfidenceOutcome.SILENCE  # VERY_LOW


# ============================================================================
# CONFIDENCE RESULT — output ของการ evaluate
# ============================================================================

@dataclass(frozen=True)
class ConfidenceResult:
    """
    ผลการประเมิน confidence ครั้งหนึ่ง

    Attributes:
        score    : raw score [0.0, 1.0]
        level    : ConfidenceLevel
        outcome  : ConfidenceOutcome (gate decision)
        conflict : ConflictType ที่ตรวจพบ
        reason   : เหตุผล (สำหรับ logging / audit)
        factors  : dict ของ input factors ที่ใช้คำนวณ
        eval_id  : unique id ของการ evaluate ครั้งนี้
        timestamp: เวลาที่ evaluate
    """
    score:     float
    level:     ConfidenceLevel
    outcome:   ConfidenceOutcome
    conflict:  ConflictType        = ConflictType.NONE
    reason:    str                 = ""
    factors:   Dict[str, float]    = field(default_factory=dict)
    eval_id:   str                 = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float               = field(default_factory=time.time)

    # ── Convenience properties ────────────────────────────────────

    @property
    def can_commit(self) -> bool:
        return self.outcome == ConfidenceOutcome.COMMIT

    @property
    def should_ask(self) -> bool:
        return self.outcome == ConfidenceOutcome.ASK

    @property
    def should_silence(self) -> bool:
        return self.outcome == ConfidenceOutcome.SILENCE

    @property
    def should_reject(self) -> bool:
        return self.outcome == ConfidenceOutcome.REJECT

    @property
    def is_conditional(self) -> bool:
        return self.outcome == ConfidenceOutcome.CONDITIONAL

    @property
    def has_conflict(self) -> bool:
        return self.conflict != ConflictType.NONE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eval_id":   self.eval_id,
            "score":     self.score,
            "level":     self.level.value,
            "outcome":   self.outcome.value,
            "conflict":  self.conflict.value,
            "reason":    self.reason,
            "factors":   self.factors,
            "timestamp": self.timestamp,
        }

    def __str__(self) -> str:
        conflict_str = f" [{self.conflict}]" if self.has_conflict else ""
        return (
            f"Confidence[{self.level}]{conflict_str} "
            f"score={self.score:.3f} → {self.outcome} "
            f"| {self.reason}"
        )


# ── Common results ────────────────────────────────────────────────────────

def make_reject(conflict: ConflictType, reason: str) -> ConfidenceResult:
    return ConfidenceResult(
        score=0.0, level=ConfidenceLevel.VERY_LOW,
        outcome=ConfidenceOutcome.REJECT,
        conflict=conflict, reason=reason,
    )

def make_silence(reason: str) -> ConfidenceResult:
    return ConfidenceResult(
        score=0.0, level=ConfidenceLevel.VERY_LOW,
        outcome=ConfidenceOutcome.SILENCE,
        conflict=ConflictType.RULE_CONFLICT, reason=reason,
    )

IDENTITY_CONFLICT_RESULT = make_reject(
    ConflictType.IDENTITY_CONFLICT, "identity conflict detected"
)
SYSTEM_ERROR_RESULT = make_reject(
    ConflictType.SYSTEM_ERROR, "system error detected"
)
RULE_CONFLICT_RESULT = make_silence("rule conflict — silence")