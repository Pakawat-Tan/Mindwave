"""
SkillData — หน่วยเก็บ skill score ของ topic หนึ่ง

Spec:
  - Range   : 0.0000 – 100.0000
  - Precision: 4 decimal places
  - Growth  : เพิ่มได้อย่างเดียว (ห้าม delta < 0)
  - Storage : Runtime only — ไม่บันทึกเป็น Atom
  - Logging : ทุกการเปลี่ยนแปลง

Execution Priority (Phase 3):
  Rule → Confidence → Skill → Personality → Emotion
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import time
import uuid


SKILL_MIN     = 0.0
SKILL_MAX     = 100.0
SKILL_DECIMAL = 4  # precision


# ============================================================================
# SKILL EVENT — audit trail ทุก mutation
# ============================================================================

@dataclass(frozen=True)
class SkillEvent:
    """
    บันทึกการเปลี่ยนแปลง skill — audit-ready

    Attributes:
        event_id    : unique id
        skill_name  : ชื่อ skill
        score_before: score ก่อน grow
        score_after : score หลัง grow
        delta       : ค่าที่เพิ่ม
        topic_repetition : จำนวนครั้งที่ topic ซ้ำ (trigger)
        avg_confidence   : confidence เฉลี่ย (trigger)
        timestamp   : เวลาที่เกิดเหตุการณ์
        reason      : หมายเหตุ
    """
    skill_name:        str
    score_before:      float
    score_after:       float
    delta:             float
    topic_repetition:  int
    avg_confidence:    float
    timestamp:         float = field(default_factory=time.time)
    event_id:          str   = field(default_factory=lambda: str(uuid.uuid4()))
    reason:            str   = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id":         self.event_id,
            "skill_name":       self.skill_name,
            "score_before":     self.score_before,
            "score_after":      self.score_after,
            "delta":            self.delta,
            "topic_repetition": self.topic_repetition,
            "avg_confidence":   self.avg_confidence,
            "timestamp":        self.timestamp,
            "reason":           self.reason,
        }

    def __str__(self) -> str:
        return (
            f"SkillEvent[{self.skill_name}] "
            f"{self.score_before:.4f} → {self.score_after:.4f} "
            f"(+{self.delta:.4f}) "
            f"rep={self.topic_repetition} conf={self.avg_confidence:.3f}"
        )


# ============================================================================
# SKILL DATA — runtime skill unit
# ============================================================================

@dataclass
class SkillData:
    """
    Runtime skill score สำหรับ topic หนึ่ง

    ไม่ใช้ frozen — score เปลี่ยนได้แต่ผ่าน grow() เท่านั้น
    ห้าม set score โดยตรง

    Attributes:
        skill_name  : ชื่อ skill (เช่น "python", "math", "conversation")
        topic_ids   : cluster_id ที่ skill นี้ครอบคลุม
        _score      : internal score (private)
        _events     : audit trail ทุก mutation
    """
    skill_name:  str
    topic_ids:   List[int]    = field(default_factory=list)
    _score:      float        = field(default=0.0, init=False, repr=False)
    _events:     List[SkillEvent] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.skill_name or not self.skill_name.strip():
            raise ValueError("SkillData: skill_name must not be empty")
        self._score = round(SKILL_MIN, SKILL_DECIMAL)

    # ── Read-only score ───────────────────────────────────────────

    @property
    def score(self) -> float:
        return self._score

    @property
    def is_maxed(self) -> bool:
        return self._score >= SKILL_MAX

    @property
    def events(self) -> List[SkillEvent]:
        return list(self._events)

    @property
    def event_count(self) -> int:
        return len(self._events)

    # ── Growth ────────────────────────────────────────────────────

    def grow(
        self,
        delta:             float,
        topic_repetition:  int,
        avg_confidence:    float,
        reason:            str = "",
    ) -> SkillEvent:
        """
        เพิ่ม score โดย delta

        ข้อกำหนด:
          - delta > 0 เท่านั้น (ห้าม decay)
          - score หลัง grow ต้องไม่เกิน SKILL_MAX
          - clamp ที่ SKILL_MAX ถ้า overflow

        Returns:
            SkillEvent ที่เกิดขึ้น
        """
        if delta <= 0:
            raise ValueError(
                f"SkillData.grow: delta must be > 0, got {delta}"
            )

        before = self._score
        new_score = min(round(self._score + delta, SKILL_DECIMAL), SKILL_MAX)
        actual_delta = round(new_score - before, SKILL_DECIMAL)

        self._score = new_score

        event = SkillEvent(
            skill_name        = self.skill_name,
            score_before      = before,
            score_after       = new_score,
            delta             = actual_delta,
            topic_repetition  = topic_repetition,
            avg_confidence    = avg_confidence,
            reason            = reason,
        )
        self._events.append(event)
        return event

    # ── Serialization (score + metadata only — no Atom) ──────────

    def to_dict(self) -> Dict[str, Any]:
        """Runtime snapshot — ไม่รวม event history"""
        return {
            "skill_name": self.skill_name,
            "score":      self._score,
            "topic_ids":  self.topic_ids,
            "is_maxed":   self.is_maxed,
        }

    def __str__(self) -> str:
        return (
            f"Skill[{self.skill_name}] "
            f"score={self._score:.4f} "
            f"topics={self.topic_ids}"
        )


# ============================================================================
# ARBITRATION RESULT
# ============================================================================

@dataclass(frozen=True)
class ArbitrationResult:
    """
    ผลของ Skill Arbitration

    Logic (deterministic):
      1. หา skills ที่ match topic
      2. เลือก highest score
      3. ถ้า tie → รวม score ที่เท่ากัน (sum equal highest)
      4. scale เป็น weight สำหรับ output intensity

    Attributes:
        selected_skills : skills ที่ถูกเลือก
        highest_score   : score สูงสุด
        combined_score  : sum ของ tied skills (ถ้าไม่ tie = highest_score)
        weight          : normalized weight [0.0, 1.0] สำหรับ output scaling
        topic_id        : topic ที่ arbitrate
    """
    selected_skills: tuple          # tuple[SkillData]
    highest_score:   float
    combined_score:  float
    weight:          float          # combined_score / SKILL_MAX
    topic_id:        Optional[int]  = None
    reason:          str            = ""

    @property
    def skill_names(self) -> List[str]:
        return [s.skill_name for s in self.selected_skills]

    @property
    def has_skills(self) -> bool:
        return len(self.selected_skills) > 0

    def __str__(self) -> str:
        names = ", ".join(self.skill_names) if self.selected_skills else "none"
        return (
            f"ArbitrationResult[topic={self.topic_id}] "
            f"skills=[{names}] "
            f"highest={self.highest_score:.4f} "
            f"combined={self.combined_score:.4f} "
            f"weight={self.weight:.4f}"
        )


# Default เมื่อไม่มี skill match
NO_SKILL_RESULT = ArbitrationResult(
    selected_skills = (),
    highest_score   = 0.0,
    combined_score  = 0.0,
    weight          = 0.0,
    reason          = "no matching skill",
)