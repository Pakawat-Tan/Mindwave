"""
จัดการ Skill ทั้งหมด — registry, growth, arbitration

Growth condition (ตาม spec):
    topic_repetition >= repetition_threshold
    AND avg_confidence >= confidence_threshold

Arbitration (deterministic):
    1. หา skills ที่ match topic_id
    2. เลือก highest score
    3. ถ้า tie → sum equal highest
    4. คืน ArbitrationResult พร้อม weight

Execution Priority:
    Rule → Confidence → Skill → Personality → Emotion
    (SkillController อยู่ลำดับ 3 — รับ context จาก Confidence)
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Core.Condition.ConditionController import ConditionController

import logging
from typing import Optional, List, Dict

import sys, os
from Core.Skill.SkillData import (
    SkillData, SkillEvent, ArbitrationResult,
    NO_SKILL_RESULT, SKILL_MAX
)


# Default thresholds (ปรับได้ผ่าน NumericPolicy)
DEFAULT_REPETITION_THRESHOLD  = 3
DEFAULT_CONFIDENCE_THRESHOLD  = 0.6


class SkillController:

    def __init__(
        self,
        condition=None,
        repetition_threshold:  int   = DEFAULT_REPETITION_THRESHOLD,
        confidence_threshold:  float = DEFAULT_CONFIDENCE_THRESHOLD,
    ):
        """
        Args:
            repetition_threshold : topic ต้องซ้ำกี่ครั้งถึง trigger growth
            confidence_threshold : confidence ต้องสูงกว่าเท่าไหร่
        """
        self._skills:   Dict[str, SkillData] = {}
        self._rep_threshold  = repetition_threshold
        self._conf_threshold = confidence_threshold
        self._logger = logging.getLogger("mindwave.skill")
        self._condition = condition

    # ─────────────────────────────────────────────────────────────
    # Registry
    # ─────────────────────────────────────────────────────────────

    def register(
        self,
        skill_name: str,
        topic_ids:  Optional[List[int]] = None,
    ) -> SkillData:
        """
        ลงทะเบียน skill ใหม่

        ถ้าชื่อซ้ำ → คืน skill เดิม (ไม่ reset)
        """
        if skill_name in self._skills:
            return self._skills[skill_name]

        skill = SkillData(
            skill_name = skill_name,
            topic_ids  = topic_ids or [],
        )
        self._skills[skill_name] = skill
        self._logger.info(
            f"[SkillController] REGISTER '{skill_name}' "
            f"topics={topic_ids or []}"
        )
        return skill

    def get(self, skill_name: str) -> Optional[SkillData]:
        return self._skills.get(skill_name)

    def list(self) -> List[SkillData]:
        return list(self._skills.values())

    def has(self, skill_name: str) -> bool:
        return skill_name in self._skills

    # ─────────────────────────────────────────────────────────────
    # Growth
    # ─────────────────────────────────────────────────────────────

    def try_grow(
        self,
        skill_name:        str,
        delta:             float,
        topic_repetition:  int,
        avg_confidence:    float,
        reason:            str = "",
    ) -> Optional[SkillEvent]:
        """
        ลอง grow skill ถ้า condition ผ่าน

        Growth condition:
            topic_repetition >= repetition_threshold
            AND avg_confidence >= confidence_threshold

        Args:
            skill_name       : ชื่อ skill ที่จะ grow
            delta            : ค่าที่เพิ่ม (ต้อง > 0)
            topic_repetition : จำนวนครั้งที่ topic ซ้ำในช่วงนี้
            avg_confidence   : confidence เฉลี่ยของ topic นี้

        Returns:
            SkillEvent ถ้า grow สำเร็จ
            None ถ้า condition ไม่ผ่าน หรือ skill ไม่พบ
        """
        # ── Condition Gate ─────────────────────────────────────
        if self._condition is not None:
            _allowed, _reason = self._condition.is_skill_allowed()
            if not _allowed:
                self._logger.warning(
                    f"[SkillController] try_grow BLOCKED reason={_reason}"
                )
                return False

        skill = self._skills.get(skill_name)
        if skill is None:
            # auto-register skill ใหม่แทนที่จะ warning
            self._logger.debug(
                f"[SkillController] try_grow: auto-register '{skill_name}'"
            )
            self.register(skill_name)
            skill = self._skills.get(skill_name)
            if skill is None:
                return None

        if skill.is_maxed:
            self._logger.debug(
                f"[SkillController] try_grow: '{skill_name}' already maxed"
            )
            return None

        # ── Growth condition ──────────────────────────────────────
        if topic_repetition < self._rep_threshold:
            self._logger.debug(
                f"[SkillController] try_grow: '{skill_name}' "
                f"rep={topic_repetition} < threshold={self._rep_threshold}"
            )
            return None

        if avg_confidence < self._conf_threshold:
            self._logger.debug(
                f"[SkillController] try_grow: '{skill_name}' "
                f"conf={avg_confidence:.3f} < threshold={self._conf_threshold:.3f}"
            )
            return None

        # ── Grow ──────────────────────────────────────────────────
        event = skill.grow(
            delta            = delta,
            topic_repetition = topic_repetition,
            avg_confidence   = avg_confidence,
            reason           = reason,
        )
        self._logger.info(
            f"[SkillController] GREW '{skill_name}' "
            f"{event.score_before:.4f} → {event.score_after:.4f} "
            f"(+{event.delta:.4f})"
        )
        return event

    def force_grow(
        self,
        skill_name:       str,
        delta:            float,
        reviewer_id:      str,
        reason:           str = "",
    ) -> SkillEvent:
        """
        บังคับ grow โดยข้าม condition — ต้องมี reviewer_id

        ใช้สำหรับ admin / test / manual correction
        """
        if not reviewer_id or not reviewer_id.strip():
            raise PermissionError(
                "[SkillController] force_grow requires reviewer_id"
            )
        skill = self._skills.get(skill_name)
        if skill is None:
            raise KeyError(
                f"[SkillController] force_grow: skill '{skill_name}' not registered"
            )
        event = skill.grow(
            delta            = delta,
            topic_repetition = -1,  # sentinel = forced
            avg_confidence   = -1.0,
            reason           = f"forced by {reviewer_id}: {reason}",
        )
        self._logger.warning(
            f"[SkillController] FORCE_GREW '{skill_name}' "
            f"+{event.delta:.4f} by='{reviewer_id}'"
        )
        return event

    # ─────────────────────────────────────────────────────────────
    # Arbitration (Deterministic)
    # ─────────────────────────────────────────────────────────────

    def arbitrate(
        self,
        topic_id: Optional[int] = None,
        skill_names: Optional[List[str]] = None,
    ) -> ArbitrationResult:
        """
        เลือก skill ที่ดีที่สุดสำหรับ topic

        Algorithm (deterministic):
          1. หา candidates ที่ match topic_id หรือ skill_names
          2. เลือก highest score
          3. ถ้า tie → รวม score ทั้งหมดที่ equal highest (sum)
          4. weight = combined_score / SKILL_MAX

        Args:
            topic_id    : เลือก skills ที่ครอบคลุม topic_id นี้
            skill_names : หรือระบุชื่อ skills โดยตรง

        Returns:
            ArbitrationResult
        """
        # ── Condition Gate ─────────────────────────────────────
        if self._condition is not None:
            _allowed, _reason = self._condition.is_skill_allowed()
            if not _allowed:
                self._logger.warning(
                    f"[SkillController] arbitrate BLOCKED reason={_reason}"
                )
                return ArbitrationResult(
                    selected_skills = (),
                    highest_score   = 0.0,
                    combined_score  = 0.0,
                    weight          = 0.0,
                    reason          = f"condition blocked: {_reason}",
                )

        # ── Find candidates ───────────────────────────────────────
        candidates: List[SkillData] = []

        if skill_names:
            candidates = [
                s for name in skill_names
                if (s := self._skills.get(name)) is not None
            ]
        elif topic_id is not None:
            candidates = [
                s for s in self._skills.values()
                if topic_id in s.topic_ids
            ]
        else:
            candidates = list(self._skills.values())

        if not candidates:
            return NO_SKILL_RESULT

        # ── Find highest ──────────────────────────────────────────
        highest_score = max(s.score for s in candidates)

        # ── Tie → sum equal highest ───────────────────────────────
        tied = [s for s in candidates if s.score == highest_score]
        combined_score = round(
            sum(s.score for s in tied), 4
        ) if len(tied) > 1 else highest_score

        weight = round(combined_score / SKILL_MAX, 4)

        result = ArbitrationResult(
            selected_skills = tuple(tied),
            highest_score   = highest_score,
            combined_score  = combined_score,
            weight          = weight,
            topic_id        = topic_id,
            reason          = (
                f"tie({len(tied)})" if len(tied) > 1
                else f"best={tied[0].skill_name}"
            ),
        )
        self._logger.debug(
            f"[SkillController] ARBITRATE topic={topic_id} → {result}"
        )
        return result

    # ─────────────────────────────────────────────────────────────
    # Thresholds (ปรับได้จาก PolicyController)
    # ─────────────────────────────────────────────────────────────

    def set_thresholds(
        self,
        repetition: Optional[int]   = None,
        confidence: Optional[float] = None,
    ) -> None:
        """ปรับ threshold — เรียกจาก PolicyController"""
        if repetition is not None:
            self._rep_threshold = repetition
        if confidence is not None:
            self._conf_threshold = confidence
        self._logger.info(
            f"[SkillController] THRESHOLD "
            f"rep={self._rep_threshold} conf={self._conf_threshold}"
        )

    @property
    def repetition_threshold(self) -> int:
        return self._rep_threshold

    @property
    def confidence_threshold(self) -> float:
        return self._conf_threshold

    # ─────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        scores = [s.score for s in self._skills.values()]
        return {
            "skill_count":   len(self._skills),
            "maxed_count":   sum(1 for s in self._skills.values() if s.is_maxed),
            "avg_score":     round(sum(scores) / len(scores), 4) if scores else 0.0,
            "total_events":  sum(s.event_count for s in self._skills.values()),
            "rep_threshold": self._rep_threshold,
            "conf_threshold": self._conf_threshold,
        }