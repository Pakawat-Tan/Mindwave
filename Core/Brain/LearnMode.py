"""
Core/Brain/LearnMode.py

Learn Mode — การเรียนรู้แบบตั้งใจ (/learn)

ต่างจาก Interaction Mode:
  Interaction : อัปเดต short-term, belief ชั่วคราว, emotional heuristics
  Learn       : structured belief update, long-term direct, ไม่มี emotional heuristics

Flow:
  /learn <text>
    → parse_input()      — แยก subject/value/confidence
    → update_belief()    — อัปเดต belief แบบ structured
    → consolidate()      — ตัดสินใจ long-term ทันที
    → record_session()   — บันทึก learn session

จาก document:
  belief_mean += learning_rate * (input_value - belief_mean)
  belief_variance += noise_estimate
  update_count += 1
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mindwave.learn_mode")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

LEARN_LEARNING_RATE     = 0.6    # สูงกว่า interaction (0.3) — ตั้งใจเรียน
NOISE_ESTIMATE_DEFAULT  = 0.15   # noise เริ่มต้น — ต้องซ้ำหลายครั้งกว่าจะเสถียร
CONSOLIDATE_THRESHOLD   = 3      # ซ้ำกี่ครั้งถึง consolidate ลง long-term
VARIANCE_STABLE_MAX     = 0.10   # variance ต่ำกว่านี้ถือว่าเสถียร


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Belief:
    """
    โครงสร้างความเชื่อแบบ probabilistic

    ไม่มี True/False — มีแค่ mean + variance
    """
    subject:       str
    belief_mean:   float          # ค่าเฉลี่ยของความเชื่อ (0.0–1.0)
    belief_variance: float        # ความไม่แน่นอน (0.0 = แน่ใจมาก)
    update_count:  int   = 0
    conflict_rate: float = 0.0    # อัตราที่ข้อมูลขัดแย้ง
    last_updated:  float = field(default_factory=time.time)
    source:        str   = "learn"

    @property
    def confidence_score(self) -> float:
        """confidence = mean ลบ variance"""
        return max(0.0, min(1.0, self.belief_mean - self.belief_variance))

    @property
    def is_stable(self) -> bool:
        return self.belief_variance <= VARIANCE_STABLE_MAX

    @property
    def needs_consolidation(self) -> bool:
        return self.update_count >= CONSOLIDATE_THRESHOLD and self.is_stable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject":         self.subject,
            "belief_mean":     round(self.belief_mean, 4),
            "belief_variance": round(self.belief_variance, 4),
            "update_count":    self.update_count,
            "conflict_rate":   round(self.conflict_rate, 4),
            "confidence_score": round(self.confidence_score, 4),
            "is_stable":       self.is_stable,
        }


@dataclass(frozen=True)
class LearnSession:
    """บันทึก 1 /learn session"""
    session_id:   str
    raw_input:    str
    subject:      str
    input_value:  float          # ค่าที่ parse ได้ (0.0–1.0)
    belief_before: Dict[str, float]
    belief_after:  Dict[str, float]
    consolidated:  bool
    timestamp:    float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id":   self.session_id,
            "raw_input":    self.raw_input[:80],
            "subject":      self.subject,
            "input_value":  round(self.input_value, 3),
            "consolidated":  self.consolidated,
            "belief_before": {k: round(v, 3) for k, v in self.belief_before.items()},
            "belief_after":  {k: round(v, 3) for k, v in self.belief_after.items()},
        }


# ─────────────────────────────────────────────────────────────────────────────
# LearnMode
# ─────────────────────────────────────────────────────────────────────────────

class LearnMode:
    """
    การเรียนรู้แบบตั้งใจ — ไม่มี emotional heuristics

    ใช้:
      result = learn_mode.learn("neural network คือ graph ของ nodes")
      print(result.subject, result.consolidated)
    """

    def __init__(self, learning_rate: float = LEARN_LEARNING_RATE):
        self._lr       = learning_rate
        self._beliefs: Dict[str, Belief] = {}   # subject → Belief
        self._sessions: List[LearnSession] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Main: learn จาก text
    # ─────────────────────────────────────────────────────────────────────────

    def learn(self, text: str) -> LearnSession:
        """
        เรียนรู้จาก text แบบ structured

        Args:
            text: ข้อความที่ต้องการเรียนรู้

        Returns:
            LearnSession — ผลของ session นี้
        """
        # 1. parse
        subject, input_value = self._parse_input(text)

        # 2. snapshot before
        before = {}
        if subject in self._beliefs:
            b = self._beliefs[subject]
            before = {
                "mean":     b.belief_mean,
                "variance": b.belief_variance,
            }

        # 3. update belief
        belief = self._update_belief(subject, input_value, text)

        # 4. snapshot after
        after = {
            "mean":     belief.belief_mean,
            "variance": belief.belief_variance,
        }

        # 5. consolidate ถ้าเสถียรพอ
        consolidated = self._consolidate_if_ready(belief)

        session = LearnSession(
            session_id    = f"ls_{int(time.time()*1000) % 999999}",
            raw_input     = text,
            subject       = subject,
            input_value   = input_value,
            belief_before = before,
            belief_after  = after,
            consolidated  = consolidated,
        )
        self._sessions.append(session)

        logger.info(
            f"[LearnMode] LEARN subject='{subject}' "
            f"mean={belief.belief_mean:.3f} var={belief.belief_variance:.3f} "
            f"consolidated={consolidated}"
        )
        return session

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Parse Input
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_input(self, text: str) -> Tuple[str, float]:
        """
        แยก subject + input_value จาก text

        รองรับ format "context:input" จาก auto-learn
        """
        # แยก context prefix ออกถ้ามี
        if ":" in text:
            parts = text.split(":", 1)
            context_hint = parts[0].strip()
            core_text    = parts[1].strip()
        else:
            context_hint = ""
            core_text    = text.strip()

        text_lower = core_text.lower()

        # input_value จาก certainty markers
        input_value = 0.7  # default

        high_certainty = {
            "แน่ใจ", "ชัดเจน", "definitely", "always", "ตลอดเวลา",
            "คือ", "is", "are", "was", "were", "คือการ", "หมายถึง",
        }
        medium_certainty = {
            "อาจจะ", "maybe", "น่าจะ", "probably", "likely",
            "ส่วนใหญ่", "often", "บางครั้ง",
        }
        low_certainty = {
            "ไม่แน่", "uncertain", "unclear", "อาจ", "might",
            "บางที", "sometimes", "ไม่แน่ใจ",
        }

        words = set(text_lower.split())
        if words & high_certainty:
            input_value = 0.85
        elif words & medium_certainty:
            input_value = 0.55
        elif words & low_certainty:
            input_value = 0.35

        # subject = context + คำสำคัญ 2 คำแรก (ไม่รวม stop words)
        stop_words = {
            "คือ", "is", "are", "the", "a", "an", "ที่", "และ",
            "หรือ", "of", "in", "on", "at", "to", "for",
            "ครับ", "ค่ะ", "นะ", "นะครับ",
        }
        tokens = [w for w in core_text.split() if w.lower() not in stop_words]
        core_subject = " ".join(tokens[:2]).strip() if tokens else core_text[:20]

        # รวม context ถ้ามี เพื่อให้ subject แตกต่างกันระหว่าง contexts
        if context_hint and context_hint != "general":
            subject = f"[{context_hint}] {core_subject}"
        else:
            subject = core_subject

        if len(subject) < 2:
            subject = core_text[:30]

        return subject, input_value

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Update Belief
    # ─────────────────────────────────────────────────────────────────────────

    def _update_belief(
        self,
        subject:     str,
        input_value: float,
        source_text: str,
    ) -> Belief:
        """
        อัปเดต belief ตาม document:
          belief_mean += lr * (input_value - belief_mean)
          belief_variance += noise_estimate
          update_count += 1

        ถ้าขัดแย้ง → เพิ่ม variance ก่อน ห้ามเขียนทับทันที
        """
        if subject not in self._beliefs:
            # Belief ใหม่
            self._beliefs[subject] = Belief(
                subject        = subject,
                belief_mean    = input_value,
                belief_variance = NOISE_ESTIMATE_DEFAULT,
                update_count   = 1,
                source         = "learn",
            )
            return self._beliefs[subject]

        b = self._beliefs[subject]

        # ตรวจ conflict — input ห่างจาก mean มากเกิน
        delta       = input_value - b.belief_mean
        is_conflict = abs(delta) > 0.3

        if is_conflict:
            # เพิ่ม variance ก่อน — ห้ามเขียนทับทันที
            new_variance  = min(0.9, b.belief_variance + NOISE_ESTIMATE_DEFAULT * 2)
            new_mean      = b.belief_mean + self._lr * 0.3 * delta  # อัปเดตช้าลง
            new_conflict  = min(1.0, b.conflict_rate + 0.1)
            logger.debug(
                f"[LearnMode] CONFLICT subject='{subject}' "
                f"delta={delta:.3f} → variance up"
            )
        else:
            # ปกติ — อัปเดตตาม document
            new_mean     = b.belief_mean + self._lr * delta
            new_variance = max(0.01, b.belief_variance - 0.01)  # ลดลงเมื่อสอดคล้อง
            new_conflict = b.conflict_rate

        self._beliefs[subject] = Belief(
            subject        = subject,
            belief_mean    = max(0.0, min(1.0, new_mean)),
            belief_variance = new_variance,
            update_count   = b.update_count + 1,
            conflict_rate  = new_conflict,
            last_updated   = time.time(),
            source         = "learn",
        )
        return self._beliefs[subject]

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Consolidate
    # ─────────────────────────────────────────────────────────────────────────

    def _consolidate_if_ready(self, belief: Belief) -> bool:
        """
        Auto-Consolidation ตาม document:
          ถ้าซ้ำเกิน threshold + variance ลดลง → long-term

        Returns:
            True ถ้า consolidate แล้ว
        """
        if not belief.needs_consolidation:
            return False

        logger.info(
            f"[LearnMode] CONSOLIDATE subject='{belief.subject}' "
            f"count={belief.update_count} var={belief.belief_variance:.3f}"
        )
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Query
    # ─────────────────────────────────────────────────────────────────────────

    def get_belief(self, subject: str) -> Optional[Belief]:
        """ดู belief ของ subject"""
        # exact match ก่อน
        if subject in self._beliefs:
            return self._beliefs[subject]
        # partial match
        s_lower = subject.lower()
        for key, belief in self._beliefs.items():
            if s_lower in key.lower() or key.lower() in s_lower:
                return belief
        return None

    def get_consolidated(self) -> List[Belief]:
        """คืน beliefs ที่ consolidate แล้ว (พร้อมลง long-term)"""
        return [b for b in self._beliefs.values() if b.needs_consolidation]

    def summary(self) -> str:
        """สรุป beliefs ทั้งหมด — ใช้แสดงใน Main.py"""
        if not self._beliefs:
            return "ยังไม่มีข้อมูลที่เรียนรู้ไว้"

        lines = [f"สิ่งที่เรียนรู้ ({len(self._beliefs)} หัวข้อ):"]
        for subject, b in sorted(
            self._beliefs.items(),
            key=lambda x: -x[1].confidence_score
        ):
            stability = "✓ เสถียร" if b.is_stable else "~ ยังไม่แน่"
            consolidated = " [long-term]" if b.needs_consolidation else ""
            lines.append(
                f"  • {subject:<30} "
                f"conf={b.confidence_score:.2f} "
                f"var={b.belief_variance:.2f} "
                f"{stability}{consolidated}"
            )
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        consolidated = self.get_consolidated()
        conflict_subjects = [
            b.subject for b in self._beliefs.values()
            if b.conflict_rate > 0.2
        ]
        return {
            "total_beliefs":   len(self._beliefs),
            "sessions":        len(self._sessions),
            "consolidated":    len(consolidated),
            "stable_beliefs":  sum(1 for b in self._beliefs.values() if b.is_stable),
            "conflicts":       len(conflict_subjects),
            "conflict_subjects": conflict_subjects[:5],
        }

    @property
    def beliefs(self) -> Dict[str, Belief]:
        return dict(self._beliefs)

    @property
    def sessions(self) -> List[LearnSession]:
        return list(self._sessions)