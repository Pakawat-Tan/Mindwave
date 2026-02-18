"""
BeliefSystem — ระบบความเชื่อกลางของ Brain

ตาม document:
  - ความเชื่อทุกอย่างเป็นแบบ Probabilistic
  - ห้ามมีสถานะ True/False แบบ binary
  - เมื่อข้อมูลขัดแย้ง → เพิ่ม variance ก่อน ห้ามเขียนทับทันที
  - ต้องมีหลักฐานซ้ำหลายครั้งก่อนปรับ belief_mean อย่างมีนัยสำคัญ

ต่างจาก LearnMode:
  LearnMode    = session-based, ประมวลผลแต่ละ input
  BeliefSystem = persistent store ของ Brain ทั้งหมด
                 ทุก module อ่าน/เขียนผ่านที่นี่

Flow:
  input → update(subject, value)
        → conflict check
        → belief_mean update (ช้าถ้า conflict)
        → variance update
        → confidence_score recalculate
        → persist to store
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mindwave.belief")


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NOISE_ESTIMATE_INIT    = 0.20   # variance เริ่มต้น — ยังไม่แน่ใจ
NOISE_DECAY_NORMAL     = 0.02   # variance ลดลงเมื่อข้อมูลสอดคล้อง
NOISE_GROWTH_CONFLICT  = 0.08   # variance เพิ่มขึ้นเมื่อขัดแย้ง
CONFLICT_THRESHOLD     = 0.30   # delta ที่ถือว่าขัดแย้ง
STABLE_VARIANCE_MAX    = 0.10   # variance ต่ำกว่านี้ = เสถียร
STRONG_BELIEF_MIN      = 0.75   # confidence สูงกว่านี้ = เชื่อมั่น


# ─────────────────────────────────────────────────────────────────────────────
# BeliefEntry
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BeliefEntry:
    """
    ความเชื่อ 1 หัวข้อ — probabilistic เสมอ

    ไม่มี True/False — มีแค่ mean ± variance
    """
    subject:          str
    belief_mean:      float        # 0.0–1.0
    belief_variance:  float        # 0.0 = แน่ใจ, 1.0 = ไม่แน่เลย
    update_count:     int   = 0
    conflict_count:   int   = 0
    conflict_rate:    float = 0.0  # conflict_count / update_count
    last_value:       float = 0.5  # input ล่าสุด
    last_updated:     float = field(default_factory=time.time)
    created_at:       float = field(default_factory=time.time)
    context:          str   = "general"
    source:           str   = "auto"   # "auto" / "learn" / "feedback"

    # ── Derived ──────────────────────────────────────────────────

    @property
    def confidence_score(self) -> float:
        """ความมั่นใจ = mean ลบ variance (clamp 0–1)"""
        return max(0.0, min(1.0, self.belief_mean - self.belief_variance * 0.5))

    @property
    def is_stable(self) -> bool:
        return self.belief_variance <= STABLE_VARIANCE_MAX

    @property
    def is_strong(self) -> bool:
        return self.confidence_score >= STRONG_BELIEF_MIN

    @property
    def is_conflicted(self) -> bool:
        return self.conflict_rate > 0.3

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject":         self.subject,
            "belief_mean":     round(self.belief_mean, 4),
            "belief_variance": round(self.belief_variance, 4),
            "confidence_score": round(self.confidence_score, 4),
            "update_count":    self.update_count,
            "conflict_rate":   round(self.conflict_rate, 4),
            "is_stable":       self.is_stable,
            "is_strong":       self.is_strong,
            "context":         self.context,
            "source":          self.source,
            "last_updated":    round(self.last_updated, 2),
        }


# ─────────────────────────────────────────────────────────────────────────────
# UpdateResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class UpdateResult:
    """ผลของการ update belief"""
    subject:       str
    is_new:        bool
    was_conflict:  bool
    delta_mean:    float   # belief_mean เปลี่ยนไปเท่าไหร่
    delta_var:     float   # belief_variance เปลี่ยนไปเท่าไหร่
    confidence:    float
    is_stable:     bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject":      self.subject,
            "is_new":       self.is_new,
            "conflict":     self.was_conflict,
            "delta_mean":   round(self.delta_mean, 4),
            "delta_var":    round(self.delta_var, 4),
            "confidence":   round(self.confidence, 4),
            "stable":       self.is_stable,
        }


# ─────────────────────────────────────────────────────────────────────────────
# BeliefSystem
# ─────────────────────────────────────────────────────────────────────────────

class BeliefSystem:
    """
    ระบบความเชื่อกลางของ Brain

    ทุก module update ผ่าน update()
    ทุก module อ่านผ่าน get() / query()
    """

    def __init__(
        self,
        learning_rate: float = 0.3,
        persist_path:  str   = "Core/Data/beliefs.json",
    ):
        self._lr           = learning_rate
        self._persist_path = Path(persist_path)
        self._beliefs:     Dict[str, BeliefEntry] = {}
        self._history:     List[Dict[str, Any]]   = []  # update log

        # โหลด beliefs ที่บันทึกไว้ (ถ้ามี)
        self._load()
        logger.info(
            f"[BeliefSystem] INIT lr={learning_rate} "
            f"loaded={len(self._beliefs)} beliefs"
        )

    # ─────────────────────────────────────────────────────────────
    # Update — หลัก
    # ─────────────────────────────────────────────────────────────

    def update(
        self,
        subject:      str,
        input_value:  float,
        context:      str  = "general",
        source:       str  = "auto",
        learning_rate: Optional[float] = None,
    ) -> UpdateResult:
        """
        อัปเดต belief ตาม document:
          belief_mean += lr * (input_value - belief_mean)
          belief_variance += noise_estimate
          update_count += 1

        ถ้าขัดแย้ง:
          - เพิ่ม variance ก่อน
          - ลด learning_rate ลง
          - ห้ามเขียนทับ mean ทันที
        """
        lr = learning_rate if learning_rate is not None else self._lr

        is_new = subject not in self._beliefs
        if is_new:
            entry = BeliefEntry(
                subject         = subject,
                belief_mean     = input_value,
                belief_variance = NOISE_ESTIMATE_INIT,
                update_count    = 1,
                last_value      = input_value,
                context         = context,
                source          = source,
            )
            self._beliefs[subject] = entry
            result = UpdateResult(
                subject      = subject,
                is_new       = True,
                was_conflict = False,
                delta_mean   = 0.0,
                delta_var    = 0.0,
                confidence   = entry.confidence_score,
                is_stable    = entry.is_stable,
            )
            self._record(subject, input_value, result)
            return result

        # ── existing belief ───────────────────────────────────────
        b = self._beliefs[subject]
        old_mean = b.belief_mean
        old_var  = b.belief_variance
        delta    = input_value - old_mean
        is_conflict = abs(delta) > CONFLICT_THRESHOLD

        if is_conflict:
            # ขัดแย้ง — เพิ่ม variance ก่อน, ลด lr
            new_variance   = min(0.95, old_var + NOISE_GROWTH_CONFLICT)
            effective_lr   = lr * 0.3   # อัปเดตช้า
            new_conflict_c = b.conflict_count + 1
            logger.debug(
                f"[BeliefSystem] CONFLICT '{subject}' "
                f"delta={delta:+.3f} var {old_var:.3f}→{new_variance:.3f}"
            )
        else:
            # สอดคล้อง — variance ลดลง
            new_variance   = max(0.01, old_var - NOISE_DECAY_NORMAL)
            effective_lr   = lr
            new_conflict_c = b.conflict_count

        new_mean         = old_mean + effective_lr * delta
        new_mean         = max(0.0, min(1.0, new_mean))
        new_update_count = b.update_count + 1
        new_conflict_rate = new_conflict_c / new_update_count

        updated = BeliefEntry(
            subject         = subject,
            belief_mean     = new_mean,
            belief_variance = new_variance,
            update_count    = new_update_count,
            conflict_count  = new_conflict_c,
            conflict_rate   = new_conflict_rate,
            last_value      = input_value,
            last_updated    = time.time(),
            created_at      = b.created_at,
            context         = context or b.context,
            source          = source,
        )
        self._beliefs[subject] = updated

        result = UpdateResult(
            subject      = subject,
            is_new       = False,
            was_conflict = is_conflict,
            delta_mean   = new_mean - old_mean,
            delta_var    = new_variance - old_var,
            confidence   = updated.confidence_score,
            is_stable    = updated.is_stable,
        )
        self._record(subject, input_value, result)
        return result

    def update_from_feedback(
        self,
        subject:   str,
        polarity:  str,   # "positive" / "negative" / "neutral"
        strength:  float,
        context:   str = "general",
    ) -> Optional[UpdateResult]:
        """
        อัปเดต belief จาก FeedbackSignal

        positive → เพิ่ม belief_mean
        negative → ลด belief_mean + เพิ่ม variance
        """
        if subject not in self._beliefs:
            return None

        b = self._beliefs[subject]
        if polarity == "positive":
            value = min(1.0, b.belief_mean + 0.1 * strength)
        elif polarity == "negative":
            value = max(0.0, b.belief_mean - 0.15 * strength)
        else:
            return None

        return self.update(subject, value, context=context, source="feedback")

    # ─────────────────────────────────────────────────────────────
    # Read
    # ─────────────────────────────────────────────────────────────

    def get(self, subject: str) -> Optional[BeliefEntry]:
        """ดู belief ของ subject (exact match)"""
        return self._beliefs.get(subject)

    def query(self, keyword: str, context: str = "") -> List[BeliefEntry]:
        """ค้นหา beliefs ที่ subject มี keyword"""
        kw = keyword.lower()
        results = []
        for key, b in self._beliefs.items():
            if kw in key.lower():
                if not context or b.context == context:
                    results.append(b)
        return sorted(results, key=lambda x: -x.confidence_score)

    def strongest(self, context: str = "", n: int = 5) -> List[BeliefEntry]:
        """คืน beliefs ที่ confidence สูงสุด"""
        beliefs = list(self._beliefs.values())
        if context:
            beliefs = [b for b in beliefs if b.context == context]
        return sorted(beliefs, key=lambda x: -x.confidence_score)[:n]

    def conflicted(self) -> List[BeliefEntry]:
        """คืน beliefs ที่มี conflict_rate สูง"""
        return [b for b in self._beliefs.values() if b.is_conflicted]

    def stable(self) -> List[BeliefEntry]:
        """คืน beliefs ที่เสถียรแล้ว"""
        return [b for b in self._beliefs.values() if b.is_stable]

    # ─────────────────────────────────────────────────────────────
    # Persist
    # ─────────────────────────────────────────────────────────────

    def save(self) -> bool:
        """บันทึก beliefs ลง disk"""
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {k: asdict(v) for k, v in self._beliefs.items()}
            self._persist_path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug(f"[BeliefSystem] SAVE {len(data)} beliefs")
            return True
        except Exception as e:
            logger.error(f"[BeliefSystem] SAVE FAILED: {e}")
            return False

    def _load(self) -> None:
        """โหลด beliefs จาก disk"""
        try:
            if not self._persist_path.exists():
                return
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for key, val in data.items():
                # ลบ key ที่ dataclass ไม่รู้จัก
                valid = {k: v for k, v in val.items()
                         if k in BeliefEntry.__dataclass_fields__}
                self._beliefs[key] = BeliefEntry(**valid)
            logger.info(f"[BeliefSystem] LOADED {len(self._beliefs)} beliefs")
        except Exception as e:
            logger.warning(f"[BeliefSystem] LOAD FAILED: {e}")

    # ─────────────────────────────────────────────────────────────
    # History / Stats
    # ─────────────────────────────────────────────────────────────

    def _record(self, subject: str, value: float, result: UpdateResult) -> None:
        self._history.append({
            "subject":   subject,
            "value":     round(value, 3),
            "conflict":  result.was_conflict,
            "conf":      round(result.confidence, 3),
            "ts":        round(time.time(), 1),
        })
        if len(self._history) > 500:
            self._history = self._history[-500:]

    def stats(self) -> Dict[str, Any]:
        beliefs = list(self._beliefs.values())
        return {
            "total":          len(beliefs),
            "stable":         sum(1 for b in beliefs if b.is_stable),
            "strong":         sum(1 for b in beliefs if b.is_strong),
            "conflicted":     sum(1 for b in beliefs if b.is_conflicted),
            "avg_confidence": round(
                sum(b.confidence_score for b in beliefs) / max(1, len(beliefs)), 3
            ),
            "avg_variance":   round(
                sum(b.belief_variance for b in beliefs) / max(1, len(beliefs)), 3
            ),
            "total_updates":  len(self._history),
        }

    def summary(self, n: int = 10) -> str:
        """สรุป top-n beliefs"""
        top = self.strongest(n=n)
        if not top:
            return "ยังไม่มี beliefs ในระบบ"
        lines = [f"Beliefs ({len(self._beliefs)} total):"]
        for b in top:
            status = "✓" if b.is_stable else "~"
            conflict = " ⚠conflict" if b.is_conflicted else ""
            lines.append(
                f"  {status} {b.subject:<32} "
                f"conf={b.confidence_score:.2f} "
                f"var={b.belief_variance:.2f} "
                f"n={b.update_count}{conflict}"
            )
        return "\n".join(lines)

    @property
    def beliefs(self) -> Dict[str, BeliefEntry]:
        return dict(self._beliefs)

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)

    def __len__(self) -> int:
        return len(self._beliefs)