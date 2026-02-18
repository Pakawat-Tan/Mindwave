"""
FeedbackInference — Implicit Feedback จากพฤติกรรม user

Brain สังเกตพฤติกรรมเองโดยไม่ต้องให้ user กด +/-

5 Signals:
  1. repeat detection  — ถามซ้ำ/คล้ายกัน = ไม่เข้าใจ         → penalty
  2. follow-up quality — ถามต่อเนื่อง = เข้าใจบางส่วน          → small reward
  3. context switch    — เปลี่ยน context = จบ topic แล้ว        → neutral/reward
  4. silence signal    — ไม่ถามต่อ (session end) = ตอบครบ      → reward
  5. confusion words   — งง/หมายความว่า/ไม่เข้าใจ = สับสน      → penalty

Effects:
  Immediate   — ปรับ confidence_bias + skill_weight ทันที
  Long-term   — สะสม FeedbackAtom → MetaCognition.calibrate()
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("mindwave.feedback")


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackType(Enum):
    REPEAT      = "repeat"        # ถามซ้ำ = ไม่เข้าใจ
    FOLLOW_UP   = "follow_up"     # ถามต่อ = เข้าใจบางส่วน
    CTX_SWITCH  = "ctx_switch"    # เปลี่ยน topic = จบแล้ว
    SILENCE     = "silence"       # ไม่ถามต่อ = ตอบครบ
    CONFUSION   = "confusion"     # คำสับสน = ไม่เข้าใจ


class FeedbackPolarity(Enum):
    POSITIVE = "positive"   # Brain ตอบดี → เพิ่ม weight
    NEGATIVE = "negative"   # Brain ตอบไม่ดี → ลด weight
    NEUTRAL  = "neutral"    # ไม่ชัดเจน


# ─────────────────────────────────────────────────────────────────────────────
# Keyword Lexicons
# ─────────────────────────────────────────────────────────────────────────────

CONFUSION_KEYWORDS = {
    # Thai
    "งง", "หมายความว่า", "ไม่เข้าใจ", "ไม่ชัด", "อธิบายอีกที",
    "พูดใหม่", "คืออะไร", "แปลว่า", "หมายถึง", "ไม่แน่ใจ",
    "ขอถามอีกครั้ง", "ช่วยอธิบาย", "ยังไม่เข้าใจ", "ทำไม",
    # English
    "confused", "don't understand", "what do you mean",
    "unclear", "explain again", "what is", "could you clarify",
    "i'm lost", "huh", "what?", "pardon",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FeedbackSignal:
    """Signal ที่ตรวจพบจากพฤติกรรม"""
    signal_type: FeedbackType
    polarity:    FeedbackPolarity
    strength:    float           # 0.0–1.0
    context:     str
    source_text: str             # ข้อความที่ trigger
    ref_log_id:  str             # BrainLog ที่ feedback อ้างถึง
    timestamp:   float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type":       self.signal_type.value,
            "polarity":   self.polarity.value,
            "strength":   round(self.strength, 3),
            "context":    self.context,
            "ref_log_id": self.ref_log_id,
        }


@dataclass
class FeedbackAtom:
    """สะสมผลระยะยาว — ให้ MetaCognition calibrate"""
    atom_id:      str   = field(default_factory=lambda: f"fb_{int(time.time()*1000)}")
    context:      str   = ""
    signals:      List[FeedbackSignal] = field(default_factory=list)
    net_reward:   float = 0.0     # ผลรวม reward ของ session
    session_len:  int   = 0       # จำนวน interactions
    created_at:   float = field(default_factory=time.time)

    @property
    def avg_reward(self) -> float:
        return self.net_reward / max(1, self.session_len)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atom_id":    self.atom_id,
            "context":    self.context,
            "net_reward": round(self.net_reward, 3),
            "avg_reward": round(self.avg_reward, 3),
            "session_len": self.session_len,
            "signal_count": len(self.signals),
        }


@dataclass(frozen=True)
class ImmediateEffect:
    """ผลทันที — ส่งกลับให้ BrainController ปรับ"""
    confidence_delta: float   # บวก/ลบ confidence_bias
    skill_delta:      float   # บวก/ลบ skill weight
    reason:           str
    signal_type:      FeedbackType

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confidence_delta": round(self.confidence_delta, 3),
            "skill_delta":      round(self.skill_delta, 3),
            "reason":           self.reason,
            "signal_type":      self.signal_type.value,
        }


# ─────────────────────────────────────────────────────────────────────────────
# FeedbackInference
# ─────────────────────────────────────────────────────────────────────────────

class FeedbackInference:
    """
    สังเกตพฤติกรรม user แล้วสรุป feedback เอง

    เรียกใช้:
      feedback = brain.feedback
      signal = feedback.infer(current_input, prev_log, context)
      if signal:
          brain.apply_feedback(signal.effect)
    """

    # Reward/Penalty magnitudes
    _REWARDS = {
        FeedbackType.SILENCE:    +0.08,
        FeedbackType.CTX_SWITCH: +0.03,
        FeedbackType.FOLLOW_UP:  +0.02,
        FeedbackType.REPEAT:     -0.06,
        FeedbackType.CONFUSION:  -0.10,
    }

    # Similarity threshold สำหรับ repeat detection
    _REPEAT_THRESHOLD = 0.6

    def __init__(self, session_window: int = 20):
        self._window        = session_window
        self._history:      Deque[Tuple[str, str, float]] = deque(maxlen=session_window)
        # (text, context, timestamp)

        self._signals:      List[FeedbackSignal] = []
        self._atoms:        List[FeedbackAtom]   = []
        self._current_atom: FeedbackAtom         = FeedbackAtom()

        # cumulative bias (ส่งให้ MetaCognition ทีหลัง)
        self._cumulative_confidence_delta: float = 0.0
        self._cumulative_skill_delta:      float = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # Main: infer feedback จาก input ใหม่
    # ─────────────────────────────────────────────────────────────────────────

    def infer(
        self,
        current_text: str,
        context:      str,
        prev_log:     Any,          # BrainLog ล่าสุด (optional)
        prev_context: str = "",
    ) -> Optional[FeedbackSignal]:
        """
        วิเคราะห์ input ใหม่เทียบกับ history → หา signal

        Args:
            current_text: input ปัจจุบัน
            context:      context ปัจจุบัน
            prev_log:     BrainLog ของ interaction ก่อนหน้า
            prev_context: context ก่อนหน้า

        Returns:
            FeedbackSignal ถ้าพบ, None ถ้าไม่พบ
        """
        ref_id = prev_log.log_id if prev_log else ""

        # ── Priority: confusion > repeat > follow_up > ctx_switch ──
        signal = (
            self._detect_confusion(current_text, context, ref_id) or
            self._detect_repeat(current_text, context, ref_id) or
            self._detect_follow_up(current_text, context, prev_context, ref_id) or
            self._detect_ctx_switch(context, prev_context, ref_id)
        )

        # บันทึก history
        self._history.append((current_text, context, time.time()))

        if signal:
            self._signals.append(signal)
            self._current_atom.signals.append(signal)
            self._current_atom.net_reward += self._REWARDS[signal.signal_type]
            self._current_atom.session_len += 1
            logger.info(
                f"[FeedbackInference] SIGNAL {signal.signal_type.value} "
                f"polarity={signal.polarity.value} strength={signal.strength:.2f}"
            )

        return signal

    # ─────────────────────────────────────────────────────────────────────────
    # Immediate Effect
    # ─────────────────────────────────────────────────────────────────────────

    def get_immediate_effect(
        self, signal: FeedbackSignal
    ) -> ImmediateEffect:
        """
        คำนวณผลทันที จาก signal

        Returns:
            ImmediateEffect — ส่งให้ BrainController ปรับ confidence + skill
        """
        reward = self._REWARDS[signal.signal_type] * signal.strength

        # confidence_delta: penalty ลด bias (ทำให้ระวังมากขึ้น)
        # reward เพิ่ม confidence
        confidence_delta = reward * 0.5

        # skill_delta: แรงกว่า — skill ปรับเร็ว
        skill_delta = reward

        self._cumulative_confidence_delta += confidence_delta
        self._cumulative_skill_delta      += skill_delta

        effect = ImmediateEffect(
            confidence_delta = confidence_delta,
            skill_delta      = skill_delta,
            reason           = f"{signal.signal_type.value}: {signal.source_text[:40]}",
            signal_type      = signal.signal_type,
        )
        logger.debug(
            f"[FeedbackInference] IMMEDIATE "
            f"conf_delta={confidence_delta:+.3f} "
            f"skill_delta={skill_delta:+.3f}"
        )
        return effect

    # ─────────────────────────────────────────────────────────────────────────
    # Long-term: seal session → FeedbackAtom
    # ─────────────────────────────────────────────────────────────────────────

    def seal_session(
        self,
        silence_reward: bool = False,
    ) -> FeedbackAtom:
        """
        ปิด session → สร้าง FeedbackAtom สำหรับ long-term learning

        เรียกเมื่อ:
          - user หยุดคุย (silence signal)
          - เปลี่ยน context ใหม่
          - /reset

        Args:
            silence_reward: ถ้า True → ให้ bonus reward (ตอบครบ)

        Returns:
            FeedbackAtom
        """
        if silence_reward:
            self._current_atom.net_reward += self._REWARDS[FeedbackType.SILENCE]
            silence_sig = FeedbackSignal(
                signal_type = FeedbackType.SILENCE,
                polarity    = FeedbackPolarity.POSITIVE,
                strength    = 0.8,
                context     = self._current_atom.context,
                source_text = "[session_end]",
                ref_log_id  = "",
            )
            self._current_atom.signals.append(silence_sig)
            self._signals.append(silence_sig)

        atom = self._current_atom
        self._atoms.append(atom)

        logger.info(
            f"[FeedbackInference] SEAL_SESSION "
            f"net_reward={atom.net_reward:.3f} "
            f"signals={len(atom.signals)}"
        )

        # เริ่ม atom ใหม่
        self._current_atom = FeedbackAtom()
        return atom

    def get_long_term_delta(self) -> Tuple[float, float]:
        """
        คืน cumulative delta สำหรับส่งให้ MetaCognition

        Returns:
            (confidence_delta, skill_delta)
        """
        c = self._cumulative_confidence_delta
        s = self._cumulative_skill_delta
        # reset หลังอ่าน
        self._cumulative_confidence_delta = 0.0
        self._cumulative_skill_delta      = 0.0
        return c, s

    # ─────────────────────────────────────────────────────────────────────────
    # Signal Detectors
    # ─────────────────────────────────────────────────────────────────────────

    def _detect_confusion(
        self, text: str, context: str, ref_id: str
    ) -> Optional[FeedbackSignal]:
        """ตรวจ confusion words"""
        text_lower = text.lower()
        found = [kw for kw in CONFUSION_KEYWORDS if kw in text_lower]
        if not found:
            return None

        strength = min(1.0, 0.5 + len(found) * 0.2)
        return FeedbackSignal(
            signal_type = FeedbackType.CONFUSION,
            polarity    = FeedbackPolarity.NEGATIVE,
            strength    = strength,
            context     = context,
            source_text = text,
            ref_log_id  = ref_id,
        )

    def _detect_repeat(
        self, text: str, context: str, ref_id: str
    ) -> Optional[FeedbackSignal]:
        """ตรวจถามซ้ำ — Jaccard similarity กับ history"""
        if len(self._history) < 1:
            return None

        words = set(text.lower().split())
        if not words:
            return None

        max_sim = 0.0
        for prev_text, prev_ctx, _ in self._history:
            if prev_ctx != context:
                continue
            prev_words = set(prev_text.lower().split())
            if not prev_words:
                continue
            sim = len(words & prev_words) / len(words | prev_words)
            if sim > max_sim:
                max_sim = sim

        if max_sim < self._REPEAT_THRESHOLD:
            return None

        strength = min(1.0, max_sim)
        return FeedbackSignal(
            signal_type = FeedbackType.REPEAT,
            polarity    = FeedbackPolarity.NEGATIVE,
            strength    = strength,
            context     = context,
            source_text = text,
            ref_log_id  = ref_id,
        )

    def _detect_follow_up(
        self, text: str, context: str,
        prev_context: str, ref_id: str
    ) -> Optional[FeedbackSignal]:
        """ตรวจถามต่อใน context เดิม — เข้าใจบางส่วน"""
        if not prev_context or context != prev_context:
            return None
        if len(self._history) < 1:
            return None

        # ถ้าคำถามใหม่ไม่ซ้ำ (ไม่ใช่ repeat) แต่อยู่ context เดิม = follow-up
        words = set(text.lower().split())
        if not words:
            return None

        last_text = self._history[-1][0] if self._history else ""
        last_words = set(last_text.lower().split())
        if last_words:
            sim = len(words & last_words) / len(words | last_words)
            if sim >= self._REPEAT_THRESHOLD:
                return None  # ซ้ำมากไป → repeat ดีกว่า

        return FeedbackSignal(
            signal_type = FeedbackType.FOLLOW_UP,
            polarity    = FeedbackPolarity.POSITIVE,
            strength    = 0.5,
            context     = context,
            source_text = text,
            ref_log_id  = ref_id,
        )

    def _detect_ctx_switch(
        self, context: str, prev_context: str, ref_id: str
    ) -> Optional[FeedbackSignal]:
        """ตรวจเปลี่ยน context — จบ topic แล้ว"""
        if not prev_context or context == prev_context:
            return None

        return FeedbackSignal(
            signal_type = FeedbackType.CTX_SWITCH,
            polarity    = FeedbackPolarity.POSITIVE,
            strength    = 0.4,
            context     = prev_context,  # feedback ต่อ context เก่า
            source_text = f"switched: {prev_context} → {context}",
            ref_log_id  = ref_id,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        by_polarity: Dict[str, int] = {
            "positive": 0, "negative": 0, "neutral": 0
        }
        for sig in self._signals:
            by_type[sig.signal_type.value] = (
                by_type.get(sig.signal_type.value, 0) + 1
            )
            by_polarity[sig.polarity.value] += 1

        return {
            "total_signals":    len(self._signals),
            "sealed_atoms":     len(self._atoms),
            "current_session":  len(self._current_atom.signals),
            "by_type":          by_type,
            "by_polarity":      by_polarity,
            "cumulative_conf":  round(self._cumulative_confidence_delta, 3),
            "cumulative_skill": round(self._cumulative_skill_delta, 3),
        }

    @property
    def signals(self) -> List[FeedbackSignal]:
        return list(self._signals)

    @property
    def atoms(self) -> List[FeedbackAtom]:
        return list(self._atoms)

    @property
    def current_atom(self) -> FeedbackAtom:
        return self._current_atom