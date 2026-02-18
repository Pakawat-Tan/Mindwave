"""
Pattern Recognition — หา patterns ใน interactions

Features:
  1. sequence detection  — หา pattern ซ้ำในลำดับ interaction
  2. temporal pattern    — pattern ตามช่วงเวลา (วัน/สัปดาห์)
  3. user behavior       — จดจำพฤติกรรม user
  4. context pattern     — context ไหนมักตามหลัง context ไหน
  5. error pattern       — pattern ของ errors ที่เกิดซ้ำ
  6. success pattern     — อะไรที่ทำแล้วสำเร็จบ่อย

Flow:
  BrainController.logs → PatternRecognition.analyze()
    → detect_sequences, detect_temporal, detect_behavior,
       detect_context_transitions, detect_errors, detect_success
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger("mindwave.pattern")


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class PatternType(Enum):
    SEQUENCE = "sequence"
    TEMPORAL = "temporal"
    BEHAVIOR = "behavior"
    CONTEXT  = "context"
    ERROR    = "error"
    SUCCESS  = "success"


class TimeWindow(Enum):
    HOURLY  = "hourly"
    DAILY   = "daily"
    WEEKLY  = "weekly"
    MONTHLY = "monthly"


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SequencePattern:
    """Pattern ที่พบในลำดับ interactions"""
    sequence:   Tuple[str, ...]  # ("math", "general", "math")
    frequency:  int
    avg_confidence: float
    success_rate: float  # % ของ sequence ที่จบด้วย commit
    first_seen: float
    last_seen:  float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence":      list(self.sequence),
            "frequency":     self.frequency,
            "avg_confidence": round(self.avg_confidence, 3),
            "success_rate":  round(self.success_rate, 3),
        }


@dataclass(frozen=True)
class TemporalPattern:
    """Pattern ตามช่วงเวลา"""
    window:     TimeWindow
    peak_hours: List[int]  # [9, 14, 18] ชั่วโมงที่ active มาก
    peak_days:  List[int]  # [1, 3, 5] วันที่ active (0=Mon)
    activity_dist: Dict[str, int]  # {"morning": 10, "afternoon": 15, ...}
    avg_interactions_per_day: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window":      self.window.value,
            "peak_hours":  self.peak_hours,
            "peak_days":   self.peak_days,
            "activity_dist": self.activity_dist,
            "avg_per_day": round(self.avg_interactions_per_day, 2),
        }


@dataclass(frozen=True)
class BehaviorPattern:
    """พฤติกรรม user"""
    preferred_contexts: List[str]  # contexts ที่ใช้บ่อย
    avg_session_length: int        # จำนวน interactions ต่อ session
    question_rate:      float      # % ของ interactions ที่เป็นคำถาม
    learning_preference: str       # "exploratory" / "focused"
    interaction_style:   str       # "verbose" / "concise"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preferred_contexts": self.preferred_contexts,
            "avg_session_length": self.avg_session_length,
            "question_rate":      round(self.question_rate, 3),
            "learning_preference": self.learning_preference,
            "interaction_style":   self.interaction_style,
        }


@dataclass(frozen=True)
class ContextTransition:
    """Pattern ของ context transitions"""
    from_context: str
    to_context:   str
    frequency:    int
    avg_confidence: float
    avg_time_gap:   float  # seconds ระหว่าง transitions

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from": self.from_context,
            "to":   self.to_context,
            "frequency": self.frequency,
            "avg_confidence": round(self.avg_confidence, 3),
            "avg_time_gap": round(self.avg_time_gap, 2),
        }


@dataclass(frozen=True)
class ErrorPattern:
    """Pattern ของ errors"""
    trigger:    str  # อะไรที่ทำให้เกิด error
    frequency:  int
    contexts:   List[str]
    avg_confidence: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger":    self.trigger,
            "frequency":  self.frequency,
            "contexts":   self.contexts,
            "avg_confidence": round(self.avg_confidence, 3),
            "description": self.description,
        }


@dataclass(frozen=True)
class SuccessPattern:
    """Pattern ที่ทำแล้วสำเร็จบ่อย"""
    context:    str
    approach:   str  # "confident" / "cautious" / "exploratory"
    frequency:  int
    avg_confidence: float
    success_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "context":    self.context,
            "approach":   self.approach,
            "frequency":  self.frequency,
            "avg_confidence": round(self.avg_confidence, 3),
            "success_rate":  round(self.success_rate, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# PatternRecognition
# ─────────────────────────────────────────────────────────────────────────────

class PatternRecognition:
    """
    หา patterns ใน interactions

    รับ BrainLogs → วิเคราะห์ → คืน patterns
    """

    def __init__(self, min_frequency: int = 2):
        self._min_freq = min_frequency  # pattern ต้องเกิดอย่างน้อยกี่ครั้ง

        self._sequences:   List[SequencePattern]     = []
        self._temporal:    List[TemporalPattern]     = []
        self._behaviors:   List[BehaviorPattern]     = []
        self._transitions: List[ContextTransition]   = []
        self._errors:      List[ErrorPattern]        = []
        self._successes:   List[SuccessPattern]      = []

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Sequence Detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_sequences(
        self,
        logs: List[Any],
        window_size: int = 3,
    ) -> List[SequencePattern]:
        """
        หา pattern ซ้ำในลำดับ interactions

        Args:
            logs: BrainLogs
            window_size: ขนาด window ที่ดู (3 = ดู 3 contexts ติดกัน)

        Returns:
            List[SequencePattern]
        """
        if len(logs) < window_size:
            return []

        # extract sequences
        seq_data: Dict[Tuple[str, ...], List[Any]] = defaultdict(list)
        for i in range(len(logs) - window_size + 1):
            window = logs[i:i + window_size]
            seq = tuple(log.context for log in window)
            seq_data[seq].extend(window)

        # filter by min frequency
        patterns = []
        for seq, seq_logs in seq_data.items():
            freq = len(seq_logs) // window_size
            if freq < self._min_freq:
                continue

            avg_conf = sum(log.confidence for log in seq_logs) / len(seq_logs)
            success = sum(
                1 for log in seq_logs if log.outcome == "commit"
            )
            success_rate = success / len(seq_logs)

            pattern = SequencePattern(
                sequence       = seq,
                frequency      = freq,
                avg_confidence = avg_conf,
                success_rate   = success_rate,
                first_seen     = seq_logs[0].timestamp,
                last_seen      = seq_logs[-1].timestamp,
            )
            patterns.append(pattern)

        self._sequences.extend(patterns)
        if patterns:
            logger.info(f"[PatternRecognition] DETECT_SEQUENCES {len(patterns)} patterns")
        return patterns

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Temporal Pattern
    # ─────────────────────────────────────────────────────────────────────────

    def detect_temporal(
        self,
        logs: List[Any],
        window: TimeWindow = TimeWindow.DAILY,
    ) -> TemporalPattern:
        """
        หา pattern ตามช่วงเวลา

        Args:
            logs: BrainLogs
            window: ระดับเวลาที่สนใจ

        Returns:
            TemporalPattern
        """
        if not logs:
            return TemporalPattern(
                window=window, peak_hours=[], peak_days=[],
                activity_dist={}, avg_interactions_per_day=0.0,
            )

        # count by hour
        hour_counts: Dict[int, int] = defaultdict(int)
        day_counts:  Dict[int, int] = defaultdict(int)

        for log in logs:
            dt = datetime.fromtimestamp(log.timestamp)
            hour_counts[dt.hour] += 1
            day_counts[dt.weekday()] += 1

        # peak hours (top 3)
        peak_hours = sorted(hour_counts, key=hour_counts.get, reverse=True)[:3]

        # peak days (top 3)
        peak_days = sorted(day_counts, key=day_counts.get, reverse=True)[:3]

        # activity distribution
        activity_dist = {
            "morning":   sum(hour_counts[h] for h in range(6, 12)),
            "afternoon": sum(hour_counts[h] for h in range(12, 18)),
            "evening":   sum(hour_counts[h] for h in range(18, 24)),
            "night":     sum(hour_counts[h] for h in range(0, 6)),
        }

        # avg per day
        if logs:
            first = datetime.fromtimestamp(logs[0].timestamp)
            last  = datetime.fromtimestamp(logs[-1].timestamp)
            days  = max(1, (last - first).days + 1)
            avg_per_day = len(logs) / days
        else:
            avg_per_day = 0.0

        pattern = TemporalPattern(
            window                   = window,
            peak_hours               = peak_hours,
            peak_days                = peak_days,
            activity_dist            = activity_dist,
            avg_interactions_per_day = avg_per_day,
        )
        self._temporal.append(pattern)
        logger.info(
            f"[PatternRecognition] DETECT_TEMPORAL "
            f"peak_hours={peak_hours} peak_days={peak_days}"
        )
        return pattern

    # ─────────────────────────────────────────────────────────────────────────
    # 3. User Behavior
    # ─────────────────────────────────────────────────────────────────────────

    def detect_behavior(self, logs: List[Any]) -> BehaviorPattern:
        """
        จดจำพฤติกรรม user

        - preferred contexts
        - session length
        - question rate (outcome = ask)
        - learning preference
        - interaction style
        """
        if not logs:
            return BehaviorPattern(
                preferred_contexts=[], avg_session_length=0,
                question_rate=0.0, learning_preference="unknown",
                interaction_style="unknown",
            )

        # preferred contexts (top 3)
        ctx_counts: Dict[str, int] = defaultdict(int)
        for log in logs:
            ctx_counts[log.context] += 1
        preferred = sorted(ctx_counts, key=ctx_counts.get, reverse=True)[:3]

        # avg session length (ประมาณจาก time gaps)
        sessions = 1
        for i in range(1, len(logs)):
            gap = logs[i].timestamp - logs[i-1].timestamp
            if gap > 1800:  # 30 min = new session
                sessions += 1
        avg_session = len(logs) / sessions

        # question rate
        questions = sum(1 for log in logs if log.outcome == "ask")
        question_rate = questions / len(logs)

        # learning preference
        learned = sum(1 for log in logs if log.learned)
        learning_rate = learned / len(logs)
        if learning_rate > 0.5:
            learning_pref = "exploratory"
        else:
            learning_pref = "focused"

        # interaction style (ประมาณจาก avg text length)
        # ถ้าไม่มี text length → ใช้ outcome แทน
        verbose_outcomes = sum(
            1 for log in logs if log.outcome in ("commit", "conditional")
        )
        if verbose_outcomes / len(logs) > 0.6:
            style = "verbose"
        else:
            style = "concise"

        pattern = BehaviorPattern(
            preferred_contexts  = preferred,
            avg_session_length  = int(avg_session),
            question_rate       = question_rate,
            learning_preference = learning_pref,
            interaction_style   = style,
        )
        self._behaviors.append(pattern)
        logger.info(
            f"[PatternRecognition] DETECT_BEHAVIOR "
            f"pref={preferred} style={style}"
        )
        return pattern

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Context Transitions
    # ─────────────────────────────────────────────────────────────────────────

    def detect_context_transitions(
        self,
        logs: List[Any],
    ) -> List[ContextTransition]:
        """
        context ไหนมักตามหลัง context ไหน

        Returns:
            List[ContextTransition]
        """
        if len(logs) < 2:
            return []

        # count transitions
        trans_data: Dict[Tuple[str, str], List[Tuple[Any, Any]]] = defaultdict(list)
        for i in range(len(logs) - 1):
            from_ctx = logs[i].context
            to_ctx   = logs[i+1].context
            if from_ctx != to_ctx:  # สนใจแค่เปลี่ยน context
                trans_data[(from_ctx, to_ctx)].append((logs[i], logs[i+1]))

        # filter by frequency
        patterns = []
        for (from_ctx, to_ctx), pairs in trans_data.items():
            freq = len(pairs)
            if freq < self._min_freq:
                continue

            avg_conf = sum(
                (l1.confidence + l2.confidence) / 2
                for l1, l2 in pairs
            ) / freq

            avg_gap = sum(
                l2.timestamp - l1.timestamp for l1, l2 in pairs
            ) / freq

            pattern = ContextTransition(
                from_context   = from_ctx,
                to_context     = to_ctx,
                frequency      = freq,
                avg_confidence = avg_conf,
                avg_time_gap   = avg_gap,
            )
            patterns.append(pattern)

        self._transitions.extend(patterns)
        if patterns:
            logger.info(
                f"[PatternRecognition] DETECT_TRANSITIONS {len(patterns)} patterns"
            )
        return patterns

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Error Pattern
    # ─────────────────────────────────────────────────────────────────────────

    def detect_errors(self, logs: List[Any]) -> List[ErrorPattern]:
        """
        pattern ของ errors ที่เกิดซ้ำ

        - outcome = reject / silence
        - confidence สูงแต่ผิด
        """
        if not logs:
            return []

        # errors = reject/silence
        errors = [log for log in logs if log.outcome in ("reject", "silence")]
        if not errors:
            return []

        # group by context
        ctx_errors: Dict[str, List[Any]] = defaultdict(list)
        for log in errors:
            ctx_errors[log.context].append(log)

        patterns = []
        for ctx, ctx_logs in ctx_errors.items():
            freq = len(ctx_logs)
            if freq < self._min_freq:
                continue

            avg_conf = sum(log.confidence for log in ctx_logs) / freq

            pattern = ErrorPattern(
                trigger         = f"context={ctx}",
                frequency       = freq,
                contexts        = [ctx],
                avg_confidence  = avg_conf,
                description     = f"{freq} errors in {ctx} context",
            )
            patterns.append(pattern)

        self._errors.extend(patterns)
        if patterns:
            logger.warning(
                f"[PatternRecognition] DETECT_ERRORS {len(patterns)} patterns"
            )
        return patterns

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Success Pattern
    # ─────────────────────────────────────────────────────────────────────────

    def detect_success(self, logs: List[Any]) -> List[SuccessPattern]:
        """
        อะไรที่ทำแล้วสำเร็จบ่อย

        - outcome = commit
        - confidence สูง + commit
        """
        if not logs:
            return []

        successes = [log for log in logs if log.outcome == "commit"]
        if not successes:
            return []

        # group by context + approach
        success_data: Dict[Tuple[str, str], List[Any]] = defaultdict(list)
        for log in successes:
            # classify approach
            if log.confidence > 0.8:
                approach = "confident"
            elif log.confidence < 0.5:
                approach = "cautious"
            else:
                approach = "exploratory"

            success_data[(log.context, approach)].append(log)

        patterns = []
        for (ctx, approach), ctx_logs in success_data.items():
            freq = len(ctx_logs)
            if freq < self._min_freq:
                continue

            avg_conf = sum(log.confidence for log in ctx_logs) / freq

            # success rate (ทั้งหมดเป็น commit แล้ว → 1.0)
            success_rate = 1.0

            pattern = SuccessPattern(
                context        = ctx,
                approach       = approach,
                frequency      = freq,
                avg_confidence = avg_conf,
                success_rate   = success_rate,
            )
            patterns.append(pattern)

        self._successes.extend(patterns)
        if patterns:
            logger.info(
                f"[PatternRecognition] DETECT_SUCCESS {len(patterns)} patterns"
            )
        return patterns

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "sequences":   len(self._sequences),
            "temporal":    len(self._temporal),
            "behaviors":   len(self._behaviors),
            "transitions": len(self._transitions),
            "errors":      len(self._errors),
            "successes":   len(self._successes),
            "min_frequency": self._min_freq,
        }

    @property
    def sequences(self) -> List[SequencePattern]:
        return list(self._sequences)

    @property
    def temporal(self) -> List[TemporalPattern]:
        return list(self._temporal)

    @property
    def behaviors(self) -> List[BehaviorPattern]:
        return list(self._behaviors)

    @property
    def transitions(self) -> List[ContextTransition]:
        return list(self._transitions)

    @property
    def errors(self) -> List[ErrorPattern]:
        return list(self._errors)

    @property
    def successes(self) -> List[SuccessPattern]:
        return list(self._successes)