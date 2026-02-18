"""
MetaCognition — Brain คิดเกี่ยวกับการคิดของตัวเอง

Features:
  1. self-reflection     — วิเคราะห์การตัดสินใจจาก BrainLog
  2. confidence calibration — ปรับ confidence weights ตามผลลัพธ์จริง
  3. error awareness     — ตรวจจับว่าตอบผิด + บันทึก error pattern
  4. learning tracking   — ติดตาม trend ของการเรียนรู้
  5. strategy selection  — แนะนำ strategy ตาม context + history

Flow:
  BrainController.respond() → logs สะสม
  MetaCognition.reflect(logs) → วิเคราะห์
    → calibrate, detect_errors, track_learning, suggest_strategy
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any

logger = logging.getLogger("mindwave.metacognition")


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class Strategy(Enum):
    """Strategy ที่ Brain ควรใช้"""
    AGGRESSIVE   = "aggressive"    # มั่นใจสูง → commit เร็ว
    CAUTIOUS     = "cautious"      # ไม่แน่ใจ → ask มากขึ้น
    EXPLORATORY  = "exploratory"   # ทดลองสิ่งใหม่
    CONSERVATIVE = "conservative"  # ยึดติดกับที่รู้
    ADAPTIVE     = "adaptive"      # ปรับตาม context


class LearningTrend(Enum):
    """ทิศทางการเรียนรู้"""
    IMPROVING  = "improving"   # ดีขึ้นเรื่อยๆ
    DEGRADING  = "degrading"   # แย่ลง
    PLATEAU    = "plateau"     # คงที่
    UNSTABLE   = "unstable"    # ขึ้นลงไม่แน่นอน
    UNKNOWN    = "unknown"     # ข้อมูลไม่พอ


class ErrorType(Enum):
    """ประเภทของ error"""
    OVERCONFIDENT   = "overconfident"    # confidence สูงแต่ผิด
    UNDERCONFIDENT  = "underconfident"   # confidence ต่ำแต่ถูก
    INCONSISTENT    = "inconsistent"     # context เดียวกันตอบไม่เหมือนกัน
    RULE_VIOLATION  = "rule_violation"   # ฝ่า rule/policy
    LOW_QUALITY     = "low_quality"      # outcome ไม่ดี


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReflectionResult:
    """ผลของ self-reflection"""
    log_count:        int
    avg_confidence:   float
    outcome_dist:     Dict[str, int]  # {"commit": 10, "ask": 3, ...}
    context_coverage: Dict[str, int]  # {"math": 5, "general": 8, ...}
    quality_score:    float           # 0.0-1.0
    timestamp:        float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "log_count":      self.log_count,
            "avg_confidence": round(self.avg_confidence, 3),
            "outcome_dist":   self.outcome_dist,
            "quality_score":  round(self.quality_score, 3),
        }


@dataclass(frozen=True)
class CalibrationUpdate:
    """การปรับ confidence calibration"""
    before_bias:     float  # bias ก่อนปรับ (predicted - actual)
    after_bias:      float  # bias หลังปรับ
    adjustment:      float  # จำนวนที่ปรับ
    sample_size:     int
    timestamp:       float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "before_bias": round(self.before_bias, 3),
            "after_bias":  round(self.after_bias, 3),
            "adjustment":  round(self.adjustment, 3),
            "sample_size": self.sample_size,
        }


@dataclass(frozen=True)
class ErrorPattern:
    """Pattern ของ error ที่พบ"""
    error_type:   ErrorType
    context:      str
    frequency:    int
    avg_confidence: float
    description:  str
    timestamp:    float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type":   self.error_type.value,
            "context":      self.context,
            "frequency":    self.frequency,
            "avg_confidence": round(self.avg_confidence, 3),
            "description":  self.description,
        }


@dataclass(frozen=True)
class LearningTrack:
    """การติดตามการเรียนรู้"""
    trend:           LearningTrend
    confidence_trend: float  # slope: positive=ดีขึ้น, negative=แย่ลง
    interaction_count: int
    learned_count:   int
    learning_rate:   float   # learned / interactions
    timestamp:       float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trend":           self.trend.value,
            "confidence_trend": round(self.confidence_trend, 3),
            "learning_rate":   round(self.learning_rate, 3),
            "interaction_count": self.interaction_count,
        }


@dataclass(frozen=True)
class StrategyRecommendation:
    """แนะนำ strategy"""
    recommended:  Strategy
    confidence:   float      # 0.0-1.0 ความมั่นใจในคำแนะนำ
    reason:       str
    alternatives: List[Strategy] = field(default_factory=list)
    timestamp:    float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recommended":  self.recommended.value,
            "confidence":   round(self.confidence, 3),
            "reason":       self.reason,
            "alternatives": [s.value for s in self.alternatives],
        }


# ─────────────────────────────────────────────────────────────────────────────
# MetaCognition
# ─────────────────────────────────────────────────────────────────────────────

class MetaCognition:
    """
    Brain คิดเกี่ยวกับการคิดของตัวเอง

    รับ BrainLogs → วิเคราะห์ → ปรับปรุง
    """

    def __init__(self):
        self._reflections:  List[ReflectionResult]      = []
        self._calibrations: List[CalibrationUpdate]     = []
        self._errors:       List[ErrorPattern]          = []
        self._tracks:       List[LearningTrack]         = []
        self._strategies:   List[StrategyRecommendation] = []

        # calibration state
        self._confidence_bias:  float = 0.0  # ค่า bias ปัจจุบัน
        self._calibration_count: int  = 0

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Self-Reflection
    # ─────────────────────────────────────────────────────────────────────────

    def reflect(self, logs: List[Any]) -> ReflectionResult:
        """
        วิเคราะห์ BrainLogs

        Args:
            logs: List[BrainLog] จาก BrainController

        Returns:
            ReflectionResult
        """
        if not logs:
            return ReflectionResult(
                log_count=0, avg_confidence=0.0,
                outcome_dist={}, context_coverage={},
                quality_score=0.0,
            )

        # outcome distribution
        outcome_dist: Dict[str, int] = {}
        for log in logs:
            outcome_dist[log.outcome] = outcome_dist.get(log.outcome, 0) + 1

        # context coverage
        context_coverage: Dict[str, int] = {}
        for log in logs:
            context_coverage[log.context] = context_coverage.get(log.context, 0) + 1

        # avg confidence
        avg_conf = sum(log.confidence for log in logs) / len(logs)

        # quality score (อิงจาก outcome: commit=1.0, conditional=0.7, ask=0.5, reject=0.0)
        quality_weights = {
            "commit": 1.0, "conditional": 0.7,
            "ask": 0.5, "silence": 0.3, "reject": 0.0,
        }
        quality_score = sum(
            quality_weights.get(log.outcome, 0.0) for log in logs
        ) / len(logs)

        result = ReflectionResult(
            log_count        = len(logs),
            avg_confidence   = avg_conf,
            outcome_dist     = outcome_dist,
            context_coverage = context_coverage,
            quality_score    = quality_score,
        )
        self._reflections.append(result)

        logger.info(
            f"[MetaCognition] REFLECT {len(logs)} logs "
            f"quality={quality_score:.3f} "
            f"avg_conf={avg_conf:.3f}"
        )
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Confidence Calibration
    # ─────────────────────────────────────────────────────────────────────────

    def calibrate_confidence(
        self,
        logs: List[Any],
        actual_outcomes: Optional[List[float]] = None,
    ) -> CalibrationUpdate:
        """
        ปรับ confidence bias ตามผลลัพธ์จริง

        Args:
            logs: BrainLogs
            actual_outcomes: ผลลัพธ์จริง (0.0-1.0) ถ้ามี
                             None → ใช้ quality_score จาก outcome

        Returns:
            CalibrationUpdate
        """
        if not logs:
            return CalibrationUpdate(
                before_bias=0.0, after_bias=0.0,
                adjustment=0.0, sample_size=0,
            )

        # ถ้าไม่มี actual outcomes → ประมาณจาก outcome type
        if actual_outcomes is None:
            quality_map = {
                "commit": 1.0, "conditional": 0.7,
                "ask": 0.5, "silence": 0.3, "reject": 0.0,
            }
            actual_outcomes = [
                quality_map.get(log.outcome, 0.5) for log in logs
            ]

        # คำนวณ bias: predicted confidence - actual outcome
        predicted = [log.confidence for log in logs]
        bias = sum(
            pred - actual
            for pred, actual in zip(predicted, actual_outcomes)
        ) / len(logs)

        before_bias = self._confidence_bias

        # ปรับ bias (exponential moving average)
        alpha = 0.3  # learning rate
        self._confidence_bias = (1 - alpha) * self._confidence_bias + alpha * bias

        adjustment = self._confidence_bias - before_bias
        self._calibration_count += 1

        update = CalibrationUpdate(
            before_bias  = before_bias,
            after_bias   = self._confidence_bias,
            adjustment   = adjustment,
            sample_size  = len(logs),
        )
        self._calibrations.append(update)

        logger.info(
            f"[MetaCognition] CALIBRATE bias={self._confidence_bias:.3f} "
            f"adjustment={adjustment:+.3f}"
        )
        return update

    @property
    def confidence_bias(self) -> float:
        """ค่า bias ปัจจุบัน — ใช้ปรับ confidence ของ Brain"""
        return self._confidence_bias

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Error Awareness
    # ─────────────────────────────────────────────────────────────────────────

    def detect_errors(self, logs: List[Any]) -> List[ErrorPattern]:
        """
        ตรวจจับ error patterns

        - OVERCONFIDENT: confidence สูงแต่ outcome = reject/silence
        - UNDERCONFIDENT: confidence ต่ำแต่ outcome = commit
        - INCONSISTENT: context เดียวกันตอบไม่เหมือนกัน
        """
        if not logs:
            return []

        errors: List[ErrorPattern] = []

        # detect overconfident
        overconf = [
            log for log in logs
            if log.confidence > 0.8 and log.outcome in ("reject", "silence")
        ]
        if overconf:
            ctx_freq: Dict[str, int] = {}
            for log in overconf:
                ctx_freq[log.context] = ctx_freq.get(log.context, 0) + 1
            for ctx, freq in ctx_freq.items():
                pattern = ErrorPattern(
                    error_type   = ErrorType.OVERCONFIDENT,
                    context      = ctx,
                    frequency    = freq,
                    avg_confidence = sum(
                        l.confidence for l in overconf if l.context == ctx
                    ) / freq,
                    description  = f"High confidence but outcome was reject/silence",
                )
                errors.append(pattern)
                self._errors.append(pattern)

        # detect underconfident
        underconf = [
            log for log in logs
            if log.confidence < 0.3 and log.outcome == "commit"
        ]
        if underconf:
            ctx_freq = {}
            for log in underconf:
                ctx_freq[log.context] = ctx_freq.get(log.context, 0) + 1
            for ctx, freq in ctx_freq.items():
                pattern = ErrorPattern(
                    error_type   = ErrorType.UNDERCONFIDENT,
                    context      = ctx,
                    frequency    = freq,
                    avg_confidence = sum(
                        l.confidence for l in underconf if l.context == ctx
                    ) / freq,
                    description  = f"Low confidence but outcome was commit",
                )
                errors.append(pattern)
                self._errors.append(pattern)

        # detect inconsistent (context เดียวกัน outcome ต่างกัน)
        ctx_outcomes: Dict[str, set] = {}
        for log in logs:
            if log.context not in ctx_outcomes:
                ctx_outcomes[log.context] = set()
            ctx_outcomes[log.context].add(log.outcome)

        for ctx, outcomes in ctx_outcomes.items():
            if len(outcomes) > 2:  # มากกว่า 2 outcomes ต่าง
                pattern = ErrorPattern(
                    error_type   = ErrorType.INCONSISTENT,
                    context      = ctx,
                    frequency    = len(outcomes),
                    avg_confidence = sum(
                        l.confidence for l in logs if l.context == ctx
                    ) / sum(1 for l in logs if l.context == ctx),
                    description  = f"Inconsistent outcomes: {outcomes}",
                )
                errors.append(pattern)
                self._errors.append(pattern)

        if errors:
            logger.warning(
                f"[MetaCognition] DETECT_ERRORS {len(errors)} patterns"
            )
        return errors

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Learning Tracking
    # ─────────────────────────────────────────────────────────────────────────

    def track_learning(self, logs: List[Any]) -> LearningTrack:
        """
        ติดตาม trend ของการเรียนรู้

        - confidence trend: confidence เพิ่มขึ้นหรือลดลง
        - learning rate: % ของ interactions ที่เรียนรู้
        """
        if len(logs) < 5:
            return LearningTrack(
                trend=LearningTrend.UNKNOWN,
                confidence_trend=0.0,
                interaction_count=len(logs),
                learned_count=0,
                learning_rate=0.0,
            )

        # confidence trend: เปรียบเทียบ half แรกกับ half หลัง
        mid = len(logs) // 2
        first_half = logs[:mid]
        second_half = logs[mid:]

        avg_first = sum(l.confidence for l in first_half) / len(first_half)
        avg_second = sum(l.confidence for l in second_half) / len(second_half)

        conf_trend = avg_second - avg_first

        # learning rate
        learned_count = sum(1 for log in logs if log.learned)
        learning_rate = learned_count / len(logs)

        # ตัดสิน trend
        if conf_trend > 0.05:
            trend = LearningTrend.IMPROVING
        elif conf_trend < -0.05:
            trend = LearningTrend.DEGRADING
        elif abs(conf_trend) < 0.02:
            trend = LearningTrend.PLATEAU
        else:
            trend = LearningTrend.UNSTABLE

        track = LearningTrack(
            trend             = trend,
            confidence_trend  = conf_trend,
            interaction_count = len(logs),
            learned_count     = learned_count,
            learning_rate     = learning_rate,
        )
        self._tracks.append(track)

        logger.info(
            f"[MetaCognition] TRACK_LEARNING trend={trend.value} "
            f"conf_trend={conf_trend:+.3f} "
            f"lr={learning_rate:.3f}"
        )
        return track

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Strategy Selection
    # ─────────────────────────────────────────────────────────────────────────

    def suggest_strategy(
        self,
        logs:    List[Any],
        context: str = "",
    ) -> StrategyRecommendation:
        """
        แนะนำ strategy ตาม logs + context

        Logic:
          - quality สูง + confidence สูง → AGGRESSIVE
          - quality ต่ำ + errors มาก → CAUTIOUS
          - learning rate สูง → EXPLORATORY
          - learning rate ต่ำ → CONSERVATIVE
          - unstable → ADAPTIVE
        """
        if not logs:
            return StrategyRecommendation(
                recommended = Strategy.ADAPTIVE,
                confidence  = 0.3,
                reason      = "insufficient data",
            )

        reflection = self.reflect(logs)
        track = self.track_learning(logs)
        errors = self.detect_errors(logs)

        quality = reflection.quality_score
        avg_conf = reflection.avg_confidence
        learning_rate = track.learning_rate
        error_count = len(errors)

        # decision tree
        if quality > 0.8 and avg_conf > 0.7:
            strategy = Strategy.AGGRESSIVE
            reason = "high quality + high confidence"
            conf = 0.9
        elif error_count > 3 or quality < 0.4:
            strategy = Strategy.CAUTIOUS
            reason = f"errors={error_count} quality={quality:.2f}"
            conf = 0.8
        elif learning_rate > 0.7:
            strategy = Strategy.EXPLORATORY
            reason = f"high learning rate={learning_rate:.2f}"
            conf = 0.75
        elif learning_rate < 0.2:
            strategy = Strategy.CONSERVATIVE
            reason = f"low learning rate={learning_rate:.2f}"
            conf = 0.7
        else:
            strategy = Strategy.ADAPTIVE
            reason = "mixed signals — adapt to context"
            conf = 0.6

        # alternatives
        alternatives = [
            s for s in Strategy if s != strategy
        ][:2]

        rec = StrategyRecommendation(
            recommended  = strategy,
            confidence   = conf,
            reason       = reason,
            alternatives = alternatives,
        )
        self._strategies.append(rec)

        logger.info(
            f"[MetaCognition] SUGGEST_STRATEGY {strategy.value} "
            f"conf={conf:.2f} reason='{reason}'"
        )
        return rec

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "reflections":     len(self._reflections),
            "calibrations":    len(self._calibrations),
            "errors_detected": len(self._errors),
            "learning_tracks": len(self._tracks),
            "strategies":      len(self._strategies),
            "confidence_bias": round(self._confidence_bias, 3),
            "last_reflection": (
                self._reflections[-1].to_dict()
                if self._reflections else None
            ),
            "last_strategy": (
                self._strategies[-1].to_dict()
                if self._strategies else None
            ),
        }

    @property
    def reflections(self) -> List[ReflectionResult]:
        return list(self._reflections)

    @property
    def calibrations(self) -> List[CalibrationUpdate]:
        return list(self._calibrations)

    @property
    def errors(self) -> List[ErrorPattern]:
        return list(self._errors)

    @property
    def tracks(self) -> List[LearningTrack]:
        return list(self._tracks)

    @property
    def strategies(self) -> List[StrategyRecommendation]:
        return list(self._strategies)