"""
Emotion Inference — ตรวจจับและติดตามอารมณ์

Features:
  1. text sentiment      — วิเคราะห์อารมณ์จากข้อความ
  2. behavior analysis   — อารมณ์จากพฤติกรรม (frequency, timing)
  3. emotion tracking    — ติดตามอารมณ์ต่อเนื่อง
  4. emotion influence   — อารมณ์มีผลต่อ response
  5. emotional state     — สถานะอารมณ์ปัจจุบัน
  6. emotion detection   — ตรวจจับอารมณ์หลัก (joy, sadness, anger, ...)

Algorithm:
  - Text sentiment: keyword-based + punctuation
  - Behavior: interaction patterns (frequency, timing, errors)
  - Tracking: exponential moving average
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Deque

logger = logging.getLogger("mindwave.emotion")


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class Emotion(Enum):
    """อารมณ์หลัก"""
    JOY       = "joy"
    SADNESS   = "sadness"
    ANGER     = "anger"
    FEAR      = "fear"
    SURPRISE  = "surprise"
    NEUTRAL   = "neutral"
    FRUSTRATION = "frustration"
    EXCITEMENT  = "excitement"


class Sentiment(Enum):
    """Sentiment แบบง่าย"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL  = "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EmotionScore:
    """คะแนนอารมณ์"""
    emotion:    Emotion
    confidence: float  # 0.0-1.0
    source:     str    # "text" / "behavior" / "combined"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "emotion":    self.emotion.value,
            "confidence": round(self.confidence, 3),
            "source":     self.source,
        }


@dataclass
class EmotionalState:
    """สถานะอารมณ์ปัจจุบัน"""
    primary_emotion:  Emotion
    intensity:        float  # 0.0-1.0
    sentiment:        Sentiment
    emotion_scores:   Dict[Emotion, float]  # scores ทุก emotion
    timestamp:        float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_emotion": self.primary_emotion.value,
            "intensity":       round(self.intensity, 3),
            "sentiment":       self.sentiment.value,
            "emotion_scores":  {
                e.value: round(s, 3)
                for e, s in self.emotion_scores.items()
            },
            "timestamp":       self.timestamp,
        }


@dataclass(frozen=True)
class BehaviorIndicator:
    """ตัวบ่งชี้พฤติกรรม"""
    indicator_type: str  # "high_frequency" / "rapid_succession" / "error_rate" / "timing"
    value:          float
    emotion_hint:   Emotion
    confidence:     float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type":         self.indicator_type,
            "value":        round(self.value, 3),
            "emotion_hint": self.emotion_hint.value,
            "confidence":   round(self.confidence, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Keyword Lexicons
# ─────────────────────────────────────────────────────────────────────────────

EMOTION_KEYWORDS = {
    Emotion.JOY: {
        "happy", "joy", "great", "awesome", "wonderful", "fantastic",
        "excited", "love", "excellent", "amazing", "perfect",
        "ดีใจ", "สนุก", "ยินดี", "ชอบ", "เยี่ยม",
    },
    Emotion.SADNESS: {
        "sad", "unhappy", "depressed", "disappointed", "miserable",
        "down", "blue", "upset", "hurt",
        "เศร้า", "เสียใจ", "ผิดหวัง", "ไม่สบายใจ",
    },
    Emotion.ANGER: {
        "angry", "mad", "furious", "annoyed", "irritated",
        "frustrated", "hate", "terrible", "awful",
        "โกรธ", "รำคาญ", "ขุ่นเคือง", "น่าหงุดหงิด",
    },
    Emotion.FEAR: {
        "scared", "afraid", "frightened", "worried", "anxious",
        "nervous", "concerned", "uneasy",
        "กลัว", "เกรงว่า", "วิตก", "กังวล",
    },
    Emotion.SURPRISE: {
        "surprised", "shocked", "amazed", "astonished", "wow",
        "unexpected", "sudden",
        "แปลกใจ", "ตกใจ", "ประหลาดใจ",
    },
    Emotion.FRUSTRATION: {
        "frustrating", "stuck", "confused", "difficult", "hard",
        "complicated", "struggle", "can't",
        "งง", "สับสน", "ยาก", "ติดขัด",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# EmotionInference
# ─────────────────────────────────────────────────────────────────────────────

class EmotionInference:
    """
    ตรวจจับและติดตามอารมณ์

    Flow:
      text + behavior → detect emotions → track state → influence response
    """

    def __init__(self, tracking_window: int = 10):
        self._tracking_window = tracking_window  # จำนวน interactions ที่ track

        self._current_state:  Optional[EmotionalState] = None
        self._emotion_history: Deque[EmotionScore]    = deque(maxlen=tracking_window)
        self._behavior_history: List[BehaviorIndicator] = []

        # tracking scores (exponential moving average)
        self._emotion_ema: Dict[Emotion, float] = defaultdict(float)
        self._alpha = 0.3  # EMA learning rate

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Text Sentiment
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_sentiment(self, text: str) -> Sentiment:
        """
        วิเคราะห์ sentiment จากข้อความ

        Args:
            text: ข้อความที่ต้องการวิเคราะห์

        Returns:
            Sentiment (POSITIVE / NEGATIVE / NEUTRAL)
        """
        if not text:
            return Sentiment.NEUTRAL

        text_lower = text.lower()

        # count positive/negative keywords
        positive_count = sum(
            1 for kw in EMOTION_KEYWORDS[Emotion.JOY]
            if kw in text_lower
        )
        negative_count = sum(
            1 for emotion in [Emotion.SADNESS, Emotion.ANGER, Emotion.FEAR, Emotion.FRUSTRATION]
            for kw in EMOTION_KEYWORDS[emotion]
            if kw in text_lower
        )

        # punctuation hints
        if "!" in text:
            positive_count += 0.5  # อาจเป็น excitement
        if "?" in text and "??" in text:
            negative_count += 0.3  # confusion

        if positive_count > negative_count:
            return Sentiment.POSITIVE
        elif negative_count > positive_count:
            return Sentiment.NEGATIVE
        else:
            return Sentiment.NEUTRAL

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Behavior Analysis
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_behavior(
        self,
        interactions: List[Any],
        time_window: float = 300.0,  # 5 minutes
    ) -> List[BehaviorIndicator]:
        """
        วิเคราะห์อารมณ์จากพฤติกรรม

        Args:
            interactions: BrainLogs
            time_window: ช่วงเวลาที่สนใจ (seconds)

        Returns:
            List[BehaviorIndicator]
        """
        if not interactions:
            return []

        now = time.time()
        recent = [
            log for log in interactions
            if (now - log.timestamp) <= time_window
        ]

        if not recent:
            return []

        indicators = []

        # 1. high frequency → EXCITEMENT / FRUSTRATION
        freq = len(recent) / (time_window / 60)  # interactions per minute
        if freq > 5:  # > 5 interactions/min
            emotion = Emotion.EXCITEMENT if freq < 10 else Emotion.FRUSTRATION
            indicators.append(BehaviorIndicator(
                indicator_type = "high_frequency",
                value          = freq,
                emotion_hint   = emotion,
                confidence     = min(0.7, freq / 15),
            ))

        # 2. rapid succession → URGENCY / FRUSTRATION
        if len(recent) >= 2:
            gaps = [
                recent[i].timestamp - recent[i-1].timestamp
                for i in range(1, len(recent))
            ]
            avg_gap = sum(gaps) / len(gaps)
            if avg_gap < 10:  # < 10 sec between interactions
                indicators.append(BehaviorIndicator(
                    indicator_type = "rapid_succession",
                    value          = avg_gap,
                    emotion_hint   = Emotion.FRUSTRATION,
                    confidence     = 0.6,
                ))

        # 3. error rate → FRUSTRATION
        errors = sum(1 for log in recent if log.outcome in ("reject", "silence"))
        if errors > 0:
            error_rate = errors / len(recent)
            if error_rate > 0.3:
                indicators.append(BehaviorIndicator(
                    indicator_type = "error_rate",
                    value          = error_rate,
                    emotion_hint   = Emotion.FRUSTRATION,
                    confidence     = min(0.8, error_rate),
                ))

        self._behavior_history.extend(indicators)
        return indicators

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Emotion Tracking
    # ─────────────────────────────────────────────────────────────────────────

    def track_emotion(self, emotion_score: EmotionScore) -> None:
        """
        ติดตามอารมณ์ต่อเนื่อง

        Update EMA สำหรับแต่ละ emotion
        """
        self._emotion_history.append(emotion_score)

        # update EMA
        emotion = emotion_score.emotion
        current = self._emotion_ema[emotion]
        self._emotion_ema[emotion] = (
            (1 - self._alpha) * current +
            self._alpha * emotion_score.confidence
        )

        # decay other emotions slightly
        for e in Emotion:
            if e != emotion:
                self._emotion_ema[e] *= 0.95

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Emotion Influence
    # ─────────────────────────────────────────────────────────────────────────

    def get_influence(self, emotion: Emotion) -> float:
        """
        อารมณ์มีผลต่อ response ยังไง

        Returns:
            influence factor (-1.0 to +1.0)
            positive = ทำให้ response aggressive/confident
            negative = ทำให้ response cautious
        """
        influence_map = {
            Emotion.JOY:        +0.3,
            Emotion.EXCITEMENT: +0.5,
            Emotion.SADNESS:    -0.2,
            Emotion.ANGER:      +0.2,  # aggressive
            Emotion.FEAR:       -0.5,  # very cautious
            Emotion.FRUSTRATION: -0.3,
            Emotion.SURPRISE:    0.0,
            Emotion.NEUTRAL:     0.0,
        }
        base = influence_map.get(emotion, 0.0)

        # scale by intensity
        if self._current_state:
            base *= self._current_state.intensity

        return base

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Emotional State
    # ─────────────────────────────────────────────────────────────────────────

    def get_emotional_state(self) -> EmotionalState:
        """
        สถานะอารมณ์ปัจจุบัน

        Returns:
            EmotionalState
        """
        if self._current_state:
            return self._current_state

        # fallback
        return EmotionalState(
            primary_emotion = Emotion.NEUTRAL,
            intensity       = 0.0,
            sentiment       = Sentiment.NEUTRAL,
            emotion_scores  = {},
        )

    def update_emotional_state(self) -> EmotionalState:
        """
        อัพเดท emotional state จาก EMA scores

        Returns:
            EmotionalState ใหม่
        """
        if not self._emotion_ema:
            state = EmotionalState(
                primary_emotion = Emotion.NEUTRAL,
                intensity       = 0.0,
                sentiment       = Sentiment.NEUTRAL,
                emotion_scores  = {},
            )
            self._current_state = state
            return state

        # primary emotion = highest EMA
        primary = max(self._emotion_ema, key=self._emotion_ema.get)
        intensity = self._emotion_ema[primary]

        # sentiment
        if primary in (Emotion.JOY, Emotion.EXCITEMENT, Emotion.SURPRISE):
            sentiment = Sentiment.POSITIVE
        elif primary in (Emotion.SADNESS, Emotion.ANGER, Emotion.FEAR, Emotion.FRUSTRATION):
            sentiment = Sentiment.NEGATIVE
        else:
            sentiment = Sentiment.NEUTRAL

        state = EmotionalState(
            primary_emotion = primary,
            intensity       = intensity,
            sentiment       = sentiment,
            emotion_scores  = dict(self._emotion_ema),
        )
        self._current_state = state

        logger.info(
            f"[EmotionInference] STATE_UPDATE "
            f"emotion={primary.value} intensity={intensity:.2f}"
        )
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Emotion Detection
    # ─────────────────────────────────────────────────────────────────────────

    def detect_emotion(
        self,
        text: str,
        behavior_indicators: List[BehaviorIndicator] = None,
    ) -> EmotionScore:
        """
        ตรวจจับอารมณ์หลักจาก text + behavior

        Args:
            text: ข้อความ
            behavior_indicators: ตัวบ่งชี้พฤติกรรม (optional)

        Returns:
            EmotionScore
        """
        text_lower = text.lower()

        # count keywords for each emotion
        emotion_counts: Dict[Emotion, float] = defaultdict(float)
        for emotion, keywords in EMOTION_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            emotion_counts[emotion] = count

        # behavior hints
        if behavior_indicators:
            for indicator in behavior_indicators:
                emotion_counts[indicator.emotion_hint] += indicator.confidence

        # find dominant emotion
        if not emotion_counts or max(emotion_counts.values()) == 0:
            detected = Emotion.NEUTRAL
            confidence = 0.3
        else:
            detected = max(emotion_counts, key=emotion_counts.get)
            total = sum(emotion_counts.values())
            confidence = min(0.9, emotion_counts[detected] / (total + 1))

        score = EmotionScore(
            emotion    = detected,
            confidence = confidence,
            source     = "combined" if behavior_indicators else "text",
        )

        # track automatically
        self.track_emotion(score)
        self.update_emotional_state()

        logger.debug(
            f"[EmotionInference] DETECT emotion={detected.value} "
            f"conf={confidence:.2f} source={score.source}"
        )
        return score

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        return {
            "current_state":    (
                self._current_state.to_dict()
                if self._current_state else None
            ),
            "emotion_history":  len(self._emotion_history),
            "behavior_indicators": len(self._behavior_history),
            "emotion_ema":      {
                e.value: round(s, 3)
                for e, s in self._emotion_ema.items()
                if s > 0.01
            },
            "tracking_window":  self._tracking_window,
        }

    @property
    def emotion_history(self) -> List[EmotionScore]:
        return list(self._emotion_history)

    @property
    def behavior_indicators(self) -> List[BehaviorIndicator]:
        return list(self._behavior_history)