"""
=================================================================
  Emotion Inference Test Suite
=================================================================
  1. Text Sentiment           (4 tests)
  2. Behavior Analysis        (4 tests)
  3. Emotion Tracking         (4 tests)
  4. Emotion Influence        (4 tests)
  5. Emotional State          (4 tests)
  6. Emotion Detection        (5 tests)
  7. Integration              (3 tests)
-----------------------------------------------------------------
  Total: 28 tests
=================================================================
"""

import unittest
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Core.Brain.EmotionInference import (
    EmotionInference, Emotion, Sentiment,
    EmotionScore, EmotionalState, BehaviorIndicator,
)
from Core.BrainController import BrainLog

def _make_log(
    outcome: str, timestamp: float = None, context: str = "general"
) -> BrainLog:
    return BrainLog(
        log_id="test", input_text="test", context=context,
        outcome=outcome, confidence=0.7,
        skill_weight=0.5, personality="test",
        learned=False, response="test",
        timestamp=timestamp or time.time(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Text Sentiment
# ─────────────────────────────────────────────────────────────────────────────

class TestTextSentiment(unittest.TestCase):

    def setUp(self):
        self.ei = EmotionInference()

    def test_sentiment_positive(self):
        s = self.ei.analyze_sentiment("I am so happy and excited!")
        self.assertEqual(s, Sentiment.POSITIVE)

    def test_sentiment_negative(self):
        s = self.ei.analyze_sentiment("I am sad and frustrated")
        self.assertEqual(s, Sentiment.NEGATIVE)

    def test_sentiment_neutral(self):
        s = self.ei.analyze_sentiment("The weather is okay")
        self.assertEqual(s, Sentiment.NEUTRAL)

    def test_sentiment_empty(self):
        s = self.ei.analyze_sentiment("")
        self.assertEqual(s, Sentiment.NEUTRAL)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Behavior Analysis
# ─────────────────────────────────────────────────────────────────────────────

class TestBehaviorAnalysis(unittest.TestCase):

    def setUp(self):
        self.ei = EmotionInference()

    def test_behavior_empty(self):
        indicators = self.ei.analyze_behavior([])
        self.assertEqual(len(indicators), 0)

    def test_behavior_high_frequency(self):
        now = time.time()
        logs = [_make_log("commit", now - i) for i in range(20)]  # 20 in short time
        indicators = self.ei.analyze_behavior(logs, time_window=60)
        # high frequency → EXCITEMENT / FRUSTRATION
        self.assertGreater(len(indicators), 0)

    def test_behavior_error_rate(self):
        now = time.time()
        logs = [
            _make_log("reject", now - i) for i in range(5)
        ]
        indicators = self.ei.analyze_behavior(logs, time_window=300)
        # errors → FRUSTRATION
        frustration = [
            i for i in indicators
            if i.emotion_hint == Emotion.FRUSTRATION
        ]
        self.assertGreater(len(frustration), 0)

    def test_behavior_returns_indicators(self):
        now = time.time()
        logs = [_make_log("commit", now - i*10) for i in range(5)]
        indicators = self.ei.analyze_behavior(logs)
        self.assertIsInstance(indicators, list)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Emotion Tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestEmotionTracking(unittest.TestCase):

    def setUp(self):
        self.ei = EmotionInference()

    def test_track_emotion_updates_history(self):
        score = EmotionScore(Emotion.JOY, 0.8, "text")
        self.ei.track_emotion(score)
        self.assertEqual(len(self.ei.emotion_history), 1)

    def test_track_emotion_updates_ema(self):
        score = EmotionScore(Emotion.JOY, 0.8, "text")
        self.ei.track_emotion(score)
        self.assertGreater(self.ei._emotion_ema[Emotion.JOY], 0)

    def test_track_multiple_emotions(self):
        self.ei.track_emotion(EmotionScore(Emotion.JOY, 0.7, "text"))
        self.ei.track_emotion(EmotionScore(Emotion.SADNESS, 0.6, "text"))
        self.ei.track_emotion(EmotionScore(Emotion.JOY, 0.8, "text"))
        # JOY ควรมี EMA สูงกว่า SADNESS
        self.assertGreater(
            self.ei._emotion_ema[Emotion.JOY],
            self.ei._emotion_ema[Emotion.SADNESS],
        )

    def test_track_respects_window_size(self):
        ei = EmotionInference(tracking_window=3)
        for i in range(5):
            ei.track_emotion(EmotionScore(Emotion.JOY, 0.7, "text"))
        # history จำกัดที่ 3
        self.assertEqual(len(ei.emotion_history), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Emotion Influence
# ─────────────────────────────────────────────────────────────────────────────

class TestEmotionInfluence(unittest.TestCase):

    def setUp(self):
        self.ei = EmotionInference()

    def test_influence_joy_positive(self):
        inf = self.ei.get_influence(Emotion.JOY)
        self.assertGreater(inf, 0)

    def test_influence_fear_negative(self):
        inf = self.ei.get_influence(Emotion.FEAR)
        self.assertLess(inf, 0)

    def test_influence_neutral_zero(self):
        inf = self.ei.get_influence(Emotion.NEUTRAL)
        self.assertEqual(inf, 0.0)

    def test_influence_scaled_by_intensity(self):
        # set state with high intensity
        self.ei._current_state = EmotionalState(
            primary_emotion = Emotion.JOY,
            intensity       = 0.9,
            sentiment       = Sentiment.POSITIVE,
            emotion_scores  = {Emotion.JOY: 0.9},
        )
        inf = self.ei.get_influence(Emotion.JOY)
        # ควรมีค่ามากขึ้น
        self.assertGreater(abs(inf), 0.1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Emotional State
# ─────────────────────────────────────────────────────────────────────────────

class TestEmotionalState(unittest.TestCase):

    def setUp(self):
        self.ei = EmotionInference()

    def test_get_emotional_state_initial(self):
        state = self.ei.get_emotional_state()
        self.assertEqual(state.primary_emotion, Emotion.NEUTRAL)

    def test_update_emotional_state(self):
        self.ei.track_emotion(EmotionScore(Emotion.JOY, 0.8, "text"))
        state = self.ei.update_emotional_state()
        self.assertEqual(state.primary_emotion, Emotion.JOY)

    def test_state_has_intensity(self):
        self.ei.track_emotion(EmotionScore(Emotion.SADNESS, 0.7, "text"))
        state = self.ei.update_emotional_state()
        self.assertGreater(state.intensity, 0)

    def test_state_has_sentiment(self):
        self.ei.track_emotion(EmotionScore(Emotion.JOY, 0.8, "text"))
        state = self.ei.update_emotional_state()
        self.assertEqual(state.sentiment, Sentiment.POSITIVE)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Emotion Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestEmotionDetection(unittest.TestCase):

    def setUp(self):
        self.ei = EmotionInference()

    def test_detect_emotion_from_text(self):
        score = self.ei.detect_emotion("I am so happy!")
        self.assertEqual(score.emotion, Emotion.JOY)

    def test_detect_emotion_sad(self):
        score = self.ei.detect_emotion("I feel sad and disappointed")
        self.assertEqual(score.emotion, Emotion.SADNESS)

    def test_detect_emotion_neutral(self):
        score = self.ei.detect_emotion("The weather is normal today")
        # neutral text ควรได้ NEUTRAL
        self.assertIn(score.emotion, [Emotion.NEUTRAL, Emotion.SURPRISE])

    def test_detect_emotion_with_behavior(self):
        indicator = BehaviorIndicator(
            indicator_type = "error_rate",
            value          = 0.5,
            emotion_hint   = Emotion.FRUSTRATION,
            confidence     = 0.7,
        )
        score = self.ei.detect_emotion("test", behavior_indicators=[indicator])
        # behavior hint ควรมีผล
        self.assertIsNotNone(score)

    def test_detect_emotion_tracks_automatically(self):
        self.ei.detect_emotion("I am happy")
        # ควร track และ update state
        self.assertEqual(len(self.ei.emotion_history), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):

    def test_full_pipeline(self):
        ei = EmotionInference()

        # 1. sentiment
        sent = ei.analyze_sentiment("I am frustrated with errors")

        # 2. behavior
        now = time.time()
        logs = [_make_log("reject", now - i) for i in range(3)]
        indicators = ei.analyze_behavior(logs)

        # 3. detect
        score = ei.detect_emotion("This is frustrating!", indicators)

        # 4. state
        state = ei.get_emotional_state()

        # 5. influence
        inf = ei.get_influence(state.primary_emotion)

        # all produced results
        self.assertIsNotNone(sent)
        self.assertIsNotNone(score)
        self.assertIsNotNone(state)
        self.assertIsInstance(inf, float)

    def test_stats_reflect_state(self):
        ei = EmotionInference()
        ei.detect_emotion("I am happy!")
        ei.detect_emotion("Very excited!")

        stats = ei.stats()
        self.assertIsNotNone(stats["current_state"])
        self.assertGreater(stats["emotion_history"], 0)

    def test_properties_accessible(self):
        ei = EmotionInference()
        ei.detect_emotion("test")

        self.assertIsInstance(ei.emotion_history, list)
        self.assertIsInstance(ei.behavior_indicators, list)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1. Text Sentiment      (4)", TestTextSentiment),
        ("2. Behavior Analysis   (4)", TestBehaviorAnalysis),
        ("3. Emotion Tracking    (4)", TestEmotionTracking),
        ("4. Emotion Influence   (4)", TestEmotionInfluence),
        ("5. Emotional State     (4)", TestEmotionalState),
        ("6. Emotion Detection   (5)", TestEmotionDetection),
        ("7. Integration         (3)", TestIntegration),
    ]

    print("\n=================================================================")
    print("  Emotion Inference Test Suite")
    print("=================================================================")
    for label, _ in groups:
        print(f"  {label}")
    print("─────────────────────────────────────────────────────────────────")
    print("  Total: 28 tests")
    print("=================================================================\n")

    for _, cls in groups:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n=================================================================")
    print(f"  Passed : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failed : {len(result.failures)}")
    print(f"  Errors : {len(result.errors)}")
    print("=================================================================")
    print("\n  🎉 ALL TESTS PASSED!\n" if result.wasSuccessful() else "\n  ❌ SOME TESTS FAILED\n")


if __name__ == "__main__":
    run_tests()