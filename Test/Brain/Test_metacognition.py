"""
=================================================================
  MetaCognition Test Suite
=================================================================
  1. Self-Reflection           (5 tests)
  2. Confidence Calibration    (5 tests)
  3. Error Awareness           (6 tests)
  4. Learning Tracking         (5 tests)
  5. Strategy Selection        (5 tests)
  6. Integration               (4 tests)
-----------------------------------------------------------------
  Total: 30 tests
=================================================================
"""

import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Core.Brain.MetaCognition import (
    MetaCognition, Strategy, LearningTrend, ErrorType,
    ReflectionResult, CalibrationUpdate, ErrorPattern,
    LearningTrack, StrategyRecommendation,
)
from Core.BrainController import BrainController, BrainLog

def _make_log(outcome: str, confidence: float, context: str = "general",
               learned: bool = False) -> BrainLog:
    return BrainLog(
        log_id="test", input_text="test", context=context,
        outcome=outcome, confidence=confidence,
        skill_weight=0.5, personality="test",
        learned=learned, response="test response",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Self-Reflection
# ─────────────────────────────────────────────────────────────────────────────

class TestReflection(unittest.TestCase):

    def setUp(self):
        self.mc = MetaCognition()

    def test_reflect_empty_logs(self):
        r = self.mc.reflect([])
        self.assertEqual(r.log_count, 0)

    def test_reflect_returns_result(self):
        logs = [_make_log("commit", 0.8) for _ in range(3)]
        r = self.mc.reflect(logs)
        self.assertIsInstance(r, ReflectionResult)

    def test_reflect_outcome_distribution(self):
        logs = [
            _make_log("commit", 0.8),
            _make_log("commit", 0.7),
            _make_log("ask", 0.5),
        ]
        r = self.mc.reflect(logs)
        self.assertEqual(r.outcome_dist["commit"], 2)
        self.assertEqual(r.outcome_dist["ask"], 1)

    def test_reflect_context_coverage(self):
        logs = [
            _make_log("commit", 0.8, "math"),
            _make_log("commit", 0.7, "math"),
            _make_log("ask", 0.5, "general"),
        ]
        r = self.mc.reflect(logs)
        self.assertEqual(r.context_coverage["math"], 2)
        self.assertEqual(r.context_coverage["general"], 1)

    def test_reflect_quality_score(self):
        logs = [_make_log("commit", 0.9) for _ in range(10)]
        r = self.mc.reflect(logs)
        self.assertGreater(r.quality_score, 0.9)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Confidence Calibration
# ─────────────────────────────────────────────────────────────────────────────

class TestCalibration(unittest.TestCase):

    def setUp(self):
        self.mc = MetaCognition()

    def test_calibrate_empty_logs(self):
        c = self.mc.calibrate_confidence([])
        self.assertEqual(c.sample_size, 0)

    def test_calibrate_returns_update(self):
        logs = [_make_log("commit", 0.8) for _ in range(5)]
        c = self.mc.calibrate_confidence(logs)
        self.assertIsInstance(c, CalibrationUpdate)

    def test_calibrate_adjusts_bias(self):
        logs = [_make_log("commit", 0.9) for _ in range(5)]
        before = self.mc.confidence_bias
        self.mc.calibrate_confidence(logs)
        # bias เปลี่ยน
        self.assertNotEqual(self.mc.confidence_bias, before)

    def test_calibrate_with_actual_outcomes(self):
        logs = [_make_log("commit", 0.8) for _ in range(3)]
        actual = [1.0, 1.0, 0.8]
        c = self.mc.calibrate_confidence(logs, actual)
        self.assertEqual(c.sample_size, 3)

    def test_confidence_bias_property(self):
        bias = self.mc.confidence_bias
        self.assertIsInstance(bias, float)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Error Awareness
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorAwareness(unittest.TestCase):

    def setUp(self):
        self.mc = MetaCognition()

    def test_detect_errors_empty(self):
        errors = self.mc.detect_errors([])
        self.assertEqual(len(errors), 0)

    def test_detect_overconfident(self):
        logs = [
            _make_log("reject", 0.9, "math"),
            _make_log("silence", 0.85, "math"),
        ]
        errors = self.mc.detect_errors(logs)
        self.assertTrue(
            any(e.error_type == ErrorType.OVERCONFIDENT for e in errors)
        )

    def test_detect_underconfident(self):
        logs = [
            _make_log("commit", 0.2, "general"),
            _make_log("commit", 0.15, "general"),
        ]
        errors = self.mc.detect_errors(logs)
        self.assertTrue(
            any(e.error_type == ErrorType.UNDERCONFIDENT for e in errors)
        )

    def test_detect_inconsistent(self):
        logs = [
            _make_log("commit", 0.7, "math"),
            _make_log("ask", 0.6, "math"),
            _make_log("silence", 0.5, "math"),
        ]
        errors = self.mc.detect_errors(logs)
        self.assertTrue(
            any(e.error_type == ErrorType.INCONSISTENT for e in errors)
        )

    def test_error_pattern_has_context(self):
        logs = [_make_log("reject", 0.9, "science")]
        errors = self.mc.detect_errors(logs)
        if errors:
            self.assertEqual(errors[0].context, "science")

    def test_errors_stored(self):
        logs = [_make_log("reject", 0.9)]
        self.mc.detect_errors(logs)
        self.assertGreater(len(self.mc.errors), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Learning Tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestLearningTracking(unittest.TestCase):

    def setUp(self):
        self.mc = MetaCognition()

    def test_track_empty_logs(self):
        track = self.mc.track_learning([])
        self.assertEqual(track.trend, LearningTrend.UNKNOWN)

    def test_track_returns_track(self):
        logs = [_make_log("commit", 0.5 + i*0.05, learned=True) for i in range(10)]
        track = self.mc.track_learning(logs)
        self.assertIsInstance(track, LearningTrack)

    def test_track_improving_trend(self):
        logs = [_make_log("commit", 0.4 + i*0.1, learned=True) for i in range(10)]
        track = self.mc.track_learning(logs)
        self.assertEqual(track.trend, LearningTrend.IMPROVING)

    def test_track_learning_rate(self):
        logs = [_make_log("commit", 0.7, learned=True) for _ in range(8)]
        logs += [_make_log("commit", 0.7, learned=False) for _ in range(2)]
        track = self.mc.track_learning(logs)
        self.assertAlmostEqual(track.learning_rate, 0.8)

    def test_track_stored(self):
        logs = [_make_log("commit", 0.6, learned=True) for _ in range(10)]
        self.mc.track_learning(logs)
        self.assertEqual(len(self.mc.tracks), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Strategy Selection
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategySelection(unittest.TestCase):

    def setUp(self):
        self.mc = MetaCognition()

    def test_suggest_empty_logs(self):
        rec = self.mc.suggest_strategy([])
        self.assertEqual(rec.recommended, Strategy.ADAPTIVE)

    def test_suggest_returns_recommendation(self):
        logs = [_make_log("commit", 0.9, learned=True) for _ in range(10)]
        rec = self.mc.suggest_strategy(logs)
        self.assertIsInstance(rec, StrategyRecommendation)

    def test_suggest_aggressive_on_high_quality(self):
        logs = [_make_log("commit", 0.9, learned=True) for _ in range(10)]
        rec = self.mc.suggest_strategy(logs)
        self.assertEqual(rec.recommended, Strategy.AGGRESSIVE)

    def test_suggest_cautious_on_errors(self):
        logs = [_make_log("reject", 0.9, "math") for _ in range(5)]
        rec = self.mc.suggest_strategy(logs)
        self.assertEqual(rec.recommended, Strategy.CAUTIOUS)

    def test_suggest_has_reason(self):
        logs = [_make_log("commit", 0.7, learned=True) for _ in range(10)]
        rec = self.mc.suggest_strategy(logs)
        self.assertNotEqual(rec.reason, "")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):

    def test_full_pipeline(self):
        mc = MetaCognition()
        brain = BrainController()

        # generate logs
        for i in range(20):
            brain.respond(f"input {i}", "math" if i % 2 == 0 else "general")

        logs = brain.logs

        # run full analysis
        reflection = mc.reflect(logs)
        calibration = mc.calibrate_confidence(logs)
        errors = mc.detect_errors(logs)
        track = mc.track_learning(logs)
        strategy = mc.suggest_strategy(logs)

        # all produced results
        self.assertIsNotNone(reflection)
        self.assertIsNotNone(calibration)
        self.assertIsNotNone(track)
        self.assertIsNotNone(strategy)

    def test_stats_reflect_state(self):
        mc = MetaCognition()
        logs = [_make_log("commit", 0.8, learned=True) for _ in range(10)]
        mc.reflect(logs)
        mc.calibrate_confidence(logs)
        mc.track_learning(logs)
        mc.suggest_strategy(logs)

        stats = mc.stats()
        self.assertEqual(stats["reflections"], 2)  # reflect called twice (in suggest too)
        self.assertEqual(stats["calibrations"], 1)
        self.assertGreater(stats["learning_tracks"], 0)

    def test_sequential_analysis(self):
        mc = MetaCognition()
        brain = BrainController()

        # batch 1
        for _ in range(5):
            brain.respond("test", "math")
        mc.reflect(brain.logs)

        # batch 2
        for _ in range(5):
            brain.respond("test", "general")
        mc.reflect(brain.logs)

        self.assertEqual(len(mc.reflections), 2)

    def test_properties_accessible(self):
        mc = MetaCognition()
        logs = [_make_log("commit", 0.7, learned=True) for _ in range(5)]
        mc.reflect(logs)
        mc.calibrate_confidence(logs)
        mc.detect_errors(logs)

        self.assertIsInstance(mc.reflections, list)
        self.assertIsInstance(mc.calibrations, list)
        self.assertIsInstance(mc.errors, list)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1. Self-Reflection        (5)", TestReflection),
        ("2. Confidence Calibration  (5)", TestCalibration),
        ("3. Error Awareness         (6)", TestErrorAwareness),
        ("4. Learning Tracking       (5)", TestLearningTracking),
        ("5. Strategy Selection      (5)", TestStrategySelection),
        ("6. Integration             (4)", TestIntegration),
    ]

    print("\n=================================================================")
    print("  MetaCognition Test Suite")
    print("=================================================================")
    for label, _ in groups:
        print(f"  {label}")
    print("─────────────────────────────────────────────────────────────────")
    print("  Total: 30 tests")
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