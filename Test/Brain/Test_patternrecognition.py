"""
=================================================================
  Pattern Recognition Test Suite
=================================================================
  1. Sequence Detection         (4 tests)
  2. Temporal Pattern           (4 tests)
  3. User Behavior              (4 tests)
  4. Context Transitions        (4 tests)
  5. Error Pattern              (4 tests)
  6. Success Pattern            (4 tests)
  7. Integration                (4 tests)
-----------------------------------------------------------------
  Total: 28 tests
=================================================================
"""

import unittest
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Core.Brain.PatternRecognition import (
    PatternRecognition, TimeWindow,
    SequencePattern, TemporalPattern, BehaviorPattern,
    ContextTransition, ErrorPattern, SuccessPattern,
)
from Core.BrainController import BrainLog

def _make_log(
    context: str, outcome: str, confidence: float,
    learned: bool = False, timestamp: float = None
) -> BrainLog:
    return BrainLog(
        log_id="test", input_text="test", context=context,
        outcome=outcome, confidence=confidence,
        skill_weight=0.5, personality="test",
        learned=learned, response="test response",
        timestamp=timestamp or time.time(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Sequence Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestSequenceDetection(unittest.TestCase):

    def setUp(self):
        self.pr = PatternRecognition(min_frequency=2)

    def test_detect_sequences_empty(self):
        patterns = self.pr.detect_sequences([])
        self.assertEqual(len(patterns), 0)

    def test_detect_sequences_finds_repeating(self):
        logs = [
            _make_log("math", "commit", 0.8),
            _make_log("general", "commit", 0.7),
            _make_log("math", "commit", 0.8),
            _make_log("math", "commit", 0.8),
            _make_log("general", "commit", 0.7),
            _make_log("math", "commit", 0.8),
        ]
        patterns = self.pr.detect_sequences(logs, window_size=2)
        # ("math", "general") ซ้ำ 2 ครั้ง
        self.assertGreater(len(patterns), 0)

    def test_sequence_has_frequency(self):
        logs = [
            _make_log("A", "commit", 0.8),
            _make_log("B", "commit", 0.7),
            _make_log("A", "commit", 0.8),
            _make_log("B", "commit", 0.7),
        ]
        patterns = self.pr.detect_sequences(logs, window_size=2)
        if patterns:
            self.assertGreaterEqual(patterns[0].frequency, 2)

    def test_sequence_filters_by_min_freq(self):
        pr = PatternRecognition(min_frequency=5)
        logs = [_make_log("A", "commit", 0.8) for _ in range(3)]
        patterns = pr.detect_sequences(logs, window_size=2)
        # frequency < 5 → ไม่ผ่าน
        self.assertEqual(len(patterns), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Temporal Pattern
# ─────────────────────────────────────────────────────────────────────────────

class TestTemporalPattern(unittest.TestCase):

    def setUp(self):
        self.pr = PatternRecognition()

    def test_detect_temporal_empty(self):
        pattern = self.pr.detect_temporal([])
        self.assertEqual(len(pattern.peak_hours), 0)

    def test_detect_temporal_returns_pattern(self):
        now = time.time()
        logs = [_make_log("math", "commit", 0.8, timestamp=now + i*3600) for i in range(10)]
        pattern = self.pr.detect_temporal(logs)
        self.assertIsInstance(pattern, TemporalPattern)

    def test_temporal_has_peak_hours(self):
        now = time.time()
        logs = [_make_log("math", "commit", 0.8, timestamp=now + i*3600) for i in range(5)]
        pattern = self.pr.detect_temporal(logs)
        self.assertLessEqual(len(pattern.peak_hours), 3)

    def test_temporal_activity_distribution(self):
        now = time.time()
        logs = [_make_log("math", "commit", 0.8, timestamp=now + i*3600) for i in range(10)]
        pattern = self.pr.detect_temporal(logs)
        self.assertIn("morning", pattern.activity_dist)


# ─────────────────────────────────────────────────────────────────────────────
# 3. User Behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestUserBehavior(unittest.TestCase):

    def setUp(self):
        self.pr = PatternRecognition()

    def test_detect_behavior_empty(self):
        pattern = self.pr.detect_behavior([])
        self.assertEqual(len(pattern.preferred_contexts), 0)

    def test_detect_behavior_returns_pattern(self):
        logs = [_make_log("math", "commit", 0.8) for _ in range(5)]
        pattern = self.pr.detect_behavior(logs)
        self.assertIsInstance(pattern, BehaviorPattern)

    def test_behavior_preferred_contexts(self):
        logs = [
            _make_log("math", "commit", 0.8) for _ in range(10)
        ] + [
            _make_log("general", "commit", 0.7) for _ in range(3)
        ]
        pattern = self.pr.detect_behavior(logs)
        self.assertIn("math", pattern.preferred_contexts)

    def test_behavior_question_rate(self):
        logs = [
            _make_log("math", "ask", 0.5) for _ in range(3)
        ] + [
            _make_log("math", "commit", 0.8) for _ in range(7)
        ]
        pattern = self.pr.detect_behavior(logs)
        self.assertAlmostEqual(pattern.question_rate, 0.3, delta=0.1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Context Transitions
# ─────────────────────────────────────────────────────────────────────────────

class TestContextTransitions(unittest.TestCase):

    def setUp(self):
        self.pr = PatternRecognition(min_frequency=2)

    def test_detect_transitions_empty(self):
        patterns = self.pr.detect_context_transitions([])
        self.assertEqual(len(patterns), 0)

    def test_detect_transitions_finds_changes(self):
        logs = [
            _make_log("math", "commit", 0.8),
            _make_log("general", "commit", 0.7),
            _make_log("math", "commit", 0.8),
            _make_log("general", "commit", 0.7),
        ]
        patterns = self.pr.detect_context_transitions(logs)
        # math → general ซ้ำ 2 ครั้ง
        self.assertGreater(len(patterns), 0)

    def test_transition_has_from_to(self):
        logs = [
            _make_log("A", "commit", 0.8),
            _make_log("B", "commit", 0.7),
            _make_log("A", "commit", 0.8),
            _make_log("B", "commit", 0.7),
        ]
        patterns = self.pr.detect_context_transitions(logs)
        if patterns:
            self.assertIn(patterns[0].from_context, ["A", "B"])

    def test_transition_ignores_same_context(self):
        logs = [_make_log("math", "commit", 0.8) for _ in range(5)]
        patterns = self.pr.detect_context_transitions(logs)
        # ไม่มี transition (context เดียวกันตลอด)
        self.assertEqual(len(patterns), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Error Pattern
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorPattern(unittest.TestCase):

    def setUp(self):
        self.pr = PatternRecognition(min_frequency=2)

    def test_detect_errors_empty(self):
        patterns = self.pr.detect_errors([])
        self.assertEqual(len(patterns), 0)

    def test_detect_errors_finds_rejects(self):
        logs = [
            _make_log("math", "reject", 0.9),
            _make_log("math", "reject", 0.85),
            _make_log("math", "silence", 0.8),
        ]
        patterns = self.pr.detect_errors(logs)
        self.assertGreater(len(patterns), 0)

    def test_error_pattern_has_context(self):
        logs = [
            _make_log("science", "reject", 0.9),
            _make_log("science", "silence", 0.85),
        ]
        patterns = self.pr.detect_errors(logs)
        if patterns:
            self.assertEqual(patterns[0].contexts, ["science"])

    def test_errors_ignore_success(self):
        logs = [_make_log("math", "commit", 0.8) for _ in range(5)]
        patterns = self.pr.detect_errors(logs)
        # ไม่มี error
        self.assertEqual(len(patterns), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Success Pattern
# ─────────────────────────────────────────────────────────────────────────────

class TestSuccessPattern(unittest.TestCase):

    def setUp(self):
        self.pr = PatternRecognition(min_frequency=2)

    def test_detect_success_empty(self):
        patterns = self.pr.detect_success([])
        self.assertEqual(len(patterns), 0)

    def test_detect_success_finds_commits(self):
        logs = [
            _make_log("math", "commit", 0.9),
            _make_log("math", "commit", 0.85),
            _make_log("general", "commit", 0.8),
        ]
        patterns = self.pr.detect_success(logs)
        self.assertGreater(len(patterns), 0)

    def test_success_classifies_approach(self):
        logs = [
            _make_log("math", "commit", 0.95),
            _make_log("math", "commit", 0.9),
        ]
        patterns = self.pr.detect_success(logs)
        if patterns:
            self.assertEqual(patterns[0].approach, "confident")

    def test_success_ignores_failures(self):
        logs = [_make_log("math", "reject", 0.8) for _ in range(5)]
        patterns = self.pr.detect_success(logs)
        # ไม่มี success
        self.assertEqual(len(patterns), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):

    def test_full_pipeline(self):
        pr = PatternRecognition(min_frequency=2)
        logs = []
        now = time.time()

        # สร้าง pattern ที่หลากหลาย
        for i in range(20):
            ctx = "math" if i % 2 == 0 else "general"
            outcome = "commit" if i % 3 != 0 else "ask"
            conf = 0.7 + (i % 3) * 0.1
            logs.append(_make_log(
                ctx, outcome, conf, learned=(i % 4 == 0),
                timestamp=now + i*3600
            ))

        # run all detections
        seq = pr.detect_sequences(logs, window_size=2)
        temp = pr.detect_temporal(logs)
        behav = pr.detect_behavior(logs)
        trans = pr.detect_context_transitions(logs)
        err = pr.detect_errors(logs)
        succ = pr.detect_success(logs)

        # all produced results
        self.assertIsNotNone(seq)
        self.assertIsNotNone(temp)
        self.assertIsNotNone(behav)
        self.assertIsNotNone(trans)

    def test_stats_reflect_state(self):
        pr = PatternRecognition(min_frequency=2)
        logs = [
            _make_log("math", "commit", 0.8) for _ in range(10)
        ]
        pr.detect_sequences(logs, window_size=2)
        pr.detect_temporal(logs)
        pr.detect_behavior(logs)

        stats = pr.stats()
        self.assertGreaterEqual(stats["sequences"], 0)
        self.assertEqual(stats["temporal"], 1)
        self.assertEqual(stats["behaviors"], 1)

    def test_properties_accessible(self):
        pr = PatternRecognition()
        logs = [_make_log("math", "commit", 0.9) for _ in range(5)]
        pr.detect_success(logs)

        self.assertIsInstance(pr.sequences, list)
        self.assertIsInstance(pr.successes, list)

    def test_min_frequency_filtering(self):
        pr = PatternRecognition(min_frequency=10)
        logs = [_make_log("math", "commit", 0.8) for _ in range(3)]
        seq = pr.detect_sequences(logs, window_size=2)
        # frequency < 10 → ไม่ผ่าน
        self.assertEqual(len(seq), 0)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1. Sequence Detection    (4)", TestSequenceDetection),
        ("2. Temporal Pattern      (4)", TestTemporalPattern),
        ("3. User Behavior         (4)", TestUserBehavior),
        ("4. Context Transitions   (4)", TestContextTransitions),
        ("5. Error Pattern         (4)", TestErrorPattern),
        ("6. Success Pattern       (4)", TestSuccessPattern),
        ("7. Integration           (4)", TestIntegration),
    ]

    print("\n=================================================================")
    print("  Pattern Recognition Test Suite")
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