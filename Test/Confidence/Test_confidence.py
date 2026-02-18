"""
=================================================================
  Confidence System Test Suite
=================================================================
  1. ConfidenceLevel & score_to_level    (5 tests)
  2. ConfidenceOutcome & level_to_outcome (5 tests)
  3. ConfidenceResult properties         (5 tests)
  4. Hard conflict — identity            (3 tests)
  5. Hard conflict — system error        (2 tests)
  6. Hard conflict — rule blocked        (3 tests)
  7. evaluate() — commit path            (3 tests)
  8. evaluate() — conditional path       (2 tests)
  9. evaluate() — ask / silence path     (3 tests)
 10. Weights                             (3 tests)
 11. History & stats                     (4 tests)
-----------------------------------------------------------------
  Total: 38 tests
=================================================================
"""

import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Core.Confidence.ConfidenceData import (
    ConfidenceLevel, ConfidenceOutcome, ConflictType,
    ConfidenceResult, score_to_level, level_to_outcome,
    IDENTITY_CONFLICT_RESULT, SYSTEM_ERROR_RESULT,
)
from Core.Confidence.ConfidenceController import ConfidenceController


def _cc(**kwargs) -> ConfidenceController:
    return ConfidenceController(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 1. ConfidenceLevel & score_to_level
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreToLevel(unittest.TestCase):

    def test_high_at_075(self):
        self.assertEqual(score_to_level(0.75), ConfidenceLevel.HIGH)

    def test_high_at_1(self):
        self.assertEqual(score_to_level(1.0), ConfidenceLevel.HIGH)

    def test_medium_at_050(self):
        self.assertEqual(score_to_level(0.50), ConfidenceLevel.MEDIUM)

    def test_low_at_025(self):
        self.assertEqual(score_to_level(0.25), ConfidenceLevel.LOW)

    def test_very_low_below_025(self):
        self.assertEqual(score_to_level(0.10), ConfidenceLevel.VERY_LOW)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ConfidenceOutcome & level_to_outcome
# ─────────────────────────────────────────────────────────────────────────────

class TestLevelToOutcome(unittest.TestCase):

    def test_high_no_conflict_commit(self):
        self.assertEqual(
            level_to_outcome(ConfidenceLevel.HIGH), ConfidenceOutcome.COMMIT
        )

    def test_medium_no_conflict_conditional(self):
        self.assertEqual(
            level_to_outcome(ConfidenceLevel.MEDIUM), ConfidenceOutcome.CONDITIONAL
        )

    def test_low_no_conflict_ask(self):
        self.assertEqual(
            level_to_outcome(ConfidenceLevel.LOW), ConfidenceOutcome.ASK
        )

    def test_identity_conflict_overrides_to_reject(self):
        """identity conflict → REJECT แม้ level = HIGH"""
        self.assertEqual(
            level_to_outcome(ConfidenceLevel.HIGH, ConflictType.IDENTITY_CONFLICT),
            ConfidenceOutcome.REJECT
        )

    def test_rule_conflict_overrides_to_silence(self):
        """rule conflict → SILENCE แม้ level = HIGH"""
        self.assertEqual(
            level_to_outcome(ConfidenceLevel.HIGH, ConflictType.RULE_CONFLICT),
            ConfidenceOutcome.SILENCE
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. ConfidenceResult properties
# ─────────────────────────────────────────────────────────────────────────────

class TestConfidenceResult(unittest.TestCase):

    def _make(self, outcome: ConfidenceOutcome) -> ConfidenceResult:
        return ConfidenceResult(
            score=0.5, level=ConfidenceLevel.MEDIUM, outcome=outcome
        )

    def test_can_commit(self):
        self.assertTrue(self._make(ConfidenceOutcome.COMMIT).can_commit)

    def test_should_ask(self):
        self.assertTrue(self._make(ConfidenceOutcome.ASK).should_ask)

    def test_should_silence(self):
        self.assertTrue(self._make(ConfidenceOutcome.SILENCE).should_silence)

    def test_should_reject(self):
        self.assertTrue(self._make(ConfidenceOutcome.REJECT).should_reject)

    def test_is_conditional(self):
        self.assertTrue(self._make(ConfidenceOutcome.CONDITIONAL).is_conditional)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Hard conflict — identity
# ─────────────────────────────────────────────────────────────────────────────

class TestIdentityConflict(unittest.TestCase):

    def setUp(self):
        self.cc = _cc()

    def test_reject_identity_conflict_direct(self):
        """reject_identity_conflict() → REJECT"""
        r = self.cc.reject_identity_conflict()
        self.assertTrue(r.should_reject)
        self.assertEqual(r.conflict, ConflictType.IDENTITY_CONFLICT)

    def test_evaluate_with_identity_conflict_flag(self):
        """evaluate(identity_conflict=True) → REJECT"""
        r = self.cc.evaluate(identity_conflict=True)
        self.assertTrue(r.should_reject)

    def test_evaluate_with_zero_identity_score(self):
        """evaluate(identity_score=0.0) → REJECT"""
        r = self.cc.evaluate(identity_score=0.0)
        self.assertTrue(r.should_reject)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Hard conflict — system error
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemError(unittest.TestCase):

    def setUp(self):
        self.cc = _cc()

    def test_reject_system_error_direct(self):
        """reject_system_error() → REJECT"""
        r = self.cc.reject_system_error()
        self.assertTrue(r.should_reject)
        self.assertEqual(r.conflict, ConflictType.SYSTEM_ERROR)

    def test_evaluate_with_system_error_flag(self):
        """evaluate(system_error=True) → REJECT"""
        r = self.cc.evaluate(system_error=True)
        self.assertTrue(r.should_reject)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Hard conflict — rule blocked
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleConflict(unittest.TestCase):

    def setUp(self):
        self.cc = _cc()

    def test_silence_rule_conflict_direct(self):
        """silence_rule_conflict() → SILENCE"""
        r = self.cc.silence_rule_conflict()
        self.assertTrue(r.should_silence)
        self.assertEqual(r.conflict, ConflictType.RULE_CONFLICT)

    def test_evaluate_with_rule_blocked_flag(self):
        """evaluate(rule_blocked=True) → SILENCE"""
        r = self.cc.evaluate(rule_blocked=True)
        self.assertTrue(r.should_silence)

    def test_evaluate_with_zero_rule_score(self):
        """evaluate(rule_score=0.0) → SILENCE"""
        r = self.cc.evaluate(rule_score=0.0)
        self.assertTrue(r.should_silence)


# ─────────────────────────────────────────────────────────────────────────────
# 7. evaluate() — commit path
# ─────────────────────────────────────────────────────────────────────────────

class TestCommitPath(unittest.TestCase):

    def setUp(self):
        self.cc = _cc()

    def test_all_high_factors_commit(self):
        """ทุก factor สูง → COMMIT"""
        r = self.cc.evaluate(
            rule_score=1.0, context_score=1.0,
            skill_score=1.0, identity_score=1.0
        )
        self.assertTrue(r.can_commit)

    def test_commit_level_is_high(self):
        """COMMIT → level = HIGH"""
        r = self.cc.evaluate(
            rule_score=1.0, context_score=1.0,
            skill_score=1.0, identity_score=1.0
        )
        self.assertEqual(r.level, ConfidenceLevel.HIGH)

    def test_no_conflict_on_commit(self):
        """COMMIT → ไม่มี conflict"""
        r = self.cc.evaluate(
            rule_score=1.0, context_score=1.0,
            skill_score=1.0, identity_score=1.0
        )
        self.assertFalse(r.has_conflict)


# ─────────────────────────────────────────────────────────────────────────────
# 8. evaluate() — conditional path
# ─────────────────────────────────────────────────────────────────────────────

class TestConditionalPath(unittest.TestCase):

    def setUp(self):
        self.cc = _cc()

    def test_medium_score_conditional(self):
        """score ≥ 0.50 และ < 0.75 → CONDITIONAL"""
        r = self.cc.evaluate(
            rule_score=1.0, context_score=0.5,
            skill_score=0.3, identity_score=1.0
        )
        self.assertTrue(r.is_conditional)

    def test_conditional_level_is_medium(self):
        """CONDITIONAL → level = MEDIUM"""
        r = self.cc.evaluate(
            rule_score=1.0, context_score=0.5,
            skill_score=0.3, identity_score=1.0
        )
        self.assertEqual(r.level, ConfidenceLevel.MEDIUM)


# ─────────────────────────────────────────────────────────────────────────────
# 9. evaluate() — ask / silence path
# ─────────────────────────────────────────────────────────────────────────────

class TestAskSilencePath(unittest.TestCase):

    def setUp(self):
        self.cc = _cc()

    def test_low_score_ask(self):
        """score ≥ 0.25 และ < 0.50 → ASK"""
        r = self.cc.evaluate(
            rule_score=0.8, context_score=0.1,
            skill_score=0.1, identity_score=0.8
        )
        self.assertTrue(r.should_ask)

    def test_very_low_score_silence(self):
        """score < 0.25 → SILENCE"""
        # 0.3*0.35 + 0.0*0.25 + 0.0*0.20 + 0.3*0.20 = 0.105+0.06 = 0.165
        r = self.cc.evaluate(
            rule_score=0.3, context_score=0.0,
            skill_score=0.0, identity_score=0.3
        )
        self.assertTrue(r.should_silence)

    def test_low_confidence_conflict_set(self):
        """low level → conflict = LOW_CONFIDENCE"""
        r = self.cc.evaluate(
            rule_score=0.8, context_score=0.1,
            skill_score=0.1, identity_score=0.8
        )
        self.assertEqual(r.conflict, ConflictType.LOW_CONFIDENCE)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Weights
# ─────────────────────────────────────────────────────────────────────────────

class TestWeights(unittest.TestCase):

    def test_default_weights_sum_to_one(self):
        """default weights รวมกัน = 1.0"""
        cc = _cc()
        self.assertAlmostEqual(sum(cc.weights.values()), 1.0, places=3)

    def test_set_valid_weights(self):
        """set_weights() ด้วย weights รวม = 1.0 → สำเร็จ"""
        cc = _cc()
        cc.set_weights({
            "rule_score": 0.4, "context_score": 0.3,
            "skill_score": 0.2, "identity_score": 0.1
        })
        self.assertAlmostEqual(sum(cc.weights.values()), 1.0)

    def test_set_invalid_weights_raises(self):
        """set_weights() weights ไม่รวม 1.0 → ValueError"""
        cc = _cc()
        with self.assertRaises(ValueError):
            cc.set_weights({"rule_score": 0.5, "context_score": 0.5,
                            "skill_score": 0.5, "identity_score": 0.5})


# ─────────────────────────────────────────────────────────────────────────────
# 11. History & stats
# ─────────────────────────────────────────────────────────────────────────────

class TestHistoryStats(unittest.TestCase):

    def setUp(self):
        self.cc = _cc()

    def test_last_result_after_evaluate(self):
        """evaluate() → last_result ถูกอัปเดต"""
        r = self.cc.evaluate(rule_score=1.0, context_score=1.0,
                             skill_score=1.0, identity_score=1.0)
        self.assertEqual(self.cc.last_result.eval_id, r.eval_id)

    def test_history_accumulates(self):
        """evaluate หลายครั้ง → history เพิ่ม"""
        for _ in range(5):
            self.cc.evaluate()
        self.assertEqual(len(self.cc.history(10)), 5)

    def test_stats_commit_rate(self):
        """stats commit_rate ถูกต้อง"""
        self.cc.evaluate(rule_score=1.0, context_score=1.0,
                         skill_score=1.0, identity_score=1.0)
        self.cc.evaluate(identity_conflict=True)
        s = self.cc.stats()
        self.assertEqual(s["total_evaluations"], 2)
        self.assertAlmostEqual(s["commit_rate"], 0.5)

    def test_clear_history(self):
        """clear_history() → history ว่าง"""
        self.cc.evaluate()
        self.cc.clear_history()
        self.assertIsNone(self.cc.last_result)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1.  ConfidenceLevel & score_to_level     (5)", TestScoreToLevel),
        ("2.  ConfidenceOutcome & level_to_outcome (5)", TestLevelToOutcome),
        ("3.  ConfidenceResult properties          (5)", TestConfidenceResult),
        ("4.  Hard conflict — identity             (3)", TestIdentityConflict),
        ("5.  Hard conflict — system error         (2)", TestSystemError),
        ("6.  Hard conflict — rule blocked         (3)", TestRuleConflict),
        ("7.  evaluate() — commit path             (3)", TestCommitPath),
        ("8.  evaluate() — conditional path        (2)", TestConditionalPath),
        ("9.  evaluate() — ask / silence path      (3)", TestAskSilencePath),
        ("10. Weights                              (3)", TestWeights),
        ("11. History & stats                      (4)", TestHistoryStats),
    ]

    print("\n=================================================================")
    print("  Confidence System Test Suite")
    print("=================================================================")
    for label, _ in groups:
        print(f"  {label}")
    print("─────────────────────────────────────────────────────────────────")
    print("  Total: 38 tests")
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