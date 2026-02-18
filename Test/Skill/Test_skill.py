"""
=================================================================
  Skill System Test Suite
=================================================================
  1. SkillData Construction          (4 tests)
  2. SkillData Growth                (6 tests)
  3. SkillEvent audit trail          (3 tests)
  4. ArbitrationResult               (3 tests)
  5. SkillController — Registry      (4 tests)
  6. SkillController — try_grow      (6 tests)
  7. SkillController — force_grow    (3 tests)
  8. SkillController — arbitration   (6 tests)
  9. SkillController — thresholds    (3 tests)
 10. SkillController — stats         (2 tests)
-----------------------------------------------------------------
  Total: 40 tests
=================================================================
"""

import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Core.Skill.SkillData import (
    SkillData, SkillEvent, ArbitrationResult,
    NO_SKILL_RESULT, SKILL_MAX, SKILL_MIN
)
from Core.Skill.SkillController import SkillController


REVIEWER = "reviewer_001"


def _sc(**kwargs) -> SkillController:
    return SkillController(**kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SkillData Construction
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillDataConstruction(unittest.TestCase):

    def test_create_skill_default_score_zero(self):
        """SkillData ใหม่ → score = 0.0"""
        s = SkillData(skill_name="python")
        self.assertAlmostEqual(s.score, 0.0)

    def test_empty_name_raises(self):
        """skill_name ว่าง → ValueError"""
        with self.assertRaises(ValueError):
            SkillData(skill_name="")

    def test_topic_ids_stored(self):
        """topic_ids เก็บถูกต้อง"""
        s = SkillData(skill_name="math", topic_ids=[1, 2, 3])
        self.assertEqual(s.topic_ids, [1, 2, 3])

    def test_is_maxed_false_initially(self):
        """ใหม่ → is_maxed = False"""
        s = SkillData(skill_name="python")
        self.assertFalse(s.is_maxed)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SkillData Growth
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillDataGrowth(unittest.TestCase):

    def test_grow_increases_score(self):
        """grow(1.0) → score = 1.0"""
        s = SkillData(skill_name="python")
        s.grow(delta=1.0, topic_repetition=5, avg_confidence=0.8)
        self.assertAlmostEqual(s.score, 1.0)

    def test_grow_precision_4_decimal(self):
        """score มี precision 4 decimal"""
        s = SkillData(skill_name="python")
        s.grow(delta=0.12345, topic_repetition=5, avg_confidence=0.8)
        self.assertEqual(s.score, round(0.12345, 4))

    def test_grow_negative_delta_raises(self):
        """delta <= 0 → ValueError"""
        s = SkillData(skill_name="python")
        with self.assertRaises(ValueError):
            s.grow(delta=-1.0, topic_repetition=5, avg_confidence=0.8)

    def test_grow_zero_delta_raises(self):
        """delta = 0 → ValueError"""
        s = SkillData(skill_name="python")
        with self.assertRaises(ValueError):
            s.grow(delta=0.0, topic_repetition=5, avg_confidence=0.8)

    def test_grow_capped_at_max(self):
        """score ไม่เกิน SKILL_MAX"""
        s = SkillData(skill_name="python")
        s.grow(delta=99.0, topic_repetition=5, avg_confidence=0.8)
        s.grow(delta=99.0, topic_repetition=5, avg_confidence=0.8)
        self.assertAlmostEqual(s.score, SKILL_MAX)

    def test_is_maxed_after_cap(self):
        """score = 100 → is_maxed = True"""
        s = SkillData(skill_name="python")
        s.grow(delta=100.0, topic_repetition=5, avg_confidence=0.8)
        self.assertTrue(s.is_maxed)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SkillEvent audit trail
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillEvent(unittest.TestCase):

    def test_event_logged_after_grow(self):
        """grow → events มี 1 entry"""
        s = SkillData(skill_name="python")
        s.grow(delta=2.0, topic_repetition=5, avg_confidence=0.9)
        self.assertEqual(s.event_count, 1)

    def test_event_stores_correct_values(self):
        """event เก็บ score_before, score_after, delta ถูกต้อง"""
        s = SkillData(skill_name="python")
        event = s.grow(delta=3.0, topic_repetition=4, avg_confidence=0.75)
        self.assertAlmostEqual(event.score_before, 0.0)
        self.assertAlmostEqual(event.score_after,  3.0)
        self.assertAlmostEqual(event.delta,        3.0)

    def test_multiple_events_accumulate(self):
        """grow หลายครั้ง → events เพิ่มทุกครั้ง"""
        s = SkillData(skill_name="python")
        for _ in range(5):
            s.grow(delta=1.0, topic_repetition=3, avg_confidence=0.7)
        self.assertEqual(s.event_count, 5)


# ─────────────────────────────────────────────────────────────────────────────
# 4. ArbitrationResult
# ─────────────────────────────────────────────────────────────────────────────

class TestArbitrationResult(unittest.TestCase):

    def test_no_skill_result_defaults(self):
        """NO_SKILL_RESULT → weight=0, has_skills=False"""
        self.assertAlmostEqual(NO_SKILL_RESULT.weight, 0.0)
        self.assertFalse(NO_SKILL_RESULT.has_skills)

    def test_weight_is_score_over_max(self):
        """weight = combined_score / SKILL_MAX"""
        s = SkillData(skill_name="math")
        s.grow(delta=50.0, topic_repetition=5, avg_confidence=0.9)
        result = ArbitrationResult(
            selected_skills = (s,),
            highest_score   = 50.0,
            combined_score  = 50.0,
            weight          = round(50.0 / SKILL_MAX, 4),
        )
        self.assertAlmostEqual(result.weight, 0.5)

    def test_skill_names_property(self):
        """skill_names คืน list ของชื่อ"""
        s = SkillData(skill_name="coding")
        result = ArbitrationResult(
            selected_skills=(s,), highest_score=0.0,
            combined_score=0.0, weight=0.0
        )
        self.assertIn("coding", result.skill_names)


# ─────────────────────────────────────────────────────────────────────────────
# 5. SkillController — Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillRegistry(unittest.TestCase):

    def test_register_new_skill(self):
        """register ใหม่ → has() = True"""
        sc = _sc()
        sc.register("python", topic_ids=[1])
        self.assertTrue(sc.has("python"))

    def test_register_duplicate_returns_same(self):
        """register ชื่อซ้ำ → คืน skill เดิม ไม่ reset score"""
        sc = _sc()
        s1 = sc.register("python")
        sc.force_grow("python", 5.0, REVIEWER)
        s2 = sc.register("python")  # ซ้ำ
        self.assertIs(s1, s2)
        self.assertAlmostEqual(s2.score, 5.0)

    def test_get_registered_skill(self):
        """get() คืน SkillData"""
        sc = _sc()
        sc.register("math")
        self.assertIsNotNone(sc.get("math"))

    def test_get_unregistered_returns_none(self):
        """get() ชื่อที่ไม่มี → None"""
        sc = _sc()
        self.assertIsNone(sc.get("nonexistent"))


# ─────────────────────────────────────────────────────────────────────────────
# 6. SkillController — try_grow
# ─────────────────────────────────────────────────────────────────────────────

class TestTryGrow(unittest.TestCase):

    def setUp(self):
        self.sc = _sc(repetition_threshold=3, confidence_threshold=0.6)
        self.sc.register("python")

    def test_grow_when_conditions_met(self):
        """condition ผ่าน → grow สำเร็จ คืน SkillEvent"""
        event = self.sc.try_grow("python", 1.0, topic_repetition=3, avg_confidence=0.7)
        self.assertIsNotNone(event)
        self.assertAlmostEqual(self.sc.get("python").score, 1.0)

    def test_grow_fails_low_repetition(self):
        """repetition < threshold → ไม่ grow"""
        event = self.sc.try_grow("python", 1.0, topic_repetition=2, avg_confidence=0.9)
        self.assertIsNone(event)
        self.assertAlmostEqual(self.sc.get("python").score, 0.0)

    def test_grow_fails_low_confidence(self):
        """confidence < threshold → ไม่ grow"""
        event = self.sc.try_grow("python", 1.0, topic_repetition=5, avg_confidence=0.5)
        self.assertIsNone(event)

    def test_grow_fails_unregistered_skill(self):
        """skill ไม่ได้ register → คืน None"""
        event = self.sc.try_grow("unknown", 1.0, topic_repetition=5, avg_confidence=0.9)
        self.assertIsNone(event)

    def test_grow_stops_at_max(self):
        """grow ถึง max แล้ว → ไม่ grow อีก"""
        self.sc.force_grow("python", 100.0, REVIEWER)
        event = self.sc.try_grow("python", 1.0, topic_repetition=5, avg_confidence=0.9)
        self.assertIsNone(event)
        self.assertAlmostEqual(self.sc.get("python").score, SKILL_MAX)

    def test_grow_exact_threshold_passes(self):
        """repetition = threshold, confidence = threshold → pass"""
        event = self.sc.try_grow("python", 0.5, topic_repetition=3, avg_confidence=0.6)
        self.assertIsNotNone(event)


# ─────────────────────────────────────────────────────────────────────────────
# 7. SkillController — force_grow
# ─────────────────────────────────────────────────────────────────────────────

class TestForceGrow(unittest.TestCase):

    def setUp(self):
        self.sc = _sc()
        self.sc.register("math")

    def test_force_grow_with_reviewer_succeeds(self):
        """force_grow + reviewer_id → grow สำเร็จ"""
        event = self.sc.force_grow("math", 10.0, REVIEWER)
        self.assertAlmostEqual(self.sc.get("math").score, 10.0)

    def test_force_grow_without_reviewer_raises(self):
        """force_grow ไม่มี reviewer_id → PermissionError"""
        with self.assertRaises(PermissionError):
            self.sc.force_grow("math", 5.0, reviewer_id="")

    def test_force_grow_unregistered_raises(self):
        """force_grow skill ที่ไม่มี → KeyError"""
        with self.assertRaises(KeyError):
            self.sc.force_grow("nonexistent", 5.0, REVIEWER)


# ─────────────────────────────────────────────────────────────────────────────
# 8. SkillController — arbitration
# ─────────────────────────────────────────────────────────────────────────────

class TestArbitration(unittest.TestCase):

    def setUp(self):
        self.sc = _sc()
        self.sc.register("python",  topic_ids=[1, 2])
        self.sc.register("math",    topic_ids=[2, 3])
        self.sc.register("writing", topic_ids=[4])
        self.sc.force_grow("python",  30.0, REVIEWER)
        self.sc.force_grow("math",    50.0, REVIEWER)
        self.sc.force_grow("writing", 20.0, REVIEWER)

    def test_arbitrate_by_topic_returns_highest(self):
        """topic_id=2 → math (50.0) ชนะ python (30.0)"""
        result = self.sc.arbitrate(topic_id=2)
        self.assertEqual(result.skill_names, ["math"])
        self.assertAlmostEqual(result.highest_score, 50.0)

    def test_arbitrate_no_match_returns_no_skill(self):
        """topic ที่ไม่มี skill match → NO_SKILL_RESULT"""
        result = self.sc.arbitrate(topic_id=99)
        self.assertFalse(result.has_skills)

    def test_arbitrate_tie_sums_scores(self):
        """สอง skills score เท่ากัน → combined = sum"""
        self.sc.register("skill_a", topic_ids=[7])
        self.sc.register("skill_b", topic_ids=[7])
        self.sc.force_grow("skill_a", 40.0, REVIEWER)
        self.sc.force_grow("skill_b", 40.0, REVIEWER)
        result = self.sc.arbitrate(topic_id=7)
        self.assertAlmostEqual(result.combined_score, 80.0)
        self.assertEqual(len(result.selected_skills), 2)

    def test_arbitrate_weight_is_normalized(self):
        """weight = combined_score / 100"""
        result = self.sc.arbitrate(topic_id=3)  # math=50 เท่านั้น
        self.assertAlmostEqual(result.weight, 0.5)

    def test_arbitrate_by_skill_names(self):
        """ระบุ skill_names โดยตรง → arbitrate ได้"""
        result = self.sc.arbitrate(skill_names=["python", "math"])
        self.assertAlmostEqual(result.highest_score, 50.0)

    def test_arbitrate_all_when_no_filter(self):
        """ไม่ระบุ topic หรือ names → arbitrate ทุก skills"""
        result = self.sc.arbitrate()
        self.assertAlmostEqual(result.highest_score, 50.0)


# ─────────────────────────────────────────────────────────────────────────────
# 9. SkillController — thresholds
# ─────────────────────────────────────────────────────────────────────────────

class TestThresholds(unittest.TestCase):

    def test_default_thresholds(self):
        """default: rep=3, conf=0.6"""
        sc = _sc()
        self.assertEqual(sc.repetition_threshold, 3)
        self.assertAlmostEqual(sc.confidence_threshold, 0.6)

    def test_set_thresholds(self):
        """set_thresholds() เปลี่ยนค่าได้"""
        sc = _sc()
        sc.set_thresholds(repetition=5, confidence=0.8)
        self.assertEqual(sc.repetition_threshold, 5)
        self.assertAlmostEqual(sc.confidence_threshold, 0.8)

    def test_custom_threshold_at_init(self):
        """กำหนด threshold ตอน init"""
        sc = SkillController(repetition_threshold=10, confidence_threshold=0.9)
        self.assertEqual(sc.repetition_threshold, 10)


# ─────────────────────────────────────────────────────────────────────────────
# 10. SkillController — stats
# ─────────────────────────────────────────────────────────────────────────────

class TestStats(unittest.TestCase):

    def test_stats_empty(self):
        """ไม่มี skill → stats ว่าง"""
        sc = _sc()
        s = sc.stats()
        self.assertEqual(s["skill_count"], 0)
        self.assertAlmostEqual(s["avg_score"], 0.0)

    def test_stats_after_grow(self):
        """หลัง grow → stats สะท้อนค่าจริง"""
        sc = _sc()
        sc.register("python")
        sc.register("math")
        sc.force_grow("python", 100.0, REVIEWER)
        s = sc.stats()
        self.assertEqual(s["skill_count"], 2)
        self.assertEqual(s["maxed_count"], 1)
        self.assertAlmostEqual(s["avg_score"], 50.0)
        self.assertGreater(s["total_events"], 0)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1.  SkillData Construction         (4)", TestSkillDataConstruction),
        ("2.  SkillData Growth               (6)", TestSkillDataGrowth),
        ("3.  SkillEvent audit trail         (3)", TestSkillEvent),
        ("4.  ArbitrationResult              (3)", TestArbitrationResult),
        ("5.  SkillController — Registry     (4)", TestSkillRegistry),
        ("6.  SkillController — try_grow     (6)", TestTryGrow),
        ("7.  SkillController — force_grow   (3)", TestForceGrow),
        ("8.  SkillController — arbitration  (6)", TestArbitration),
        ("9.  SkillController — thresholds   (3)", TestThresholds),
        ("10. SkillController — stats        (2)", TestStats),
    ]

    print("\n=================================================================")
    print("  Skill System Test Suite")
    print("=================================================================")
    for label, _ in groups:
        print(f"  {label}")
    print("─────────────────────────────────────────────────────────────────")
    print("  Total: 40 tests")
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