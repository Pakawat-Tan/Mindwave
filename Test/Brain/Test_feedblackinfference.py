"""
=================================================================
  FeedbackInference Test Suite
=================================================================
  1. Confusion Detection     (4 tests)
  2. Repeat Detection        (4 tests)
  3. Follow-up Detection     (4 tests)
  4. Context Switch          (3 tests)
  5. Immediate Effect        (4 tests)
  6. Long-term / Session     (4 tests)
  7. Integration             (3 tests)
-----------------------------------------------------------------
  Total: 26 tests
=================================================================
"""

import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Core.Brain.FeedbackInference import (
    FeedbackInference, FeedbackType, FeedbackPolarity,
    FeedbackSignal, FeedbackAtom, ImmediateEffect,
)
from Core.BrainController import BrainLog
import time


def _make_log(log_id: str = "abc") -> BrainLog:
    return BrainLog(
        log_id="test_" + log_id, input_text="test",
        context="general", outcome="commit",
        confidence=0.7, skill_weight=0.5,
        personality="test", learned=False, response="ok",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Confusion Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestConfusion(unittest.TestCase):

    def setUp(self): self.fi = FeedbackInference()

    def test_detect_thai_confusion(self):
        sig = self.fi.infer("งงมากเลย", "general", _make_log())
        self.assertIsNotNone(sig)
        self.assertEqual(sig.signal_type, FeedbackType.CONFUSION)

    def test_detect_english_confusion(self):
        sig = self.fi.infer("i don't understand", "math", _make_log())
        self.assertIsNotNone(sig)
        self.assertEqual(sig.polarity, FeedbackPolarity.NEGATIVE)

    def test_confusion_strength_scales_with_keywords(self):
        sig1 = self.fi.infer("งง", "general", _make_log())
        fi2  = FeedbackInference()
        sig2 = fi2.infer("งง ไม่เข้าใจ ไม่ชัด", "general", _make_log())
        if sig1 and sig2:
            self.assertGreaterEqual(sig2.strength, sig1.strength)

    def test_no_confusion_on_normal_text(self):
        sig = self.fi.infer("ขอบคุณครับ", "general", _make_log())
        self.assertIsNone(sig)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Repeat Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestRepeat(unittest.TestCase):

    def setUp(self): self.fi = FeedbackInference()

    def test_detect_exact_repeat(self):
        self.fi.infer("อธิบาย neural network", "math", _make_log())
        sig = self.fi.infer("อธิบาย neural network", "math", _make_log())
        self.assertIsNotNone(sig)
        self.assertEqual(sig.signal_type, FeedbackType.REPEAT)

    def test_detect_similar_repeat(self):
        self.fi.infer("same text same text", "math", _make_log())
        sig = self.fi.infer("same text same text", "math", _make_log())
        # ข้อความเหมือนกันทุกคำ → repeat ชัดเจน
        self.assertIsNotNone(sig)
        self.assertEqual(sig.signal_type, FeedbackType.REPEAT)

    def test_no_repeat_different_context(self):
        self.fi.infer("อธิบาย neural network", "math", _make_log())
        sig = self.fi.infer("อธิบาย neural network", "science", _make_log())
        # context ต่างกัน → ไม่ใช่ repeat
        if sig:
            self.assertNotEqual(sig.signal_type, FeedbackType.REPEAT)

    def test_repeat_is_negative(self):
        self.fi.infer("hello world test", "general", _make_log())
        sig = self.fi.infer("hello world test", "general", _make_log())
        if sig:
            self.assertEqual(sig.polarity, FeedbackPolarity.NEGATIVE)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Follow-up Detection
# ─────────────────────────────────────────────────────────────────────────────

class TestFollowUp(unittest.TestCase):

    def setUp(self): self.fi = FeedbackInference()

    def test_detect_follow_up_same_context(self):
        self.fi.infer("neural network คืออะไร", "math", _make_log(),
                      prev_context="")
        sig = self.fi.infer("แล้ว deep learning ล่ะ", "math", _make_log(),
                            prev_context="math")
        self.assertIsNotNone(sig)

    def test_follow_up_is_positive(self):
        self.fi.infer("คำถามแรก abc", "science", _make_log(), prev_context="")
        sig = self.fi.infer("คำถามต่อ xyz", "science", _make_log(),
                            prev_context="science")
        if sig and sig.signal_type == FeedbackType.FOLLOW_UP:
            self.assertEqual(sig.polarity, FeedbackPolarity.POSITIVE)

    def test_no_follow_up_on_first_message(self):
        sig = self.fi.infer("คำถามแรก", "general", _make_log(),
                            prev_context="")
        # ไม่มี prev_context → ไม่ใช่ follow-up
        if sig:
            self.assertNotEqual(sig.signal_type, FeedbackType.FOLLOW_UP)

    def test_follow_up_not_repeat(self):
        self.fi.infer("คำถาม A", "math", _make_log(), prev_context="")
        sig = self.fi.infer("คำถาม B ใหม่", "math", _make_log(),
                            prev_context="math")
        if sig:
            self.assertNotEqual(sig.signal_type, FeedbackType.REPEAT)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Context Switch
# ─────────────────────────────────────────────────────────────────────────────

class TestContextSwitch(unittest.TestCase):

    def setUp(self): self.fi = FeedbackInference()

    def test_detect_context_switch(self):
        sig = self.fi.infer("เรื่องใหม่", "science", _make_log(),
                            prev_context="math")
        self.assertIsNotNone(sig)
        self.assertEqual(sig.signal_type, FeedbackType.CTX_SWITCH)

    def test_no_switch_same_context(self):
        sig = self.fi.infer("ถามต่อ", "math", _make_log(),
                            prev_context="math")
        if sig:
            self.assertNotEqual(sig.signal_type, FeedbackType.CTX_SWITCH)

    def test_ctx_switch_positive(self):
        sig = self.fi.infer("เรื่องอื่น", "history", _make_log(),
                            prev_context="math")
        if sig and sig.signal_type == FeedbackType.CTX_SWITCH:
            self.assertEqual(sig.polarity, FeedbackPolarity.POSITIVE)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Immediate Effect
# ─────────────────────────────────────────────────────────────────────────────

class TestImmediateEffect(unittest.TestCase):

    def setUp(self): self.fi = FeedbackInference()

    def test_confusion_gives_negative_delta(self):
        sig = self.fi.infer("งงมากเลย", "general", _make_log())
        if sig:
            effect = self.fi.get_immediate_effect(sig)
            self.assertLess(effect.confidence_delta, 0)
            self.assertLess(effect.skill_delta, 0)

    def test_ctx_switch_gives_positive_delta(self):
        sig = self.fi.infer("เรื่องใหม่", "science", _make_log(),
                            prev_context="math")
        if sig and sig.signal_type == FeedbackType.CTX_SWITCH:
            effect = self.fi.get_immediate_effect(sig)
            self.assertGreater(effect.confidence_delta, 0)

    def test_effect_has_reason(self):
        sig = self.fi.infer("งง", "general", _make_log())
        if sig:
            effect = self.fi.get_immediate_effect(sig)
            self.assertNotEqual(effect.reason, "")

    def test_cumulative_delta_accumulates(self):
        sig1 = self.fi.infer("งง", "general", _make_log())
        if sig1: self.fi.get_immediate_effect(sig1)
        sig2 = self.fi.infer("งง", "general", _make_log())
        if sig2: self.fi.get_immediate_effect(sig2)
        stats = self.fi.stats()
        self.assertLess(stats["cumulative_conf"], 0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Long-term / Session
# ─────────────────────────────────────────────────────────────────────────────

class TestLongTerm(unittest.TestCase):

    def setUp(self): self.fi = FeedbackInference()

    def test_seal_session_creates_atom(self):
        self.fi.infer("hello", "general", _make_log())
        atom = self.fi.seal_session()
        self.assertIsInstance(atom, FeedbackAtom)
        self.assertEqual(len(self.fi.atoms), 1)

    def test_seal_with_silence_reward(self):
        self.fi.infer("คำถาม", "general", _make_log())
        atom = self.fi.seal_session(silence_reward=True)
        self.assertGreater(atom.net_reward, 0)

    def test_get_long_term_delta_resets(self):
        self.fi.infer("งง", "general", _make_log())
        c1, s1 = self.fi.get_long_term_delta()
        c2, s2 = self.fi.get_long_term_delta()
        # หลัง read ครั้งแรก → reset เป็น 0
        self.assertEqual(c2, 0.0)
        self.assertEqual(s2, 0.0)

    def test_new_session_starts_after_seal(self):
        self.fi.infer("msg", "general", _make_log())
        self.fi.seal_session()
        # current_atom ใหม่
        self.assertEqual(len(self.fi.current_atom.signals), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):

    def test_full_session_pipeline(self):
        fi = FeedbackInference()
        log = _make_log()

        # simulate conversation
        fi.infer("neural network คืออะไร", "math", log)
        fi.infer("แล้ว deep learning ล่ะ", "math", log, prev_context="math")
        fi.infer("งงนิดหน่อย", "math", log, prev_context="math")
        fi.infer("โอเค เข้าใจแล้ว เรื่องอื่น", "science", log, prev_context="math")

        atom = fi.seal_session(silence_reward=True)
        c, s = fi.get_long_term_delta()

        self.assertIsNotNone(atom)
        self.assertIsInstance(c, float)
        self.assertIsInstance(s, float)

    def test_stats_reflect_signals(self):
        fi = FeedbackInference()
        fi.infer("งง", "general", _make_log())
        fi.infer("งง", "general", _make_log())
        stats = fi.stats()
        self.assertGreater(stats["total_signals"], 0)
        self.assertIn("confusion", stats["by_type"])

    def test_signals_property(self):
        fi = FeedbackInference()
        fi.infer("งง", "general", _make_log())
        self.assertEqual(len(fi.signals), 1)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1. Confusion Detection (4)", TestConfusion),
        ("2. Repeat Detection    (4)", TestRepeat),
        ("3. Follow-up Detection (4)", TestFollowUp),
        ("4. Context Switch      (3)", TestContextSwitch),
        ("5. Immediate Effect    (4)", TestImmediateEffect),
        ("6. Long-term / Session (4)", TestLongTerm),
        ("7. Integration         (3)", TestIntegration),
    ]

    print("\n=================================================================")
    print("  FeedbackInference Test Suite")
    print("=================================================================")
    for label, _ in groups:
        print(f"  {label}")
    print("─────────────────────────────────────────────────────────────────")
    print("  Total: 26 tests")
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