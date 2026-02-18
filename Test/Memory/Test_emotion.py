"""
=================================================================
  EmotionData Test Suite  (VAD Model)
=================================================================
  1. Construction & Clamping     (5 tests)
  2. Tendency Derivation         (5 tests)
  3. Derived Properties          (5 tests)
  4. Serialization               (3 tests)
  5. Factory & Sentinel          (2 tests)
-----------------------------------------------------------------
  Total: 20 tests
=================================================================
"""

import unittest
import json
import math
from Core.Memory.Emotion import EmotionData, create_emotion, NEUTRAL_EMOTION


# ============================================================================
# 1. Construction & Clamping
# ============================================================================

class TestEmotionConstruction(unittest.TestCase):

    def test_valence_clamped_above(self):
        """valence > 1.0 → clamp เป็น 1.0"""
        e = EmotionData(valence=2.5, arousal=0.5, dominance=0.5)
        self.assertEqual(e.valence, 1.0)

    def test_valence_clamped_below(self):
        """valence < -1.0 → clamp เป็น -1.0"""
        e = EmotionData(valence=-3.0, arousal=0.5, dominance=0.5)
        self.assertEqual(e.valence, -1.0)

    def test_arousal_clamped_below_zero(self):
        """arousal < 0.0 → clamp เป็น 0.0"""
        e = EmotionData(valence=0.0, arousal=-0.5, dominance=0.5)
        self.assertEqual(e.arousal, 0.0)

    def test_dominance_clamped_above_one(self):
        """dominance > 1.0 → clamp เป็น 1.0"""
        e = EmotionData(valence=0.0, arousal=0.5, dominance=1.8)
        self.assertEqual(e.dominance, 1.0)

    def test_confidence_clamped(self):
        """confidence ถูก clamp ทั้งบนและล่าง"""
        hi = EmotionData(valence=0.0, arousal=0.0, dominance=0.5, confidence=5.0)
        lo = EmotionData(valence=0.0, arousal=0.0, dominance=0.5, confidence=-1.0)
        self.assertEqual(hi.confidence, 1.0)
        self.assertEqual(lo.confidence, 0.0)


# ============================================================================
# 2. Tendency Derivation
# ============================================================================

class TestEmotionTendency(unittest.TestCase):
    """tendency label ต้องมาจาก VAD coordinates ไม่ใช่ manual input"""

    def test_high_valence_high_arousal_is_excited(self):
        """V=+0.8, A=0.8 → excited"""
        e = EmotionData(valence=0.8, arousal=0.8, dominance=0.5)
        self.assertEqual(e.tendency, "excited")

    def test_high_valence_low_arousal_is_relaxed(self):
        """V=+0.6, A=0.1 → relaxed"""
        e = EmotionData(valence=0.6, arousal=0.1, dominance=0.5)
        self.assertEqual(e.tendency, "relaxed")

    def test_negative_valence_high_arousal_high_dominance_is_angry(self):
        """V=-0.7, A=0.8, D=0.8 → angry"""
        e = EmotionData(valence=-0.7, arousal=0.8, dominance=0.8)
        self.assertEqual(e.tendency, "angry")

    def test_negative_valence_low_arousal_is_sad(self):
        """V=-0.6, A=0.2 → sad"""
        e = EmotionData(valence=-0.6, arousal=0.2, dominance=0.5)
        self.assertEqual(e.tendency, "sad")

    def test_near_zero_valence_mid_arousal_is_neutral(self):
        """V=0.0, A=0.5 → neutral"""
        e = EmotionData(valence=0.0, arousal=0.5, dominance=0.5)
        self.assertEqual(e.tendency, "neutral")


# ============================================================================
# 3. Derived Properties
# ============================================================================

class TestEmotionDerivedProperties(unittest.TestCase):

    def test_is_positive(self):
        """valence > 0.1 → is_positive = True"""
        e = EmotionData(valence=0.5, arousal=0.3, dominance=0.5)
        self.assertTrue(e.is_positive)
        self.assertFalse(e.is_negative)

    def test_is_negative(self):
        """valence < -0.1 → is_negative = True"""
        e = EmotionData(valence=-0.5, arousal=0.3, dominance=0.5)
        self.assertTrue(e.is_negative)
        self.assertFalse(e.is_positive)

    def test_is_neutral(self):
        """|valence| ≤ 0.1 → is_neutral = True"""
        e = EmotionData(valence=0.05, arousal=0.3, dominance=0.5)
        self.assertTrue(e.is_neutral)

    def test_intensity_at_origin_is_zero(self):
        """V=0, A=0, D=0 → intensity = 0.0"""
        e = EmotionData(valence=0.0, arousal=0.0, dominance=0.0)
        self.assertAlmostEqual(e.intensity, 0.0)

    def test_intensity_increases_with_vad_magnitude(self):
        """ยิ่ง VAD ห่างจาก origin มาก → intensity สูงกว่า"""
        low  = EmotionData(valence=0.1, arousal=0.1, dominance=0.1)
        high = EmotionData(valence=0.9, arousal=0.9, dominance=0.9)
        self.assertGreater(high.intensity, low.intensity)


# ============================================================================
# 4. Serialization
# ============================================================================

class TestEmotionSerialization(unittest.TestCase):

    def _sample(self) -> EmotionData:
        return EmotionData(valence=0.7, arousal=0.6, dominance=0.4,
                           confidence=0.85, context="test context")

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict → from_dict → ค่าครบถ้วน"""
        original = self._sample()
        restored = EmotionData.from_dict(original.to_dict())
        self.assertAlmostEqual(restored.valence,    original.valence)
        self.assertAlmostEqual(restored.arousal,    original.arousal)
        self.assertAlmostEqual(restored.dominance,  original.dominance)
        self.assertAlmostEqual(restored.confidence, original.confidence)
        self.assertEqual(restored.context, original.context)

    def test_to_json_from_json_roundtrip(self):
        """to_json → from_json → ค่าครบถ้วน"""
        original = self._sample()
        restored = EmotionData.from_json(original.to_json())
        self.assertAlmostEqual(restored.valence,   original.valence)
        self.assertAlmostEqual(restored.dominance, original.dominance)

    def test_to_json_contains_all_vad_keys(self):
        """to_json() ต้องมี key: valence, arousal, dominance"""
        parsed = json.loads(self._sample().to_json())
        for key in ("valence", "arousal", "dominance", "confidence", "context"):
            self.assertIn(key, parsed)


# ============================================================================
# 5. Factory & Sentinel
# ============================================================================

class TestEmotionFactory(unittest.TestCase):

    def test_create_emotion_defaults_are_neutral_origin(self):
        """create_emotion() ไม่ระบุ args → V=0, A=0, D=0.5"""
        e = create_emotion()
        self.assertEqual(e.valence,   0.0)
        self.assertEqual(e.arousal,   0.0)
        self.assertEqual(e.dominance, 0.5)

    def test_neutral_emotion_sentinel_is_not_positive_or_negative(self):
        """NEUTRAL_EMOTION sentinel → is_neutral = True"""
        self.assertTrue(NEUTRAL_EMOTION.is_neutral)
        self.assertFalse(NEUTRAL_EMOTION.is_positive)
        self.assertFalse(NEUTRAL_EMOTION.is_negative)


# ============================================================================
# RUNNER
# ============================================================================

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1. Construction & Clamping     (5 tests)", TestEmotionConstruction),
        ("2. Tendency Derivation         (5 tests)", TestEmotionTendency),
        ("3. Derived Properties          (5 tests)", TestEmotionDerivedProperties),
        ("4. Serialization               (3 tests)", TestEmotionSerialization),
        ("5. Factory & Sentinel          (2 tests)", TestEmotionFactory),
    ]

    print("\n=================================================================")
    print("  EmotionData Test Suite  (VAD Model)")
    print("=================================================================")
    for label, _ in groups:
        print(f"  {label}")
    print("─────────────────────────────────────────────────────────────────")
    print("  Total: 20 tests")
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