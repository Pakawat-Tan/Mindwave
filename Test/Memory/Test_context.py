"""
=================================================================
  MemoryController × Topic/Emotion Integration Test Suite  v3
=================================================================
  1. write() with Topic only          (5 tests)
  2. read_with_context()              (4 tests)
  3. Path derives from TopicData      (3 tests)
  4. VAD Weighting Formula            (5 tests)
  5. read_for_response()              (5 tests)
-----------------------------------------------------------------
  Total: 22 tests
=================================================================
"""

import unittest
import json

from Core.Memory.MemoryController import MemoryController, AtomContext, WeightedAtom
from Core.Memory.Structure.AtomStructure import AtomData
from Core.Memory.Emotion import EmotionData, NEUTRAL_EMOTION
from Core.Memory.Topic   import TopicData

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mc() -> MemoryController:
    return MemoryController(base_path="/tmp/test_mc_v3")

def _atom(text: str = "hello") -> AtomData:
    return AtomData(payload=text.encode())

def _topic(cluster_id=1, keywords=None, coherence=0.7, label=None) -> TopicData:
    return TopicData(
        cluster_id   = cluster_id,
        top_keywords = keywords or ["python", "code"],
        coherence    = coherence,
        label        = label,
    )

def _emotion(v=0.0, a=0.0, d=0.5) -> EmotionData:
    return EmotionData(valence=v, arousal=a, dominance=d)


# ─────────────────────────────────────────────────────────────────────────────
# 1. write() with Topic only
# ─────────────────────────────────────────────────────────────────────────────

class TestWriteTopicOnly(unittest.TestCase):

    def setUp(self): self.mc = _mc()

    def test_write_returns_atom_id(self):
        """write() พร้อม topic → คืน atom_id"""
        atom_id = self.mc.write(_atom(), _topic(), importance=0.5)
        self.assertIsNotNone(atom_id)
        self.assertIsInstance(atom_id, str)

    def test_write_embeds_topic_in_metadata(self):
        """metadata ต้องมี 'topic' key"""
        atom_id = self.mc.write(_atom(), _topic(keywords=["ai","ml"]), importance=0.5)
        meta = json.loads(self.mc.read(atom_id).metadata.decode())
        self.assertIn("topic", meta)
        self.assertEqual(meta["topic"]["top_keywords"], ["ai", "ml"])

    def test_write_has_no_emotion_in_metadata(self):
        """metadata ต้องไม่มี 'emotion' key — emotion ไม่ถูกเก็บใน atom"""
        atom_id = self.mc.write(_atom(), _topic(), importance=0.5)
        meta = json.loads(self.mc.read(atom_id).metadata.decode())
        self.assertNotIn("emotion", meta)

    def test_write_low_importance_returns_none(self):
        """importance < 0.3 → ไม่เก็บ คืน None"""
        self.assertIsNone(self.mc.write(_atom(), _topic(), importance=0.1))

    def test_write_metadata_has_importance_and_tier(self):
        """metadata ต้องมี importance + tier"""
        atom_id = self.mc.write(_atom(), _topic(), importance=0.75)
        meta = json.loads(self.mc.read(atom_id).metadata.decode())
        self.assertAlmostEqual(meta["importance"], 0.75)
        self.assertEqual(meta["tier"], "long")


# ─────────────────────────────────────────────────────────────────────────────
# 2. read_with_context()
# ─────────────────────────────────────────────────────────────────────────────

class TestReadWithContext(unittest.TestCase):

    def setUp(self): self.mc = _mc()

    def test_returns_atom_context_type(self):
        """read_with_context() คืน AtomContext"""
        atom_id = self.mc.write(_atom(), _topic(), importance=0.5)
        self.assertIsInstance(self.mc.read_with_context(atom_id), AtomContext)

    def test_context_topic_roundtrip(self):
        """TopicData ที่อ่านกลับต้องเท่ากับที่เขียน"""
        t = _topic(cluster_id=77, keywords=["deep","neural"], coherence=0.91)
        atom_id = self.mc.write(_atom(), t, importance=0.5)
        ctx = self.mc.read_with_context(atom_id)
        self.assertEqual(ctx.topic.cluster_id,   77)
        self.assertEqual(ctx.topic.top_keywords, ["deep", "neural"])
        self.assertAlmostEqual(ctx.topic.coherence, 0.91)

    def test_context_has_no_emotion_field_by_default(self):
        """AtomContext ไม่มี emotion field — emotion ไม่เก็บใน atom"""
        atom_id = self.mc.write(_atom(), _topic(), importance=0.5)
        ctx = self.mc.read_with_context(atom_id)
        self.assertFalse(hasattr(ctx, "emotion"))

    def test_read_with_context_not_found_returns_none(self):
        """atom_id ที่ไม่มี → คืน None"""
        self.assertIsNone(self.mc.read_with_context("x" * 64))


# ─────────────────────────────────────────────────────────────────────────────
# 3. Path derives from TopicData
# ─────────────────────────────────────────────────────────────────────────────

class TestPathDerivation(unittest.TestCase):

    def setUp(self): self.mc = _mc()

    def test_category_uses_label_when_available(self):
        """topic มี label → category = label"""
        atom_id = self.mc.write(_atom(), _topic(label="Programming"), importance=0.5)
        meta = json.loads(self.mc.read(atom_id).metadata.decode())
        self.assertEqual(meta["category"], "Programming")

    def test_category_uses_cluster_id_when_no_label(self):
        """topic ไม่มี label → category = 'cluster_{id}'"""
        atom_id = self.mc.write(_atom(), _topic(cluster_id=42), importance=0.5)
        meta = json.loads(self.mc.read(atom_id).metadata.decode())
        self.assertEqual(meta["category"], "cluster_42")

    def test_primary_uses_top_keyword(self):
        """primary = top_keyword (อันดับแรก)"""
        atom_id = self.mc.write(_atom(), _topic(keywords=["machine","learning"]), importance=0.5)
        meta = json.loads(self.mc.read(atom_id).metadata.decode())
        self.assertEqual(meta["primary"], "machine")


# ─────────────────────────────────────────────────────────────────────────────
# 4. VAD Weighting Formula
# ─────────────────────────────────────────────────────────────────────────────

class TestEmotionWeightFormula(unittest.TestCase):

    def _score(self, v, a, d, importance=0.7, coherence=0.7, tier_rank=2):
        em = EmotionData(valence=v, arousal=a, dominance=d)
        return MemoryController._emotion_weight(importance, coherence, tier_rank, em)

    def test_high_arousal_gives_higher_score_than_low(self):
        """arousal สูง → score สูงกว่า arousal ต่ำ (importance + coherence เท่ากัน)"""
        high = self._score(v=0.0, a=0.9, d=0.5)
        low  = self._score(v=0.0, a=0.1, d=0.5)
        self.assertGreater(high, low)

    def test_negative_valence_favors_high_importance(self):
        """valence ลบ → atom ที่ importance สูงได้ score มากกว่า coherence สูง"""
        # atom A: importance สูง coherence ต่ำ
        score_imp = MemoryController._emotion_weight(
            importance=0.9, coherence=0.2, tier_rank=2,
            emotion=EmotionData(valence=-0.8, arousal=0.5, dominance=0.5)
        )
        # atom B: importance ต่ำ coherence สูง
        score_coh = MemoryController._emotion_weight(
            importance=0.2, coherence=0.9, tier_rank=2,
            emotion=EmotionData(valence=-0.8, arousal=0.5, dominance=0.5)
        )
        self.assertGreater(score_imp, score_coh)

    def test_positive_valence_favors_high_coherence(self):
        """valence บวก → atom ที่ coherence สูงได้ score มากกว่า importance สูง"""
        score_imp = MemoryController._emotion_weight(
            importance=0.9, coherence=0.2, tier_rank=2,
            emotion=EmotionData(valence=0.8, arousal=0.5, dominance=0.5)
        )
        score_coh = MemoryController._emotion_weight(
            importance=0.2, coherence=0.9, tier_rank=2,
            emotion=EmotionData(valence=0.8, arousal=0.5, dominance=0.5)
        )
        self.assertGreater(score_coh, score_imp)

    def test_high_dominance_boosts_deep_tier(self):
        """dominance สูง → immortal tier ได้ score สูงกว่า dominance ต่ำ"""
        high_d = self._score(v=0.0, a=0.5, d=1.0, tier_rank=4)  # immortal
        low_d  = self._score(v=0.0, a=0.5, d=0.0, tier_rank=4)
        self.assertGreater(high_d, low_d)

    def test_neutral_emotion_gives_moderate_score(self):
        """NEUTRAL emotion → score อยู่ในช่วง (0, 1.5)"""
        score = MemoryController._emotion_weight(
            importance=0.5, coherence=0.5, tier_rank=2, emotion=NEUTRAL_EMOTION
        )
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.5)


# ─────────────────────────────────────────────────────────────────────────────
# 5. read_for_response()
# ─────────────────────────────────────────────────────────────────────────────

class TestReadForResponse(unittest.TestCase):

    def setUp(self):
        self.mc = _mc()
        # เขียน atoms หลายตัวที่ต่าง importance / coherence
        self.id_low  = self.mc.write(
            _atom("low"),  _topic(coherence=0.3, keywords=["low"]),  importance=0.35
        )  # short tier, low coherence
        self.id_mid  = self.mc.write(
            _atom("mid"),  _topic(coherence=0.6, keywords=["mid"]),  importance=0.55
        )  # middle tier
        self.id_high = self.mc.write(
            _atom("high"), _topic(coherence=0.9, keywords=["high"]), importance=0.8
        )  # long tier, high coherence

    def test_returns_list_of_weighted_atoms(self):
        """read_for_response() คืน list of WeightedAtom"""
        ids = [self.id_low, self.id_mid, self.id_high]
        results = self.mc.read_for_response(ids, _emotion())
        self.assertIsInstance(results, list)
        for item in results:
            self.assertIsInstance(item, WeightedAtom)

    def test_results_sorted_by_score_descending(self):
        """ผลลัพธ์เรียงตาม score สูง → ต่ำ"""
        ids = [self.id_low, self.id_mid, self.id_high]
        results = self.mc.read_for_response(ids, _emotion())
        scores = [r.score for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_limit_caps_results(self):
        """limit=2 → คืนไม่เกิน 2 atoms"""
        ids = [self.id_low, self.id_mid, self.id_high]
        results = self.mc.read_for_response(ids, _emotion(), limit=2)
        self.assertLessEqual(len(results), 2)

    def test_negative_emotion_ranks_high_importance_first(self):
        """emotion negative → atom importance สูง (id_high) ขึ้นมาก่อน"""
        ids = [self.id_low, self.id_mid, self.id_high]
        angry = _emotion(v=-0.9, a=0.8, d=0.8)
        results = self.mc.read_for_response(ids, angry)
        self.assertEqual(results[0].context.atom_id, self.id_high)

    def test_neutral_emotion_fallback(self):
        """emotion=None → ใช้ NEUTRAL_EMOTION ไม่ error"""
        ids = [self.id_low, self.id_high]
        results = self.mc.read_for_response(ids, emotion=None)
        self.assertGreater(len(results), 0)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1. write() with Topic only          (5 tests)", TestWriteTopicOnly),
        ("2. read_with_context()              (4 tests)", TestReadWithContext),
        ("3. Path derives from TopicData      (3 tests)", TestPathDerivation),
        ("4. VAD Weighting Formula            (5 tests)", TestEmotionWeightFormula),
        ("5. read_for_response()              (5 tests)", TestReadForResponse),
    ]

    print("\n=================================================================")
    print("  MemoryController × Topic/Emotion Integration Test Suite  v3")
    print("=================================================================")
    for label, _ in groups:
        print(f"  {label}")
    print("─────────────────────────────────────────────────────────────────")
    print("  Total: 22 tests")
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