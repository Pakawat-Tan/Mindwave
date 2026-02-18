"""
=================================================================
  TopicData Test Suite  (Emergent Cluster Model)
=================================================================
  1. Construction & Normalization  (5 tests)
  2. Derived Properties            (4 tests)
  3. Label Management              (3 tests)
  4. Similarity                    (5 tests)
  5. Serialization                 (4 tests)
  6. Factory                       (1 test)
-----------------------------------------------------------------
  Total: 22 tests
=================================================================
"""

import unittest
import json
from Core.Memory.Topic import TopicData, create_topic

# ============================================================================
# 1. Construction & Normalization
# ============================================================================

class TestTopicConstruction(unittest.TestCase):

    def test_keywords_stripped_and_lowercased(self):
        """keywords → strip whitespace + lowercase"""
        t = TopicData(cluster_id=1, top_keywords=["  Python  ", "CODE"], coherence=0.7)
        self.assertIn("python", t.top_keywords)
        self.assertIn("code",   t.top_keywords)

    def test_keywords_deduplicated(self):
        """keywords ที่ซ้ำ (case-insensitive) → ลบซ้ำ"""
        t = TopicData(cluster_id=1,
                      top_keywords=["python", "Python", "PYTHON", "code"],
                      coherence=0.7)
        self.assertEqual(t.top_keywords, ["python", "code"])

    def test_coherence_clamped_above(self):
        """coherence > 1.0 → clamp เป็น 1.0"""
        t = TopicData(cluster_id=1, top_keywords=[], coherence=2.5)
        self.assertEqual(t.coherence, 1.0)

    def test_coherence_clamped_below(self):
        """coherence < 0.0 → clamp เป็น 0.0"""
        t = TopicData(cluster_id=1, top_keywords=[], coherence=-0.3)
        self.assertEqual(t.coherence, 0.0)

    def test_document_count_clamped_to_zero(self):
        """document_count < 0 → clamp เป็น 0"""
        t = TopicData(cluster_id=1, top_keywords=[], coherence=0.5, document_count=-5)
        self.assertEqual(t.document_count, 0)


# ============================================================================
# 2. Derived Properties
# ============================================================================

class TestTopicDerivedProperties(unittest.TestCase):

    def test_has_label_false_by_default(self):
        """label ไม่ได้กำหนด → has_label = False"""
        t = TopicData(cluster_id=1, top_keywords=["ai"], coherence=0.6)
        self.assertFalse(t.has_label)

    def test_is_coherent_above_threshold(self):
        """coherence >= 0.5 → is_coherent = True"""
        t_hi = TopicData(cluster_id=1, top_keywords=[], coherence=0.8)
        t_lo = TopicData(cluster_id=2, top_keywords=[], coherence=0.3)
        self.assertTrue(t_hi.is_coherent)
        self.assertFalse(t_lo.is_coherent)

    def test_top_keyword_returns_first(self):
        """top_keyword → อันดับแรกของ top_keywords"""
        t = TopicData(cluster_id=1, top_keywords=["machine", "learning"], coherence=0.7)
        self.assertEqual(t.top_keyword, "machine")

    def test_top_keyword_none_when_empty(self):
        """top_keywords ว่าง → top_keyword = None"""
        t = TopicData(cluster_id=1, top_keywords=[], coherence=0.5)
        self.assertIsNone(t.top_keyword)


# ============================================================================
# 3. Label Management
# ============================================================================

class TestTopicLabelManagement(unittest.TestCase):

    def test_assign_label_sets_label(self):
        """assign_label() → has_label = True"""
        t = TopicData(cluster_id=1, top_keywords=["python"], coherence=0.7)
        t.assign_label("Programming")
        self.assertTrue(t.has_label)
        self.assertEqual(t.label, "Programming")

    def test_assign_empty_label_raises(self):
        """assign_label('') → ValueError"""
        t = TopicData(cluster_id=1, top_keywords=[], coherence=0.5)
        with self.assertRaises(ValueError):
            t.assign_label("")
        with self.assertRaises(ValueError):
            t.assign_label("   ")

    def test_clear_label_resets_to_none(self):
        """clear_label() → label = None, has_label = False"""
        t = TopicData(cluster_id=1, top_keywords=["ai"], coherence=0.6, label="Tech")
        t.clear_label()
        self.assertIsNone(t.label)
        self.assertFalse(t.has_label)


# ============================================================================
# 4. Similarity
# ============================================================================

class TestTopicSimilarity(unittest.TestCase):

    def test_cosine_similarity_identical_embeddings(self):
        """embedding เหมือนกันทุกมิติ → cosine = 1.0"""
        emb = [1.0, 0.5, 0.8]
        a = TopicData(cluster_id=1, top_keywords=[], coherence=0.7, embedding=emb)
        b = TopicData(cluster_id=2, top_keywords=[], coherence=0.7, embedding=emb)
        self.assertAlmostEqual(a.cosine_similarity(b), 1.0)

    def test_cosine_similarity_orthogonal_embeddings(self):
        """embedding ตั้งฉากกัน → cosine = 0.0"""
        a = TopicData(cluster_id=1, top_keywords=[], coherence=0.7, embedding=[1.0, 0.0])
        b = TopicData(cluster_id=2, top_keywords=[], coherence=0.7, embedding=[0.0, 1.0])
        self.assertAlmostEqual(a.cosine_similarity(b), 0.0)

    def test_cosine_similarity_returns_none_without_embedding(self):
        """ไม่มี embedding → cosine_similarity = None"""
        a = TopicData(cluster_id=1, top_keywords=[], coherence=0.5)
        b = TopicData(cluster_id=2, top_keywords=[], coherence=0.5, embedding=[1.0])
        self.assertIsNone(a.cosine_similarity(b))
        self.assertIsNone(b.cosine_similarity(a))

    def test_cosine_similarity_raises_on_dimension_mismatch(self):
        """dimension ต่างกัน → ValueError"""
        a = TopicData(cluster_id=1, top_keywords=[], coherence=0.5, embedding=[1.0, 0.0])
        b = TopicData(cluster_id=2, top_keywords=[], coherence=0.5, embedding=[1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            a.cosine_similarity(b)

    def test_keyword_overlap_jaccard(self):
        """keyword_overlap = |intersection| / |union|"""
        a = TopicData(cluster_id=1, top_keywords=["ai", "ml", "deep"], coherence=0.7)
        b = TopicData(cluster_id=2, top_keywords=["ai", "ml", "nlp"], coherence=0.7)
        # intersection = {ai, ml} = 2, union = {ai, ml, deep, nlp} = 4 → 0.5
        self.assertAlmostEqual(a.keyword_overlap(b), 0.5)


# ============================================================================
# 5. Serialization
# ============================================================================

class TestTopicSerialization(unittest.TestCase):

    def _sample(self) -> TopicData:
        return TopicData(
            cluster_id     = 42,
            top_keywords   = ["neural", "network", "deep"],
            coherence      = 0.82,
            label          = "Deep Learning",
            embedding      = [0.1, 0.5, 0.9],
            document_count = 150,
        )

    def test_to_dict_from_dict_roundtrip(self):
        """to_dict → from_dict → ค่าครบทุก field"""
        original = self._sample()
        restored = TopicData.from_dict(original.to_dict())
        self.assertEqual(restored.cluster_id,     original.cluster_id)
        self.assertEqual(restored.top_keywords,   original.top_keywords)
        self.assertAlmostEqual(restored.coherence, original.coherence)
        self.assertEqual(restored.label,          original.label)
        self.assertEqual(restored.embedding,      original.embedding)
        self.assertEqual(restored.document_count, original.document_count)

    def test_to_json_from_json_roundtrip(self):
        """to_json → from_json → ค่าครบ"""
        original = self._sample()
        restored = TopicData.from_json(original.to_json())
        self.assertEqual(restored.cluster_id,   original.cluster_id)
        self.assertEqual(restored.top_keywords, original.top_keywords)

    def test_serialization_without_label_and_embedding(self):
        """label=None, embedding=None → serialize/deserialize ได้"""
        t = TopicData(cluster_id=5, top_keywords=["word"], coherence=0.4)
        restored = TopicData.from_dict(t.to_dict())
        self.assertIsNone(restored.label)
        self.assertIsNone(restored.embedding)

    def test_to_json_is_valid_json(self):
        """to_json() parse ได้ด้วย json.loads"""
        parsed = json.loads(self._sample().to_json())
        self.assertIn("cluster_id",     parsed)
        self.assertIn("top_keywords",   parsed)
        self.assertIn("coherence",      parsed)


# ============================================================================
# 6. Factory
# ============================================================================

class TestTopicFactory(unittest.TestCase):

    def test_create_topic_maps_all_args(self):
        """create_topic() ส่ง args ครบ → TopicData ถูกต้อง"""
        t = create_topic(
            cluster_id=7,
            top_keywords=["python", "code"],
            coherence=0.75,
            label="Programming",
            embedding=[0.1, 0.2],
            document_count=30,
        )
        self.assertEqual(t.cluster_id, 7)
        self.assertEqual(t.label, "Programming")
        self.assertEqual(t.document_count, 30)


# ============================================================================
# RUNNER
# ============================================================================

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1. Construction & Normalization  (5 tests)", TestTopicConstruction),
        ("2. Derived Properties            (4 tests)", TestTopicDerivedProperties),
        ("3. Label Management              (3 tests)", TestTopicLabelManagement),
        ("4. Similarity                    (5 tests)", TestTopicSimilarity),
        ("5. Serialization                 (4 tests)", TestTopicSerialization),
        ("6. Factory                       (1 test)",  TestTopicFactory),
    ]

    print("\n=================================================================")
    print("  TopicData Test Suite  (Emergent Cluster Model)")
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