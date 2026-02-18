"""
Test/Memory/test_memory_controller.py

Unit Tests สำหรับ MemoryController — แบบใช้งานจริง
ทดสอบ 5 scenarios:
    1. Write → Read (binary payload)
    2. Auto-promote ตาม importance
    3. Majority Rule → สร้าง Knowlet
    4. Cleanup expired/stale
    5. Long → Immortal ต้องมี reviewer_id

Total: 20 tests
"""

import unittest
import tempfile
import shutil
import json
import struct
import time
from pathlib import Path
from unittest.mock import patch


# ─────────────────────────────────────────────────────
# Helper: สร้าง Binary payload จำลองจากโมเดล
# ─────────────────────────────────────────────────────

def make_payload(input_text: str, learning: str, output_text: str) -> bytes:
    """
    จำลอง binary payload ที่โมเดลสร้าง
    Format: [input_len(4)][input][learning_len(4)][learning][output_len(4)][output]
    """
    parts = []
    for text in [input_text, learning, output_text]:
        encoded = text.encode("utf-8")
        parts.append(struct.pack(">I", len(encoded)) + encoded)
    return b"".join(parts)


def parse_payload(payload: bytes) -> tuple:
    """แกะ binary payload กลับเป็น (input, learning, output)"""
    results = []
    offset  = 0
    for _ in range(3):
        length  = struct.unpack(">I", payload[offset:offset + 4])[0]
        offset += 4
        text    = payload[offset:offset + length].decode("utf-8")
        offset += length
        results.append(text)
    return tuple(results)


# ─────────────────────────────────────────────────────
# Base Test Class
# ─────────────────────────────────────────────────────

class BaseMemoryTest(unittest.TestCase):

    def setUp(self):
        import sys
        # MindWave#8/  ← project root (ให้ import Core.Memory.X ได้)
        project_root = str(Path(__file__).parent.parent.parent)
        # MindWave#8/Core/Memory/  ← AtomStructure, AtomRepair, Tiers อยู่ที่นี่
        memory_root  = str(Path(__file__).parent.parent.parent / "Core" / "Memory")

        for p in [project_root, memory_root]:
            if p not in sys.path:
                sys.path.insert(0, p)

        self.test_dir = tempfile.mkdtemp()
        self.controller = self._make_controller()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _make_controller(self):
        from Core.Memory.MemoryController import MemoryController
        return MemoryController(base_path=self.test_dir)

    def _atom(self, input_text: str, learning: str, output_text: str):
        from Structure.AtomStructure import AtomData
        return AtomData(payload=make_payload(input_text, learning, output_text))


# ─────────────────────────────────────────────────────
# 1. Write → Read
# ─────────────────────────────────────────────────────

class TestWriteRead(BaseMemoryTest):

    def test_write_and_read_back_payload(self):
        """Write binary payload แล้วอ่านกลับได้ถูกต้อง"""
        atom_id = self.controller.write(
            data=self._atom("Python คืออะไร", "Python เป็นภาษา interpreted", "ตอบได้ถูกต้อง"),
            category="learning", primary="python", importance=0.6,
        )

        self.assertIsNotNone(atom_id)
        result = self.controller.read(atom_id)
        self.assertIsNotNone(result)

        inp, learn, out = parse_payload(result.payload)
        self.assertEqual(inp,   "Python คืออะไร")
        self.assertEqual(learn, "Python เป็นภาษา interpreted")
        self.assertEqual(out,   "ตอบได้ถูกต้อง")

    def test_metadata_injected(self):
        """MemoryController inject metadata category/primary/importance ถูกต้อง"""
        atom_id = self.controller.write(
            data=self._atom("in", "learn", "out"),
            category="fact", primary="gravity", importance=0.75,
        )

        meta = json.loads(self.controller.read(atom_id).metadata.decode("utf-8"))
        self.assertEqual(meta["category"],   "fact")
        self.assertEqual(meta["primary"],    "gravity")
        self.assertEqual(meta["importance"], 0.75)

    def test_importance_routes_to_short(self):
        """importance 0.3–0.49 → short tier"""
        atom_id = self.controller.write(
            data=self._atom("in", "learn", "out"),
            category="conversation", primary="hi", importance=0.4,
        )
        self.assertTrue(self.controller.exists(atom_id, tier="short"))
        self.assertFalse(self.controller.exists(atom_id, tier="middle"))

    def test_importance_routes_to_middle(self):
        """importance 0.5–0.69 → middle tier"""
        atom_id = self.controller.write(
            data=self._atom("in", "learn", "out"),
            category="learning", primary="python", importance=0.6,
        )
        self.assertTrue(self.controller.exists(atom_id, tier="middle"))

    def test_importance_routes_to_long(self):
        """importance 0.7–0.94 → long tier"""
        atom_id = self.controller.write(
            data=self._atom("in", "learn", "out"),
            category="fact", primary="math", importance=0.8,
        )
        self.assertTrue(self.controller.exists(atom_id, tier="long"))

    def test_importance_routes_to_immortal(self):
        """importance >= 0.95 → immortal tier"""
        atom_id = self.controller.write(
            data=self._atom("in", "learn", "out"),
            category="instruction", primary="identity", importance=0.97,
        )
        self.assertTrue(self.controller.exists(atom_id, tier="immortal"))

    def test_low_importance_not_stored(self):
        """importance < 0.3 → ไม่เก็บ คืน None"""
        atom_id = self.controller.write(
            data=self._atom("in", "learn", "out"),
            category="conversation", primary="test", importance=0.1,
        )
        self.assertIsNone(atom_id)

    def test_read_without_tier_finds_atom(self):
        """read โดยไม่ระบุ tier → ค้นหาทุก tier แล้วเจอ"""
        atom_id = self.controller.write(
            data=self._atom("in", "learn", "out"),
            category="fact", primary="math", importance=0.8,
        )
        result = self.controller.read(atom_id)
        self.assertIsNotNone(result)


# ─────────────────────────────────────────────────────
# 2. Auto-promote ตาม importance
# ─────────────────────────────────────────────────────

class TestAutoPromote(BaseMemoryTest):

    def _update_importance(self, atom_id: str, tier: str, new_importance: float):
        """Helper: อัปเดต importance ใน metadata"""
        from AtomStructure import AtomData
        atom = self.controller.read(atom_id, tier=tier)
        meta = json.loads(atom.metadata.decode("utf-8"))
        meta["importance"] = new_importance
        updated = AtomData(
            payload=atom.payload,
            metadata=json.dumps(meta).encode("utf-8"),
            source=atom.source,
        )
        self.controller._get_tier(tier).write(atom_id, updated)

    def test_short_to_middle_auto_promote(self):
        """Short Atom ที่ importance >= 0.5 → auto_promote ขึ้น Middle"""
        atom_id = self.controller.write(
            data=self._atom("in", "learn", "out"),
            category="learning", primary="python", importance=0.4,
        )
        self._update_importance(atom_id, "short", 0.6)

        summary = self.controller.auto_promote()
        self.assertGreaterEqual(summary["short_to_middle"], 1)
        self.assertTrue(self.controller.exists(atom_id, tier="middle"))
        self.assertFalse(self.controller.exists(atom_id, tier="short"))

    def test_middle_to_long_auto_promote(self):
        """Middle Atom ที่ importance >= 0.7 → auto_promote ขึ้น Long"""
        atom_id = self.controller.write(
            data=self._atom("in", "learn", "out"),
            category="fact", primary="physics", importance=0.6,
        )
        self._update_importance(atom_id, "middle", 0.75)

        summary = self.controller.auto_promote()
        self.assertGreaterEqual(summary["middle_to_long"], 1)
        self.assertTrue(self.controller.exists(atom_id, tier="long"))

    def test_stats_after_promote(self):
        """stats() สะท้อนจำนวน Atom ที่ถูกต้องหลัง promote"""
        self.controller.write(
            data=self._atom("in", "learn", "out"),
            category="learning", primary="python", importance=0.4,
        )
        stats = self.controller.stats()
        self.assertEqual(stats["short"],  1)
        self.assertEqual(stats["middle"], 0)


# ─────────────────────────────────────────────────────
# 3. Majority Rule → สร้าง Knowlet
# ─────────────────────────────────────────────────────

class TestMajorityRule(BaseMemoryTest):

    def _write_many(self, category: str, primary: str, n: int, importance: float = 0.6):
        for i in range(n):
            self.controller.write(
                data=self._atom(f"in {i}", f"learn {i}", f"out {i}"),
                category=category, primary=primary, importance=importance,
            )

    def test_majority_passes_creates_knowlet(self):
        """Atom > 50% context เดียวกัน → try_create คืน KnowletData"""
        self._write_many("learning", "python", 6)      # 6/8 = 75%
        self._write_many("learning", "javascript", 2)

        knowlet = self.controller._knowlet.try_create(
            tier="middle", category="learning", primary="python",
            summary="โมเดลมีความสามารถด้าน Python ในระดับสูง",
            confidence=0.85,
        )

        self.assertIsNotNone(knowlet)
        self.assertEqual(knowlet.category, "learning")
        self.assertGreater(knowlet.confidence, knowlet.parent_confidence)

    def test_majority_fails_returns_none(self):
        """Atom <= 50% context เดียวกัน → try_create คืน None"""
        self._write_many("learning", "python", 3)      # 3/7 = 43%
        self._write_many("learning", "javascript", 4)

        knowlet = self.controller._knowlet.try_create(
            tier="middle", category="learning", primary="python",
            summary="ยังไม่ถึง majority", confidence=0.85,
        )
        self.assertIsNone(knowlet)

    def test_knowlet_confidence_must_exceed_parent(self):
        """Knowlet confidence <= parent → ValueError"""
        from Core.Memory.Structure.KnowletStructure import KnowletData
        with self.assertRaises(ValueError):
            KnowletData.create(
                parent_ids=["abc"],
                category="learning", primary="python",
                summary="test",
                confidence=0.5,
                parent_confidence=0.8,   # confidence < parent → error
            )

    def test_promote_knowlet_without_reviewer_raises(self):
        """promote Knowlet โดยไม่มี reviewer_id → PermissionError"""
        self._write_many("learning", "python", 6)
        self._write_many("learning", "other", 1)

        knowlet = self.controller._knowlet.try_create(
            tier="middle", category="learning", primary="python",
            summary="Python mastery", confidence=0.9,
        )
        self.assertIsNotNone(knowlet)

        with self.assertRaises(PermissionError):
            knowlet.promote(reviewer_id="")

    def test_promote_knowlet_with_reviewer_succeeds(self):
        """promote Knowlet ที่มี reviewer_id → is_promoted = True"""
        self._write_many("learning", "python", 6)
        self._write_many("learning", "other", 1)

        knowlet = self.controller._knowlet.try_create(
            tier="middle", category="learning", primary="python",
            summary="Python mastery", confidence=0.9,
        )

        promoted = self.controller._knowlet.promote(
            knowlet_id=knowlet.knowlet_id,
            category="learning", primary="python",
            reviewer_id="reviewer_001",
        )

        self.assertTrue(promoted.is_promoted)
        self.assertEqual(promoted.reviewer_id, "reviewer_001")


# ─────────────────────────────────────────────────────
# 4. Cleanup expired/stale
# ─────────────────────────────────────────────────────

class TestCleanup(BaseMemoryTest):

    def test_cleanup_removes_expired_middle(self):
        """Middle Atom ที่ expired → ถูกลบโดย cleanup"""
        atom_id = self.controller.write(
            data=self._atom("in", "learn", "out"),
            category="learning", primary="test", importance=0.6,
        )

        with patch.object(self.controller._middle.__class__,
                          "list_expired", return_value=[atom_id]):
            summary = self.controller.cleanup()

        self.assertGreaterEqual(summary["middle"], 1)
        self.assertFalse(self.controller.exists(atom_id, tier="middle"))

    def test_cleanup_preserves_promotable_short(self):
        """Short Atom ที่ stale แต่ promotable → ไม่ถูกลบ"""
        from AtomStructure import AtomData

        atom_id = self.controller.write(
            data=self._atom("in", "learn", "out"),
            category="learning", primary="test", importance=0.4,
        )

        # update importance ให้ promotable
        atom = self.controller.read(atom_id, tier="short")
        meta = json.loads(atom.metadata.decode("utf-8"))
        meta["importance"] = 0.6
        self.controller._short.write(atom_id, AtomData(
            payload=atom.payload,
            metadata=json.dumps(meta).encode("utf-8"),
        ))

        with patch.object(self.controller._short.__class__,
                          "list_stale", return_value=[atom_id]):
            self.controller.cleanup()

        self.assertTrue(self.controller.exists(atom_id, tier="short"))

    def test_clear_session(self):
        """clear_session() ล้าง Short-term ทั้งหมด"""
        for i in range(5):
            self.controller.write(
                data=self._atom(f"in {i}", f"learn {i}", f"out {i}"),
                category="conversation", primary=f"topic_{i}", importance=0.4,
            )

        self.assertEqual(self.controller.stats()["short"], 5)
        count = self.controller.clear_session()
        self.assertEqual(count, 5)
        self.assertEqual(self.controller.stats()["short"], 0)


# ─────────────────────────────────────────────────────
# 5. Long → Immortal ต้องมี reviewer_id
# ─────────────────────────────────────────────────────

class TestImmortalPromotion(BaseMemoryTest):

    def _long_atom(self) -> str:
        return self.controller.write(
            data=self._atom("identity input", "core learning", "identity output"),
            category="instruction", primary="identity", importance=0.8,
        )

    def test_long_to_immortal_without_reviewer_raises(self):
        """Long → Immortal โดยไม่มี reviewer_id → PermissionError"""
        atom_id = self._long_atom()
        with self.assertRaises(PermissionError):
            self.controller.promote(atom_id, from_tier="long", reviewer_id=None)

    def test_long_to_immortal_with_reviewer_succeeds(self):
        """Long → Immortal ที่มี reviewer_id → สำเร็จ"""
        atom_id = self._long_atom()
        success = self.controller.promote(
            atom_id, from_tier="long", reviewer_id="reviewer_001"
        )
        self.assertTrue(success)
        self.assertTrue(self.controller.exists(atom_id, tier="immortal"))
        self.assertFalse(self.controller.exists(atom_id, tier="long"))

    def test_immortal_delete_raises(self):
        """Immortal Atom ลบไม่ได้เด็ดขาด → PermissionError"""
        atom_id = self._long_atom()
        self.controller.promote(atom_id, from_tier="long", reviewer_id="reviewer_001")
        with self.assertRaises(PermissionError):
            self.controller._immortal.delete(atom_id)

    def test_immortal_clear_raises(self):
        """Immortal Tier ล้างไม่ได้ → PermissionError"""
        with self.assertRaises(PermissionError):
            self.controller._immortal.clear()

    def test_short_middle_long_no_reviewer_needed(self):
        """Short → Middle → Long ไม่ต้องมี reviewer_id"""
        atom_id = self.controller.write(
            data=self._atom("in", "learn", "out"),
            category="learning", primary="python", importance=0.4,
        )

        self.assertTrue(self.controller.promote(atom_id, from_tier="short"))
        self.assertTrue(self.controller.exists(atom_id, tier="middle"))

        self.assertTrue(self.controller.promote(atom_id, from_tier="middle"))
        self.assertTrue(self.controller.exists(atom_id, tier="long"))


# ─────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────

def create_test_suite():
    suite = unittest.TestSuite()
    for cls in [TestWriteRead, TestAutoPromote, TestMajorityRule,
                TestCleanup, TestImmortalPromotion]:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))
    return suite


if __name__ == "__main__":
    print("\n" + "=" * 65)
    print("  MemoryController Test Suite")
    print("=" * 65)
    print("  1. Write → Read           (8 tests)")
    print("  2. Auto-promote           (3 tests)")
    print("  3. Majority Rule → Knowlet (5 tests)")
    print("  4. Cleanup                (3 tests)")
    print("  5. Long → Immortal        (5 tests)")
    print(f"{'─' * 65}")
    print("  Total: 24 tests")
    print("=" * 65 + "\n")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(create_test_suite())

    print("\n" + "=" * 65)
    print(f"  Passed : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Failed : {len(result.failures)}")
    print(f"  Errors : {len(result.errors)}")
    print("=" * 65)
    print("\n  🎉 ALL TESTS PASSED!\n" if result.wasSuccessful() else "\n  ⚠️  SOME TESTS FAILED\n")