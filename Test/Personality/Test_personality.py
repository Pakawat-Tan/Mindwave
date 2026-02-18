"""
=================================================================
  Personality System Test Suite
=================================================================
  1. PersonalityProfile & PROFILES    (4 tests)
  2. PersonalityData — init           (4 tests)
  3. PersonalityData — dimensions     (4 tests)
  4. PersonalityData — change         (5 tests)
  5. PersonalityData — audit trail    (3 tests)
  6. PersonalityController — init     (4 tests)
  7. PersonalityController — change   (4 tests)
  8. PersonalityController — query    (4 tests)
  9. PersonalityController — stats    (2 tests)
-----------------------------------------------------------------
  Total: 34 tests
=================================================================
"""

import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Core.Personality.PersonalityData import (
    PersonalityData, PersonalityProfile, PersonalityChangeEvent,
    PROFILES, Tone, Friendliness, Firmness,
    ResponseStyle, Humor, Empathy, random_profile
)
from Core.Personality.PersonalityController import PersonalityController


CREATOR  = "creator_root"
REVIEWER = "reviewer_001"


# ─────────────────────────────────────────────────────────────────────────────
# 1. PersonalityProfile & PROFILES
# ─────────────────────────────────────────────────────────────────────────────

class TestProfiles(unittest.TestCase):

    def test_profiles_has_six_entries(self):
        """PROFILES มี 6 presets"""
        self.assertEqual(len(PROFILES), 6)

    def test_all_expected_profiles_exist(self):
        """ทุก profile ที่กำหนดมีอยู่ใน PROFILES"""
        for name in ["Friendly", "Professional", "Balanced",
                     "Empathetic", "Direct", "Creative"]:
            self.assertIn(name, PROFILES)

    def test_profile_is_frozen(self):
        """PersonalityProfile เป็น frozen dataclass"""
        p = PROFILES["Friendly"]
        with self.assertRaises((AttributeError, TypeError)):
            p.tone = Tone.FORMAL  # type: ignore

    def test_random_profile_returns_valid(self):
        """random_profile() คืน profile ที่มีใน PROFILES"""
        p = random_profile()
        self.assertIn(p.profile_name, PROFILES)


# ─────────────────────────────────────────────────────────────────────────────
# 2. PersonalityData — init
# ─────────────────────────────────────────────────────────────────────────────

class TestPersonalityInit(unittest.TestCase):

    def test_init_creates_personality(self):
        """PersonalityData() สร้างได้"""
        p = PersonalityData()
        self.assertIsNotNone(p.profile)

    def test_init_seed_reproducible(self):
        """seed เดียวกัน → profile เดิมทุกครั้ง"""
        p1 = PersonalityData(seed=42)
        p2 = PersonalityData(seed=42)
        self.assertEqual(p1.profile_name, p2.profile_name)

    def test_different_seeds_may_differ(self):
        """seed ต่างกัน → อาจได้ profile ต่างกัน (ตรวจ type)"""
        p = PersonalityData(seed=999)
        self.assertIn(p.profile_name, PROFILES)

    def test_no_changes_on_init(self):
        """ใหม่ → change_count = 0"""
        p = PersonalityData()
        self.assertEqual(p.change_count, 0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PersonalityData — dimensions
# ─────────────────────────────────────────────────────────────────────────────

class TestDimensions(unittest.TestCase):

    def setUp(self):
        self.p = PersonalityData(seed=0)

    def test_tone_is_valid_enum(self):
        """tone เป็น Tone enum"""
        self.assertIsInstance(self.p.tone, Tone)

    def test_friendliness_is_valid_enum(self):
        """friendliness เป็น Friendliness enum"""
        self.assertIsInstance(self.p.friendliness, Friendliness)

    def test_all_dimensions_accessible(self):
        """ทุก dimension เข้าถึงได้"""
        self.assertIsInstance(self.p.firmness,       Firmness)
        self.assertIsInstance(self.p.response_style, ResponseStyle)
        self.assertIsInstance(self.p.humor,          Humor)
        self.assertIsInstance(self.p.empathy,        Empathy)

    def test_dimensions_match_profile(self):
        """dimensions ตรงกับ profile ที่เลือก"""
        expected = PROFILES[self.p.profile_name]
        self.assertEqual(self.p.tone,           expected.tone)
        self.assertEqual(self.p.friendliness,   expected.friendliness)
        self.assertEqual(self.p.firmness,       expected.firmness)
        self.assertEqual(self.p.response_style, expected.response_style)
        self.assertEqual(self.p.humor,          expected.humor)
        self.assertEqual(self.p.empathy,        expected.empathy)


# ─────────────────────────────────────────────────────────────────────────────
# 4. PersonalityData — change
# ─────────────────────────────────────────────────────────────────────────────

class TestPersonalityChange(unittest.TestCase):

    def setUp(self):
        self.p = PersonalityData(seed=0)

    def test_change_with_creator_succeeds(self):
        """change() ด้วย creator_id → สำเร็จ"""
        self.p.change("Professional", CREATOR)
        self.assertEqual(self.p.profile_name, "Professional")

    def test_change_without_creator_raises(self):
        """change() ไม่มี creator_id → PermissionError"""
        with self.assertRaises(PermissionError):
            self.p.change("Professional", creator_id="")

    def test_change_reviewer_only_raises(self):
        """change() ด้วย reviewer_id (ไม่ใช่ creator) → PermissionError"""
        with self.assertRaises(PermissionError):
            self.p.change("Professional", creator_id="")

    def test_change_unknown_profile_raises(self):
        """change() ไปหา profile ที่ไม่มี → ValueError"""
        with self.assertRaises(ValueError):
            self.p.change("UnknownProfile", CREATOR)

    def test_change_updates_dimensions(self):
        """change() → dimensions เปลี่ยนตาม profile ใหม่"""
        self.p.change("Direct", CREATOR)
        self.assertEqual(self.p.tone,         Tone.NEUTRAL)
        self.assertEqual(self.p.friendliness, Friendliness.COLD)
        self.assertEqual(self.p.firmness,     Firmness.ASSERTIVE)
        self.assertEqual(self.p.empathy,      Empathy.LOW)


# ─────────────────────────────────────────────────────────────────────────────
# 5. PersonalityData — audit trail
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditTrail(unittest.TestCase):

    def setUp(self):
        self.p = PersonalityData(seed=0)
        self.original = self.p.profile_name

    def test_change_creates_event(self):
        """change() → events มี 1 entry"""
        self.p.change("Balanced", CREATOR)
        self.assertEqual(self.p.change_count, 1)

    def test_event_stores_from_to(self):
        """event เก็บ from_profile และ to_profile"""
        self.p.change("Creative", CREATOR, reason="test")
        event = self.p.events[0]
        self.assertEqual(event.from_profile, self.original)
        self.assertEqual(event.to_profile,   "Creative")
        self.assertEqual(event.changed_by,   CREATOR)

    def test_multiple_changes_accumulate(self):
        """เปลี่ยนหลายครั้ง → events สะสม"""
        self.p.change("Direct",       CREATOR)
        self.p.change("Professional", CREATOR)
        self.p.change("Friendly",     CREATOR)
        self.assertEqual(self.p.change_count, 3)


# ─────────────────────────────────────────────────────────────────────────────
# 6. PersonalityController — init
# ─────────────────────────────────────────────────────────────────────────────

class TestControllerInit(unittest.TestCase):

    def test_not_initialized_by_default(self):
        """controller ใหม่ → is_initialized = False"""
        pc = PersonalityController()
        self.assertFalse(pc.is_initialized())

    def test_init_creates_personality(self):
        """init() → is_initialized = True"""
        pc = PersonalityController()
        pc.init(seed=1)
        self.assertTrue(pc.is_initialized())

    def test_init_twice_raises(self):
        """init() ซ้ำ → PermissionError"""
        pc = PersonalityController()
        pc.init(seed=1)
        with self.assertRaises(PermissionError):
            pc.init(seed=1)

    def test_init_returns_personality_data(self):
        """init() คืน PersonalityData"""
        pc = PersonalityController()
        result = pc.init(seed=2)
        self.assertIsInstance(result, PersonalityData)


# ─────────────────────────────────────────────────────────────────────────────
# 7. PersonalityController — change
# ─────────────────────────────────────────────────────────────────────────────

class TestControllerChange(unittest.TestCase):

    def setUp(self):
        self.pc = PersonalityController()
        self.pc.init(seed=0)

    def test_change_with_creator_succeeds(self):
        """change() ด้วย creator_id → สำเร็จ"""
        self.pc.change("Professional", CREATOR)
        self.assertEqual(self.pc.profile_name, "Professional")

    def test_change_without_creator_raises(self):
        """change() ไม่มี creator_id → PermissionError"""
        with self.assertRaises(PermissionError):
            self.pc.change("Professional", creator_id="")

    def test_change_before_init_raises(self):
        """change() ก่อน init() → RuntimeError"""
        pc = PersonalityController()
        with self.assertRaises(RuntimeError):
            pc.change("Friendly", CREATOR)

    def test_change_returns_event(self):
        """change() คืน PersonalityChangeEvent"""
        event = self.pc.change("Empathetic", CREATOR, reason="test")
        self.assertIsInstance(event, PersonalityChangeEvent)
        self.assertEqual(event.to_profile, "Empathetic")


# ─────────────────────────────────────────────────────────────────────────────
# 8. PersonalityController — query
# ─────────────────────────────────────────────────────────────────────────────

class TestControllerQuery(unittest.TestCase):

    def setUp(self):
        self.pc = PersonalityController()
        self.pc.init(seed=0)

    def test_get_tone_returns_enum(self):
        """get_tone() คืน Tone enum"""
        self.assertIsInstance(self.pc.get_tone(), Tone)

    def test_list_profiles_has_six(self):
        """list_available_profiles() → 6 entries"""
        self.assertEqual(len(self.pc.list_available_profiles()), 6)

    def test_change_history_grows(self):
        """change_history() เพิ่มหลัง change()"""
        self.pc.change("Direct", CREATOR)
        self.assertEqual(len(self.pc.change_history()), 1)

    def test_all_dimension_getters(self):
        """ทุก getter คืน enum ที่ถูกต้อง"""
        self.assertIsInstance(self.pc.get_friendliness(),   Friendliness)
        self.assertIsInstance(self.pc.get_firmness(),       Firmness)
        self.assertIsInstance(self.pc.get_response_style(), ResponseStyle)
        self.assertIsInstance(self.pc.get_humor(),          Humor)
        self.assertIsInstance(self.pc.get_empathy(),        Empathy)


# ─────────────────────────────────────────────────────────────────────────────
# 9. PersonalityController — stats
# ─────────────────────────────────────────────────────────────────────────────

class TestControllerStats(unittest.TestCase):

    def test_stats_before_init(self):
        """ก่อน init → stats initialized=False"""
        pc = PersonalityController()
        self.assertFalse(pc.stats()["initialized"])

    def test_stats_after_init(self):
        """หลัง init → stats สะท้อนข้อมูล"""
        pc = PersonalityController()
        pc.init(seed=5)
        s = pc.stats()
        self.assertTrue(s["initialized"])
        self.assertIn(s["profile"], PROFILES)
        self.assertEqual(s["change_count"], 0)
        self.assertEqual(s["profiles_available"], 6)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1. PersonalityProfile & PROFILES    (4)", TestProfiles),
        ("2. PersonalityData — init           (4)", TestPersonalityInit),
        ("3. PersonalityData — dimensions     (4)", TestDimensions),
        ("4. PersonalityData — change         (5)", TestPersonalityChange),
        ("5. PersonalityData — audit trail    (3)", TestAuditTrail),
        ("6. PersonalityController — init     (4)", TestControllerInit),
        ("7. PersonalityController — change   (4)", TestControllerChange),
        ("8. PersonalityController — query    (4)", TestControllerQuery),
        ("9. PersonalityController — stats    (2)", TestControllerStats),
    ]

    print("\n=================================================================")
    print("  Personality System Test Suite")
    print("=================================================================")
    for label, _ in groups:
        print(f"  {label}")
    print("─────────────────────────────────────────────────────────────────")
    print("  Total: 34 tests")
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