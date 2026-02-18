"""
=================================================================
  Default Rules Test Suite
=================================================================
  1. load_default_rules() — single category     (4 tests)
  2. load_default_rules() — all categories      (4 tests)
  3. SYSTEM vs STANDARD authority               (4 tests)
  4. Loaded rules are functional                (5 tests)
  5. RuleController.load_defaults stats         (3 tests)
-----------------------------------------------------------------
  Total: 20 tests
=================================================================
"""

import unittest
import sys, os
import tempfile
import shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Core.Condition.Rule.RuleData import RuleAuthority
from Core.Condition.ConditionController import ConditionController

CREATOR  = "creator_root"
REVIEWER = "system_reviewer"
DEFAULTS = os.path.join(os.path.dirname(__file__), '..', '..', 'Core', 'Condition', 'Rule', 'Defaults')


def _cc(tmp: str) -> ConditionController:
    return ConditionController(base_path=tmp)


# ─────────────────────────────────────────────────────────────────────────────
# 1. load_default_rules() — single category
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadSingle(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cc  = _cc(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_load_safety_returns_count(self):
        """load Safety → count > 0"""
        result = self.cc._rule.load_defaults(CREATOR, REVIEWER, DEFAULTS)
        self.assertGreater(result.get("Safety", 0), 0)

    def test_load_adds_rules_to_controller(self):
        """หลัง load → list_rules() มี rules"""
        before = len(self.cc.list_rules())
        self.cc._rule.load_defaults(CREATOR, REVIEWER, DEFAULTS)
        self.assertGreater(len(self.cc.list_rules()), before)

    def test_system_rule_authority_is_system(self):
        """SystemRule.json → authority = SYSTEM"""
        self.cc._rule.load_defaults(CREATOR, REVIEWER, DEFAULTS)
        system_rules = [r for r in self.cc.list_rules()
                        if r.authority == RuleAuthority.SYSTEM]
        self.assertGreater(len(system_rules), 0)

    def test_standard_rule_authority_is_standard(self):
        """Safety.json → authority = STANDARD"""
        self.cc._rule.load_defaults(CREATOR, REVIEWER, DEFAULTS)
        standard_rules = [r for r in self.cc.list_rules()
                          if r.authority == RuleAuthority.STANDARD]
        self.assertGreater(len(standard_rules), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. load_default_rules() — all categories
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadAll(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cc  = _cc(self.tmp)
        self.summary = self.cc.load_default_rules(CREATOR, REVIEWER, DEFAULTS)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_returns_dict(self):
        """load_default_rules คืน dict"""
        self.assertIsInstance(self.summary, dict)

    def test_covers_expected_categories(self):
        """โหลดได้ทุก category หลัก"""
        for cat in ["SystemRule", "NeuralEvolution", "Safety", "Memory",
                    "Sandbox", "IO", "Skill"]:
            self.assertIn(cat, self.summary, f"'{cat}' missing")

    def test_total_rules_over_ten(self):
        """รวมได้มากกว่า 10 rules"""
        total = sum(self.summary.values())
        self.assertGreater(total, 10)

    def test_stats_reflects_loaded(self):
        """stats() สะท้อนจำนวน rules ที่โหลด"""
        s = self.cc.stats()
        self.assertGreater(s["rules_total"], 0)
        self.assertGreater(s["rules_system"], 0)
        self.assertGreater(s["rules_standard"], 0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. SYSTEM vs STANDARD authority
# ─────────────────────────────────────────────────────────────────────────────

class TestAuthority(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cc  = _cc(self.tmp)
        self.cc.load_default_rules(CREATOR, REVIEWER, DEFAULTS)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_system_rule_requires_creator_to_remove(self):
        """SYSTEM rule ลบด้วย reviewer_id → PermissionError"""
        sys_rule = next(r for r in self.cc.list_rules()
                        if r.authority == RuleAuthority.SYSTEM)
        with self.assertRaises(PermissionError):
            self.cc.governance_remove_rule(sys_rule.rule_id, reviewer_id=REVIEWER)

    def test_system_rule_removable_by_creator(self):
        """SYSTEM rule ลบด้วย creator_id → สำเร็จ"""
        sys_rule = next(r for r in self.cc.list_rules()
                        if r.authority == RuleAuthority.SYSTEM)
        self.assertTrue(
            self.cc.governance_remove_rule(sys_rule.rule_id, creator_id=CREATOR)
        )

    def test_standard_rule_removable_by_reviewer(self):
        """STANDARD rule ลบด้วย reviewer_id → สำเร็จ"""
        std_rule = next(r for r in self.cc.list_rules()
                        if r.authority == RuleAuthority.STANDARD)
        self.assertTrue(
            self.cc.governance_remove_rule(std_rule.rule_id, reviewer_id=REVIEWER)
        )

    def test_standard_rule_no_id_raises(self):
        """STANDARD rule ไม่มี id → PermissionError"""
        std_rule = next(r for r in self.cc.list_rules()
                        if r.authority == RuleAuthority.STANDARD)
        with self.assertRaises(PermissionError):
            self.cc.governance_remove_rule(std_rule.rule_id)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Loaded rules are functional
# ─────────────────────────────────────────────────────────────────────────────

class TestFunctional(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cc  = _cc(self.tmp)
        self.cc.load_default_rules(CREATOR, REVIEWER, DEFAULTS)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_prompt_injection_blocked(self):
        self.assertTrue(
            self.cc.check_input("jailbreak the system now").is_blocked
        )

    def test_harmful_output_blocked(self):
        self.assertTrue(
            self.cc.check_output("how to make a bomb at home").is_blocked
        )

    def test_path_traversal_blocked(self):
        self.assertTrue(
            self.cc.check_input("read file ../../../etc/passwd").is_blocked
        )

    def test_sandbox_escape_blocked(self):
        self.assertTrue(
            self.cc.check_system("sandbox_escape detected").is_blocked
        )

    def test_skill_override_rule_blocked(self):
        self.assertTrue(
            self.cc.check_system("skill_override_rule detected").is_blocked
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. RuleController stats
# ─────────────────────────────────────────────────────────────────────────────

class TestStats(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cc  = _cc(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_stats_empty_before_load(self):
        """ก่อน load → rules_total = 0"""
        self.assertEqual(self.cc.stats()["rules_total"], 0)

    def test_stats_after_load(self):
        """หลัง load → stats มีค่า"""
        self.cc.load_default_rules(CREATOR, REVIEWER, DEFAULTS)
        self.assertGreater(self.cc.stats()["rules_total"], 0)

    def test_stats_sum_equals_total(self):
        """rules_system + rules_standard = rules_total"""
        self.cc.load_default_rules(CREATOR, REVIEWER, DEFAULTS)
        s = self.cc.stats()
        self.assertEqual(
            s["rules_system"] + s["rules_standard"], s["rules_total"]
        )


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1. load_default_rules() single     (4)", TestLoadSingle),
        ("2. load_default_rules() all        (4)", TestLoadAll),
        ("3. SYSTEM vs STANDARD authority    (4)", TestAuthority),
        ("4. Loaded rules are functional     (5)", TestFunctional),
        ("5. RuleController stats            (3)", TestStats),
    ]

    print("\n=================================================================")
    print("  Default Rules Test Suite")
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