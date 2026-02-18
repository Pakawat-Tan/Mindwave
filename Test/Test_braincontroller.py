"""
=================================================================
  BrainController Test Suite
=================================================================
  1. BrainLog                        (3 tests)
  2. ContractResult                  (3 tests)
  3. BrainController — Init          (4 tests)
  4. BrainController — respond()     (5 tests)
  5. BrainController — Skill Contract(4 tests)
  6. BrainController — Relay         (3 tests)
  7. BrainController — Lock/Unlock   (4 tests)
  8. BrainController — Logs          (3 tests)
  9. BrainController — Status        (3 tests)
-----------------------------------------------------------------
  Total: 32 tests
=================================================================
"""

import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Core.BrainController import BrainController, BrainLog, ContractResult
from Core.Confidence.ConfidenceData import ConfidenceOutcome

REVIEWER = "reviewer_001"


def _brain() -> BrainController:
    return BrainController()


# ─────────────────────────────────────────────────────────────────────────────
# 1. BrainLog
# ─────────────────────────────────────────────────────────────────────────────

class TestBrainLog(unittest.TestCase):

    def test_create_log(self):
        log = BrainLog(input_text="hello", context="general", outcome="commit")
        self.assertEqual(log.outcome, "commit")

    def test_log_is_frozen(self):
        log = BrainLog()
        with self.assertRaises((AttributeError, TypeError)):
            log.outcome = "changed"  # type: ignore

    def test_log_to_dict(self):
        log = BrainLog(input_text="test", context="math")
        d = log.to_dict()
        self.assertIn("log_id", d)
        self.assertIn("outcome", d)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ContractResult
# ─────────────────────────────────────────────────────────────────────────────

class TestContractResult(unittest.TestCase):

    def test_commit_can_respond(self):
        c = ContractResult(final_outcome=ConfidenceOutcome.COMMIT)
        self.assertTrue(c.can_respond)

    def test_conditional_can_respond(self):
        c = ContractResult(final_outcome=ConfidenceOutcome.CONDITIONAL)
        self.assertTrue(c.can_respond)

    def test_silence_cannot_respond(self):
        c = ContractResult(final_outcome=ConfidenceOutcome.SILENCE)
        self.assertFalse(c.can_respond)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Init
# ─────────────────────────────────────────────────────────────────────────────

class TestInit(unittest.TestCase):

    def test_init_creates_brain(self):
        b = _brain()
        self.assertIsNotNone(b)

    def test_personality_initialized(self):
        b = _brain()
        self.assertTrue(b.personality.is_initialized())

    def test_mode_active_on_init(self):
        b = _brain()
        self.assertEqual(b.mode, "active")

    def test_modules_accessible(self):
        b = _brain()
        self.assertIsNotNone(b.condition)
        self.assertIsNotNone(b.confidence)
        self.assertIsNotNone(b.skill)
        self.assertIsNotNone(b.neural)
        self.assertIsNotNone(b.reviewer)


# ─────────────────────────────────────────────────────────────────────────────
# 4. respond()
# ─────────────────────────────────────────────────────────────────────────────

class TestRespond(unittest.TestCase):

    def setUp(self):
        self.b = _brain()

    def test_respond_returns_dict(self):
        r = self.b.respond("hello", "general")
        self.assertIsInstance(r, dict)

    def test_respond_has_required_keys(self):
        r = self.b.respond("hello", "general")
        for key in ["response", "outcome", "confidence", "log_id"]:
            self.assertIn(key, r)

    def test_respond_logs_interaction(self):
        self.b.respond("hello", "general")
        self.assertEqual(len(self.b.logs), 1)

    def test_respond_multiple_times(self):
        for i in range(3):
            self.b.respond(f"input {i}", "general")
        self.assertEqual(len(self.b.logs), 3)

    def test_respond_locked_returns_reject(self):
        self.b.lock(REVIEWER)
        r = self.b.respond("hello", "general")
        self.assertEqual(r["outcome"], "reject")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Skill Contract
# ─────────────────────────────────────────────────────────────────────────────

class TestSkillContract(unittest.TestCase):

    def setUp(self):
        self.b = _brain()

    def test_contract_returns_result(self):
        c = self.b._run_skill_contract("hello", "general", None)
        self.assertIsInstance(c, ContractResult)

    def test_contract_confidence_score_in_range(self):
        c = self.b._run_skill_contract("test", "math", None)
        self.assertGreaterEqual(c.confidence_score, 0.0)
        self.assertLessEqual(c.confidence_score,    1.0)

    def test_contract_personality_set(self):
        c = self.b._run_skill_contract("test", "general", None)
        self.assertNotEqual(c.personality, "")

    def test_contract_has_final_outcome(self):
        c = self.b._run_skill_contract("test", "general", None)
        self.assertIsInstance(c.final_outcome, ConfidenceOutcome)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Relay
# ─────────────────────────────────────────────────────────────────────────────

class TestRelay(unittest.TestCase):

    def test_relay_to_valid_target(self):
        b = _brain()
        result = b.relay("memory", {"data": "test"})
        self.assertEqual(result["data"], "test")

    def test_relay_to_io_raises(self):
        b = _brain()
        with self.assertRaises(PermissionError):
            b.relay("io", "direct message")

    def test_relay_to_IO_uppercase_raises(self):
        b = _brain()
        with self.assertRaises(PermissionError):
            b.relay("IO", "direct message")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Lock/Unlock
# ─────────────────────────────────────────────────────────────────────────────

class TestLockUnlock(unittest.TestCase):

    def test_lock_changes_mode(self):
        b = _brain()
        b.lock(REVIEWER)
        self.assertEqual(b.mode, "locked")

    def test_unlock_restores_active(self):
        b = _brain()
        b.lock(REVIEWER)
        b.unlock(REVIEWER)
        self.assertEqual(b.mode, "active")

    def test_lock_without_reviewer_raises(self):
        b = _brain()
        with self.assertRaises(PermissionError):
            b.lock("")

    def test_unlock_without_reviewer_raises(self):
        b = _brain()
        b.lock(REVIEWER)
        with self.assertRaises(PermissionError):
            b.unlock("")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Logs
# ─────────────────────────────────────────────────────────────────────────────

class TestLogs(unittest.TestCase):

    def test_last_log_after_respond(self):
        b = _brain()
        b.respond("hello", "general")
        self.assertIsNotNone(b.last_log())

    def test_last_log_is_brain_log(self):
        b = _brain()
        b.respond("hello", "general")
        self.assertIsInstance(b.last_log(), BrainLog)

    def test_logs_accumulate(self):
        b = _brain()
        b.respond("a", "general")
        b.respond("b", "general")
        b.respond("c", "general")
        self.assertEqual(len(b.logs), 3)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Status
# ─────────────────────────────────────────────────────────────────────────────

class TestStatus(unittest.TestCase):

    def test_status_returns_dict(self):
        b = _brain()
        s = b.status()
        self.assertIsInstance(s, dict)

    def test_status_has_modules(self):
        b = _brain()
        s = b.status()
        self.assertIn("modules", s)
        self.assertTrue(s["modules"]["condition"])

    def test_status_reflects_mode(self):
        b = _brain()
        b.lock(REVIEWER)
        self.assertEqual(b.status()["mode"], "locked")


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1. BrainLog                    (3)", TestBrainLog),
        ("2. ContractResult              (3)", TestContractResult),
        ("3. BrainController — Init      (4)", TestInit),
        ("4. BrainController — respond() (5)", TestRespond),
        ("5. BrainController — Contract  (4)", TestSkillContract),
        ("6. BrainController — Relay     (3)", TestRelay),
        ("7. BrainController — Lock      (4)", TestLockUnlock),
        ("8. BrainController — Logs      (3)", TestLogs),
        ("9. BrainController — Status    (3)", TestStatus),
    ]

    print("\n=================================================================")
    print("  BrainController Test Suite")
    print("=================================================================")
    for label, _ in groups:
        print(f"  {label}")
    print("─────────────────────────────────────────────────────────────────")
    print("  Total: 32 tests")
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