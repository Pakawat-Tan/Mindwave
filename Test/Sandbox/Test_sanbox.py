"""
=================================================================
  Sandbox Test Suite
=================================================================
  1. SandboxData                     (5 tests)
  2. SCL                             (8 tests)
  3. SandboxController — Boot        (3 tests)
  4. SandboxController — Safety      (3 tests)
  5. SandboxController — Simulation  (4 tests)
  6. SandboxController — Respond     (4 tests)
  7. SandboxController — Rule Test   (3 tests)
  8. SandboxController — Replay      (3 tests)
  9. SandboxController — Promote     (3 tests)
  10. SandboxController — SCL        (4 tests)
  11. SandboxWorld multi-instance    (3 tests)
-----------------------------------------------------------------
  Total: 43 tests
=================================================================
"""

import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Core.Sandbox.SandboxController import SandboxController, SimulationResult
from Core.Sandbox.SandboxData import (
    SandboxAtom, SandboxStatus, AtomType,
    ExperimentState, SandboxWorld,
)
from Core.Sandbox.SCL import SCL, ConflictRecord
from Core.BrainController import BrainController, BrainLog
from Core.Condition.ConditionController import ConditionController
from Core.Condition.Rule.RuleData import (
    RuleData, RuleAction, RuleScope, MatchType
)

REVIEWER = "reviewer_001"

def _sandbox(scl=None, world=None) -> SandboxController:
    return SandboxController(scl=scl, world=world)

def _block_rule(scope: RuleScope, pattern: str = "blocked") -> RuleData:
    return RuleData(
        scope=scope, action=RuleAction.BLOCK,
        match_type=MatchType.PATTERN, pattern=pattern,
        description=f"test block {scope.value}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. SandboxData
# ─────────────────────────────────────────────────────────────────────────────

class TestSandboxData(unittest.TestCase):

    def test_sandbox_atom_create(self):
        a = SandboxAtom(instance_id="A", context="math", confidence=0.8)
        self.assertEqual(a.context, "math")
        self.assertEqual(a.status, SandboxStatus.ACTIVE)

    def test_atom_is_promotable(self):
        a = SandboxAtom(confidence=0.5)
        self.assertTrue(a.is_promotable)

    def test_atom_not_promotable_low_conf(self):
        a = SandboxAtom(confidence=0.1)
        self.assertFalse(a.is_promotable)

    def test_experiment_state_hashes_instance(self):
        s = ExperimentState.create("instance_A", "test hypo", "good")
        self.assertNotEqual(s.source_instance, "instance_A")
        self.assertEqual(len(s.source_instance), 16)

    def test_sandbox_world_register(self):
        w = SandboxWorld()
        w.register_instance("A")
        w.register_instance("B")
        self.assertEqual(w.instance_count, 2)


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCL
# ─────────────────────────────────────────────────────────────────────────────

class TestSCL(unittest.TestCase):

    def setUp(self):
        self.scl = SCL()
        self.scl.register("instance_A")
        self.scl.register("instance_B")

    def test_register_instances(self):
        self.assertEqual(self.scl.instance_count, 2)

    def test_publish_state(self):
        s = ExperimentState.create("instance_A", "hypothesis A", "good")
        result = self.scl.publish("instance_A", s)
        self.assertTrue(result)

    def test_read_excludes_own(self):
        s = ExperimentState.create("instance_A", "hypo A", "ok")
        self.scl.publish("instance_A", s)
        states = self.scl.read_for("instance_A")
        self.assertEqual(len(states), 0)  # ไม่เห็น state ของตัวเอง

    def test_read_sees_others(self):
        s = ExperimentState.create("instance_A", "hypo A", "ok")
        self.scl.publish("instance_A", s)
        states = self.scl.read_for("instance_B")
        self.assertEqual(len(states), 1)

    def test_forbidden_tags_raises(self):
        s = ExperimentState.create("instance_A", "hypo", "ok", tags=["identity"])
        with self.assertRaises(ValueError):
            self.scl.publish("instance_A", s)

    def test_unregistered_publish_raises(self):
        s = ExperimentState.create("unknown", "hypo", "ok")
        with self.assertRaises(PermissionError):
            self.scl.publish("unknown", s)

    def test_conflict_detected(self):
        s1 = ExperimentState.create("instance_A", "same hypo", "good",
                                     confidence_delta=0.5)
        s2 = ExperimentState.create("instance_B", "same hypo", "bad",
                                     confidence_delta=-0.5)
        self.scl.publish("instance_A", s1)
        self.scl.publish("instance_B", s2)
        self.assertGreater(len(self.scl._conflicts), 0)

    def test_stats(self):
        s = self.scl.stats()
        self.assertIn("registered", s)
        self.assertEqual(s["registered"], 2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Boot
# ─────────────────────────────────────────────────────────────────────────────

class TestBoot(unittest.TestCase):

    def test_sandbox_creates(self):
        s = _sandbox()
        self.assertIsNotNone(s)

    def test_sandbox_is_active(self):
        s = _sandbox()
        self.assertTrue(s.is_active)

    def test_sandbox_has_instance_id(self):
        s = _sandbox()
        self.assertNotEqual(s.instance_id, "")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Safety Check
# ─────────────────────────────────────────────────────────────────────────────

class TestSafety(unittest.TestCase):

    def test_safe_input(self):
        s = _sandbox()
        safe, reason = s.is_safe("hello world")
        self.assertTrue(safe)

    def test_blocked_input(self):
        condition = ConditionController()
        brain = BrainController(condition=condition)
        sb = SandboxController(brain=brain)
        condition.governance_add_rule(
            _block_rule(RuleScope.INPUT, "danger"), reviewer_id="system"
        )
        safe, reason = sb.is_safe("danger word here")
        self.assertFalse(safe)

    def test_safety_reason_provided(self):
        condition = ConditionController()
        brain = BrainController(condition=condition)
        sb = SandboxController(brain=brain)
        condition.governance_add_rule(
            _block_rule(RuleScope.INPUT, "forbidden"), reviewer_id="system"
        )
        safe, reason = sb.is_safe("forbidden text")
        self.assertFalse(safe)
        self.assertNotEqual(reason, "")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Simulation
# ─────────────────────────────────────────────────────────────────────────────

class TestSimulation(unittest.TestCase):

    def setUp(self):
        self.sb = _sandbox()

    def test_simulate_returns_result(self):
        r = self.sb.simulate("hello", "general")
        self.assertIsInstance(r, SimulationResult)

    def test_simulate_not_saved_to_atoms(self):
        before = len(self.sb.atoms())
        self.sb.simulate("hello", "general")
        self.assertEqual(len(self.sb.atoms()), before)  # ไม่บันทึก atom

    def test_simulate_has_outcome(self):
        r = self.sb.simulate("hello", "general")
        self.assertIn(r.outcome, ["commit", "conditional", "ask", "silence", "reject"])

    def test_simulate_safety_check(self):
        condition = ConditionController()
        brain = BrainController(condition=condition)
        sb = SandboxController(brain=brain)
        condition.governance_add_rule(
            _block_rule(RuleScope.INPUT, "unsafe"), reviewer_id="system"
        )
        r = sb.simulate("unsafe input", "general")
        self.assertFalse(r.is_safe)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Respond
# ─────────────────────────────────────────────────────────────────────────────

class TestRespond(unittest.TestCase):

    def setUp(self):
        self.sb = _sandbox()

    def test_respond_returns_dict(self):
        r = self.sb.respond("hello", "general")
        self.assertIsInstance(r, dict)

    def test_respond_creates_sandbox_atom(self):
        self.sb.respond("hello", "general")
        self.assertEqual(len(self.sb.atoms()), 1)

    def test_respond_blocked_creates_conflict_atom(self):
        condition = ConditionController()
        brain = BrainController(condition=condition)
        sb = SandboxController(brain=brain)
        condition.governance_add_rule(
            _block_rule(RuleScope.INPUT, "blocked"), reviewer_id="system"
        )
        r = sb.respond("blocked word", "general")
        self.assertEqual(r["outcome"], "reject")
        atoms = sb.atoms()
        self.assertTrue(any(a.atom_type == AtomType.CONFLICT for a in atoms))

    def test_deactivated_sandbox_rejects(self):
        self.sb.deactivate()
        r = self.sb.respond("hello", "general")
        self.assertEqual(r["outcome"], "reject")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Rule Testing
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleTesting(unittest.TestCase):

    def test_test_rule_no_match(self):
        sb = _sandbox()
        rule = _block_rule(RuleScope.INPUT, "dangerous")
        results = sb.test_rule(rule, ["hello", "safe text"])
        self.assertTrue(all(not r["matched"] for r in results))

    def test_test_rule_match(self):
        sb = _sandbox()
        rule = _block_rule(RuleScope.INPUT, "dangerous")
        results = sb.test_rule(rule, ["this is dangerous"])
        self.assertTrue(results[0]["matched"])

    def test_test_rule_live_requires_reviewer(self):
        sb = _sandbox()
        rule = _block_rule(RuleScope.INPUT, "test")
        with self.assertRaises(PermissionError):
            sb.test_rule_live(rule, ["test"], reviewer_id="")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Replay
# ─────────────────────────────────────────────────────────────────────────────

class TestReplay(unittest.TestCase):

    def test_replay_dry_run(self):
        sb = _sandbox()
        brain = sb._brain
        brain.respond("hello", "general")
        logs = brain.logs
        results = sb.replay(logs, dry_run=True)
        self.assertEqual(len(results), len(logs))

    def test_replay_has_match_key(self):
        sb = _sandbox()
        sb._brain.respond("test", "general")
        results = sb.replay(sb._brain.logs, dry_run=True)
        self.assertIn("match", results[0])

    def test_replay_live(self):
        sb = _sandbox()
        sb._brain.respond("hello", "math")
        results = sb.replay(sb._brain.logs, dry_run=False)
        self.assertIn("atom_id", results[0])


# ─────────────────────────────────────────────────────────────────────────────
# 9. Promote
# ─────────────────────────────────────────────────────────────────────────────

class TestPromote(unittest.TestCase):

    def test_promote_valid_atom(self):
        sb = _sandbox()
        sb.respond("hello", "general")
        atom = sb.atoms()[0]
        atom.confidence = 0.8  # ทำให้ promotable
        proposal_id = sb.promote(atom.atom_id, REVIEWER, "test promote")
        self.assertIsNotNone(proposal_id)

    def test_promote_low_confidence_raises(self):
        sb = _sandbox()
        sb.respond("hello", "general")
        atom = sb.atoms()[0]
        # force confidence ต่ำ
        object.__setattr__(atom, 'confidence', 0.1) if hasattr(atom, '__dataclass_fields__') else setattr(atom, 'confidence', 0.1)
        # SandboxAtom ไม่ frozen — set ได้
        atom.confidence = 0.05
        with self.assertRaises(ValueError):
            sb.promote(atom.atom_id, REVIEWER)

    def test_promote_requires_reviewer(self):
        sb = _sandbox()
        sb.respond("hello", "general")
        atom = sb.atoms()[0]
        with self.assertRaises(PermissionError):
            sb.promote(atom.atom_id, "")


# ─────────────────────────────────────────────────────────────────────────────
# 10. SCL inter-instance
# ─────────────────────────────────────────────────────────────────────────────

class TestSCLIntegration(unittest.TestCase):

    def setUp(self):
        self.scl = SCL()
        self.world = SandboxWorld()
        self.sbA = SandboxController(
            instance_id="instance_A", scl=self.scl, world=self.world
        )
        self.sbB = SandboxController(
            instance_id="instance_B", scl=self.scl, world=self.world
        )

    def test_respond_publishes_to_scl(self):
        self.sbA.respond("hello", "general")
        states = self.sbB.read_experiments()
        self.assertGreater(len(states), 0)

    def test_instances_isolated_atoms(self):
        self.sbA.respond("hello", "math")
        self.sbB.respond("hello", "math")
        # atoms แยกกัน
        self.assertEqual(len(self.sbA.atoms()), 1)
        self.assertEqual(len(self.sbB.atoms()), 1)

    def test_publish_hypothesis(self):
        self.sbA.publish_hypothesis("learning improves", "positive", 0.3)
        states = self.sbB.read_experiments()
        self.assertEqual(len(states), 1)

    def test_no_scl_read_returns_empty(self):
        sb_alone = SandboxController()
        result = sb_alone.read_experiments()
        self.assertEqual(result, [])


# ─────────────────────────────────────────────────────────────────────────────
# 11. SandboxWorld multi-instance
# ─────────────────────────────────────────────────────────────────────────────

class TestSandboxWorld(unittest.TestCase):

    def test_world_tracks_instances(self):
        world = SandboxWorld()
        scl = SCL()
        SandboxController(instance_id="A", world=world, scl=scl)
        SandboxController(instance_id="B", world=world, scl=scl)
        SandboxController(instance_id="C", world=world, scl=scl)
        self.assertEqual(world.instance_count, 3)

    def test_deactivate_removes_from_world(self):
        world = SandboxWorld()
        scl = SCL()
        sb = SandboxController(instance_id="X", world=world, scl=scl)
        self.assertEqual(world.instance_count, 1)
        sb.deactivate()
        self.assertEqual(world.instance_count, 0)

    def test_stats_reflect_world(self):
        world = SandboxWorld()
        scl = SCL()
        sb = SandboxController(instance_id="Y", world=world, scl=scl)
        sb.respond("hello", "general")
        s = sb.stats()
        self.assertEqual(s["atoms_total"], 1)
        self.assertEqual(s["world_id"], world.world_id)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1.  SandboxData                   (5)", TestSandboxData),
        ("2.  SCL                           (8)", TestSCL),
        ("3.  Boot                          (3)", TestBoot),
        ("4.  Safety Check                  (3)", TestSafety),
        ("5.  Simulation                    (4)", TestSimulation),
        ("6.  Respond                       (4)", TestRespond),
        ("7.  Rule Testing                  (3)", TestRuleTesting),
        ("8.  Replay                        (3)", TestReplay),
        ("9.  Promote                       (3)", TestPromote),
        ("10. SCL inter-instance            (4)", TestSCLIntegration),
        ("11. SandboxWorld multi-instance   (3)", TestSandboxWorld),
    ]

    print("\n=================================================================")
    print("  Sandbox Test Suite")
    print("=================================================================")
    for label, _ in groups:
        print(f"  {label}")
    print("─────────────────────────────────────────────────────────────────")
    print("  Total: 43 tests")
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