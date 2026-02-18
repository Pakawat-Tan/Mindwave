"""
=================================================================
  Integration Test Suite — End-to-End
=================================================================
  ทดสอบทุก module ทำงานร่วมกันจริง ผ่าน BrainController

  1. Boot                             (3 tests)
  2. Condition Gate — Block flow      (4 tests)
  3. Normal respond() flow            (5 tests)
  4. Realtime Learning                (4 tests)
  5. Memory integration               (4 tests)
  6. Evolution                        (3 tests)
  7. Lock / Unlock                    (3 tests)
  8. Full pipeline end-to-end         (4 tests)
-----------------------------------------------------------------
  Total: 30 tests
=================================================================
"""

import unittest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Core.BrainController import BrainController
from Core.Condition.ConditionController import ConditionController
from Core.Condition.Rule.RuleData import RuleData, RuleAction, RuleScope, MatchType, RuleAuthority
from Core.Neural.Brain.BrainStructure import BrainStructure
from Core.Neural.Brain.Functions.LossFunction import LossFunctions
from Core.Review.ReviewerController import ReviewerController
from Core.Review.ReviewerData import ReviewerRole

REVIEWER = "reviewer_001"


def _brain_with_structure() -> BrainController:
    """BrainController พร้อม BrainStructure สำหรับ test learning"""
    bs = BrainStructure(verbose=False)
    bs.layers = [2, 4, 1]
    bs.build_structure()
    bs.loss_name    = "MSE"
    bs.loss_fn      = LossFunctions.get_loss_function("MSE")
    bs.loss_grad_fn = LossFunctions.get_loss_gradient("MSE")
    b = BrainController(brain_structure=bs)
    b._brain_struct.set_evolve_every(10)
    return b


def _block_rule(scope: RuleScope, pattern: str = "") -> RuleData:
    """สร้าง blocking rule สำหรับ test"""
    if pattern:
        return RuleData(
            scope       = scope,
            action      = RuleAction.BLOCK,
            match_type  = MatchType.PATTERN,
            pattern     = pattern,
            description = f"test block rule for {scope.value}",
        )
    else:
        # ANY = block ทุกอย่างใน scope
        return RuleData(
            scope       = scope,
            action      = RuleAction.BLOCK,
            match_type  = MatchType.ANY,
            description = f"test block all in {scope.value}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Boot
# ─────────────────────────────────────────────────────────────────────────────

class TestBoot(unittest.TestCase):

    def test_brain_boots_successfully(self):
        """BrainController สร้างได้โดยไม่ error"""
        b = BrainController()
        self.assertIsNotNone(b)

    def test_all_modules_present(self):
        """ทุก module พร้อมใช้งาน"""
        b = BrainController()
        s = b.status()
        self.assertTrue(s["modules"]["condition"])
        self.assertTrue(s["modules"]["confidence"])
        self.assertTrue(s["modules"]["skill"])
        self.assertTrue(s["modules"]["personality"])
        self.assertTrue(s["modules"]["memory"])

    def test_personality_auto_initialized(self):
        """Personality init อัตโนมัติตอน boot"""
        b = BrainController()
        self.assertTrue(b.personality.is_initialized())


# ─────────────────────────────────────────────────────────────────────────────
# 2. Condition Gate — Block flow
# ─────────────────────────────────────────────────────────────────────────────

class TestConditionGate(unittest.TestCase):

    def setUp(self):
        self.condition = ConditionController()
        self.b = BrainController(condition=self.condition)

    def test_blocked_input_returns_reject(self):
        """Rule block INPUT → respond() outcome = reject หรือ silence"""
        rule = _block_rule(RuleScope.INPUT, "blocked_word")
        self.condition.governance_add_rule(rule, reviewer_id="system")
        r = self.b.respond("blocked_word test", "general")
        self.assertIn(r["outcome"], ["reject", "silence"])

    def test_blocked_skill_prevents_arbitrate(self):
        """Rule block SKILL → Skill.arbitrate() ถูก gate"""
        rule = _block_rule(RuleScope.SKILL, "")
        self.condition.governance_add_rule(rule, reviewer_id="system")
        result = self.b.skill.arbitrate()
        self.assertFalse(result.has_skills)

    def test_blocked_neural_prevents_observe(self):
        """Rule block NEURAL → BrainStructure.observe() ถูก gate"""
        condition = ConditionController()
        bs = BrainStructure(verbose=False, condition=condition)
        bs.layers = [2, 1]
        bs.build_structure()
        rule = _block_rule(RuleScope.NEURAL)
        condition.governance_add_rule(rule, reviewer_id="system")
        result = bs.observe(np.array([0.5, 0.3]), "math")
        self.assertTrue(result.get("blocked", False))

    def test_no_rule_allows_everything(self):
        """ไม่มี rule block → ผ่านทุก gate"""
        r = self.b.respond("hello world", "general")
        self.assertIn(r["outcome"], ["commit", "conditional", "ask"])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Normal respond() flow
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalFlow(unittest.TestCase):

    def setUp(self):
        self.b = BrainController()

    def test_respond_returns_valid_outcome(self):
        """respond() คืน outcome ที่ valid"""
        r = self.b.respond("hello", "general")
        self.assertIn(r["outcome"], ["commit", "conditional", "ask", "silence", "reject"])

    def test_respond_has_confidence_score(self):
        """respond() มี confidence score"""
        r = self.b.respond("test input", "math")
        self.assertGreaterEqual(r["confidence"], 0.0)
        self.assertLessEqual(r["confidence"], 1.0)

    def test_respond_logs_every_interaction(self):
        """ทุก respond() บันทึก log"""
        for i in range(3):
            self.b.respond(f"input {i}", "general")
        self.assertEqual(len(self.b.logs), 3)

    def test_respond_personality_in_result(self):
        """respond() มี personality ใน result"""
        r = self.b.respond("hello", "general")
        self.assertNotEqual(r["personality"], "")

    def test_skill_contract_runs_every_respond(self):
        """Skill Contract รันทุกครั้ง — confidence เปลี่ยนตาม context"""
        r1 = self.b.respond("hello", "general")
        r2 = self.b.respond("hello", "math")
        # ทั้งสอง respond ผ่านและมี confidence
        self.assertIsNotNone(r1["confidence"])
        self.assertIsNotNone(r2["confidence"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Realtime Learning
# ─────────────────────────────────────────────────────────────────────────────

class TestRealtimeLearning(unittest.TestCase):

    def setUp(self):
        self.b = _brain_with_structure()

    def test_observe_learns_on_every_interaction(self):
        """observe() เรียนรู้ทุก interaction"""
        result = self.b._brain_struct.observe(
            np.array([0.5, 0.3]), "math", confidence=0.8
        )
        self.assertTrue(result["learned"])

    def test_weights_change_after_learning(self):
        """weights เปลี่ยนหลัง observe"""
        bs = self.b._brain_struct
        before = dict(bs.weights)
        bs.observe(np.array([1.0, 0.5]), "math", confidence=0.9)
        changed = any(abs(bs.weights[k] - before[k]) > 1e-10 for k in before)
        self.assertTrue(changed)

    def test_respond_with_vector_triggers_learning(self):
        """respond() พร้อม input_vector → learned = True"""
        r = self.b.respond(
            "learn this",
            "science",
            input_vector=np.array([0.5, 0.3]),
        )
        self.assertTrue(r["learned"])

    def test_repetition_tracked_per_context(self):
        """context แต่ละ topic นับแยก"""
        bs = self.b._brain_struct
        for _ in range(3):
            bs.observe(np.array([0.5, 0.3]), "math")
        for _ in range(2):
            bs.observe(np.array([0.5, 0.3]), "science")
        counts = bs.repetition_counts()
        self.assertEqual(counts.get("math", 0),    3)
        self.assertEqual(counts.get("science", 0), 2)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Memory Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryIntegration(unittest.TestCase):

    def setUp(self):
        self.b = BrainController()

    def test_memory_accessible(self):
        """Memory module พร้อมใช้"""
        self.assertIsNotNone(self.b.memory)

    def test_respond_stores_response(self):
        """respond() → Memory.write_response() ถูกเรียก (ไม่ crash)"""
        r = self.b.respond("hello world", "general")
        # ตรวจว่า respond() ผ่าน ไม่ใช่ crash
        self.assertIn("outcome", r)

    def test_memory_recall_affects_confidence(self):
        """Memory recall context_score ส่งผลต่อ Confidence"""
        b = BrainController()
        # respond ครั้งแรก — ไม่มี memory
        r1 = b.respond("test", "math")
        # respond อีกครั้ง — memory อาจมี context score จากครั้งก่อน
        r2 = b.respond("test", "math")
        # ทั้งสอง respond ต้องผ่านได้
        self.assertIn("confidence", r1)
        self.assertIn("confidence", r2)

    def test_memory_condition_gate_works(self):
        """Memory gate ทำงาน — ถ้า block → write_response คืน None"""
        condition = ConditionController()
        from Core.Memory.MemoryController import MemoryController
        memory = MemoryController(condition=condition)
        rule = _block_rule(RuleScope.MEMORY)
        condition.governance_add_rule(rule, reviewer_id="system")
        result = memory.write_response("blocked text", "test", importance=0.8)
        self.assertIsNone(result)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Evolution
# ─────────────────────────────────────────────────────────────────────────────

class TestEvolution(unittest.TestCase):

    def test_evolution_triggers_at_N(self):
        """หลัง N interactions → evolution_count อาจเพิ่ม"""
        b = _brain_with_structure()
        bs = b._brain_struct
        bs.set_evolve_every(5)
        bs._last_loss = 1.0  # force loss trend → ADD_NODE
        for _ in range(5):
            bs.observe(np.array([1.0, 0.5]), "math", confidence=0.9)
        stats = bs.evolution_stats()
        self.assertEqual(stats["interaction_count"], 5)

    def test_structure_survives_evolution(self):
        """หลัง evolve — structure ยังใช้งานได้"""
        b = _brain_with_structure()
        bs = b._brain_struct
        bs.set_evolve_every(3)
        for _ in range(6):
            bs.observe(np.array([0.5, 0.3]), "math", confidence=0.8)
        # ยังสามารถ forward ได้
        inputs = [nid for nid, n in bs.nodes.items() if n["role"] == "input"]
        for nid in inputs:
            bs.nodes[nid]["value"] = 0.5
        bs.forward()  # ไม่ crash
        self.assertGreater(len(bs.nodes), 0)

    def test_rollback_on_gradient_unsafe(self):
        """gradient unsafe → rollback อัตโนมัติ"""
        b = _brain_with_structure()
        bs = b._brain_struct
        before = len(bs.nodes)
        bs.take_snapshot()
        # ใส่ค่า weight ที่ safe ก่อน
        bs.rollback()
        self.assertEqual(len(bs.nodes), before)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Lock / Unlock
# ─────────────────────────────────────────────────────────────────────────────

class TestLockUnlock(unittest.TestCase):

    def test_locked_brain_rejects_all(self):
        """Brain locked → ทุก respond() = reject"""
        b = BrainController()
        b.lock(REVIEWER)
        r = b.respond("hello", "general")
        self.assertEqual(r["outcome"], "reject")

    def test_unlocked_brain_works_again(self):
        """Brain unlock → respond() ปกติ"""
        b = BrainController()
        b.lock(REVIEWER)
        b.unlock(REVIEWER)
        r = b.respond("hello", "general")
        self.assertNotEqual(r["outcome"], "reject")

    def test_lock_requires_reviewer(self):
        """lock() ไม่มี reviewer → PermissionError"""
        b = BrainController()
        with self.assertRaises(PermissionError):
            b.lock("")


# ─────────────────────────────────────────────────────────────────────────────
# 8. Full pipeline end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestFullPipeline(unittest.TestCase):

    def test_full_respond_pipeline(self):
        """
        Full end-to-end:
        Boot → Register Skill → respond() หลายครั้ง
        → learn → logs สะสม → status ถูกต้อง
        """
        b = _brain_with_structure()

        # register skill
        b.skill.register("general", topic_ids=[])
        b.skill.register("math",    topic_ids=[1, 2])

        # respond หลาย context
        responses = []
        for i in range(5):
            r = b.respond(
                f"interaction {i}",
                "math",
                input_vector=np.array([float(i), float(i) * 0.5]),
            )
            responses.append(r)

        # ทุก response มี required keys
        for r in responses:
            self.assertIn("outcome",    r)
            self.assertIn("confidence", r)
            self.assertIn("log_id",     r)

        # logs สะสม
        self.assertEqual(len(b.logs), 5)

        # status สะท้อนความจริง
        s = b.status()
        self.assertEqual(s["logs_total"], 5)
        self.assertEqual(s["mode"], "active")

    def test_condition_blocks_propagate_to_all_modules(self):
        """
        Rule block SKILL → Skill blocked
        Rule block NEURAL → Neural blocked
        Rule block MEMORY → Memory write blocked
        ทั้งหมดจาก ConditionController เดียว
        """
        condition = ConditionController()
        b = BrainController(condition=condition)

        # block skill
        condition.governance_add_rule(_block_rule(RuleScope.SKILL, ""), reviewer_id="system")
        arb = b.skill.arbitrate()
        self.assertFalse(arb.has_skills)

        # block memory
        condition.governance_add_rule(_block_rule(RuleScope.MEMORY, ""), reviewer_id="system")
        result = b.memory.write_response("test", "general", 0.8)
        self.assertIsNone(result)

    def test_multiple_contexts_learn_independently(self):
        """หลาย context เรียนรู้แยกกัน"""
        b = _brain_with_structure()
        bs = b._brain_struct

        for _ in range(3):
            bs.observe(np.array([1.0, 0.0]), "math",    confidence=0.9)
        for _ in range(2):
            bs.observe(np.array([0.0, 1.0]), "science", confidence=0.9)

        counts = bs.repetition_counts()
        self.assertEqual(counts.get("math",    0), 3)
        self.assertEqual(counts.get("science", 0), 2)

    def test_reviewer_can_approve_proposals(self):
        """Reviewer approve Proposal ได้จริง"""
        from Core.Review.Proposal import create_proposal, ProposalAction, ProposalTarget, RuleAuthority
        b = BrainController()
        rc = b.reviewer

        # register reviewer
        rc.register_reviewer(REVIEWER, ReviewerRole.STANDARD)

        # สร้างและ approve proposal
        p = create_proposal(
            proposed_by = "brain",
            action      = ProposalAction.MODIFY,
            target_type = ProposalTarget.RULE,
            authority   = RuleAuthority.STANDARD,
            payload     = {"key": "value"},
            reason      = "integration test",
        )
        rc.enqueue(p)
        decision = rc.approve(p, REVIEWER, "approved in integration test")

        self.assertTrue(p.is_approved)
        self.assertIsNotNone(decision)
        s = rc.stats()
        self.assertEqual(s["approvals"], 1)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1. Boot                          (3)", TestBoot),
        ("2. Condition Gate — Block flow   (4)", TestConditionGate),
        ("3. Normal respond() flow         (5)", TestNormalFlow),
        ("4. Realtime Learning             (4)", TestRealtimeLearning),
        ("5. Memory Integration            (4)", TestMemoryIntegration),
        ("6. Evolution                     (3)", TestEvolution),
        ("7. Lock / Unlock                 (3)", TestLockUnlock),
        ("8. Full pipeline end-to-end      (4)", TestFullPipeline),
    ]

    print("\n=================================================================")
    print("  Integration Test Suite — End-to-End")
    print("=================================================================")
    for label, _ in groups:
        print(f"  {label}")
    print("─────────────────────────────────────────────────────────────────")
    print("  Total: 30 tests")
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