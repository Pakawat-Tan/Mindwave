"""
=================================================================
  Reviewer System Test Suite
=================================================================
  1. ReviewerData — ReviewDecision       (3 tests)
  2. ReviewerData — AuditEvent           (3 tests)
  3. ReviewerData — RollbackRecord       (2 tests)
  4. ReviewerController — Registry       (4 tests)
  5. ReviewerController — Queue          (5 tests)
  6. ReviewerController — Permission     (5 tests)
  7. ReviewerController — Approve        (5 tests)
  8. ReviewerController — Reject         (4 tests)
  9. ReviewerController — Rollback       (5 tests)
 10. ReviewerController — Audit          (4 tests)
 11. ReviewerController — Stats          (2 tests)
-----------------------------------------------------------------
  Total: 42 tests
=================================================================
"""

import unittest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Core.Review.Proposal import (
    ProposalData, ProposalAction, ProposalTarget,
    RuleAuthority, create_proposal
)
from Core.Review.ReviewerData import (
    ReviewDecision, AuditEvent, RollbackRecord,
    ReviewAction, ReviewerRole
)
from Core.Review.ReviewerController import ReviewerController


REVIEWER_STD     = "reviewer_std"
REVIEWER_CREATOR = "reviewer_creator"
MODEL_ID         = "neural_model"


def _rc() -> ReviewerController:
    rc = ReviewerController()
    rc.register_reviewer(REVIEWER_STD,     ReviewerRole.STANDARD)
    rc.register_reviewer(REVIEWER_CREATOR, ReviewerRole.CREATOR)
    return rc


def _proposal(authority: RuleAuthority = RuleAuthority.STANDARD) -> ProposalData:
    return create_proposal(
        proposed_by = MODEL_ID,
        action      = ProposalAction.MODIFY,
        target_type = ProposalTarget.RULE,
        authority   = authority,
        payload     = {"key": "value"},
        reason      = "test proposal",
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1. ReviewDecision
# ─────────────────────────────────────────────────────────────────────────────

class TestReviewDecision(unittest.TestCase):

    def test_create_decision(self):
        d = ReviewDecision(
            proposal_id  = "p001",
            action       = ReviewAction.APPROVE,
            reviewer_id  = REVIEWER_STD,
            reviewer_role = ReviewerRole.STANDARD,
        )
        self.assertEqual(d.action, ReviewAction.APPROVE)

    def test_decision_is_frozen(self):
        d = ReviewDecision()
        with self.assertRaises((AttributeError, TypeError)):
            d.reviewer_id = "changed"  # type: ignore

    def test_decision_to_dict(self):
        d = ReviewDecision(proposal_id="p001", reviewer_id=REVIEWER_STD)
        data = d.to_dict()
        self.assertIn("decision_id", data)
        self.assertIn("reviewer_id", data)


# ─────────────────────────────────────────────────────────────────────────────
# 2. AuditEvent
# ─────────────────────────────────────────────────────────────────────────────

class TestAuditEvent(unittest.TestCase):

    def test_create_event(self):
        e = AuditEvent(
            action      = ReviewAction.APPROVE,
            reviewer_id = REVIEWER_STD,
            target_id   = "p001",
            target_type = "proposal",
        )
        self.assertTrue(e.success)

    def test_event_is_frozen(self):
        e = AuditEvent()
        with self.assertRaises((AttributeError, TypeError)):
            e.success = False  # type: ignore

    def test_event_to_dict(self):
        e = AuditEvent(reviewer_id=REVIEWER_STD, target_id="p001")
        self.assertIn("event_id", e.to_dict())


# ─────────────────────────────────────────────────────────────────────────────
# 3. RollbackRecord
# ─────────────────────────────────────────────────────────────────────────────

class TestRollbackRecord(unittest.TestCase):

    def test_create_rollback(self):
        r = RollbackRecord(
            decision_id    = "d001",
            proposal_id    = "p001",
            rolled_back_by = REVIEWER_STD,
            reason         = "test",
        )
        self.assertEqual(r.rolled_back_by, REVIEWER_STD)

    def test_rollback_is_frozen(self):
        r = RollbackRecord()
        with self.assertRaises((AttributeError, TypeError)):
            r.reason = "changed"  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# 4. Registry
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistry(unittest.TestCase):

    def test_register_standard(self):
        rc = ReviewerController()
        rc.register_reviewer(REVIEWER_STD, ReviewerRole.STANDARD)
        self.assertTrue(rc.is_registered(REVIEWER_STD))

    def test_register_creator(self):
        rc = ReviewerController()
        rc.register_reviewer(REVIEWER_CREATOR, ReviewerRole.CREATOR)
        self.assertEqual(rc.get_role(REVIEWER_CREATOR), ReviewerRole.CREATOR)

    def test_register_empty_id_raises(self):
        rc = ReviewerController()
        with self.assertRaises(ValueError):
            rc.register_reviewer("")

    def test_unregistered_returns_none(self):
        rc = ReviewerController()
        self.assertIsNone(rc.get_role("nobody"))


# ─────────────────────────────────────────────────────────────────────────────
# 5. Queue
# ─────────────────────────────────────────────────────────────────────────────

class TestQueue(unittest.TestCase):

    def test_enqueue_pending_proposal(self):
        rc = _rc()
        p  = _proposal()
        rc.enqueue(p)
        self.assertEqual(rc.queue_size(), 1)

    def test_enqueue_non_pending_raises(self):
        rc = _rc()
        p  = _proposal()
        p._approve(REVIEWER_STD)
        with self.assertRaises(ValueError):
            rc.enqueue(p)

    def test_enqueue_no_duplicate(self):
        rc = _rc()
        p  = _proposal()
        rc.enqueue(p)
        rc.enqueue(p)  # ซ้ำ → ไม่เพิ่ม
        self.assertEqual(rc.queue_size(), 1)

    def test_dequeue_removes_from_queue(self):
        rc = _rc()
        p  = _proposal()
        rc.enqueue(p)
        rc.dequeue(p.proposal_id)
        self.assertEqual(rc.queue_size(), 0)

    def test_pending_by_authority(self):
        rc  = _rc()
        p1  = _proposal(RuleAuthority.STANDARD)
        p2  = _proposal(RuleAuthority.SYSTEM)
        rc.enqueue(p1)
        rc.enqueue(p2)
        std = rc.pending_by_authority(RuleAuthority.STANDARD)
        self.assertEqual(len(std), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Permission Check
# ─────────────────────────────────────────────────────────────────────────────

class TestPermission(unittest.TestCase):

    def test_empty_reviewer_raises(self):
        rc = _rc()
        p  = _proposal()
        with self.assertRaises(PermissionError):
            rc.approve(p, reviewer_id="")

    def test_unregistered_reviewer_raises(self):
        rc = _rc()
        p  = _proposal()
        with self.assertRaises(PermissionError):
            rc.approve(p, reviewer_id="nobody")

    def test_standard_cannot_approve_system(self):
        rc = _rc()
        p  = _proposal(RuleAuthority.SYSTEM)
        with self.assertRaises(PermissionError):
            rc.approve(p, REVIEWER_STD)

    def test_creator_can_approve_system(self):
        rc = _rc()
        p  = _proposal(RuleAuthority.SYSTEM)
        rc.approve(p, REVIEWER_CREATOR)
        self.assertTrue(p.is_approved)

    def test_standard_can_approve_standard(self):
        rc = _rc()
        p  = _proposal(RuleAuthority.STANDARD)
        rc.approve(p, REVIEWER_STD)
        self.assertTrue(p.is_approved)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Approve
# ─────────────────────────────────────────────────────────────────────────────

class TestApprove(unittest.TestCase):

    def setUp(self):
        self.rc = _rc()

    def test_approve_returns_decision(self):
        p = _proposal()
        d = self.rc.approve(p, REVIEWER_STD, "looks good")
        self.assertIsInstance(d, ReviewDecision)
        self.assertEqual(d.action, ReviewAction.APPROVE)

    def test_approve_marks_proposal_approved(self):
        p = _proposal()
        self.rc.approve(p, REVIEWER_STD)
        self.assertTrue(p.is_approved)

    def test_approve_removes_from_queue(self):
        p = _proposal()
        self.rc.enqueue(p)
        self.rc.approve(p, REVIEWER_STD)
        self.assertEqual(self.rc.queue_size(), 0)

    def test_approve_stores_decision(self):
        p = _proposal()
        d = self.rc.approve(p, REVIEWER_STD)
        self.assertIn(d.decision_id, self.rc._decisions)

    def test_approve_already_approved_raises(self):
        p = _proposal()
        self.rc.approve(p, REVIEWER_STD)
        with self.assertRaises(ValueError):
            self.rc.approve(p, REVIEWER_STD)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Reject
# ─────────────────────────────────────────────────────────────────────────────

class TestReject(unittest.TestCase):

    def setUp(self):
        self.rc = _rc()

    def test_reject_returns_decision(self):
        p = _proposal()
        d = self.rc.reject(p, REVIEWER_STD, "not valid")
        self.assertEqual(d.action, ReviewAction.REJECT)

    def test_reject_marks_proposal_rejected(self):
        p = _proposal()
        self.rc.reject(p, REVIEWER_STD, "reason")
        self.assertTrue(p.is_rejected)

    def test_reject_removes_from_queue(self):
        p = _proposal()
        self.rc.enqueue(p)
        self.rc.reject(p, REVIEWER_STD, "not good")
        self.assertEqual(self.rc.queue_size(), 0)

    def test_reject_already_decided_raises(self):
        p = _proposal()
        self.rc.reject(p, REVIEWER_STD, "reason")
        with self.assertRaises(ValueError):
            self.rc.reject(p, REVIEWER_STD, "again")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Rollback
# ─────────────────────────────────────────────────────────────────────────────

class TestRollback(unittest.TestCase):

    def setUp(self):
        self.rc = _rc()
        self.p  = _proposal()
        self.d  = self.rc.approve(self.p, REVIEWER_STD, "ok")

    def test_rollback_returns_record(self):
        r = self.rc.rollback(self.d.decision_id, REVIEWER_STD, "undo")
        self.assertIsInstance(r, RollbackRecord)

    def test_rollback_stores_record(self):
        self.rc.rollback(self.d.decision_id, REVIEWER_STD)
        self.assertEqual(len(self.rc._rollbacks), 1)

    def test_rollback_snapshot_available(self):
        snap = self.rc.get_rollback_snapshot(self.d.decision_id)
        self.assertIsNotNone(snap)
        self.assertIn("key", snap)

    def test_rollback_unregistered_raises(self):
        with self.assertRaises(PermissionError):
            self.rc.rollback(self.d.decision_id, "nobody")

    def test_rollback_non_approve_decision_raises(self):
        p2 = _proposal()
        d2 = self.rc.reject(p2, REVIEWER_STD, "no")
        with self.assertRaises(ValueError):
            self.rc.rollback(d2.decision_id, REVIEWER_STD)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Audit
# ─────────────────────────────────────────────────────────────────────────────

class TestAudit(unittest.TestCase):

    def setUp(self):
        self.rc = _rc()

    def test_approve_creates_audit_event(self):
        p = _proposal()
        self.rc.approve(p, REVIEWER_STD)
        log = self.rc.audit_log()
        self.assertTrue(any(e.action == ReviewAction.APPROVE for e in log))

    def test_reject_creates_audit_event(self):
        p = _proposal()
        self.rc.reject(p, REVIEWER_STD, "no")
        log = self.rc.audit_log()
        self.assertTrue(any(e.action == ReviewAction.REJECT for e in log))

    def test_audit_by_reviewer(self):
        p = _proposal()
        self.rc.approve(p, REVIEWER_STD)
        events = self.rc.audit_by_reviewer(REVIEWER_STD)
        self.assertGreater(len(events), 0)

    def test_audit_by_proposal(self):
        p = _proposal()
        self.rc.approve(p, REVIEWER_STD)
        events = self.rc.audit_by_proposal(p.proposal_id)
        self.assertGreater(len(events), 0)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Stats
# ─────────────────────────────────────────────────────────────────────────────

class TestStats(unittest.TestCase):

    def test_stats_empty(self):
        rc = ReviewerController()
        s  = rc.stats()
        self.assertEqual(s["decisions_total"], 0)
        self.assertEqual(s["queue_size"],      0)

    def test_stats_after_operations(self):
        rc = _rc()
        p1 = _proposal()
        p2 = _proposal()
        p3 = _proposal()
        rc.enqueue(p3)
        rc.approve(p1, REVIEWER_STD)
        rc.reject(p2,  REVIEWER_STD, "no")
        s = rc.stats()
        self.assertEqual(s["approvals"],   1)
        self.assertEqual(s["rejections"],  1)
        self.assertEqual(s["queue_size"],  1)
        self.assertEqual(s["reviewers_registered"], 2)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1.  ReviewDecision                (3)", TestReviewDecision),
        ("2.  AuditEvent                    (3)", TestAuditEvent),
        ("3.  RollbackRecord                (2)", TestRollbackRecord),
        ("4.  ReviewerController — Registry (4)", TestRegistry),
        ("5.  ReviewerController — Queue    (5)", TestQueue),
        ("6.  ReviewerController — Permission(5)", TestPermission),
        ("7.  ReviewerController — Approve  (5)", TestApprove),
        ("8.  ReviewerController — Reject   (4)", TestReject),
        ("9.  ReviewerController — Rollback (5)", TestRollback),
        ("10. ReviewerController — Audit    (4)", TestAudit),
        ("11. ReviewerController — Stats    (2)", TestStats),
    ]

    print("\n=================================================================")
    print("  Reviewer System Test Suite")
    print("=================================================================")
    for label, _ in groups:
        print(f"  {label}")
    print("─────────────────────────────────────────────────────────────────")
    print("  Total: 42 tests")
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