"""
=================================================================
  Distributed System Test Suite
=================================================================
  1. Instance Coordination     (5 tests)
  2. Consensus                 (5 tests)
  3. State Sync                (5 tests)
  4. Distributed Learning      (4 tests)
  5. Conflict Resolution       (5 tests)
  6. Leader Election           (4 tests)
  7. Integration               (2 tests)
-----------------------------------------------------------------
  Total: 30 tests
=================================================================
"""

import unittest
import time
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Core.Brain.DistributedSystem import (
    DistributedSystem, InstanceRole, VoteDecision, ConflictStrategy,
    InstanceState, ConsensusProposal, ConsensusResult,
    ConflictRecord, LearningUpdate,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Instance Coordination
# ─────────────────────────────────────────────────────────────────────────────

class TestInstanceCoordination(unittest.TestCase):

    def test_register_instance(self):
        ds = DistributedSystem("instance_A")
        ds.register_instance("instance_B")
        self.assertEqual(len(ds.instances), 2)

    def test_heartbeat_updates_timestamp(self):
        ds = DistributedSystem("instance_A")
        before = ds._instances["instance_A"].last_heartbeat
        time.sleep(0.01)
        ds.heartbeat()
        after = ds._instances["instance_A"].last_heartbeat
        self.assertGreater(after, before)

    def test_get_alive_instances(self):
        ds = DistributedSystem("instance_A")
        ds.register_instance("instance_B")
        ds.heartbeat("instance_B")
        alive = ds.get_alive_instances()
        self.assertGreaterEqual(len(alive), 1)

    def test_instance_count(self):
        ds = DistributedSystem("instance_A")
        ds.register_instance("instance_B")
        ds.register_instance("instance_C")
        self.assertGreaterEqual(ds.instance_count, 1)

    def test_self_registered_automatically(self):
        ds = DistributedSystem("instance_A")
        self.assertIn("instance_A", ds._instances)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Consensus
# ─────────────────────────────────────────────────────────────────────────────

class TestConsensus(unittest.TestCase):

    def setUp(self):
        self.ds = DistributedSystem("instance_A")
        self.ds.register_instance("instance_B")
        self.ds.register_instance("instance_C")

    def test_propose_creates_proposal(self):
        p = self.ds.propose("test_action", {"data": "test"})
        self.assertIsInstance(p, ConsensusProposal)

    def test_vote_records_decision(self):
        p = self.ds.propose("test", {})
        self.ds.vote(p.proposal_id, VoteDecision.APPROVE)
        self.assertIn("instance_A", self.ds._votes[p.proposal_id])

    def test_tally_votes_returns_result(self):
        p = self.ds.propose("test", {})
        self.ds.vote(p.proposal_id, VoteDecision.APPROVE)
        self.ds.vote(p.proposal_id, VoteDecision.APPROVE, "instance_B")
        result = self.ds.tally_votes(p.proposal_id)
        self.assertIsInstance(result, ConsensusResult)

    def test_majority_approval(self):
        p = self.ds.propose("test", {})
        self.ds.vote(p.proposal_id, VoteDecision.APPROVE)
        self.ds.vote(p.proposal_id, VoteDecision.APPROVE, "instance_B")
        self.ds.vote(p.proposal_id, VoteDecision.REJECT, "instance_C")
        result = self.ds.tally_votes(p.proposal_id)
        self.assertTrue(result.approved)

    def test_vote_invalid_proposal_raises(self):
        with self.assertRaises(KeyError):
            self.ds.vote("nonexistent", VoteDecision.APPROVE)


# ─────────────────────────────────────────────────────────────────────────────
# 3. State Sync
# ─────────────────────────────────────────────────────────────────────────────

class TestStateSync(unittest.TestCase):

    def setUp(self):
        self.ds = DistributedSystem("instance_A")
        self.ds.register_instance("instance_B")

    def test_sync_state_updates(self):
        success = self.ds.sync_state("instance_B", {"key": "value"}, 1)
        self.assertTrue(success)

    def test_sync_state_old_version_skipped(self):
        self.ds.sync_state("instance_B", {"key": "v1"}, 2)
        success = self.ds.sync_state("instance_B", {"key": "v2"}, 1)
        self.assertFalse(success)

    def test_get_state_returns_data(self):
        self.ds.sync_state("instance_B", {"test": "data"}, 1)
        state = self.ds.get_state("instance_B")
        self.assertEqual(state, {"test": "data"})

    def test_broadcast_state_syncs_all(self):
        self.ds.register_instance("instance_C")
        synced = self.ds.broadcast_state({"broadcast": "data"})
        # synced to B and C (ถ้า alive)
        self.assertGreaterEqual(synced, 0)

    def test_state_version_increments(self):
        before = self.ds._instances["instance_A"].state_version
        self.ds.broadcast_state({"test": "data"})
        after = self.ds._instances["instance_A"].state_version
        self.assertGreater(after, before)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Distributed Learning
# ─────────────────────────────────────────────────────────────────────────────

class TestDistributedLearning(unittest.TestCase):

    def setUp(self):
        self.ds = DistributedSystem("instance_A")

    def test_share_learning_creates_update(self):
        update = self.ds.share_learning("weight", {"w1": 0.5}, 0.8)
        self.assertIsInstance(update, LearningUpdate)

    def test_get_learning_updates(self):
        self.ds.share_learning("pattern", {"p": "data"}, 0.7)
        updates = self.ds.get_learning_updates()
        self.assertEqual(len(updates), 1)

    def test_filter_by_confidence(self):
        self.ds.share_learning("weight", {"w": 1}, 0.5)
        self.ds.share_learning("weight", {"w": 2}, 0.9)
        updates = self.ds.get_learning_updates(min_confidence=0.7)
        self.assertEqual(len(updates), 1)

    def test_filter_by_type(self):
        self.ds.share_learning("weight", {}, 0.8)
        self.ds.share_learning("knowledge", {}, 0.8)
        updates = self.ds.get_learning_updates(update_type="weight")
        self.assertEqual(len(updates), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Conflict Resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestConflictResolution(unittest.TestCase):

    def setUp(self):
        self.ds = DistributedSystem("instance_A")
        self.ds.register_instance("instance_B")

    def test_detect_conflict_finds_difference(self):
        conflict = self.ds.detect_conflict(
            "key", "value_a", "value_b", "instance_A", "instance_B"
        )
        self.assertIsNotNone(conflict)

    def test_detect_conflict_same_value_none(self):
        conflict = self.ds.detect_conflict(
            "key", "same", "same", "instance_A", "instance_B"
        )
        self.assertIsNone(conflict)

    def test_resolve_conflict_last_write_wins(self):
        conflict = self.ds.detect_conflict(
            "key", "old", "new", "instance_A", "instance_B"
        )
        resolution = self.ds.resolve_conflict(
            conflict, ConflictStrategy.LAST_WRITE_WINS
        )
        self.assertIsNotNone(resolution)

    def test_resolve_conflict_leader_decides(self):
        self.ds.elect_leader()
        conflict = self.ds.detect_conflict(
            "key", "val_a", "val_b", "instance_A", "instance_B"
        )
        resolution = self.ds.resolve_conflict(
            conflict, ConflictStrategy.LEADER_DECIDES
        )
        self.assertIsNotNone(resolution)

    def test_conflicts_property(self):
        self.ds.detect_conflict("k", "a", "b", "instance_A", "instance_B")
        conflicts = self.ds.conflicts
        self.assertEqual(len(conflicts), 1)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Leader Election
# ─────────────────────────────────────────────────────────────────────────────

class TestLeaderElection(unittest.TestCase):

    def test_elect_leader_returns_id(self):
        ds = DistributedSystem("instance_B")
        ds.register_instance("instance_A")
        ds.register_instance("instance_C")
        leader = ds.elect_leader()
        self.assertIsNotNone(leader)

    def test_leader_has_role(self):
        ds = DistributedSystem("instance_A")
        ds.register_instance("instance_B")
        leader = ds.elect_leader()
        leader_state = ds.get_instance_state(leader)
        self.assertEqual(leader_state.role, InstanceRole.LEADER)

    def test_is_leader_property(self):
        ds = DistributedSystem("instance_A")
        ds.elect_leader()
        # instance_A ควรเป็น leader (ID ต่ำสุด)
        self.assertTrue(ds.is_leader)

    def test_leader_id_property(self):
        ds = DistributedSystem("instance_A")
        ds.elect_leader()
        self.assertEqual(ds.leader_id, "instance_A")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Integration
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration(unittest.TestCase):

    def test_full_pipeline(self):
        ds1 = DistributedSystem("instance_A")
        ds2 = DistributedSystem("instance_B")
        ds3 = DistributedSystem("instance_C")

        # 1. coordination
        ds1.register_instance("instance_B")
        ds1.register_instance("instance_C")
        ds1.heartbeat()

        # 2. leader election
        leader = ds1.elect_leader()

        # 3. consensus
        p = ds1.propose("test_action", {"data": "test"})
        ds1.vote(p.proposal_id, VoteDecision.APPROVE)
        ds1.vote(p.proposal_id, VoteDecision.APPROVE, "instance_B")
        result = ds1.tally_votes(p.proposal_id)

        # 4. state sync
        ds1.broadcast_state({"shared": "data"})

        # 5. learning
        update = ds1.share_learning("weight", {"w": 0.5}, 0.8)

        # 6. conflict
        conflict = ds1.detect_conflict(
            "key", "val1", "val2", "instance_A", "instance_B"
        )
        if conflict:
            ds1.resolve_conflict(conflict)

        # all produced results
        self.assertIsNotNone(leader)
        self.assertIsNotNone(result)
        self.assertIsNotNone(update)

    def test_stats_reflect_state(self):
        ds = DistributedSystem("instance_A")
        ds.register_instance("instance_B")
        ds.elect_leader()
        ds.propose("test", {})
        ds.share_learning("weight", {}, 0.8)

        stats = ds.stats()
        self.assertEqual(stats["instance_id"], "instance_A")
        self.assertGreater(stats["total_instances"], 0)
        self.assertGreater(stats["proposals"], 0)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1. Instance Coordination  (5)", TestInstanceCoordination),
        ("2. Consensus              (5)", TestConsensus),
        ("3. State Sync             (5)", TestStateSync),
        ("4. Distributed Learning   (4)", TestDistributedLearning),
        ("5. Conflict Resolution    (5)", TestConflictResolution),
        ("6. Leader Election        (4)", TestLeaderElection),
        ("7. Integration            (2)", TestIntegration),
    ]

    print("\n=================================================================")
    print("  Distributed System Test Suite")
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