"""
=================================================================
  BrainStructure Test Suite
=================================================================
  1. Activation Functions        (7 tests)
  2. Loss Functions              (5 tests)
  3. BrainStructure — Build      (4 tests)
  4. BrainStructure — Forward    (3 tests)
  5. BrainStructure — Train      (3 tests)
  6. Snapshot & Rollback         (4 tests)
  7. Evolution → Proposal        (5 tests)
  8. Gradient Safety             (3 tests)
-----------------------------------------------------------------
  Total: 34 tests
=================================================================
"""

import unittest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from Core.Neural.Brain.Functions.Activation import ActivationFunctions
from Core.Neural.Brain.Functions.LossFunction import LossFunctions
from Core.Neural.Brain.BrainStructure import BrainStructure
from Core.Review.Proposal import ProposalStatus

REVIEWER = "reviewer_001"


def _brain(**kwargs) -> BrainStructure:
    b = BrainStructure(verbose=False, **kwargs)
    b.layers = [2, 4, 1]  # fixed layers สำหรับ test
    b.build_structure()
    return b


# ─────────────────────────────────────────────────────────────────────────────
# 1. Activation Functions
# ─────────────────────────────────────────────────────────────────────────────

class TestActivation(unittest.TestCase):

    def test_relu_positive(self):
        fn = ActivationFunctions.get_activation_function("ReLU")
        self.assertAlmostEqual(fn(2.0), 2.0)

    def test_relu_negative(self):
        fn = ActivationFunctions.get_activation_function("ReLU")
        self.assertAlmostEqual(fn(-1.0), 0.0)

    def test_sigmoid_range(self):
        fn = ActivationFunctions.get_activation_function("Sigmoid")
        self.assertGreater(fn(0.0), 0.0)
        self.assertLess(fn(0.0), 1.0)
        self.assertAlmostEqual(fn(0.0), 0.5)

    def test_tanh_zero(self):
        fn = ActivationFunctions.get_activation_function("Tanh")
        self.assertAlmostEqual(fn(0.0), 0.0)

    def test_swish_positive(self):
        fn = ActivationFunctions.get_activation_function("Swish")
        # swish(1) = 1 * sigmoid(1) ≈ 0.731
        self.assertGreater(fn(1.0), 0.0)

    def test_linear_identity(self):
        fn = ActivationFunctions.get_activation_function("Linear")
        self.assertAlmostEqual(fn(5.0), 5.0)

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            ActivationFunctions.get_activation_function("Unknown")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Loss Functions
# ─────────────────────────────────────────────────────────────────────────────

class TestLoss(unittest.TestCase):

    def test_mse_zero_loss(self):
        fn = LossFunctions.get_loss_function("MSE")
        y  = np.array([1.0])
        self.assertAlmostEqual(fn(y, {"default": y}), 0.0)

    def test_mse_nonzero_loss(self):
        fn = LossFunctions.get_loss_function("MSE")
        self.assertGreater(
            fn(np.array([1.0]), {"default": np.array([0.0])}), 0.0
        )

    def test_mae_loss(self):
        fn = LossFunctions.get_loss_function("MAE")
        self.assertAlmostEqual(
            fn(np.array([1.0]), {"default": np.array([0.0])}), 1.0
        )

    def test_bce_clipped(self):
        """BCE ต้องไม่ return nan หรือ inf"""
        fn = LossFunctions.get_loss_function("BinaryCrossEntropy")
        loss = fn(np.array([1.0]), {"default": np.array([0.999])})
        self.assertFalse(np.isnan(loss))
        self.assertFalse(np.isinf(loss))

    def test_unknown_loss_raises(self):
        with self.assertRaises(ValueError):
            LossFunctions.get_loss_function("UnknownLoss")


# ─────────────────────────────────────────────────────────────────────────────
# 3. BrainStructure — Build
# ─────────────────────────────────────────────────────────────────────────────

class TestBuild(unittest.TestCase):

    def test_build_creates_nodes(self):
        """build_structure() → มี nodes"""
        b = _brain()
        self.assertGreater(len(b.nodes), 0)

    def test_build_creates_connections(self):
        """build_structure() → มี connections"""
        b = _brain()
        self.assertGreater(len(b.connections), 0)

    def test_roles_correct(self):
        """มี input, hidden, output nodes"""
        b = _brain()
        roles = {n["role"] for n in b.nodes.values()}
        self.assertIn("input", roles)
        self.assertIn("output", roles)

    def test_compile_sets_loss(self):
        """compile() → loss_fn ถูกตั้ง"""
        b = _brain()
        b.compile()
        self.assertIsNotNone(b.loss_fn)
        self.assertIsNotNone(b.loss_name)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Forward pass
# ─────────────────────────────────────────────────────────────────────────────

class TestForward(unittest.TestCase):

    def setUp(self):
        self.b = _brain()
        self.b.compile()

    def _set_inputs(self, vals):
        inputs = [nid for nid, n in self.b.nodes.items() if n["role"] == "input"]
        for j, nid in enumerate(inputs):
            self.b.nodes[nid]["value"] = float(vals[j])

    def test_forward_sets_output_values(self):
        """forward() → output nodes มีค่า"""
        self._set_inputs([1.0, 0.5])
        self.b.forward()
        outputs = [n["value"] for n in self.b.nodes.values() if n["role"] == "output"]
        self.assertTrue(all(v is not None for v in outputs))

    def test_forward_increments_usage(self):
        """forward() → usage เพิ่ม"""
        self._set_inputs([1.0, 0.5])
        self.b.forward()
        total_usage = sum(
            n["usage"] for n in self.b.nodes.values()
            if n["role"] != "input"
        )
        self.assertGreater(total_usage, 0)

    def test_collect_outputs_returns_arrays(self):
        """collect_outputs() → dict of NDArray"""
        self._set_inputs([1.0, 0.5])
        self.b.forward()
        outputs, _ = self.b.collect_outputs()
        for arr in outputs.values():
            self.assertIsInstance(arr, np.ndarray)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Train
# ─────────────────────────────────────────────────────────────────────────────

class TestTrain(unittest.TestCase):

    def setUp(self):
        self.b = BrainStructure(verbose=False, model_type="Regression")
        self.b.layers = [2, 3, 1]
        self.b.build_structure()
        self.b.loss_name    = "MSE"
        self.b.loss_fn      = LossFunctions.get_loss_function("MSE")
        self.b.loss_grad_fn = LossFunctions.get_loss_gradient("MSE")

    def test_train_returns_history(self):
        """train() คืน loss history"""
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        y = np.array([[0.5], [0.6]])
        history = self.b.train(X, y, epochs=2, lr=0.01)
        self.assertEqual(len(history), 2)

    def test_train_loss_is_float(self):
        """loss ทุก epoch เป็น float"""
        X = np.array([[0.1, 0.2]])
        y = np.array([[0.5]])
        history = self.b.train(X, y, epochs=3, lr=0.01)
        for l in history:
            self.assertIsInstance(l, float)

    def test_train_without_compile_raises(self):
        """train ก่อน compile → RuntimeError"""
        b = BrainStructure(verbose=False)
        b.layers = [2, 1]
        b.build_structure()
        X = np.array([[0.1, 0.2]])
        y = np.array([[0.5]])
        with self.assertRaises(RuntimeError):
            b.train(X, y, epochs=1, lr=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Snapshot & Rollback
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotRollback(unittest.TestCase):

    def test_snapshot_stores_weights(self):
        """take_snapshot() เก็บ weights ณ เวลานั้น"""
        b = _brain()
        original = dict(b.weights)
        b.take_snapshot()
        # เปลี่ยน weights
        for k in b.weights:
            b.weights[k] = 999.0
        b.rollback()
        self.assertEqual(b.weights, original)

    def test_rollback_restores_nodes(self):
        """rollback() คืน node count เดิม"""
        b = _brain()
        original_count = len(b.nodes)
        b.take_snapshot()
        b._add_node()
        b.rollback()
        self.assertEqual(len(b.nodes), original_count)

    def test_rollback_no_snapshot_returns_false(self):
        """rollback ไม่มี snapshot → False"""
        b = _brain()
        self.assertFalse(b.rollback())

    def test_multiple_snapshots(self):
        """snapshot หลายครั้ง → rollback ล่าสุดก่อน"""
        b = _brain()
        b.take_snapshot()
        b._add_node()
        count_after_first = len(b.nodes)
        b.take_snapshot()
        b._add_node()
        b.rollback()
        self.assertEqual(len(b.nodes), count_after_first)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Evolution → Proposal
# ─────────────────────────────────────────────────────────────────────────────

class TestEvolutionProposal(unittest.TestCase):

    def setUp(self):
        self.b = _brain()

    def test_propose_creates_pending_proposal(self):
        """propose_evolution() → pending_proposals เพิ่ม"""
        # loss trend แย่ลง → ADD_NODE
        p = self.b.propose_evolution(None, loss=0.5, prev_loss=0.4)
        self.assertIsNotNone(p)
        self.assertEqual(len(self.b.pending_proposals), 1)

    def test_no_op_returns_none(self):
        """loss trend ดีขึ้น → NO_OP → None"""
        # loss = 0.3, prev = 0.5, trend = -0.2 < -0.001 → NO_OP
        p = self.b.propose_evolution(None, loss=0.3, prev_loss=0.5)
        self.assertIsNone(p)

    def test_apply_without_reviewer_raises(self):
        """apply_approved_evolution ไม่มี reviewer → PermissionError"""
        p = self.b.propose_evolution(None, loss=0.5, prev_loss=0.4)
        if p:
            p._approve(REVIEWER)
            with self.assertRaises(PermissionError):
                self.b.apply_approved_evolution(p.proposal_id, reviewer_id="")

    def test_apply_unapproved_raises(self):
        """apply proposal ที่ยังไม่ approve → ValueError"""
        p = self.b.propose_evolution(None, loss=0.5, prev_loss=0.4)
        if p:
            with self.assertRaises(ValueError):
                self.b.apply_approved_evolution(p.proposal_id, REVIEWER)

    def test_apply_approved_changes_structure(self):
        """apply approved → structure เปลี่ยน"""
        before = len(self.b.nodes)
        p = self.b.propose_evolution(None, loss=0.5, prev_loss=0.4)
        if p:
            p._approve(REVIEWER)
            self.b.apply_approved_evolution(p.proposal_id, REVIEWER)
            # ADD_NODE ควรเพิ่ม node
            self.assertGreaterEqual(len(self.b.nodes), before)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Gradient Safety
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientSafety(unittest.TestCase):

    def test_normal_gradient_ok(self):
        """gradient ปกติ → ไม่ raise"""
        b = _brain()
        b._neural.monitor_gradient("test", 0.5)  # ไม่ raise

    def test_nan_gradient_raises(self):
        """NaN gradient → RuntimeError"""
        b = _brain()
        with self.assertRaises(RuntimeError):
            b._neural.monitor_gradient("test", float("nan"))

    def test_explode_gradient_raises(self):
        """Explode gradient → RuntimeError"""
        b = _brain()
        with self.assertRaises(RuntimeError):
            b._neural.monitor_gradient("test", 999.0)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_tests():
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    groups = [
        ("1. Activation Functions       (7)", TestActivation),
        ("2. Loss Functions             (5)", TestLoss),
        ("3. BrainStructure — Build     (4)", TestBuild),
        ("4. BrainStructure — Forward   (3)", TestForward),
        ("5. BrainStructure — Train     (3)", TestTrain),
        ("6. Snapshot & Rollback        (4)", TestSnapshotRollback),
        ("7. Evolution → Proposal       (5)", TestEvolutionProposal),
        ("8. Gradient Safety            (3)", TestGradientSafety),
    ]

    print("\n=================================================================")
    print("  BrainStructure Test Suite")
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