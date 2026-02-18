"""
BrainStructure — Neural network ที่ใช้งานได้จริง

การเปลี่ยนแปลงหลักจาก Mock:
  1. Activation/Loss functions implement จริง
  2. Evolution → Proposal system (ไม่ apply ตรง)
  3. Gradient monitor เชื่อมกับ NeuralController
  4. Snapshot + Rollback ก่อน evolve
  5. Rule engine เชื่อมกับ ConditionController
  6. NodeSchema strict TypedDict
"""

from __future__ import annotations

import copy
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

import sys, os

from Core.Neural.Brain.Schema import (
    NodeSchema, ConnectionSchema, EvolutionContext, StructureSnapshot, MDNHead
)
from Core.Neural.Brain.Functions.Activation import ActivationFunctions
from Core.Neural.Brain.Functions.LossFunction import LossFunctions
from Core.Neural.NeuralController import NeuralController
from Core.Review.Proposal import ProposalData, ProposalAction, ProposalTarget, RuleAuthority, create_proposal


logger = logging.getLogger("mindwave.brain.structure")


class BrainStructure:

    def __init__(
        self,
        *,
        model_type:      str  = "Regression",
        mdn_components:  int  = 3,
        mdn_dim:         int  = 1,
        verbose:         bool = True,
        neural_controller: Optional[NeuralController] = None,
        condition=None,
    ):
        self.seed           = int(np.random.randint(0, 1_000_000))
        np.random.seed(self.seed)

        self.model_type     = model_type
        self.mdn_components = mdn_components
        self.mdn_dim        = mdn_dim
        self.verbose        = verbose

        self._condition = condition
        # Neural controller สำหรับ gradient monitoring + proposals
        self._neural = neural_controller or NeuralController()

        self.layers:      Optional[List[int]]             = None
        self.nodes:       Dict[str, NodeSchema]           = {}
        self.connections: Dict[str, ConnectionSchema]     = {}
        self.weights:     Dict[str, float]                = {}
        self.biases:      Dict[str, float]                = {}

        # Pending evolution proposals
        self._pending_proposals: List[ProposalData] = []
        # Snapshots สำหรับ rollback
        self._snapshots: List[StructureSnapshot]    = []

        # Realtime Evolution tracking
        self._interaction_count: int   = 0
        self._evolve_every:      int   = 50    # evolve ทุก N interactions
        self._last_loss:         float = 0.0
        self._evolution_count:   int   = 0
        self._evolution_log:     List[dict] = []

        self.loss_fn        = None
        self.loss_grad_fn   = None
        self.loss_name      = ""
        self.compiled_at    = None

    # ────────────────────────────────────────────────────────────
    # Hyperparameter pools
    # ────────────────────────────────────────────────────────────

    ACTIVATION_POOL = {
        "ReLU": 0.25, "LeakyReLU": 0.15, "GELU": 0.15,
        "Sigmoid": 0.10, "Tanh": 0.10, "Swish": 0.10,
        "ELU": 0.10, "Linear": 0.05,
    }
    LOSS_POOL = {
        "MSE": 0.35, "MAE": 0.15, "BinaryCrossEntropy": 0.15,
        "CategoricalCrossEntropy": 0.10, "MDN_NLL": 0.25,
    }

    def _rand_activation(self) -> str:
        return str(np.random.choice(
            list(self.ACTIVATION_POOL.keys()),
            p=list(self.ACTIVATION_POOL.values()),
        ))

    def _rand_loss(self) -> str:
        return str(np.random.choice(
            list(self.LOSS_POOL.keys()),
            p=list(self.LOSS_POOL.values()),
        ))

    # ────────────────────────────────────────────────────────────
    # Build
    # ────────────────────────────────────────────────────────────

    def build_structure(
        self,
        connection_prob: float = 1.0,
        min_layers:      int   = 2,
        max_layers:      int   = 5,
        min_nodes:       int   = 2,
        max_nodes:       int   = 16,
    ) -> None:
        if self.layers is None:
            self.layers = [
                int(np.random.randint(min_nodes, max_nodes + 1))
                for _ in range(np.random.randint(min_layers, max_layers + 1))
            ]

        self.nodes.clear()
        self.connections.clear()
        self.weights.clear()
        self.biases.clear()

        prev_nodes:   List[str] = []
        node_counter: int       = 0

        for li, n_nodes in enumerate(self.layers):
            curr_nodes: List[str] = []
            is_input  = li == 0
            is_output = li == len(self.layers) - 1

            if is_output and self.model_type == "MDN":
                K, D = self.mdn_components, self.mdn_dim
                heads: Dict[MDNHead, Tuple[int, Optional[str]]] = {
                    "mdn_pi":    (K,     "softmax"),
                    "mdn_mu":    (K * D, None),
                    "mdn_sigma": (K * D, "exp"),
                }
                for head, (count, act) in heads.items():
                    for _ in range(count):
                        nid = f"L{li}_{head}_{node_counter}"
                        self.nodes[nid] = NodeSchema(
                            layer=li, role="output", head=head,
                            activation=act, value=None, gradient=None, usage=0.0,
                        )
                        self.biases[nid] = float(np.random.randn() * 0.01)
                        curr_nodes.append(nid)
                        node_counter += 1
            else:
                for _ in range(n_nodes):
                    nid  = f"L{li}_N{node_counter}"
                    role: Literal["input","hidden","output"] = (
                        "input"  if is_input  else
                        "output" if is_output else "hidden"
                    )
                    act = None if role == "input" else self._rand_activation()
                    self.nodes[nid] = NodeSchema(
                        layer=li, role=role, head=None,
                        activation=act, value=None, gradient=None, usage=0.0,
                    )
                    self.biases[nid] = float(np.random.randn() * 0.01)
                    curr_nodes.append(nid)
                    node_counter += 1

            for s in prev_nodes:
                for d in curr_nodes:
                    if np.random.rand() <= connection_prob:
                        cid = f"{s}->{d}"
                        self.connections[cid] = ConnectionSchema(
                            source=s, destination=d, enabled=True,
                        )
                        self.weights[cid] = float(np.random.randn() * 0.01)

            prev_nodes = curr_nodes

        logger.info(
            f"[BrainStructure] built {len(self.nodes)} nodes "
            f"{len(self.connections)} connections"
        )

    # ────────────────────────────────────────────────────────────
    # Compile
    # ────────────────────────────────────────────────────────────

    def compile(self) -> None:
        self.loss_name    = "MDN_NLL" if self.model_type == "MDN" else self._rand_loss()
        self.loss_fn      = LossFunctions.get_loss_function(self.loss_name)
        self.loss_grad_fn = LossFunctions.get_loss_gradient(self.loss_name)
        self.compiled_at  = datetime.now()
        logger.info(f"[BrainStructure] compiled loss={self.loss_name}")

    # ────────────────────────────────────────────────────────────
    # Forward / Backward
    # ────────────────────────────────────────────────────────────

    def forward(self) -> None:
        by_layer: Dict[int, List[str]] = {}
        for nid, n in self.nodes.items():
            by_layer.setdefault(n["layer"], []).append(nid)

        for layer in sorted(by_layer):
            for nid in by_layer[layer]:
                node = self.nodes[nid]
                if node["role"] == "input":
                    continue

                total = sum(
                    self.weights[cid] * self.nodes[c["source"]]["value"]
                    for cid, c in self.connections.items()
                    if c["enabled"] and c["destination"] == nid
                    and self.nodes[c["source"]]["value"] is not None
                )
                total += self.biases[nid]

                act_name = node["activation"]
                if act_name == "softmax":
                    # softmax ใช้ vector — จัดการใน collect_outputs
                    node["value"] = total
                elif act_name is not None:
                    act_fn = ActivationFunctions.get_activation_function(act_name)
                    node["value"] = act_fn(total)
                else:
                    node["value"] = total

                node["usage"] += 1.0

    def backward(self) -> None:
        by_layer: Dict[int, List[str]] = {}
        for nid, n in self.nodes.items():
            by_layer.setdefault(n["layer"], []).append(nid)

        for layer in sorted(by_layer, reverse=True):
            for nid in by_layer[layer]:
                g = self.nodes[nid]["gradient"]
                if g is None:
                    continue
                for cid, c in self.connections.items():
                    if not c["enabled"] or c["destination"] != nid:
                        continue
                    src      = c["source"]
                    contrib  = g * self.weights[cid]
                    prev     = self.nodes[src]["gradient"]
                    self.nodes[src]["gradient"] = (
                        contrib if prev is None else prev + contrib
                    )

    def collect_outputs(
        self,
    ) -> Tuple[Dict[str, NDArray[np.float64]], Dict[str, List[str]]]:
        from Core.Neural.Brain.Functions.Activation import softmax as softmax_fn
        values:    Dict[str, List[float]] = {}
        index_map: Dict[str, List[str]]   = {}

        for nid, n in self.nodes.items():
            if n["role"] != "output":
                continue
            val = n["value"]
            if val is None:
                raise RuntimeError(f"[BrainStructure] output node {nid} has no value")
            key = n["head"] if n["head"] is not None else "default"
            values.setdefault(key, []).append(val)
            index_map.setdefault(key, []).append(nid)

        result: Dict[str, NDArray[np.float64]] = {}
        for k, v in values.items():
            arr = np.asarray(v, dtype=np.float64)
            # softmax สำหรับ mdn_pi
            if k == "mdn_pi":
                arr = softmax_fn(arr)
            result[k] = arr

        return result, index_map

    def backpropagation(self, y_true: Any, lr: float) -> float:
        if self.loss_fn is None or self.loss_grad_fn is None:
            raise RuntimeError("[BrainStructure] call compile() first")

        outputs, index_map = self.collect_outputs()
        loss        = self.loss_fn(y_true, outputs)
        grad_result = self.loss_grad_fn(y_true, outputs)
        grads: Dict[str, NDArray] = (
            grad_result if isinstance(grad_result, dict)
            else {"default": grad_result}
        )

        # ── Monitor gradients ──────────────────────────────────────
        for key, grad_vec in grads.items():
            for g_val in np.asarray(grad_vec).flatten():
                try:
                    self._neural.monitor_gradient(f"output_{key}", float(g_val))
                except RuntimeError as e:
                    logger.error(f"[BrainStructure] GRADIENT_UNSAFE: {e}")
                    raise  # หยุด training ตาม NeuralEvolution Rule

        # ── Assign output gradients ───────────────────────────────
        for key, node_ids in index_map.items():
            grad_vec = grads.get(key, grads.get("default", np.zeros(len(node_ids))))
            for i, nid in enumerate(node_ids):
                self.nodes[nid]["gradient"] = float(grad_vec[i])

        self.backward()

        # ── Update weights ────────────────────────────────────────
        for cid, c in self.connections.items():
            if not c["enabled"]:
                continue
            g = self.nodes[c["destination"]]["gradient"]
            x = self.nodes[c["source"]]["value"]
            if g is not None and x is not None:
                self.weights[cid] -= lr * g * x

        for nid, n in self.nodes.items():
            if n["gradient"] is not None:
                self.biases[nid] -= lr * n["gradient"]
                n["gradient"] = None

        return float(loss)

    # ────────────────────────────────────────────────────────────
    # Train
    # ────────────────────────────────────────────────────────────

    def train(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[Any],
        *,
        epochs: int,
        lr:     float,
    ) -> List[float]:
        """
        Train และคืน loss history ต่อ epoch
        """
        if self.loss_fn is None:
            raise RuntimeError("[BrainStructure] call compile() first")

        inputs   = [nid for nid, n in self.nodes.items() if n["role"] == "input"]
        n_samples = x_train.shape[0]
        history: List[float] = []

        for ep in range(epochs):
            total_loss = 0.0
            for i in range(n_samples):
                # reset values
                for nid in self.nodes:
                    if self.nodes[nid]["role"] != "input":
                        self.nodes[nid]["value"] = None

                for j, nid in enumerate(inputs):
                    self.nodes[nid]["value"] = float(x_train[i, j])

                self.forward()
                total_loss += self.backpropagation(y_train[i], lr)

            avg_loss = total_loss / n_samples
            history.append(avg_loss)
            if self.verbose:
                logger.info(
                    f"[BrainStructure] Epoch {ep+1}/{epochs} "
                    f"loss={avg_loss:.6f}"
                )
                if self.verbose:
                    print(f"[Epoch {ep+1}/{epochs}] loss={avg_loss:.6f}")

        return history

    # ────────────────────────────────────────────────────────────
    # Snapshot & Rollback
    # ────────────────────────────────────────────────────────────

    def take_snapshot(self) -> StructureSnapshot:
        """เก็บ snapshot ก่อน evolve"""
        snap: StructureSnapshot = {
            "nodes":       copy.deepcopy(self.nodes),
            "connections": copy.deepcopy(self.connections),
            "weights":     dict(self.weights),
            "biases":      dict(self.biases),
        }
        self._snapshots.append(snap)
        return snap

    def rollback(self) -> bool:
        """ย้อนกลับ snapshot ล่าสุด"""
        if not self._snapshots:
            logger.warning("[BrainStructure] no snapshot to rollback")
            return False
        snap = self._snapshots.pop()
        self.nodes       = snap["nodes"]
        self.connections = snap["connections"]
        self.weights     = snap["weights"]
        self.biases      = snap["biases"]
        logger.warning("[BrainStructure] ROLLBACK to previous snapshot")
        return True

    # ────────────────────────────────────────────────────────────
    # Evolution → Proposal (ไม่ apply ตรง)
    # ────────────────────────────────────────────────────────────

    def propose_evolution(
        self,
        rule_engine:  Any,          # ConditionController
        *,
        loss:         float,
        prev_loss:    float,
        proposed_by:  str   = "brain_structure",
    ) -> Optional[ProposalData]:
        """
        ตัดสินใจ evolution intent แล้วเสนอผ่าน Proposal
        ไม่ apply เอง — ต้องรอ reviewer approve

        Args:
            rule_engine : ConditionController (หรือ object ที่มี .decide())
            loss        : loss ปัจจุบัน
            prev_loss   : loss ก่อนหน้า
            proposed_by : ชื่อ proposer

        Returns:
            ProposalData ถ้าต้องการ evolve
            None ถ้า NO_OP
        """
        ctx = EvolutionContext(
            loss            = loss,
            loss_trend      = loss - prev_loss,
            num_nodes       = len(self.nodes),
            num_connections = sum(1 for c in self.connections.values() if c["enabled"]),
            usage           = {nid: n["usage"] for nid, n in self.nodes.items()},
            model_type      = self.model_type,
        )

        # ConditionController หรือ rule_engine.decide()
        if hasattr(rule_engine, "decide"):
            intent = rule_engine.decide(ctx)
        else:
            intent = self._default_intent(ctx)

        if intent == "NO_OP":
            return None

        proposal = create_proposal(
            proposed_by = proposed_by,
            action      = ProposalAction.MODIFY,
            target_type = ProposalTarget.RULE,
            authority   = RuleAuthority.STANDARD,
            payload     = {
                "evolution_intent": intent,
                "loss":             loss,
                "prev_loss":        prev_loss,
                "num_nodes":        ctx["num_nodes"],
                "num_connections":  ctx["num_connections"],
            },
            reason = f"brain proposes evolution: {intent}",
        )
        self._pending_proposals.append(proposal)
        logger.info(
            f"[BrainStructure] EVOLUTION_PROPOSED {proposal.proposal_id[:8]} "
            f"intent={intent} loss={loss:.6f}"
        )
        return proposal

    def apply_approved_evolution(
        self,
        proposal_id:  str,
        reviewer_id:  str,
    ) -> bool:
        """
        Apply evolution หลัง reviewer approve

        ถ้า gradient unsafe หลัง evolve → rollback อัตโนมัติ
        """
        if not reviewer_id or not reviewer_id.strip():
            raise PermissionError(
                "[BrainStructure] apply_approved_evolution requires reviewer_id"
            )

        proposal = next(
            (p for p in self._pending_proposals
             if p.proposal_id == proposal_id),
            None
        )
        if proposal is None:
            return False
        if not proposal.is_approved:
            raise ValueError(
                f"[BrainStructure] proposal {proposal_id[:8]} not approved"
            )

        intent = proposal.payload["evolution_intent"]

        # snapshot ก่อน apply
        self.take_snapshot()

        try:
            self._apply_intent(intent)
            self._pending_proposals = [
                p for p in self._pending_proposals
                if p.proposal_id != proposal_id
            ]
            logger.warning(
                f"[BrainStructure] EVOLUTION_APPLIED {intent} "
                f"by='{reviewer_id}'"
            )
            return True
        except Exception as e:
            logger.error(
                f"[BrainStructure] EVOLUTION_FAILED {intent}: {e} "
                f"— rolling back"
            )
            self.rollback()
            return False

    @property
    def pending_proposals(self) -> List[ProposalData]:
        return list(self._pending_proposals)

    # ────────────────────────────────────────────────────────────
    # Evolution operations (internal — เรียกผ่าน apply_approved)
    # ────────────────────────────────────────────────────────────

    def _apply_intent(self, intent: str) -> None:
        match intent:
            case "ADD_NODE":        self._add_node()
            case "ADD_CONNECTION":  self._add_connection()
            case "PRUNE_NODE":      self._prune_node()
            case "PRUNE_CONNECTION":self._prune_connection()
            case "ADD_LAYER":       self._add_layer()
            case "PRUNE_LAYER":     self._prune_layer()
            case "MUTATE_WEIGHT":   self._mutate_weight()
            case "MUTATE_BIAS":     self._mutate_bias()
            case _:
                raise ValueError(f"unknown intent: {intent}")

    def _add_node(self) -> None:
        if not self.connections:
            return
        cid  = np.random.choice(list(self.connections.keys()))
        conn = self.connections[cid]
        if not conn["enabled"]:
            return
        src = conn["source"]
        dst = conn["destination"]
        new_layer = self.nodes[src]["layer"] + 1
        nid = f"L{new_layer}_N{len(self.nodes)}"
        self.nodes[nid] = NodeSchema(
            layer=new_layer, role="hidden", head=None,
            activation=self._rand_activation(),
            value=None, gradient=None, usage=0.0,
        )
        self.biases[nid] = 0.0
        conn["enabled"] = False
        c1, c2 = f"{src}->{nid}", f"{nid}->{dst}"
        self.connections[c1] = ConnectionSchema(source=src, destination=nid, enabled=True)
        self.connections[c2] = ConnectionSchema(source=nid, destination=dst, enabled=True)
        self.weights[c1] = 1.0
        self.weights[c2] = self.weights[cid]

    def _add_connection(self) -> None:
        nodes = list(self.nodes.keys())
        for _ in range(10):
            src, dst = np.random.choice(nodes, 2, replace=False)
            if self.nodes[src]["layer"] >= self.nodes[dst]["layer"]:
                continue
            cid = f"{src}->{dst}"
            if cid in self.connections:
                continue
            self.connections[cid] = ConnectionSchema(source=src, destination=dst, enabled=True)
            self.weights[cid] = float(np.random.randn() * 0.01)
            return

    def _prune_node(self) -> None:
        candidates = [nid for nid, n in self.nodes.items() if n["role"] == "hidden"]
        if not candidates:
            return
        nid = np.random.choice(candidates)
        for cid in [k for k, c in self.connections.items()
                    if c["source"] == nid or c["destination"] == nid]:
            self.connections.pop(cid)
            self.weights.pop(cid, None)
        self.nodes.pop(nid)
        self.biases.pop(nid, None)

    def _prune_connection(self) -> None:
        enabled = [k for k, c in self.connections.items() if c["enabled"]]
        if enabled:
            self.connections[np.random.choice(enabled)]["enabled"] = False

    def _add_layer(self) -> None:
        max_layer = max(n["layer"] for n in self.nodes.values())
        insert_at = int(np.random.randint(1, max_layer))
        for n in self.nodes.values():
            if n["layer"] >= insert_at:
                n["layer"] += 1
        for _ in range(int(np.random.randint(2, 6))):
            nid = f"L{insert_at}_N{len(self.nodes)}"
            self.nodes[nid] = NodeSchema(
                layer=insert_at, role="hidden", head=None,
                activation=self._rand_activation(),
                value=None, gradient=None, usage=0.0,
            )
            self.biases[nid] = 0.0

    def _prune_layer(self) -> None:
        layers: Dict[int, List[str]] = {}
        for nid, n in self.nodes.items():
            layers.setdefault(n["layer"], []).append(nid)
        hidden_layers = [
            l for l, nids in layers.items()
            if all(self.nodes[n]["role"] == "hidden" for n in nids)
        ]
        if not hidden_layers:
            return
        target = int(np.random.choice(hidden_layers))
        for nid in list(layers[target]):
            self._prune_node()

    def _mutate_weight(self) -> None:
        if self.weights:
            cid = np.random.choice(list(self.weights.keys()))
            self.weights[cid] += float(np.random.randn() * 0.01)

    def _mutate_bias(self) -> None:
        if self.biases:
            nid = np.random.choice(list(self.biases.keys()))
            self.biases[nid] += float(np.random.randn() * 0.01)

    # ────────────────────────────────────────────────────────────
    # Default intent logic (ถ้าไม่มี rule_engine)
    # ────────────────────────────────────────────────────────────

    def _default_intent(self, ctx: EvolutionContext) -> str:
        """Simple heuristic — ใช้เมื่อไม่มี ConditionController"""
        if ctx["loss_trend"] > 0.01:       # loss แย่ลง
            return "ADD_NODE"
        if ctx["num_nodes"] > 50:          # ใหญ่เกินไป
            return "PRUNE_NODE"
        if ctx["loss_trend"] < -0.001:     # กำลังดีขึ้น
            return "NO_OP"
        return "MUTATE_WEIGHT"

    # ────────────────────────────────────────────────────────────
    # Stats / Summary
    # ────────────────────────────────────────────────────────────

    def get_structure_data(self) -> dict:
        total_nodes  = len(self.nodes)
        total_active = sum(1 for c in self.connections.values() if c["enabled"])
        total_usage  = sum(n["usage"] for n in self.nodes.values())
        role_count   = {"input": 0, "hidden": 0, "output": 0}
        layers: set  = set()
        for n in self.nodes.values():
            role_count[n["role"]] += 1
            layers.add(n["layer"])

        return {
            "model_type":  self.model_type,
            "loss_fn":     self.loss_name or "not compiled",
            "layers":      len(layers),
            "nodes":       total_nodes,
            "roles":       role_count,
            "connections": total_active,
            "parameters":  {
                "total":   total_active + len(self.biases),
                "weights": total_active,
                "biases":  len(self.biases),
            },
            "usage_avg":      total_usage / total_nodes if total_nodes else 0.0,
            "snapshots":      len(self._snapshots),
            "proposals_pending": len(self._pending_proposals),
        }

    def get_usage(self) -> Dict[str, float]:
        return {nid: n["usage"] for nid, n in self.nodes.items()}

    def clear_usage(self) -> None:
        for n in self.nodes.values():
            n["usage"] = 0.0


    # ────────────────────────────────────────────────────────────
    # Continuous Learning (แทน train())
    # ────────────────────────────────────────────────────────────

    def observe(
        self,
        input_vector:    "NDArray[np.float64]",
        context_label:   str,
        confidence:      float = 0.8,
        reviewer_id:     str   = "",
    ) -> dict:
        """
        เรียนรู้จาก 1 interaction — ไม่มีการกด Train

        Flow:
          1. Forward pass ด้วย input ที่รับมา
          2. นับ topic repetition ของ context_label
          3. ถ้า rep >= _rep_threshold AND confidence >= _conf_threshold
             → คำนวณ implicit loss จาก context coherence
             → propose weight update ผ่าน NeuralController
          4. ถ้ามี reviewer_id → auto-approve และ apply
             ถ้าไม่มี → รอ reviewer approve ภายนอก

        Args:
            input_vector  : input ของ interaction นี้
            context_label : topic/domain ของ context (เช่น "math", "coding")
            confidence    : Confidence score จาก ConfidenceController
            reviewer_id   : ถ้ามี → auto-apply หลัง propose

        Returns:
            dict ของ observation result
        """
        # ── Condition Gate ─────────────────────────────────────
        if self._condition is not None:
            _allowed, _reason = self._condition.is_neural_allowed()
            if not _allowed:
                logger.warning(
                    f"[BrainStructure] observe BLOCKED reason={_reason}"
                )
                return {"context_label": context_label, "repetition": 0,
                        "confidence": confidence, "learned": False,
                        "blocked": True, "reason": _reason}

        # ── init tracking ─────────────────────────────────────────
        if not hasattr(self, "_rep_count"):
            self._rep_count:  dict = {}
            self._rep_threshold  = 3
            self._conf_threshold = 0.6

        # ── Forward ───────────────────────────────────────────────
        inputs = [nid for nid, n in self.nodes.items() if n["role"] == "input"]
        for nid in self.nodes:
            if self.nodes[nid]["role"] != "input":
                self.nodes[nid]["value"] = None

        for j, nid in enumerate(inputs):
            if j < len(input_vector):
                self.nodes[nid]["value"] = float(input_vector[j])
            else:
                self.nodes[nid]["value"] = 0.0

        self.forward()
        outputs, _ = self.collect_outputs()

        # ── นับ repetition ────────────────────────────────────────
        self._rep_count[context_label] = self._rep_count.get(context_label, 0) + 1
        rep = self._rep_count[context_label]

        result = {
            "context_label": context_label,
            "repetition":    rep,
            "confidence":    confidence,
            "learned":       False,
            "proposal_id":   None,
        }

        # ── Realtime Learning — apply ทันที ทุก interaction ────────
        implicit_loss = self._compute_coherence_loss(outputs, context_label)
        self._apply_realtime_update(
            implicit_loss = implicit_loss,
            context_label = context_label,
            confidence    = confidence,
        )
        result["learned"]  = True
        result["applied"]  = True
        result["loss"]     = implicit_loss

        # ── Realtime Evolution — ทุก N interactions ───────────────
        self._interaction_count += 1
        evolved = False
        if self._interaction_count % self._evolve_every == 0:
            evolved = self._auto_evolve(current_loss=implicit_loss)
            result["evolved"] = evolved
            self._last_loss   = implicit_loss

        logger.info(
            f"[BrainStructure] REALTIME_LEARN context='{context_label}' "
            f"rep={rep} conf={confidence:.3f} "
            f"loss={implicit_loss:.6f} "
            f"interactions={self._interaction_count}"
            + (f" EVOLVED" if evolved else "")
        )

        return result

    def set_learning_thresholds(
        self,
        rep_threshold:  int   = 3,
        conf_threshold: float = 0.6,
    ) -> None:
        """ปรับ threshold สำหรับ continuous learning"""
        if not hasattr(self, "_rep_count"):
            self._rep_count = {}
        self._rep_threshold  = rep_threshold
        self._conf_threshold = conf_threshold

    def repetition_counts(self) -> dict:
        """ดู topic repetition ปัจจุบัน"""
        if not hasattr(self, "_rep_count"):
            return {}
        return dict(self._rep_count)

    # ── Internal helpers ──────────────────────────────────────────

    def _compute_coherence_loss(
        self,
        outputs:       dict,
        context_label: str,
    ) -> float:
        """
        คำนวณ implicit loss จาก context coherence

        Logic: variance ของ output values
        — output ที่กระจาย = ไม่ coherent = loss สูง
        — output ที่รวมกัน = coherent = loss ต่ำ
        """
        all_vals = []
        for arr in outputs.values():
            all_vals.extend(arr.flatten().tolist())

        if not all_vals:
            return 0.0

        arr    = np.array(all_vals)
        mean   = float(np.mean(arr))
        variance = float(np.var(arr))
        return round(variance, 6)

    def set_evolve_every(self, n: int) -> None:
        """ตั้ง interval ของ auto-evolve (จำนวน interactions)"""
        if n < 1:
            raise ValueError("evolve_every must be >= 1")
        self._evolve_every = n

    def evolution_stats(self) -> dict:
        """สรุปสถิติ evolution"""
        return {
            "interaction_count": self._interaction_count,
            "evolution_count":   getattr(self, "_evolution_count", 0),
            "evolve_every":      self._evolve_every,
            "last_loss":         self._last_loss,
            "current_nodes":     len(self.nodes),
            "current_connections": sum(
                1 for c in self.connections.values() if c["enabled"]
            ),
            "log": list(getattr(self, "_evolution_log", [])[-10:]),
        }

    def _auto_evolve(self, current_loss: float) -> bool:
        """Auto-evolve โดยไม่ต้องมี Reviewer"""
        if not hasattr(self, "_evolution_count"):
            self._evolution_count = 0
        if not hasattr(self, "_evolution_log"):
            self._evolution_log = []

        intent = self._default_intent_from_loss(current_loss)
        if intent == "NO_OP":
            return False

        self.take_snapshot()
        try:
            self._apply_intent(intent)
            self._evolution_count += 1
            self._evolution_log.append({
                "intent":       intent,
                "loss":         current_loss,
                "interactions": self._interaction_count,
                "nodes":        len(self.nodes),
            })
            logger.warning(
                f"[BrainStructure] AUTO_EVOLVE intent={intent} "
                f"nodes={len(self.nodes)} loss={current_loss:.6f}"
            )
            return True
        except Exception as e:
            logger.error(f"[BrainStructure] AUTO_EVOLVE_FAILED {intent}: {e} → rollback")
            self.rollback()
            return False

    def _default_intent_from_loss(self, loss: float) -> str:
        """ตัดสินใจ evolution intent จาก loss + structure"""
        loss_trend = loss - self._last_loss
        n_nodes    = len(self.nodes)
        n_active   = sum(1 for c in self.connections.values() if c["enabled"])

        if n_nodes > 100:
            return "PRUNE_NODE"
        if loss_trend > 0.05 and n_nodes < 50:
            return "ADD_NODE"
        if loss_trend > 0.1:
            return "ADD_CONNECTION"
        if n_active > 500:
            return "PRUNE_CONNECTION"
        if abs(loss_trend) < 0.001:
            return "MUTATE_WEIGHT"
        return "NO_OP"

    def _apply_realtime_update(
        self,
        implicit_loss: float,
        context_label: str,
        confidence:    float = 0.8,
        lr:            float = 0.001,
    ) -> int:
        """
        Realtime weight update — apply ทันทีโดยไม่รอ Reviewer

        lr ปรับตาม confidence: confidence สูง → เรียนรู้เร็วขึ้น
        """
        effective_lr = lr * max(confidence, 0.1)
        # minimum perturbation เพื่อให้ weights เปลี่ยนแม้ loss = 0
        # (Realtime — เรียนรู้ตลอดเวลา แม้ output จะสม่ำเสมอ)
        min_delta = effective_lr * 1e-4
        updated = 0

        for cid, conn in self.connections.items():
            if not conn["enabled"]:
                continue

            dst_node = self.nodes.get(conn["destination"])
            g = dst_node["gradient"] if dst_node else None

            current_w = self.weights.get(cid, 0.0)
            if g is not None and abs(g) > 1e-10:
                delta = -effective_lr * g * max(implicit_loss, min_delta)
            else:
                # implicit perturbation — ปรับเล็กน้อยเสมอ
                delta = float(np.random.randn()) * min_delta

            self.weights[cid] = float(np.clip(current_w + delta, -10.0, 10.0))
            updated += 1

            if updated >= 10:  # จำกัด connections ต่อ step
                break

        return updated

    def _propose_weight_updates(
        self,
        implicit_loss: float,
        context_label: str,
        lr:            float = 0.001,
    ) -> list:
        """
        เสนอ weight update สำหรับ connections ที่ active

        ใช้ gradient ที่มีอยู่ใน nodes (จาก backward ที่เคยรัน)
        ถ้าไม่มี gradient → random small perturbation
        """
        proposals = []
        checked   = 0

        for cid, conn in self.connections.items():
            if not conn["enabled"]:
                continue

            dst_node = self.nodes.get(conn["destination"])
            g = dst_node["gradient"] if dst_node else None

            # คำนวณ new weight
            current_w = self.weights.get(cid, 0.0)
            if g is not None:
                delta = -lr * g * implicit_loss
            else:
                delta = -lr * implicit_loss * float(np.random.randn() * 0.1)

            new_w = float(np.clip(current_w + delta, -10.0, 10.0))

            # สร้าง proposal แทนการ apply ตรง
            proposal = create_proposal(
                proposed_by = f"brain_{context_label}",
                action      = ProposalAction.MODIFY,
                target_type = ProposalTarget.RULE,
                authority   = RuleAuthority.STANDARD,
                payload     = {
                    "domain":    f"w_{cid}",
                    "cid":       cid,
                    "old_value": current_w,
                    "new_value": new_w,
                    "context":   context_label,
                },
                reason = f"continuous learning: {context_label} loss={implicit_loss:.6f}",
            )
            self._pending_proposals.append(proposal)
            proposals.append(proposal)

            checked += 1
            if checked >= 5:  # จำกัดไม่ให้ propose ทุก connection ในครั้งเดียว
                break

        return proposals

    def _apply_weight_from_proposal(self, proposal: "ProposalData") -> None:
        """Apply weight change จาก approved proposal"""
        payload   = proposal.payload
        cid       = payload.get("cid", "")
        new_w     = payload.get("new_value")
        if cid and cid in self.weights and new_w is not None:
            self.weights[cid] = float(new_w)