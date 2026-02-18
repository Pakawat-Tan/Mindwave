"""
NeuralTrainer — เทรน BrainStructure neural network

Train nodes/weights ด้วย gradient descent
Track usage และปรับ structure อัตโนมัติ
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mindwave.neural_trainer")


# ─────────────────────────────────────────────────────────────────────────────
# TrainingBatch
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingBatch:
    """ข้อมูล 1 batch สำหรับ training"""
    inputs:  List[float]  # input vector
    targets: List[float]  # target output
    context: str = "general"
    importance: float = 0.7

@dataclass
class TrainingResult:
    """ผลการ train 1 epoch"""
    epoch:       int
    loss:        float
    accuracy:    float
    nodes_used:  int
    weights_updated: int
    elapsed_s:   float


# ─────────────────────────────────────────────────────────────────────────────
# Activation Functions
# ─────────────────────────────────────────────────────────────────────────────

class Activation:
    """Activation functions + derivatives"""
    
    @staticmethod
    def sigmoid(x: float) -> float:
        """Sigmoid: 1 / (1 + e^-x)"""
        return 1.0 / (1.0 + math.exp(-max(-500, min(500, x))))
    
    @staticmethod
    def sigmoid_derivative(y: float) -> float:
        """Derivative: y * (1 - y)"""
        return y * (1.0 - y)
    
    @staticmethod
    def relu(x: float) -> float:
        """ReLU: max(0, x)"""
        return max(0.0, x)
    
    @staticmethod
    def relu_derivative(x: float) -> float:
        """Derivative: 1 if x > 0 else 0"""
        return 1.0 if x > 0 else 0.0
    
    @staticmethod
    def tanh(x: float) -> float:
        """Tanh: (e^x - e^-x) / (e^x + e^-x)"""
        return math.tanh(max(-500, min(500, x)))
    
    @staticmethod
    def tanh_derivative(y: float) -> float:
        """Derivative: 1 - y^2"""
        return 1.0 - y * y


# ─────────────────────────────────────────────────────────────────────────────
# NeuralTrainer
# ─────────────────────────────────────────────────────────────────────────────

class NeuralTrainer:
    """
    Train BrainStructure neural network
    
    Features:
    - Forward/Backward propagation
    - Gradient descent
    - Usage tracking
    - Auto structure adjustment
    """
    
    def __init__(
        self,
        brain_struct: Any,  # BrainStructure
        learning_rate: float = 0.01,
        activation: str = "sigmoid",  # sigmoid / relu / tanh
        enable_evolution: bool = True,   # เปิด auto-evolution
        evolve_every: int = 50,          # evolve ทุก N samples
    ):
        self._brain = brain_struct
        self._lr = learning_rate
        
        # activation function
        if activation == "sigmoid":
            self._activate = Activation.sigmoid
            self._activate_deriv = Activation.sigmoid_derivative
        elif activation == "relu":
            self._activate = Activation.relu
            self._activate_deriv = Activation.relu_derivative
        elif activation == "tanh":
            self._activate = Activation.tanh
            self._activate_deriv = Activation.tanh_derivative
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # cache สำหรับ forward pass
        self._node_outputs: Dict[str, float] = {}
        self._node_inputs: Dict[str, float] = {}
        
        # stats
        self._total_epochs = 0
        self._total_samples = 0
        self._history: List[TrainingResult] = []
        
        # ── Evolution ──────────────────────────────────────────────
        self._enable_evolution = enable_evolution
        self._evolve_every = evolve_every
        self._last_loss = 0.0
        self._loss_history: List[float] = []
        self._evolution_count = 0
        self._evolution_log: List[Dict] = []
    
    # ─────────────────────────────────────────────────────────────────────────
    # Forward Pass
    # ─────────────────────────────────────────────────────────────────────────
    
    def forward(self, inputs: List[float]) -> List[float]:
        """
        Forward propagation
        
        Args:
            inputs: input vector (length = input nodes)
        
        Returns:
            outputs: output vector (length = output nodes)
        """
        self._node_outputs.clear()
        self._node_inputs.clear()
        
        nodes = self._brain.nodes
        conns = self._brain.connections
        biases = self._brain.biases
        
        # 1. Set input layer
        input_nodes = [nid for nid, n in nodes.items() if n.get("role") == "input"]
        for i, nid in enumerate(sorted(input_nodes)):
            if i < len(inputs):
                self._node_outputs[nid] = inputs[i]
                self._node_inputs[nid] = inputs[i]
            else:
                self._node_outputs[nid] = 0.0
                self._node_inputs[nid] = 0.0
        
        # 2. Propagate through layers (layer by layer)
        max_layer = max(n.get("layer", 0) for n in nodes.values())
        for layer in range(1, max_layer + 1):
            layer_nodes = [nid for nid, n in nodes.items() if n.get("layer") == layer]
            
            for nid in layer_nodes:
                # sum incoming connections
                weighted_sum = 0.0
                for cid, conn in conns.items():
                    if conn.get("destination") == nid and conn.get("enabled"):
                        src = conn.get("source")
                        weight = conn.get("weight", 0.0)
                        if src in self._node_outputs:
                            weighted_sum += self._node_outputs[src] * weight
                
                # add bias — รองรับทั้ง float และ dict
                bias_val = biases.get(nid, 0.0)
                if isinstance(bias_val, dict):
                    # format: {"value": float}
                    bias = bias_val.get("value", 0.0)
                else:
                    # format: float
                    bias = bias_val
                weighted_sum += bias
                
                # activation
                self._node_inputs[nid] = weighted_sum
                self._node_outputs[nid] = self._activate(weighted_sum)
        
        # 3. Collect output layer
        output_nodes = [nid for nid, n in nodes.items() if n.get("role") == "output"]
        outputs = [self._node_outputs.get(nid, 0.0) for nid in sorted(output_nodes)]
        
        return outputs
    
    # ─────────────────────────────────────────────────────────────────────────
    # Backward Pass
    # ─────────────────────────────────────────────────────────────────────────
    
    def backward(self, targets: List[float]) -> Dict[str, float]:
        """
        Backward propagation — คำนวณ gradients
        
        Args:
            targets: target output vector
        
        Returns:
            node_deltas: {node_id: delta} สำหรับทุก node
        """
        nodes = self._brain.nodes
        conns = self._brain.connections
        
        node_deltas: Dict[str, float] = {}
        
        # 1. Output layer error
        output_nodes = sorted([nid for nid, n in nodes.items() if n.get("role") == "output"])
        for i, nid in enumerate(output_nodes):
            target = targets[i] if i < len(targets) else 0.0
            output = self._node_outputs.get(nid, 0.0)
            error = target - output
            # delta = error * activation_derivative
            node_deltas[nid] = error * self._activate_deriv(output)
        
        # 2. Backpropagate error to hidden/input layers
        max_layer = max(n.get("layer", 0) for n in nodes.values())
        for layer in range(max_layer - 1, -1, -1):
            layer_nodes = [nid for nid, n in nodes.items() if n.get("layer") == layer]
            
            for nid in layer_nodes:
                # sum error from downstream nodes
                error_sum = 0.0
                for cid, conn in conns.items():
                    if conn.get("source") == nid and conn.get("enabled"):
                        dst = conn.get("destination")
                        if dst in node_deltas:
                            weight = conn.get("weight", 0.0)
                            error_sum += node_deltas[dst] * weight
                
                # delta = error_sum * activation_derivative
                output = self._node_outputs.get(nid, 0.0)
                node_deltas[nid] = error_sum * self._activate_deriv(output)
        
        return node_deltas
    
    # ─────────────────────────────────────────────────────────────────────────
    # Update Weights
    # ─────────────────────────────────────────────────────────────────────────
    
    def update_weights(self, node_deltas: Dict[str, float]) -> int:
        """
        Update weights และ biases
        
        Args:
            node_deltas: {node_id: delta} จาก backward()
        
        Returns:
            count: จำนวน weights ที่อัปเดต
        """
        conns = self._brain.connections
        biases = self._brain.biases
        
        count = 0
        
        # 1. Update connection weights
        for cid, conn in conns.items():
            if not conn.get("enabled"):
                continue
            
            src = conn.get("source")
            dst = conn.get("destination")
            
            if dst in node_deltas and src in self._node_outputs:
                delta = node_deltas[dst]
                src_output = self._node_outputs[src]
                
                # gradient descent: weight += lr * delta * src_output
                weight = conn.get("weight", 0.0)
                weight += self._lr * delta * src_output
                conn["weight"] = weight
                count += 1
        
        # 2. Update biases — รองรับทั้ง float และ dict
        for nid, delta in node_deltas.items():
            if nid in biases:
                bias_val = biases[nid]
                if isinstance(bias_val, dict):
                    # format: {"value": float}
                    bias = bias_val.get("value", 0.0)
                    bias += self._lr * delta
                    biases[nid]["value"] = bias
                else:
                    # format: float
                    bias = bias_val
                    bias += self._lr * delta
                    biases[nid] = bias
                count += 1
        
        return count
    
    # ─────────────────────────────────────────────────────────────────────────
    # Train
    # ─────────────────────────────────────────────────────────────────────────
    
    def train_batch(self, batch: TrainingBatch) -> Tuple[float, float]:
        """
        Train 1 sample
        
        Returns:
            (loss, accuracy)
        """
        # forward
        outputs = self.forward(batch.inputs)
        
        # compute loss (MSE)
        loss = sum((o - t)**2 for o, t in zip(outputs, batch.targets)) / len(outputs)
        
        # accuracy (ถ้า outputs, targets เป็น binary/classification)
        # ใช้ threshold 0.5
        correct = sum(
            1 for o, t in zip(outputs, batch.targets)
            if (o > 0.5) == (t > 0.5)
        )
        accuracy = correct / len(outputs) if outputs else 0.0
        
        # backward
        deltas = self.backward(batch.targets)
        
        # update
        self.update_weights(deltas)
        
        # track usage
        self._track_usage()
        
        # ── Evolution Check ────────────────────────────────────────
        self._total_samples += 1
        self._loss_history.append(loss)
        
        if self._enable_evolution and self._total_samples % self._evolve_every == 0:
            evolved = self._try_evolve(loss)
            if evolved:
                logger.info(
                    f"[NeuralTrainer] EVOLVED at sample {self._total_samples} "
                    f"nodes={len(self._brain.nodes)}"
                )
        
        return loss, accuracy
    
    def train_epoch(self, batches: List[TrainingBatch]) -> TrainingResult:
        """
        Train 1 epoch (หลาย batches)
        
        Returns:
            TrainingResult
        """
        t0 = time.time()
        
        total_loss = 0.0
        total_acc = 0.0
        nodes_used_set = set()
        weights_updated = 0
        
        for batch in batches:
            loss, acc = self.train_batch(batch)
            total_loss += loss
            total_acc += acc
            
            # track nodes used
            nodes_used_set.update(self._node_outputs.keys())
        
        self._total_epochs += 1
        self._total_samples += len(batches)
        
        result = TrainingResult(
            epoch=self._total_epochs,
            loss=total_loss / max(1, len(batches)),
            accuracy=total_acc / max(1, len(batches)),
            nodes_used=len(nodes_used_set),
            weights_updated=weights_updated,
            elapsed_s=time.time() - t0,
        )
        
        self._history.append(result)
        logger.info(
            f"[NeuralTrainer] EPOCH {result.epoch} "
            f"loss={result.loss:.4f} acc={result.accuracy:.2f} "
            f"nodes={result.nodes_used}"
        )
        
        return result
    
    # ─────────────────────────────────────────────────────────────────────────
    # Usage Tracking
    # ─────────────────────────────────────────────────────────────────────────
    
    def _track_usage(self):
        """Track node usage หลัง forward pass"""
        nodes = self._brain.nodes
        for nid in self._node_outputs.keys():
            if nid in nodes:
                usage = nodes[nid].get("usage", 0.0)
                nodes[nid]["usage"] = usage + 1.0
    
    # ─────────────────────────────────────────────────────────────────────────
    # Evolution
    # ─────────────────────────────────────────────────────────────────────────
    
    def _try_evolve(self, current_loss: float) -> bool:
        """
        ตัดสินใจว่าควร evolve หรือไม่
        
        Returns:
            True ถ้า evolved, False ถ้าไม่
        """
        # ต้องมีประวัติอย่างน้อย 10 samples
        if len(self._loss_history) < 10:
            return False
        
        # คำนวณ loss trend
        recent_loss = sum(self._loss_history[-10:]) / 10
        loss_trend = current_loss - self._last_loss
        
        # คำนวณ structure
        n_nodes = len(self._brain.nodes)
        n_connections = len([c for c in self._brain.connections.values() if c.get("enabled")])
        
        # ตัดสินใจ intent
        intent = self._decide_evolution_intent(
            loss_trend=loss_trend,
            current_loss=current_loss,
            recent_avg=recent_loss,
            n_nodes=n_nodes,
            n_connections=n_connections,
        )
        
        if intent == "NO_OP":
            self._last_loss = current_loss
            return False
        
        # Snapshot ก่อน evolve
        self._brain.take_snapshot()
        
        try:
            # Apply evolution
            self._brain._apply_intent(intent)
            
            # Log evolution
            self._evolution_count += 1
            self._evolution_log.append({
                "sample": self._total_samples,
                "intent": intent,
                "loss": current_loss,
                "loss_trend": loss_trend,
                "nodes_before": n_nodes,
                "nodes_after": len(self._brain.nodes),
                "connections_before": n_connections,
                "connections_after": len([c for c in self._brain.connections.values() if c.get("enabled")]),
            })
            
            self._last_loss = current_loss
            logger.info(
                f"[NeuralTrainer] EVOLVED intent={intent} "
                f"nodes={n_nodes}→{len(self._brain.nodes)} "
                f"loss={current_loss:.4f} trend={loss_trend:+.4f}"
            )
            return True
            
        except Exception as e:
            logger.error(f"[NeuralTrainer] EVOLUTION_FAILED {intent}: {e} → rollback")
            self._brain.rollback()
            return False
    
    def _decide_evolution_intent(
        self,
        loss_trend: float,
        current_loss: float,
        recent_avg: float,
        n_nodes: int,
        n_connections: int,
    ) -> str:
        """
        ตัดสินใจว่าควร evolve แบบไหน
        
        Strategy:
        - Loss เพิ่ม + nodes น้อย → ADD_NODE (เพิ่มความซับซ้อน)
        - Loss เพิ่มมาก → ADD_CONNECTION (เพิ่มความยืดหยุ่น)
        - Nodes เยอะเกิน → PRUNE_NODE (ลดความซับซ้อน)
        - Connections เยอะเกิน → PRUNE_CONNECTION (ลดโอเวอร์ฟิต)
        - Loss stable แต่สูง → MUTATE_WEIGHT (explore)
        - Loss stable แต่ต่ำ → NO_OP (ดีอยู่แล้ว)
        """
        # Hard limits
        MAX_NODES = 100
        MAX_CONNECTIONS = 500
        MIN_NODES = 10
        
        # ถ้า nodes เยอะเกิน → prune
        if n_nodes > MAX_NODES:
            return "PRUNE_NODE"
        
        # ถ้า connections เยอะเกิน → prune
        if n_connections > MAX_CONNECTIONS:
            return "PRUNE_CONNECTION"
        
        # ถ้า loss เพิ่มอย่างมาก → add capacity
        if loss_trend > 0.05 and n_nodes < 50:
            return "ADD_NODE"
        
        if loss_trend > 0.1:
            return "ADD_CONNECTION"
        
        # ถ้า loss สูงแต่ stable → explore
        if abs(loss_trend) < 0.01 and current_loss > 0.1:
            return "MUTATE_WEIGHT"
        
        # ถ้า loss ลดลงเรื่อยๆ → ไม่ต้องทำอะไร
        if loss_trend < -0.01:
            return "NO_OP"
        
        # ถ้า loss ต่ำมากแล้ว → ไม่ต้องทำอะไร
        if current_loss < 0.05:
            return "NO_OP"
        
        # Default: explore
        return "MUTATE_WEIGHT"

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────
    
    def stats(self) -> Dict[str, Any]:
        """สถิติการเทรน"""
        total_usage = sum(
            n.get("usage", 0.0) for n in self._brain.nodes.values()
        )
        avg_usage = total_usage / max(1, len(self._brain.nodes))
        
        # Loss stats
        avg_loss = sum(self._loss_history) / max(1, len(self._loss_history))
        recent_loss = (
            sum(self._loss_history[-10:]) / 10 
            if len(self._loss_history) >= 10 
            else avg_loss
        )
        
        return {
            "total_epochs": self._total_epochs,
            "total_samples": self._total_samples,
            "learning_rate": self._lr,
            "avg_node_usage": avg_usage,
            "last_loss": self._history[-1].loss if self._history else 0.0,
            "last_accuracy": self._history[-1].accuracy if self._history else 0.0,
            "avg_loss": avg_loss,
            "recent_loss": recent_loss,
            # Evolution stats
            "evolution_enabled": self._enable_evolution,
            "evolution_count": self._evolution_count,
            "evolve_every": self._evolve_every,
            "current_nodes": len(self._brain.nodes),
            "current_connections": len([c for c in self._brain.connections.values() if c.get("enabled")]),
        }
    
    @property
    def history(self) -> List[TrainingResult]:
        return list(self._history)
    
    @property
    def evolution_log(self) -> List[Dict]:
        """ดูประวัติ evolution ทั้งหมด"""
        return list(self._evolution_log)