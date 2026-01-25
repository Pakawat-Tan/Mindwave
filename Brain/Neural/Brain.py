import numpy as np
from datetime import datetime
from numpy.typing import NDArray
from typing import Any, Callable, Dict, List, Literal, Optional, Union
from Brain.enum import NodeSchema, EvolutionContext
from Brain.Neural.Functions.Activation import ActivationFunctions
from Brain.Neural.Functions.LossFunction import LossFunctions

MDNHead = Literal["mdn_pi", "mdn_mu", "mdn_sigma"]

# ===========================================
# Type Schemas
# ===========================================
LossGradOutput = Union[
    NDArray[np.float64],
    Dict[str, NDArray[np.float64]],
]

LossGradFn = Callable[
    [Any, Dict[str, NDArray[np.float64]]],
    LossGradOutput,
]

# ===========================================
# Brain Structure
# ===========================================
class BrainStructure:
    def __init__(
        self,
        *,
        model_type: str = "Regression",
        mdn_components: int = 3,
        mdn_dim: int = 1,
        verbose: bool = True,
    ):
        self.seed: int = np.random.randint(0, 1_000_000)
        np.random.seed(self.seed)

        self.model_type = model_type
        self.mdn_components = mdn_components
        self.mdn_dim = mdn_dim
        self.verbose = verbose

        self.layers: List[int] | None = None
        self.nodes: Dict[str, NodeSchema] = {}
        self.connections: Dict[str, Dict[str, Any]] = {}
        self.weights: Dict[str, float] = {}
        self.biases: Dict[str, float] = {}

        self.metrics: List[str] = []

        # typed loss hooks
        self.loss_fn: Callable[[Any, Dict[str, NDArray[np.float64]]], float]
        self.loss_grad_fn: Callable[
            [Any, Dict[str, NDArray[np.float64]]],
            Union[
                NDArray[np.float64],
                Dict[str, NDArray[np.float64]],
            ],
        ]


    # ===========================================
    # Hyperparameter pools
    # ===========================================
    ACTIVATION_POOL = {
        "ReLU": 0.25,
        "LeakyReLU": 0.15,
        "GELU": 0.15,
        "Sigmoid": 0.10,
        "Tanh": 0.10,
        "Swish": 0.10,
        "ELU": 0.10,
        "Linear": 0.05,
    }

    LOSS_POOL = {
        "MSE": 0.35,
        "MAE": 0.15,
        "BinaryCrossEntropy": 0.15,
        "CategoricalCrossEntropy": 0.10,
        "MDN_NLL": 0.25,
    }

    def adapactivation_function(self) -> str:
        return str(
            np.random.choice(
                list(self.ACTIVATION_POOL.keys()),
                p=list(self.ACTIVATION_POOL.values()),
            )
        )

    def adaploss_function(self) -> str:
        return str(
            np.random.choice(
                list(self.LOSS_POOL.keys()),
                p=list(self.LOSS_POOL.values()),
            )
        )

    # ===========================================
    # Build structure
    # ===========================================
    def build_structure(
        self,
        connection_prob: float = 1.0,
        min_layers: int = 2,
        max_layers: int = 5,
        min_nodes: int = 2,
        max_nodes: int = 16,
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

        prev_nodes: List[str] = []
        node_counter = 0

        for li, n_nodes in enumerate(self.layers):
            curr_nodes: List[str] = []
            is_input = li == 0
            is_output = li == len(self.layers) - 1

            if is_output and self.model_type == "MDN":
                K, D = self.mdn_components, self.mdn_dim
                heads: Dict[MDNHead, tuple[int, Optional[str]]] = {
                    "mdn_pi": (K, "softmax"),
                    "mdn_mu": (K * D, None),
                    "mdn_sigma": (K * D, "exp"),
                }

                for head, (count, act) in heads.items():
                    for _ in range(count):
                        nid = f"L{li}_{head}_{node_counter}"
                        self.nodes[nid] = {
                            "layer": li,
                            "role": "output",
                            "head": head,
                            "activation": act,
                            "value": None,
                            "gradient": None,
                            "usage": 0.0,
                        }
                        self.biases[nid] = float(np.random.randn() * 0.01)
                        curr_nodes.append(nid)
                        node_counter += 1
            else:
                for _ in range(n_nodes):
                    nid = f"L{li}_N{node_counter}"
                    role: Literal["input", "hidden", "output"] = (
                        "input" if is_input else "output" if is_output else "hidden"
                    )
                    act = None if role == "input" else self.adapactivation_function()
                    self.nodes[nid] = {
                        "layer": li,
                        "role": role,
                        "head": None,
                        "activation": act,
                        "value": None,
                        "gradient": None,
                        "usage": 0.0,
                    }
                    self.biases[nid] = float(np.random.randn() * 0.01)
                    curr_nodes.append(nid)
                    node_counter += 1

            for s in prev_nodes:
                for d in curr_nodes:
                    if np.random.rand() <= connection_prob:
                        cid = f"{s}->{d}"
                        self.connections[cid] = {
                            "source": s,
                            "destination": d,
                            "enabled": True,
                        }
                        self.weights[cid] = float(np.random.randn() * 0.01)

            prev_nodes = curr_nodes

    # ===========================================
    # Forward / Backward
    # ===========================================
    def forward(self) -> None:
        by_layer: Dict[int, List[str]] = {}
        for nid, n in self.nodes.items():
            by_layer.setdefault(n["layer"], []).append(nid)

        for layer in sorted(by_layer):
            for nid in by_layer[layer]:
                node = self.nodes[nid]
                if node["role"] == "input":
                    continue

                total = 0.0
                for cid, c in self.connections.items():
                    if not c["enabled"] or c["destination"] != nid:
                        continue
                    src = c["source"]
                    val = self.nodes[src]["value"]
                    assert val is not None
                    total += self.weights[cid] * val

                total += self.biases[nid]
                act_name = node["activation"]
                assert act_name is not None
                act = ActivationFunctions.get_activation_function(act_name)
                node["value"] = act(total)
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

                    src = c["source"]
                    contrib: float = g * self.weights[cid]
                    src_node = self.nodes[src]

                    prev = src_node["gradient"]
                    src_node["gradient"] = contrib if prev is None else prev + contrib
                    
    # ===========================================
    # Output collection (MDN-aware)
    # ===========================================
    def collect_outputs(
        self,
    ) -> tuple[Dict[str, NDArray[np.float64]], Dict[str, List[str]]]:

        values: Dict[str, List[float]] = {}
        index_map: Dict[str, List[str]] = {}

        for nid, n in self.nodes.items():
            if n["role"] != "output":
                continue

            val = n["value"]
            if val is None:
                raise RuntimeError(f"Output node {nid} has no value")

            key = n["head"] if n["head"] is not None else "default"
            values.setdefault(key, []).append(val)
            index_map.setdefault(key, []).append(nid)

        return (
            {k: np.asarray(v, dtype=np.float64) for k, v in values.items()},
            index_map,
        )

    # ===========================================
    # Backprop wrapper
    # ===========================================
    def backpropagation(self, y_true: Any, lr: float) -> float:
        outputs, index_map = self.collect_outputs()
        loss = self.loss_fn(y_true, outputs)
        grad_result = self.loss_grad_fn(y_true, outputs)
        grads: Dict[str, NDArray[np.float64]] = grad_result if isinstance(grad_result, dict) else {"output": grad_result}

        for key, node_ids in index_map.items():
            grad_vec = grads[key]
            for i, nid in enumerate(node_ids):
                self.nodes[nid]["gradient"] = float(grad_vec[i])

        self.backward()

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
    
    # ===========================================
    # Compile / Train
    # ===========================================
    def compile(self) -> None:
        self.loss_name = (
            "MDN_NLL" if self.model_type == "MDN" else self.adaploss_function()
        )
        self.loss_fn = LossFunctions.get_loss_function(self.loss_name)
        self.loss_grad_fn = LossFunctions.get_loss_gradient(self.loss_name)
        self.compiled_at = datetime.now()

    def train(
        self,
        x_train: NDArray[np.float64],
        y_train: NDArray[Any],
        *,
        epochs: int,
        lr: float,
    ) -> None:

        inputs = [nid for nid, n in self.nodes.items() if n["role"] == "input"]
        n_samples = x_train.shape[0]

        for ep in range(epochs):
            total_loss = 0.0
            for i in range(n_samples):
                for nid in self.nodes:
                    if self.nodes[nid]["role"] != "input":
                        self.nodes[nid]["value"] = None

                for j, nid in enumerate(inputs):
                    self.nodes[nid]["value"] = float(x_train[i, j])

                self.forward()
                total_loss += self.backpropagation(y_train[i], lr)

            if self.verbose:
                print(f"[Epoch {ep+1}/{epochs}] loss={total_loss/n_samples:.6f}")


    # ===========================================
    # Evolve the brain structure.
    # ===========================================
    def add_node(self) -> None:
        if not self.connections:
            return

        cid = np.random.choice(list(self.connections.keys()))
        conn = self.connections[cid]

        if not conn["enabled"]:
            return

        src = conn["source"]
        dst = conn["destination"]

        new_layer = self.nodes[src]["layer"] + 1
        nid = f"L{new_layer}_N{len(self.nodes)}"

        self.nodes[nid] = {
            "layer": new_layer,
            "role": "hidden",
            "head": None,
            "activation": self.adapactivation_function(),
            "value": None,
            "gradient": None,
            "usage": 0.0,
        }
        self.biases[nid] = 0.0

        # disable old connection
        conn["enabled"] = False

        # add new connections
        c1 = f"{src}->{nid}"
        c2 = f"{nid}->{dst}"

        self.connections[c1] = {"source": src, "destination": nid, "enabled": True}
        self.connections[c2] = {"source": nid, "destination": dst, "enabled": True}

        self.weights[c1] = 1.0
        self.weights[c2] = self.weights[cid]


    def add_connection(self) -> None:
        nodes = list(self.nodes.keys())
        for _ in range(10):
            src, dst = np.random.choice(nodes, 2, replace=False)

            if self.nodes[src]["layer"] >= self.nodes[dst]["layer"]:
                continue

            cid = f"{src}->{dst}"
            if cid in self.connections:
                continue

            self.connections[cid] = {
                "source": src,
                "destination": dst,
                "enabled": True,
            }
            self.weights[cid] = float(np.random.randn() * 0.01)
            return

    def prune_node(self) -> None:
        candidates = [
            nid for nid, n in self.nodes.items()
            if n["role"] == "hidden"
        ]
        if not candidates:
            return

        nid = np.random.choice(candidates)

        # remove connections
        for cid in list(self.connections.keys()):
            c = self.connections[cid]
            if c["source"] == nid or c["destination"] == nid:
                self.connections.pop(cid)
                self.weights.pop(cid, None)

        self.nodes.pop(nid)
        self.biases.pop(nid, None)
    
    def add_layer(self) -> None:
        max_layer = max(n["layer"] for n in self.nodes.values())
        insert_at = np.random.randint(1, max_layer)

        for n in self.nodes.values():
            if n["layer"] >= insert_at:
                n["layer"] += 1

        for _ in range(np.random.randint(2, 6)):
            nid = f"L{insert_at}_N{len(self.nodes)}"
            self.nodes[nid] = {
                "layer": insert_at,
                "role": "hidden",
                "head": None,
                "activation": self.adapactivation_function(),
                "value": None,
                "gradient": None,
                "usage": 0.0,
            }
            self.biases[nid] = 0.0

    def prune_layer(self) -> None:
        layers: Dict[int, List[str]] = {}
        for nid, n in self.nodes.items():
            layers.setdefault(n["layer"], []).append(nid)

        hidden_layers: List[int] = [
            l for l, nodes in layers.items()
            if all(self.nodes[n]["role"] == "hidden" for n in nodes)
        ]

        if not hidden_layers:
            return

        layer: int = np.random.choice(hidden_layers)

        for nid in layers[layer]:
            self.prune_node()

    
    def prune_connection(self) -> None:
        enabled = [k for k, c in self.connections.items() if c["enabled"]]
        if not enabled:
            return

        cid = np.random.choice(enabled)
        self.connections[cid]["enabled"] = False

    
    def evolution(
        self,
        rule_engine: Any,
        *,
        loss: float,
        prev_loss: float,
    ) -> None:

        ctx: EvolutionContext = {
            "loss": loss,
            "loss_trend": loss - prev_loss,
            "num_nodes": len(self.nodes),
            "num_connections": sum(
                1 for c in self.connections.values() if c["enabled"]
            ),
            "usage": {nid: n["usage"] for nid, n in self.nodes.items()},
            "model_type": self.model_type,
        }

        intent = rule_engine.decide(ctx)

        match intent:
            case "ADD_NODE":
                self.add_node()
            case "ADD_CONNECTION":
                self.add_connection()
            case "PRUNE_NODE":
                self.prune_node()
            case "PRUNE_CONNECTION":
                self.prune_connection()
            case "ADD_LAYER":
                self.add_layer()
            case "PRUNE_LAYER":
                self.prune_layer()
            case "NO_OP":
                pass
            case _:
                pass
            
    # ===========================================
    # Mutate the brain structure.
    # ===========================================
    def mutate_node(self) -> None:
        import random
        candidates: List[NodeSchema] = [
            n for n in self.nodes.values()
            if n["role"] == "hidden" and n["activation"] is not None
        ]
        if not candidates:
            return
        node: NodeSchema = random.choice(candidates)
        node["activation"] = self.adapactivation_function()
    
    def mutate_layer(self) -> None:
        hidden_layers = {
            n["layer"]
            for n in self.nodes.values()
            if n["role"] == "hidden"
        }
        if not hidden_layers:
            return

        layer = int(np.random.choice(list(hidden_layers)))
        nodes_in_layer = [
            nid for nid, n in self.nodes.items()
            if n["layer"] == layer and n["role"] == "hidden"
        ]
        if len(nodes_in_layer) > 1 and np.random.rand() < 0.5:
            self.prune_node()
        else:
            self.add_node()

    def mutate_weight(self) -> None:
        if not self.weights:
            return
        cid = np.random.choice(list(self.weights.keys()))
        self.weights[cid] += float(np.random.randn() * 0.01)
    
    def mutate_bias(self) -> None:
        if not self.biases:
            return
        nid = np.random.choice(list(self.biases.keys()))
        self.biases[nid] += float(np.random.randn() * 0.01)

    # ===========================================
    # Summary of the brain structure.
    # ===========================================
    def summary(self) -> None:
        # ----------------------------
        # Basic counts
        # ----------------------------
        nodes = self.nodes
        conns = self.connections

        total_nodes = len(nodes)
        total_connections = sum(1 for c in conns.values() if c["enabled"])
        total_weights = total_connections
        total_biases = len(self.biases)
        total_params = total_weights + total_biases

        # ----------------------------
        # Layer & role counts
        # ----------------------------
        layer_set: set[int] = set()
        role_count = {"input": 0, "hidden": 0, "output": 0}

        for n in nodes.values():
            layer_set.add(n["layer"])
            role_count[n["role"]] += 1

        total_layers = len(layer_set)

        # ----------------------------
        # Usage stats
        # ----------------------------
        total_usage = sum(n["usage"] for n in nodes.values())
        avg_usage = total_usage / total_nodes if total_nodes > 0 else 0.0

        # ----------------------------
        # Header
        # ----------------------------
        print("\n==================== Brain Summary ====================")
        print(f"Model type          : {getattr(self, 'model_type', 'Unknown')}")
        print(f"Layers              : {total_layers}")
        print(f"Nodes               : {total_nodes}")
        print(f"  ├─ Input           : {role_count['input']}")
        print(f"  ├─ Hidden          : {role_count['hidden']}")
        print(f"  └─ Output          : {role_count['output']}")
        print(f"Active connections  : {total_connections}")
        print(f"Parameters          : {total_params}")
        print(f"  ├─ Weights         : {total_weights}")
        print(f"  └─ Biases          : {total_biases}")
        print(f"Usage (total)       : {total_usage:.2f}")
        print(f"Usage (avg/node)    : {avg_usage:.2f}")
        print("------------------------------------------------------")
        print("Layer Node                  Role       Head         Usage %    Params")
        print("----------------------------------------------------------------------")

        # ----------------------------
        # Per-node table
        # ----------------------------
        for nid, n in sorted(nodes.items(), key=lambda x: (x[1]["layer"], x[0])):
            layer = n["layer"]
            role = n["role"]
            head = n.get("head", None)
            usage_pct = (n["usage"] / total_usage * 100.0) if total_usage > 0 else 0.0

            # params per node = incoming weights + bias
            param_count = 1  # bias
            for c in conns.values():
                if c["enabled"] and c["destination"] == nid:
                    param_count += 1

            print(
                f"{layer:<5} "
                f"{nid:<22} "
                f"{role:<10} "
                f"{str(head):<12} "
                f"{usage_pct:>7.2f}% "
                f"{param_count:>9}"
            )

        # ----------------------------
        # Footer totals
        # ----------------------------
        print("======================================================")
        print("════════════════════════════════════════════════════════════")
        print(f"Total Params              : {total_params}")
        print(f"Total Trainable Params    : {total_params}")
        print(f"Total Non-trainable Params: 0")
        print("════════════════════════════════════════════════════════════\n")

    
    def visualize(self) -> None:
        print("[visualize] Not implemented (graph visualization hook)")

    # ===========================================
    # Usage statistics.
    # ===========================================
    def get_usage(self) -> Dict[str, float]:
        return {nid: n["usage"] for nid, n in self.nodes.items()}

    def clear_usage(self) -> None:
        for n in self.nodes.values():
            n["usage"] = 0.0
