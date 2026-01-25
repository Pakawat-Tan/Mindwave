"""
BrainState.py
สถานะสมอง ณ runtime
Central neural state managing weights, updates, statistics, and structure.
"""

from typing import Dict, Any, Optional
import time
import numpy as np

from .Weights.WeightSet import WeightSet
from .Weights.WeightStats import WeightStats
from .Weights.WeightLinker import WeightLinker
from .Weights.WeightStore import WeightStore


class BrainState:
    """
    Central runtime state of the neural brain.
    All learners must update weights THROUGH this class.
    """

    def __init__(self, storage_path: Optional[str] = None):
        # Core weight system
        self.weight_set = WeightSet()
        self.weight_stats = WeightStats()
        self.weight_linker = WeightLinker()
        self.weight_store = WeightStore(storage_path)

        # Runtime metadata
        self.step_count = 0
        self.last_update_time = None
        self.update_history = []

        # Structural info (used by evolution learner)
        self.structure_version = 1
        self.structure_history = []

    # ==========================
    # Weight lifecycle
    # ==========================

    def initialize_weights(self, shape_dict: Dict[str, tuple], init="xavier"):
        """
        Initialize all weights using given shapes.
        """
        for name, shape in shape_dict.items():
            if init == "xavier":
                limit = np.sqrt(6 / sum(shape))
                values = np.random.uniform(-limit, limit, size=shape)
            elif init == "zeros":
                values = np.zeros(shape)
            else:
                values = np.random.randn(*shape) * 0.01

            self.weight_set.add_weight_matrix(name, shape)
            self.weight_set.set_weight(name, values)

        self.last_update_time = time.time()

    def get_weights(self) -> WeightSet:
        """Return current weight set (read-only use recommended)."""
        return self.weight_set

    # ==========================
    # Learning updates
    # ==========================

    def apply_weight_updates(self, deltas: Dict[str, np.ndarray], source="unknown"):
        """
        Apply weight updates from learners.
        deltas = {weight_name: delta_array}
        """
        old_weights = self.weight_set.clone()

        for name, delta in deltas.items():
            if name not in self.weight_set.weights:
                continue

            self.weight_set.weights[name] += delta
            self.weight_linker.propagate_weight_change(
                weight_id=name,
                new_value=self.weight_set.weights[name]
            )

        # Track statistics
        self.weight_stats.compute_weight_stats(self.weight_set)

        for name in deltas:
            self.weight_stats.track_weight_change(
                name,
                old_weights.weights.get(name),
                self.weight_set.weights.get(name)
            )

        # Update metadata
        self.step_count += 1
        self.last_update_time = time.time()

        self.update_history.append({
            "step": self.step_count,
            "source": source,
            "changed_weights": list(deltas.keys()),
            "timestamp": self.last_update_time
        })

    # ==========================
    # Structural evolution
    # ==========================

    def apply_structure_change(self, description: str, details: Dict[str, Any]):
        """
        Record and apply a structural change (used by EvolutionLearner).
        """
        self.structure_version += 1
        self.structure_history.append({
            "version": self.structure_version,
            "description": description,
            "details": details,
            "timestamp": time.time()
        })

    # ==========================
    # Persistence
    # ==========================

    def save_state(self, name: str):
        """Save current weights and metadata."""
        self.weight_store.save_weights(self.weight_set, name)
        self.weight_store.metadata[name] = {
            "step": self.step_count,
            "structure_version": self.structure_version,
            "timestamp": time.time()
        }

    def load_state(self, name: str):
        """Load weights from storage."""
        self.weight_set = self.weight_store.load_weights(name)
        self.last_update_time = time.time()

    def backup(self, tag: Optional[str] = None):
        """Create backup snapshot."""
        tag = tag or f"backup_step_{self.step_count}"
        self.weight_store.backup_weights(self.weight_set, tag)

    # ==========================
    # Diagnostics
    # ==========================

    def get_state_summary(self) -> Dict[str, Any]:
        """Return high-level summary of brain state."""
        return {
            "step_count": self.step_count,
            "num_weights": len(self.weight_set.weights),
            "total_parameters": self.weight_set.get_total_parameters(),
            "structure_version": self.structure_version,
            "last_update_time": self.last_update_time
        }

    def detect_problems(self) -> Dict[str, Any]:
        """Detect unhealthy neural conditions."""
        return {
            "dead_neurons": self.weight_stats.detect_dead_neurons(),
            "symmetry_issues": self.weight_stats.analyze_weight_symmetry()
        }
