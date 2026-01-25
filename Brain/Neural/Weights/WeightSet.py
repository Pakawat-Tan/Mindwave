"""
WeightSet.py
ชุดน้ำหนักหนึ่งสถานะ
A set of weights representing one state.
"""

import numpy as np
from typing import Dict, Any, Optional


class WeightSet:
    """Represents a complete set of weights at one state."""

    def __init__(self, shape_dict: Optional[Dict[str, tuple]] = None):
        """
        Initialize weight set.

        Parameters
        ----------
        shape_dict : Dict[str, tuple], optional
            Mapping weight_name -> shape
        """
        self.weights: Dict[str, np.ndarray] = {}
        self.shape_dict: Dict[str, tuple] = shape_dict or {}
        self.metadata: Dict[str, Any] = {}

        # initialize weights if shape_dict provided
        for name, shape in self.shape_dict.items():
            self.weights[name] = np.zeros(shape, dtype=np.float64)

    # =====================================================
    # Core operations
    # =====================================================
    def add_weight_matrix(self, name: str, shape: tuple) -> None:
        """Add a weight matrix."""
        if name in self.weights:
            raise ValueError(f"Weight '{name}' already exists")

        self.shape_dict[name] = shape
        self.weights[name] = np.zeros(shape, dtype=np.float64)

    def get_weight(self, name: str) -> np.ndarray:
        """Get weight matrix by name."""
        if name not in self.weights:
            raise KeyError(f"Weight '{name}' not found")
        return self.weights[name]

    def set_weight(self, name: str, values: np.ndarray) -> None:
        """Set weight matrix values."""
        if name not in self.weights:
            raise KeyError(f"Weight '{name}' not found")

        expected_shape = self.shape_dict.get(name)
        if expected_shape and values.shape != expected_shape:
            raise ValueError(
                f"Shape mismatch for '{name}': "
                f"expected {expected_shape}, got {values.shape}"
            )

        self.weights[name] = values.astype(np.float64)

    # =====================================================
    # Utilities
    # =====================================================
    def clone(self) -> "WeightSet":
        """Create a deep copy of this weight set."""
        cloned = WeightSet(self.shape_dict.copy())
        for name, w in self.weights.items():
            cloned.weights[name] = np.copy(w)
        cloned.metadata = self.metadata.copy()
        return cloned

    def get_total_parameters(self) -> int:
        """Get total number of parameters."""
        total = 0
        for w in self.weights.values():
            total += int(np.prod(w.shape))
        return total

    # =====================================================
    # Brain integration helpers
    # =====================================================
    def load_from_brain(self, brain) -> None:
        """
        Load weights from Brain.weights (connection-based).
        """
        self.weights.clear()
        self.shape_dict.clear()

        for cid, val in brain.weights.items():
            name = f"W_{cid}"
            self.shape_dict[name] = (1,)
            self.weights[name] = np.array([val], dtype=np.float64)

    def apply_to_brain(self, brain) -> None:
        """
        Apply stored weights back to Brain.weights
        """
        for name, w in self.weights.items():
            if not name.startswith("W_"):
                continue
            cid = name[2:]
            if cid in brain.weights:
                brain.weights[cid] = float(w[0])

    # =====================================================
    # Debug
    # =====================================================
    def summary(self) -> None:
        """Print summary of weight set."""
        print("\n======= WeightSet Summary =======")
        print(f"Total matrices : {len(self.weights)}")
        print(f"Total params   : {self.get_total_parameters()}")
        for name, w in self.weights.items():
            print(f"- {name:<20} shape={w.shape}")
        print("================================\n")
