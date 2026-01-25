"""
WeightStats.py
วิเคราะห์พฤติกรรมน้ำหนัก
Analyzes weight behavior and statistics.
"""

import numpy as np


class WeightStats:
    """Analyzes weight statistics and behavior."""

    def __init__(self):
        """Initialize weight statistics."""
        self.statistics = {}   # latest stats
        self.history = []      # list of snapshots over time

    # =====================================================
    # Core statistics
    # =====================================================
    def compute_weight_stats(self, weight_set):
        """
        Compute statistics for a WeightSet.

        Parameters
        ----------
        weight_set : WeightSet
        """
        stats = {}

        for name, w in weight_set.weights.items():
            flat = w.flatten()

            stats[name] = {
                "mean": float(np.mean(flat)),
                "std": float(np.std(flat)),
                "min": float(np.min(flat)),
                "max": float(np.max(flat)),
                "l1_norm": float(np.sum(np.abs(flat))),
                "l2_norm": float(np.linalg.norm(flat)),
                "sparsity": float(np.mean(np.abs(flat) < 1e-6)),
                "num_params": flat.size,
            }

        self.statistics = stats
        self.history.append(stats)
        return stats

    # =====================================================
    # Distribution analysis
    # =====================================================
    def get_weight_distribution(self, weight_name, bins=50):
        """
        Get distribution (histogram) of a weight matrix.

        Returns
        -------
        hist : np.ndarray
        bin_edges : np.ndarray
        """
        if weight_name not in self.statistics:
            raise KeyError(f"Weight '{weight_name}' not analyzed yet")

        # distribution should be recomputed from latest snapshot
        for snapshot in reversed(self.history):
            if weight_name in snapshot:
                w_stats = snapshot[weight_name]
                break
        else:
            raise KeyError(f"No history for weight '{weight_name}'")

        # NOTE: actual values must come from weight_set, not stats
        # This method assumes user still has access to WeightSet
        raise RuntimeError(
            "Distribution requires raw weight values. "
            "Call track_weight_change or store raw weights if needed."
        )

    # =====================================================
    # Temporal analysis
    # =====================================================
    def track_weight_change(self, weight_name, old_weights, new_weights):
        """
        Track changes in weights over time.

        Returns
        -------
        dict with change metrics
        """
        if old_weights.shape != new_weights.shape:
            raise ValueError("Weight shape mismatch")

        delta = new_weights - old_weights

        change_stats = {
            "weight": weight_name,
            "mean_change": float(np.mean(delta)),
            "max_change": float(np.max(np.abs(delta))),
            "l2_change": float(np.linalg.norm(delta)),
            "relative_change": float(
                np.linalg.norm(delta) / (np.linalg.norm(old_weights) + 1e-8)
            ),
        }

        self.history.append({"change": change_stats})
        return change_stats

    # =====================================================
    # Structural diagnostics
    # =====================================================
    def detect_dead_neurons(self, weight_set, threshold=0.01):
        """
        Detect neurons with near-zero incoming weights.

        Returns
        -------
        dict: weight_name -> list of dead neuron indices
        """
        dead = {}

        for name, w in weight_set.weights.items():
            if w.ndim < 2:
                continue

            # neuron = row
            neuron_norms = np.linalg.norm(w, axis=1)
            dead_idx = np.where(neuron_norms < threshold)[0]

            if len(dead_idx) > 0:
                dead[name] = dead_idx.tolist()

        return dead

    def analyze_weight_symmetry(self, weight_set, tolerance=1e-3):
        """
        Analyze symmetry in weights (e.g. W ≈ W^T).

        Returns
        -------
        dict: weight_name -> symmetry_score
        """
        symmetry = {}

        for name, w in weight_set.weights.items():
            if w.ndim != 2:
                continue
            if w.shape[0] != w.shape[1]:
                continue

            diff = np.abs(w - w.T)
            score = float(np.mean(diff < tolerance))
            symmetry[name] = score

        return symmetry

    # =====================================================
    # Debug / reporting
    # =====================================================
    def summary(self):
        """Print summary of latest statistics."""
        print("\n====== Weight Statistics ======")
        for name, s in self.statistics.items():
            print(
                f"{name:<20} "
                f"μ={s['mean']:+.4f} "
                f"σ={s['std']:.4f} "
                f"sparse={s['sparsity']:.2f}"
            )
        print("================================\n")
