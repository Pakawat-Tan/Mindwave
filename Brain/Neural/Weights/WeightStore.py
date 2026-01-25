"""
WeightStore.py
จัดเก็บ/โหลดน้ำหนัก
Storage and loading of weights.
"""

import os
import json
import numpy as np
from datetime import datetime


class WeightStore:
    """Manages storage and loading of weights."""

    def __init__(self, storage_path=None):
        """Initialize weight store."""
        self.storage_path = storage_path or "weights"
        self.metadata = {}

        os.makedirs(self.storage_path, exist_ok=True)

    # =====================================================
    # Core save / load
    # =====================================================
    def save_weights(self, weight_set, filename):
        """
        Save weights to file (.npz).

        Parameters
        ----------
        weight_set : WeightSet
        filename : str
        """
        path = os.path.join(self.storage_path, filename)

        payload = {
            **weight_set.weights,
            "__metadata__": json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "shape_dict": weight_set.shape_dict,
                "custom_metadata": weight_set.metadata,
            })
        }

        np.savez_compressed(path, **payload)
        return path

    def load_weights(self, filename):
        """
        Load weights from file.

        Returns
        -------
        dict with keys: weights, metadata
        """
        path = os.path.join(self.storage_path, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        data = np.load(path, allow_pickle=True)

        weights = {}
        metadata = {}

        for key in data.files:
            if key == "__metadata__":
                metadata = json.loads(str(data[key]))
            else:
                weights[key] = data[key]

        return {
            "weights": weights,
            "metadata": metadata
        }

    # =====================================================
    # Backup system
    # =====================================================
    def backup_weights(self, weight_set, backup_name=None):
        """
        Create a backup snapshot of weights.
        """
        backup_name = backup_name or f"backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.npz"
        return self.save_weights(weight_set, backup_name)

    def restore_backup(self, backup_name):
        """
        Restore weights from backup.
        """
        return self.load_weights(backup_name)

    # =====================================================
    # Utility
    # =====================================================
    def list_saved_weights(self):
        """List all saved weight files."""
        return [
            f for f in os.listdir(self.storage_path)
            if f.endswith(".npz")
        ]
