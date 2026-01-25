"""
WeightLinker.py
mapping น้ำหนัก ↔ connection
Maps weights to network connections.
"""

from typing import Dict, Optional


class WeightLinker:
    """Maps weights to network connections."""

    def __init__(self):
        """Initialize weight linker."""
        # weight_id -> connection_id
        self.weight_map: Dict[str, str] = {}

        # connection_id -> weight_id
        self.connection_map: Dict[str, str] = {}

        # actual values (optional cache)
        self.weight_values: Dict[str, float] = {}

    # =====================================================
    # Core linking
    # =====================================================
    def link_weight_to_connection(
        self,
        weight_id: str,
        connection_id: str,
        *,
        initial_value: Optional[float] = None
    ) -> None:
        """Link a weight to a connection."""
        self.weight_map[weight_id] = connection_id
        self.connection_map[connection_id] = weight_id

        if initial_value is not None:
            self.weight_values[weight_id] = float(initial_value)

    def unlink_connection(self, connection_id: str) -> None:
        """Remove link for a connection."""
        weight_id = self.connection_map.pop(connection_id, None)
        if weight_id:
            self.weight_map.pop(weight_id, None)
            self.weight_values.pop(weight_id, None)

    # =====================================================
    # Query
    # =====================================================
    def get_weight_for_connection(self, connection_id: str) -> Optional[str]:
        """Get weight_id for a connection."""
        return self.connection_map.get(connection_id)

    def get_connection_for_weight(self, weight_id: str) -> Optional[str]:
        """Get connection_id for a weight."""
        return self.weight_map.get(weight_id)

    def get_weight_value(self, weight_id: str) -> Optional[float]:
        """Get cached weight value."""
        return self.weight_values.get(weight_id)

    # =====================================================
    # Update & propagation
    # =====================================================
    def propagate_weight_change(
        self,
        weight_id: str,
        new_value: float,
        *,
        brain=None
    ) -> None:
        """
        Propagate weight change to Brain if provided.
        """
        self.weight_values[weight_id] = float(new_value)

        connection_id = self.weight_map.get(weight_id)
        if connection_id is None:
            return

        # Optional direct Brain update
        if brain is not None and hasattr(brain, "weights"):
            if connection_id in brain.weights:
                brain.weights[connection_id] = float(new_value)

    # =====================================================
    # Debug / Visualization
    # =====================================================
    def visualize_weight_map(self) -> None:
        """Visualize weight to connection mappings."""
        print("\n====== Weight ↔ Connection Map ======")
        for weight_id, conn_id in self.weight_map.items():
            val = self.weight_values.get(weight_id, "N/A")
            print(f"{weight_id:<15} → {conn_id:<25} | value={val}")
        print("====================================\n")

    # =====================================================
    # Bulk utilities
    # =====================================================
    def sync_from_brain(self, brain) -> None:
        """
        Initialize linker from Brain.weights
        """
        self.weight_map.clear()
        self.connection_map.clear()
        self.weight_values.clear()

        for cid, val in brain.weights.items():
            wid = f"W_{cid}"
            self.link_weight_to_connection(
                weight_id=wid,
                connection_id=cid,
                initial_value=val
            )
