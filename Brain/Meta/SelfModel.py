"""
SelfModel.py
โมเดลตัวตน: ความสามารถ ข้อจำกัด สถานะโดยรวม
Represents the system's self-model including capabilities, limitations, and overall state.
"""

from datetime import datetime


class SelfModel:
    """Model of the system's own capabilities and limitations."""

    def __init__(self):
        """Initialize the self-model."""
        self.capabilities: dict = {}
        self.limitations: dict = {}
        self.state: dict = {
            "confidence": 1.0,
            "stability": 1.0,
            "load": 0.0,
            "mode": "idle",
            "last_updated": datetime.utcnow(),
        }

    # -------------------------
    # Capabilities
    # -------------------------

    def register_capability(self, name: str, description: str, level: float = 1.0):
        """
        Register or update a system capability.
        """
        self.capabilities[name] = {
            "description": description,
            "level": level,
            "updated_at": datetime.utcnow(),
        }

    def get_capabilities(self) -> dict:
        """Get the system's known capabilities."""
        return self.capabilities

    # -------------------------
    # Limitations
    # -------------------------

    def register_limitation(self, name: str, reason: str, severity: float = 1.0):
        """
        Register or update a system limitation.
        """
        self.limitations[name] = {
            "reason": reason,
            "severity": severity,
            "updated_at": datetime.utcnow(),
        }

    def get_limitations(self) -> dict:
        """Get the system's known limitations."""
        return self.limitations

    # -------------------------
    # State
    # -------------------------

    def update_state(self, **kwargs):
        """
        Update internal state values.
        """
        for key, value in kwargs.items():
            self.state[key] = value

        self.state["last_updated"] = datetime.utcnow()

    def get_overall_state(self) -> dict:
        """Get the overall state of the system."""
        return {
            "capability_count": len(self.capabilities),
            "limitation_count": len(self.limitations),
            "state": self.state,
        }

    # -------------------------
    # Meta helpers
    # -------------------------

    def assess_readiness(self) -> float:
        """
        Assess readiness score based on confidence, stability, and load.
        """
        confidence = self.state.get("confidence", 1.0)
        stability = self.state.get("stability", 1.0)
        load = self.state.get("load", 0.0)

        readiness = confidence * stability * (1.0 - load)
        return max(0.0, min(1.0, readiness))
