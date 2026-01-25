"""
GoalTracker.py
ติดตามเป้าหมาย ระยะสั้น/ยาว และความคืบหน้า
Tracks short-term and long-term goals and their progress.
"""

from datetime import datetime
from typing import Dict, List, Literal
import uuid


GoalDuration = Literal["short", "long"]


class GoalTracker:
    """Tracks and manages system goals."""

    def __init__(self):
        """Initialize the goal tracker."""
        self.short_term_goals: List[str] = []
        self.long_term_goals: List[str] = []
        self.progress: Dict[str, dict] = {}

    def add_goal(self, goal: str, duration: GoalDuration = "short") -> str:
        """
        Add a new goal.

        Args:
            goal (str): Goal description
            duration (str): 'short' or 'long'

        Returns:
            str: goal_id
        """
        goal_id = str(uuid.uuid4())

        goal_data = {
            "id": goal_id,
            "description": goal,
            "duration": duration,
            "progress": 0.0,
            "status": "active",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        self.progress[goal_id] = goal_data

        if duration == "short":
            self.short_term_goals.append(goal_id)
        else:
            self.long_term_goals.append(goal_id)

        return goal_id

    def track_progress(self, goal_id: str, value: float) -> None:
        """
        Update progress of a goal.

        Args:
            goal_id (str): Goal identifier
            value (float): Progress value (0.0 - 1.0)
        """
        if goal_id not in self.progress:
            raise ValueError(f"Goal ID {goal_id} not found")

        value = max(0.0, min(1.0, value))
        goal = self.progress[goal_id]

        goal["progress"] = value
        goal["updated_at"] = datetime.utcnow()

        if value >= 1.0:
            goal["status"] = "completed"

    def get_completed_goals(self) -> List[dict]:
        """
        Get list of completed goals.

        Returns:
            List[dict]: Completed goal records
        """
        return [
            goal
            for goal in self.progress.values()
            if goal["status"] == "completed"
        ]

    def get_active_goals(self, duration: GoalDuration | None = None) -> List[dict]:
        """
        Get active goals (optional filter by duration).
        """
        goals = [
            g for g in self.progress.values()
            if g["status"] == "active"
        ]

        if duration:
            goals = [g for g in goals if g["duration"] == duration]

        return goals
