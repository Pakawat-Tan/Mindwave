"""
ReplayLearner.py
เรียนจาก replay memory
Learning from replayed experiences.
"""

import random
from typing import Any, Dict, List


class ReplayLearner:
    """Learning through experience replay."""

    def __init__(self, replay_buffer_size=10000):
        """Initialize replay learner."""
        self.replay_buffer: List[Dict[str, Any]] = []
        self.buffer_size = replay_buffer_size
        self.replay_batch_size = 32

        # priority replay (optional)
        self.priorities: List[float] = []

    # =====================================================
    # Memory storage
    # =====================================================
    def store_experience(self, experience: Dict[str, Any]):
        """
        Store experience in replay buffer.

        experience example:
        {
            "state": ...,
            "action": ...,
            "reward": ...,
            "next_state": ...,
            "loss": float,
            "confidence": float
        }
        """
        if len(self.replay_buffer) >= self.buffer_size:
            self.replay_buffer.pop(0)
            self.priorities.pop(0)

        self.replay_buffer.append(experience)

        # default priority
        priority = abs(experience.get("reward", 0.0)) + 1e-6
        self.priorities.append(priority)

    # =====================================================
    # Sampling
    # =====================================================
    def sample_batch(self, batch_size=None):
        """
        Sample a batch of experiences from buffer.
        """
        if not self.replay_buffer:
            return []

        batch_size = batch_size or self.replay_batch_size
        batch_size = min(batch_size, len(self.replay_buffer))

        # prioritized sampling
        total_priority = sum(self.priorities)
        probs = [p / total_priority for p in self.priorities]

        indices = random.choices(
            range(len(self.replay_buffer)),
            weights=probs,
            k=batch_size,
        )

        return [self.replay_buffer[i] for i in indices]

    # =====================================================
    # Learning
    # =====================================================
    def learn_from_batch(self, batch):
        """
        Learn from a batch of replayed experiences.
        This function does NOT update weights directly.
        It prepares learning signals.
        """
        learning_signals = []

        for exp in batch:
            signal = {
                "state": exp.get("state"),
                "action": exp.get("action"),
                "reward": exp.get("reward"),
                "target": exp.get("next_state"),
                "confidence": exp.get("confidence", 1.0),
            }
            learning_signals.append(signal)

        return learning_signals

    # =====================================================
    # Prioritization
    # =====================================================
    def prioritize_experiences(self):
        """
        Update priorities based on experience quality.
        """
        for i, exp in enumerate(self.replay_buffer):
            loss = exp.get("loss", 0.0)
            reward = exp.get("reward", 0.0)
            confidence = exp.get("confidence", 1.0)

            self.priorities[i] = abs(loss) + abs(reward) * confidence + 1e-6

    # =====================================================
    # Stats
    # =====================================================
    def get_buffer_stats(self):
        """Get statistics about replay buffer."""
        if not self.replay_buffer:
            return {
                "size": 0,
                "avg_reward": 0.0,
                "avg_priority": 0.0,
            }

        rewards = [exp.get("reward", 0.0) for exp in self.replay_buffer]

        return {
            "size": len(self.replay_buffer),
            "avg_reward": sum(rewards) / len(rewards),
            "avg_priority": sum(self.priorities) / len(self.priorities),
        }
