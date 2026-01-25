"""
EvolutionLearner.py
ปรับโครงสร้างสมอง
Evolves the brain structure itself.
"""

import random
from datetime import datetime


class EvolutionLearner:
    """Evolves neural network structure and architecture."""

    def __init__(self):
        """Initialize evolution learner."""
        self.population = []          # list of architectures
        self.fitness_history = []     # fitness over time
        self.mutation_rate = 0.1
        self.current_structure = {
            "layers": {},
            "connections": set()
        }

    # =====================================================
    # Structural proposal
    # =====================================================
    def propose_structural_change(self):
        """
        Propose a structural mutation.
        Returned object should be evaluated by rules.
        """
        actions = ["add_neuron", "remove_neuron", "modify_connection"]
        action = random.choice(actions)

        proposal = {
            "type": action,
            "timestamp": datetime.utcnow().isoformat(),
            "details": {}
        }

        if action == "add_neuron":
            layer_id = random.choice(list(self.current_structure["layers"].keys()))
            proposal["details"] = {"layer_id": layer_id}

        elif action == "remove_neuron":
            neuron_id = self._random_neuron()
            proposal["details"] = {"neuron_id": neuron_id}

        elif action == "modify_connection":
            proposal["details"] = {
                "from": self._random_neuron(),
                "to": self._random_neuron(),
                "action": random.choice(["add", "remove"])
            }

        return proposal

    # =====================================================
    # Structural operations
    # =====================================================
    def add_neuron(self, layer_id):
        """Add a neuron to the network."""
        neuron_id = f"{layer_id}_n{len(self.current_structure['layers'].get(layer_id, []))}"

        self.current_structure.setdefault("layers", {}).setdefault(layer_id, []).append(neuron_id)
        return neuron_id

    def remove_neuron(self, neuron_id):
        """Remove a neuron from the network."""
        for layer, neurons in self.current_structure["layers"].items():
            if neuron_id in neurons:
                neurons.remove(neuron_id)

        # Remove associated connections
        self.current_structure["connections"] = {
            c for c in self.current_structure["connections"]
            if neuron_id not in c
        }
        return True

    def modify_connections(self, from_neuron, to_neuron, action="add"):
        """Modify connections between neurons."""
        conn = (from_neuron, to_neuron)

        if action == "add":
            self.current_structure["connections"].add(conn)
        elif action == "remove":
            self.current_structure["connections"].discard(conn)

        return conn

    # =====================================================
    # Fitness evaluation
    # =====================================================
    def evaluate_structural_fitness(self, metrics=None):
        """
        Evaluate fitness of current structure.

        metrics example:
        {
            "accuracy": 0.82,
            "loss": 0.3,
            "energy_cost": 0.1
        }
        """
        if not metrics:
            return 0.0

        fitness = (
            metrics.get("accuracy", 0.0)
            - metrics.get("loss", 0.0)
            - metrics.get("energy_cost", 0.0)
        )

        record = {
            "fitness": fitness,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics
        }

        self.fitness_history.append(record)
        return fitness

    # =====================================================
    # Genetic operations
    # =====================================================
    def crossover_architectures(self, arch1, arch2):
        """
        Combine two architectures into a new one.
        """
        new_arch = {
            "layers": {},
            "connections": set()
        }

        for layer in set(arch1["layers"]) | set(arch2["layers"]):
            neurons1 = arch1["layers"].get(layer, [])
            neurons2 = arch2["layers"].get(layer, [])
            new_arch["layers"][layer] = random.choice([neurons1, neurons2]).copy()

        new_arch["connections"] = set(
            random.sample(
                list(arch1["connections"] | arch2["connections"]),
                k=min(len(arch1["connections"]), len(arch2["connections"]))
            )
        )

        return new_arch

    # =====================================================
    # Utilities
    # =====================================================
    def _random_neuron(self):
        """Select a random neuron id."""
        all_neurons = [
            n for neurons in self.current_structure["layers"].values()
            for n in neurons
        ]
        return random.choice(all_neurons) if all_neurons else None
