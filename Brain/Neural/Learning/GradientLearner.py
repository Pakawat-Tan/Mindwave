"""
GradientLearner.py
backprop ปกติ
Standard backpropagation learning.
"""

import numpy as np


class GradientLearner:
    """Standard gradient-based backpropagation learning."""

    def __init__(self, learning_rate=0.01):
        """Initialize gradient learner."""
        self.learning_rate = learning_rate
        self.velocity = {}      # momentum buffer
        self.momentum = 0.9

    # =====================================================
    # Gradient computation
    # =====================================================
    def compute_gradients(self, loss_grad, weights):
        """
        Compute gradients of loss w.r.t weights.

        Parameters
        ----------
        loss_grad : dict
            gradient of loss for each weight
        weights : dict
            current weight matrices

        Returns
        -------
        dict
            gradients for each weight
        """
        gradients = {}

        for name, w in weights.items():
            if name in loss_grad:
                gradients[name] = loss_grad[name]
            else:
                gradients[name] = np.zeros_like(w)

        return gradients

    # =====================================================
    # Weight update
    # =====================================================
    def update_weights(self, weights, gradients):
        """
        Update weights using gradients + momentum.

        Returns
        -------
        dict
            updated weights
        """
        updated = {}

        for name, w in weights.items():
            grad = gradients.get(name, np.zeros_like(w))

            # Initialize velocity
            if name not in self.velocity:
                self.velocity[name] = np.zeros_like(w)

            # Momentum update
            self.velocity[name] = (
                self.momentum * self.velocity[name]
                - self.learning_rate * grad
            )

            updated[name] = w + self.velocity[name]

        return updated

    # =====================================================
    # Backpropagation
    # =====================================================
    def backpropagate(self, output_gradient, layers):
        """
        Backpropagate gradients through layers.

        Parameters
        ----------
        output_gradient : np.ndarray
            gradient at output layer
        layers : list
            list of layers with forward cache

        Returns
        -------
        dict
            gradients per weight
        """
        grads = {}
        delta = output_gradient

        for layer in reversed(layers):
            if not hasattr(layer, "backward"):
                continue

            delta, layer_grads = layer.backward(delta)

            for name, g in layer_grads.items():
                grads[name] = g

        return grads

    # =====================================================
    # Hyperparameters
    # =====================================================
    def set_learning_rate(self, lr):
        """Set learning rate."""
        self.learning_rate = float(lr)

    # =====================================================
    # Momentum utilities
    # =====================================================
    def add_momentum(self, gradients, previous_gradients):
        """
        Apply momentum to raw gradients.

        Returns
        -------
        dict
            smoothed gradients
        """
        smoothed = {}

        for name, grad in gradients.items():
            prev = previous_gradients.get(name, np.zeros_like(grad))
            smoothed[name] = (
                self.momentum * prev + (1 - self.momentum) * grad
            )

        return smoothed
