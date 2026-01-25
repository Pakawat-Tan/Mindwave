import numpy as np
from typing import Dict

class ActivationFunctions:
    """
    Collection of activation functions for Mindwave.

    Design principles
    -----------------
    - Activation only (no backpropagation logic)
    - Stateless functions
    - Supports scalar and numpy.ndarray
    - Designed for runtime usage, adaptive systems, and plugin-based extension

    Conceptual role in Mindwave
    ---------------------------
    Activation functions represent *neuronal behavior*.
    They may be adapted, replaced, or mutated over time
    based on node usage, performance, or meta-learning signals.
    """

    # ======================================================
    # Basic / Identity-like activations
    # ======================================================

    @staticmethod
    def Linear(x: int | float | np.ndarray) -> int | float | np.ndarray:
        """
        Linear (Identity) activation.

        Mathematical form
        -----------------
        f(x) = x

        Cognitive interpretation
        -------------------------
        - No transformation
        - Pure signal relay
        - Often used in input nodes or regression outputs

        Parameters
        ----------
        x : int | float | np.ndarray
            Input value(s)

        Returns
        -------
        int | float | np.ndarray
            Output identical to input
        """
        return x

    @staticmethod
    def Binary(x: int | float | np.ndarray) -> int | np.ndarray:
        """
        Binary step activation.

        Mathematical form
        -----------------
        f(x) = 1 if x >= 0 else 0

        Cognitive interpretation
        -------------------------
        - Hard decision
        - Threshold-based firing neuron
        - Useful for rule-like or symbolic behavior

        Parameters
        ----------
        x : int | float | np.ndarray
            Input value(s)

        Returns
        -------
        int | np.ndarray
            Binary output (0 or 1)
        """
        if isinstance(x, np.ndarray):
            return (x >= 0).astype(int)
        return 1 if x >= 0 else 0

    @staticmethod
    def Sign(x: int | float | np.ndarray) -> int | float | np.ndarray:
        """
        Sign activation.

        Mathematical form
        -----------------
        f(x) = -1 if x < 0, 0 if x == 0, 1 if x > 0

        Cognitive interpretation
        -------------------------
        - Direction-sensitive neuron
        - Useful for polarity detection and symbolic transitions

        Parameters
        ----------
        x : int | float | np.ndarray
            Input value(s)

        Returns
        -------
        int | float | np.ndarray
            Sign of input
        """
        return np.sign(x)

    # ======================================================
    # Sigmoid family
    # ======================================================

    @staticmethod
    def Sigmoid(x: int | float | np.ndarray) -> float | np.ndarray:
        """
        Sigmoid (Logistic) activation.

        Mathematical form
        -----------------
        f(x) = 1 / (1 + e^-x)

        Cognitive interpretation
        -------------------------
        - Smooth probabilistic firing
        - Represents confidence or belief strength
        - Common in decision and gating mechanisms

        Parameters
        ----------
        x : int | float | np.ndarray
            Input value(s)

        Returns
        -------
        float | np.ndarray
            Output in range (0, 1)
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def HardSigmoid(x: int | float | np.ndarray) -> float | np.ndarray:
        """
        Hard Sigmoid activation (piecewise linear approximation).

        Mathematical form
        -----------------
        f(x) = clip((x + 1) / 2, 0, 1)

        Cognitive interpretation
        -------------------------
        - Faster, less expressive approximation of Sigmoid
        - Useful for lightweight adaptive systems

        Parameters
        ----------
        x : int | float | np.ndarray
            Input value(s)

        Returns
        -------
        float | np.ndarray
            Output in range [0, 1]
        """
        return np.clip((x + 1) / 2, 0, 1)

    # ======================================================
    # Tanh family
    # ======================================================

    @staticmethod
    def Tanh(x: int | float | np.ndarray) -> float | np.ndarray:
        """
        Hyperbolic tangent activation.

        Mathematical form
        -----------------
        f(x) = tanh(x)

        Cognitive interpretation
        -------------------------
        - Zero-centered activation
        - Balanced excitatory / inhibitory behavior
        - Often used in internal hidden representations

        Parameters
        ----------
        x : int | float | np.ndarray
            Input value(s)

        Returns
        -------
        float | np.ndarray
            Output in range (-1, 1)
        """
        return np.tanh(x)

    @staticmethod
    def HardTanh(x: int | float | np.ndarray) -> int | float | np.ndarray:
        """
        Hard Tanh activation.

        Mathematical form
        -----------------
        f(x) = clip(x, -1, 1)

        Cognitive interpretation
        -------------------------
        - Saturated emotional response
        - Prevents extreme signal amplification

        Parameters
        ----------
        x : int | float | np.ndarray
            Input value(s)

        Returns
        -------
        int | float | np.ndarray
            Output limited to [-1, 1]
        """
        return np.clip(x, -1, 1)

    # ======================================================
    # ReLU family
    # ======================================================

    @staticmethod
    def ReLU(x: int | float | np.ndarray) -> int | float | np.ndarray:
        """
        Rectified Linear Unit (ReLU).

        Mathematical form
        -----------------
        f(x) = max(0, x)

        Cognitive interpretation
        -------------------------
        - Sparse activation
        - Energy-efficient firing
        - Strong feature detector

        Parameters
        ----------
        x : int | float | np.ndarray
            Input value(s)

        Returns
        -------
        int | float | np.ndarray
            Non-negative output
        """
        return np.maximum(0, x)

    @staticmethod
    def LeakyReLU(
        x: int | float | np.ndarray,
        alpha: float = 0.01
    ) -> float | np.ndarray:
        """
        Leaky ReLU activation.

        Mathematical form
        -----------------
        f(x) = x if x >= 0 else alpha * x

        Cognitive interpretation
        -------------------------
        - Avoids dead neurons
        - Maintains weak signal flow for negative inputs

        Parameters
        ----------
        x : int | float | np.ndarray
            Input value(s)
        alpha : float
            Negative slope coefficient

        Returns
        -------
        float | np.ndarray
            Activated output
        """
        return np.where(x >= 0, x, alpha * x)

    # ======================================================
    # Exponential / smooth activations
    # ======================================================

    @staticmethod
    def ELU(
        x: int | float | np.ndarray,
        alpha: float = 1.0
    ) -> float | np.ndarray:
        """
        Exponential Linear Unit (ELU).

        Mathematical form
        -----------------
        f(x) = x if x >= 0 else alpha * (exp(x) - 1)

        Cognitive interpretation
        -------------------------
        - Smooth negative saturation
        - Encourages stable mean activation

        Parameters
        ----------
        x : int | float | np.ndarray
            Input value(s)
        alpha : float
            Saturation value for negative inputs

        Returns
        -------
        float | np.ndarray
            Activated output
        """
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def SoftPlus(x: int | float | np.ndarray) -> float | np.ndarray:
        """
        SoftPlus activation.

        Mathematical form
        -----------------
        f(x) = ln(1 + e^x)

        Cognitive interpretation
        -------------------------
        - Smooth ReLU
        - Represents gradual excitation buildup

        Parameters
        ----------
        x : int | float | np.ndarray
            Input value(s)

        Returns
        -------
        float | np.ndarray
            Positive output
        """
        return np.log1p(np.exp(x))

    @staticmethod
    def SoftSign(x: int | float | np.ndarray) -> float | np.ndarray:
        """
        SoftSign activation.

        Mathematical form
        -----------------
        f(x) = x / (1 + |x|)

        Cognitive interpretation
        -------------------------
        - Gentle saturation
        - Resistant to outliers

        Parameters
        ----------
        x : int | float | np.ndarray
            Input value(s)

        Returns
        -------
        float | np.ndarray
            Output in range (-1, 1)
        """
        return x / (1 + np.abs(x))

    # ======================================================
    # Probabilistic
    # ======================================================

    @staticmethod
    def Softmax(x: np.ndarray) -> np.ndarray:
        """
        Softmax activation.

        Mathematical form
        -----------------
        f(x_i) = exp(x_i) / sum(exp(x))

        Cognitive interpretation
        -------------------------
        - Competitive decision making
        - Attention or class probability distribution

        Parameters
        ----------
        x : np.ndarray
            Input vector or batch of vectors

        Returns
        -------
        np.ndarray
            Probability distribution (sum = 1)
        """
        x = x.astype(float)
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    # ======================================================
    # Statistical utility (non-firing)
    # ======================================================

    @staticmethod
    def Mean(x: int | float | np.ndarray) -> float:
        """
        Compute mean value.

        Cognitive interpretation
        -------------------------
        - Aggregation neuron
        - Represents average belief or signal level
        """
        return float(np.mean(x))

    @staticmethod
    def Variance(x: int | float | np.ndarray) -> float:
        """
        Compute variance.

        Cognitive interpretation
        -------------------------
        - Measures uncertainty or instability
        """
        return float(np.var(x))

    @staticmethod
    def StandardDeviation(x: int | float | np.ndarray) -> float:
        """
        Compute standard deviation.

        Cognitive interpretation
        -------------------------
        - Dispersion awareness
        - Often used in uncertainty-driven adaptation
        """
        return float(np.std(x))
    
    # ======================================================
    # Special Activations
    # ======================================================
    @staticmethod
    def MDN_Gaussian(
        x: np.ndarray,
        num_components: int,
        epsilon: float = 1e-6
    ) -> Dict[str, np.ndarray]:
        """
        Gaussian Mixture Density activation.

        Expected input
        --------------
        x shape = (..., num_components * 3)

        Splits into:
        - pi logits
        - mu
        - sigma logits

        Mathematical form
        -----------------
        pi     = softmax(pi_logits)
        mu     = mu
        sigma  = exp(sigma_logits) + epsilon

        Cognitive interpretation
        -------------------------
        - Neuron emits a *probabilistic belief*
        - Represents uncertainty-aware prediction
        - Used for multi-modal reasoning and ambiguity

        Returns
        -------
        dict with keys:
        - "pi"    : mixture weights
        - "mu"    : means
        - "sigma" : standard deviations
        """

        x = np.asarray(x)
        last_dim = x.shape[-1]

        expected_dim = num_components * 3
        if last_dim != expected_dim:
            raise ValueError(
                f"MDN_Gaussian expects last dim {expected_dim}, got {last_dim}"
            )

        # Split
        pi_logits, mu, sigma_logits = np.split(x, 3, axis=-1)

        # Activations
        pi = ActivationFunctions.Softmax(pi_logits)
        sigma = np.exp(sigma_logits) + epsilon

        return {
            "pi": pi,
            "mu": mu,
            "sigma": sigma
        } # type: ignore

    # ======================================================
    # Dynamic resolver
    # ======================================================

    @staticmethod
    def get_activation_function(name: str):
        """
        Resolve activation function by name.

        Purpose
        -------
        Enables:
        - String-based configuration
        - Adaptive mutation
        - Plugin-style extensibility

        Parameters
        ----------
        name : str
            Activation function name

        Returns
        -------
        Callable
            Corresponding activation function

        Raises
        ------
        ValueError
            If activation name is unknown
        """
        try:
            return getattr(ActivationFunctions, name)
        except AttributeError:
            raise ValueError(f"Unknown activation function: {name}")
