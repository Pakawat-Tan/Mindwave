import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable, Dict, cast

# ==================================================
# Type aliases
# ==================================================

LossFn = Callable[..., float]
LossGradFn = Callable[..., NDArray[np.float64]]

class LossFunctions:
    """
    Collection of loss functions.

    Design principles
    -----------------
    - Forward-only (no gradient / backprop here)
    - Supports scalar and numpy.ndarray
    - BrainStructure only references by name
    - Trainer / Controller decides how to use
    """

    # ==================================================
    # Adaptive loss pool (for evolution / exploration)
    # ==================================================

    LOSS_POOL: Dict[str, float] = {
        "MSE": 0.20,
        "MAE": 0.15,
        "Huber": 0.10,
        "BinaryCrossEntropy": 0.15,
        "CategoricalCrossEntropy": 0.15,
        "KLDiv": 0.05,
        "MDN_NLL": 0.15,
        "Zero": 0.05,
    }

    # ==================================================
    # Loss selection utilities
    # ==================================================
    @classmethod
    def adaptive_loss(cls) -> str:
        names: list[str] = list(cls.LOSS_POOL.keys())
        weights: list[float] = list(cls.LOSS_POOL.values())

        choice: Any = np.random.choice(names, p=weights)
        return str(choice)

    @classmethod
    def get_loss_function(cls, name: str) -> LossFn:
        fn: Any = getattr(cls, name, None)
        if fn is None:
            raise ValueError(f"Unknown loss function: {name}")
        return cast(LossFn, fn)

    @classmethod
    def get_loss_gradient(cls, name: str) -> LossGradFn:
        grad_fn: Any = getattr(cls, f"{name}_grad", None)
        if grad_fn is None:
            raise ValueError(f"No gradient defined for loss: {name}")
        return cast(LossGradFn, grad_fn)

    # ==================================================
    # Basic regression losses
    # ==================================================
    @staticmethod
    def MSE(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def MAE(y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def Huber(
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
        delta: float = 1.0
    ) -> float:
        error = y_true - y_pred
        abs_error = np.abs(error)
        quadratic = np.minimum(abs_error, delta)
        linear = abs_error - quadratic
        loss = 0.5 * quadratic ** 2 + delta * linear
        return float(np.mean(loss))

    # ==================================================
    # Classification losses
    # ==================================================
    @staticmethod
    def BinaryCrossEntropy(
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:
        eps: float = 1e-9
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        loss = (
            y_true * np.log(y_pred) +
            (1.0 - y_true) * np.log(1.0 - y_pred)
        )
        return float(-np.mean(loss))

    @staticmethod
    def CategoricalCrossEntropy(
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64]
    ) -> float:
        eps: float = 1e-9
        y_pred_clipped: NDArray[np.float64] = np.clip(y_pred, eps, 1.0)
        log_pred: NDArray[np.float64] = np.log(y_pred_clipped)
        loss_vec: NDArray[np.float64] = -np.sum(y_true * log_pred, axis=-1)

        return float(np.mean(loss_vec))

    @staticmethod
    def KLDiv(
        p_true: NDArray[np.float64],
        p_pred: NDArray[np.float64]
    ) -> float:
        eps: float = 1e-9

        p_true_c: NDArray[np.float64] = np.clip(p_true, eps, 1.0)
        p_pred_c: NDArray[np.float64] = np.clip(p_pred, eps, 1.0)
        log_ratio: NDArray[np.float64] = np.log(p_true_c / p_pred_c)
        kl_vec: NDArray[np.float64] = np.sum(p_true_c * log_ratio, axis=-1)

        return float(np.mean(kl_vec))

    # ===================================
    # Sparse Categorical Crossentropy
    # ===================================
    @staticmethod
    def SparseCategoricalCrossentropy(
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.float64]
    ) -> float:
        eps: float = 1e-9
        y_pred = np.clip(y_pred, eps, 1.0 - eps)

        batch_size: int = int(y_true.shape[0])
        batch_indices: NDArray[np.int64] = np.arange(
            batch_size, dtype=np.int64
        )

        correct_class_probs = y_pred[batch_indices, y_true]
        return float(-np.mean(np.log(correct_class_probs)))

    @staticmethod
    def SparseCategoricalCrossentropy_grad(
        y_true: NDArray[np.int64],
        y_pred: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        grad: NDArray[np.float64] = y_pred.copy()
        batch_size: int = int(y_true.shape[0])
        batch_indices: NDArray[np.int64] = np.arange(
            batch_size, dtype=np.int64
        )

        grad[batch_indices, y_true] -= 1.0
        grad /= float(batch_size)

        return grad

    # ==================================================
    # Mixture Density Network (MDN)
    # ==================================================
    @staticmethod
    def MDN_NLL(
        y_true: NDArray[np.float64],
        pi: NDArray[np.float64],
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64]
    ) -> float:
        eps: float = 1e-9

        y_true_exp: NDArray[np.float64] = np.expand_dims(y_true, axis=1)
        sigma = np.clip(sigma, eps, None)
        pi = np.clip(pi, eps, 1.0)

        two_pi: float = 2.0 * float(np.pi)
        norm_const: float = float(np.sqrt(two_pi))
        coeff: NDArray[np.float64] = 1.0 / (norm_const * sigma)
        exponent: NDArray[np.float64] = np.exp(-0.5 * ((y_true_exp - mu) / sigma) ** 2)

        gaussian: NDArray[np.float64] = np.prod(coeff * exponent, axis=-1)
        weighted: NDArray[np.float64] = pi * gaussian

        likelihood: NDArray[np.float64] = np.sum(weighted, axis=1)
        nll: NDArray[np.float64] = -np.log(likelihood + eps)

        return float(np.mean(nll))

    # ==================================================
    # Special / utility losses
    # ==================================================
    @staticmethod
    def Zero(*args: Any, **kwargs: Any) -> float:
        return 0.0
