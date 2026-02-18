"""
Activation functions — implemented จริง ใช้งานได้

Supported:
  ReLU, LeakyReLU, GELU, Sigmoid, Tanh,
  Swish, ELU, Linear, Softmax, Exp
"""

from __future__ import annotations
import math
import numpy as np
from numpy.typing import NDArray
from typing import Callable


# ============================================================================
# TYPE
# ============================================================================

ActivationFn = Callable[[float | NDArray], float | NDArray]


# ============================================================================
# IMPLEMENTATIONS
# ============================================================================

def relu(x: float) -> float:
    return max(0.0, x)

def relu_grad(x: float) -> float:
    return 1.0 if x > 0 else 0.0


def leaky_relu(x: float, alpha: float = 0.01) -> float:
    return x if x > 0 else alpha * x

def leaky_relu_grad(x: float, alpha: float = 0.01) -> float:
    return 1.0 if x > 0 else alpha


def gelu(x: float) -> float:
    """Gaussian Error Linear Unit — exact formula"""
    return 0.5 * x * (1.0 + math.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    ))

def gelu_grad(x: float) -> float:
    cdf = 0.5 * (1.0 + math.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    ))
    pdf = math.exp(-0.5 * x ** 2) / math.sqrt(2.0 * math.pi)
    return cdf + x * pdf


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)

def sigmoid_grad(x: float) -> float:
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh_fn(x: float) -> float:
    return math.tanh(x)

def tanh_grad(x: float) -> float:
    return 1.0 - math.tanh(x) ** 2


def swish(x: float) -> float:
    """Swish = x * sigmoid(x)"""
    return x * sigmoid(x)

def swish_grad(x: float) -> float:
    s = sigmoid(x)
    return s + x * s * (1.0 - s)


def elu(x: float, alpha: float = 1.0) -> float:
    return x if x > 0 else alpha * (math.exp(x) - 1.0)

def elu_grad(x: float, alpha: float = 1.0) -> float:
    return 1.0 if x > 0 else alpha * math.exp(x)


def linear(x: float) -> float:
    return x

def linear_grad(x: float) -> float:
    return 1.0


def exp_fn(x: float) -> float:
    """ใช้ใน MDN sigma head — clamp เพื่อป้องกัน overflow"""
    return math.exp(min(x, 80.0))

def exp_grad(x: float) -> float:
    return exp_fn(x)


def softmax(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerically stable softmax"""
    shifted = arr - np.max(arr)
    exps    = np.exp(shifted)
    return exps / np.sum(exps)


# ============================================================================
# REGISTRY
# ============================================================================

class ActivationFunctions:

    _SCALAR: dict[str, tuple[ActivationFn, ActivationFn]] = {
        "ReLU":      (relu,      relu_grad),
        "LeakyReLU": (leaky_relu, leaky_relu_grad),
        "GELU":      (gelu,      gelu_grad),
        "Sigmoid":   (sigmoid,   sigmoid_grad),
        "Tanh":      (tanh_fn,   tanh_grad),
        "Swish":     (swish,     swish_grad),
        "ELU":       (elu,       elu_grad),
        "Linear":    (linear,    linear_grad),
        "exp":       (exp_fn,    exp_grad),
        "softmax":   (None,      None),   # vector-only
    }

    @classmethod
    def get_activation_function(cls, name: str) -> ActivationFn:
        """คืน forward function"""
        if name == "softmax":
            return softmax  # type: ignore
        entry = cls._SCALAR.get(name)
        if entry is None:
            raise ValueError(
                f"[ActivationFunctions] unknown activation '{name}'. "
                f"Available: {list(cls._SCALAR.keys())}"
            )
        return entry[0]

    @classmethod
    def get_gradient_function(cls, name: str) -> ActivationFn:
        """คืน gradient function"""
        if name == "softmax":
            raise NotImplementedError("softmax gradient ใช้ Jacobian — handle ใน loss")
        entry = cls._SCALAR.get(name)
        if entry is None:
            raise ValueError(
                f"[ActivationFunctions] unknown activation '{name}'"
            )
        return entry[1]

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._SCALAR.keys())