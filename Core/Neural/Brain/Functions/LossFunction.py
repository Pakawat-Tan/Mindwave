"""
Loss functions — implemented จริง

Supported:
  MSE, MAE, BinaryCrossEntropy,
  CategoricalCrossEntropy, MDN_NLL
"""

from __future__ import annotations
import math
import numpy as np
from numpy.typing import NDArray
from typing import Any, Callable, Dict, Union


LossGradOutput = Union[
    NDArray[np.float64],
    Dict[str, NDArray[np.float64]],
]


# ============================================================================
# SCALAR HELPERS
# ============================================================================

_EPS = 1e-8


# ── MSE ──────────────────────────────────────────────────────────────────────

def mse_loss(y_true: Any, outputs: Dict[str, NDArray]) -> float:
    y_pred = outputs.get("default", list(outputs.values())[0])
    diff   = np.asarray(y_true, dtype=np.float64) - y_pred
    return float(np.mean(diff ** 2))

def mse_grad(y_true: Any, outputs: Dict[str, NDArray]) -> NDArray:
    y_pred = outputs.get("default", list(outputs.values())[0])
    diff   = np.asarray(y_true, dtype=np.float64) - y_pred
    return -2.0 * diff / max(len(diff), 1)


# ── MAE ──────────────────────────────────────────────────────────────────────

def mae_loss(y_true: Any, outputs: Dict[str, NDArray]) -> float:
    y_pred = outputs.get("default", list(outputs.values())[0])
    diff   = np.asarray(y_true, dtype=np.float64) - y_pred
    return float(np.mean(np.abs(diff)))

def mae_grad(y_true: Any, outputs: Dict[str, NDArray]) -> NDArray:
    y_pred = outputs.get("default", list(outputs.values())[0])
    diff   = np.asarray(y_true, dtype=np.float64) - y_pred
    return -np.sign(diff) / max(len(diff), 1)


# ── Binary Cross Entropy ──────────────────────────────────────────────────────

def bce_loss(y_true: Any, outputs: Dict[str, NDArray]) -> float:
    y_pred = np.clip(
        outputs.get("default", list(outputs.values())[0]),
        _EPS, 1.0 - _EPS
    )
    yt = np.asarray(y_true, dtype=np.float64)
    return float(-np.mean(yt * np.log(y_pred) + (1 - yt) * np.log(1 - y_pred)))

def bce_grad(y_true: Any, outputs: Dict[str, NDArray]) -> NDArray:
    y_pred = np.clip(
        outputs.get("default", list(outputs.values())[0]),
        _EPS, 1.0 - _EPS
    )
    yt = np.asarray(y_true, dtype=np.float64)
    return (-yt / y_pred + (1 - yt) / (1 - y_pred)) / max(len(y_pred), 1)


# ── Categorical Cross Entropy ─────────────────────────────────────────────────

def cce_loss(y_true: Any, outputs: Dict[str, NDArray]) -> float:
    y_pred = np.clip(
        outputs.get("default", list(outputs.values())[0]),
        _EPS, 1.0
    )
    yt = np.asarray(y_true, dtype=np.float64)
    return float(-np.sum(yt * np.log(y_pred)))

def cce_grad(y_true: Any, outputs: Dict[str, NDArray]) -> NDArray:
    """softmax + CCE combined gradient = y_pred - y_true"""
    y_pred = outputs.get("default", list(outputs.values())[0])
    yt     = np.asarray(y_true, dtype=np.float64)
    return y_pred - yt


# ── MDN NLL ───────────────────────────────────────────────────────────────────

def mdn_nll_loss(y_true: Any, outputs: Dict[str, NDArray]) -> float:
    """
    Mixture Density Network Negative Log Likelihood

    Expects outputs keys: mdn_pi, mdn_mu, mdn_sigma
    y_true: scalar or 1D array
    """
    pi    = outputs["mdn_pi"]      # mixture weights (softmax applied)
    mu    = outputs["mdn_mu"]      # means
    sigma = np.maximum(outputs["mdn_sigma"], _EPS)  # std devs (exp applied)
    y     = np.asarray(y_true, dtype=np.float64).flatten()

    K = len(pi)
    D = len(mu) // K

    log_probs = []
    for k in range(K):
        mu_k    = mu[k*D:(k+1)*D]
        sig_k   = sigma[k*D:(k+1)*D]
        diff    = y[:D] - mu_k
        exponent = -0.5 * np.sum((diff / sig_k) ** 2)
        log_norm = -np.sum(np.log(sig_k)) - 0.5 * D * math.log(2 * math.pi)
        log_probs.append(math.log(max(pi[k], _EPS)) + log_norm + exponent)

    log_sum = log_probs[0]
    for lp in log_probs[1:]:
        log_sum = math.log(math.exp(log_sum - lp) + 1.0) + lp

    return float(-log_sum)


def mdn_nll_grad(
    y_true: Any,
    outputs: Dict[str, NDArray],
) -> Dict[str, NDArray]:
    """Gradient สำหรับ MDN NLL — คืน dict per head"""
    pi    = outputs["mdn_pi"]
    mu    = outputs["mdn_mu"]
    sigma = np.maximum(outputs["mdn_sigma"], _EPS)
    y     = np.asarray(y_true, dtype=np.float64).flatten()

    K = len(pi)
    D = len(mu) // K

    responsibilities = np.zeros(K)
    for k in range(K):
        mu_k  = mu[k*D:(k+1)*D]
        sig_k = sigma[k*D:(k+1)*D]
        diff  = y[:D] - mu_k
        log_n = (-0.5 * np.sum((diff / sig_k) ** 2)
                 - np.sum(np.log(sig_k))
                 - 0.5 * D * math.log(2 * math.pi))
        responsibilities[k] = pi[k] * math.exp(log_n)

    r_sum = responsibilities.sum() + _EPS
    responsibilities /= r_sum

    d_pi    = pi - responsibilities
    d_mu    = np.zeros_like(mu)
    d_sigma = np.zeros_like(sigma)

    for k in range(K):
        mu_k  = mu[k*D:(k+1)*D]
        sig_k = sigma[k*D:(k+1)*D]
        diff  = y[:D] - mu_k
        rk    = responsibilities[k]
        d_mu   [k*D:(k+1)*D] = rk * (-diff / (sig_k ** 2))
        d_sigma[k*D:(k+1)*D] = rk * (-1.0 / sig_k + diff ** 2 / (sig_k ** 3))

    return {"mdn_pi": d_pi, "mdn_mu": d_mu, "mdn_sigma": d_sigma}


# ============================================================================
# REGISTRY
# ============================================================================

class LossFunctions:

    _REGISTRY: dict[str, tuple[Callable, Callable]] = {
        "MSE":                    (mse_loss,  mse_grad),
        "MAE":                    (mae_loss,  mae_grad),
        "BinaryCrossEntropy":     (bce_loss,  bce_grad),
        "CategoricalCrossEntropy":(cce_loss,  cce_grad),
        "MDN_NLL":                (mdn_nll_loss, mdn_nll_grad),
    }

    @classmethod
    def get_loss_function(cls, name: str) -> Callable:
        entry = cls._REGISTRY.get(name)
        if entry is None:
            raise ValueError(
                f"[LossFunctions] unknown loss '{name}'. "
                f"Available: {list(cls._REGISTRY.keys())}"
            )
        return entry[0]

    @classmethod
    def get_loss_gradient(cls, name: str) -> Callable:
        entry = cls._REGISTRY.get(name)
        if entry is None:
            raise ValueError(
                f"[LossFunctions] unknown loss '{name}'"
            )
        return entry[1]

    @classmethod
    def available(cls) -> list[str]:
        return list(cls._REGISTRY.keys())