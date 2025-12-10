from __future__ import annotations

import inspect
import math
from typing import Any, Callable, Tuple

import numpy as np


def normal_cdf(z: float) -> float:
    """Standard Normal CDF Φ(z)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def level_to_conf_int(level: float) -> Tuple[float, float]:
    """Central interval for a Normal-equivalent ±level sigma."""
    lo = normal_cdf(-float(level))
    hi = normal_cdf(+float(level))
    return (lo, hi)


def infer_param_names(func: Callable[..., Any]) -> Tuple[str, ...]:
    """Infer parameter names from a function signature.

    Conventions:
    - first arg is independent variable container (x)
    - remaining positional/keyword parameters are fit parameters

    v1 restriction:
    - no *args/**kwargs in model functions
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) < 2:
        raise TypeError("Model function must have at least (x, p1, ...).")

    bad_kinds = {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
    for p in params:
        if p.kind in bad_kinds:
            raise TypeError("v1: *args/**kwargs are not supported in model functions.")

    names = [p.name for p in params[1:]]
    if len(set(names)) != len(names):
        raise TypeError("Duplicate parameter names in function signature.")
    return tuple(names)


def prod(shape: Tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return int(n)


def flatten_batch(arr: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Flatten batch dimensions of an array shaped batch_shape + (N,).

    Returns
    -------
    arr_flat : ndarray, shape (B, N) where B=prod(batch_shape)
    batch_shape : tuple
        Empty tuple means scalar/single dataset.
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr[None, :], ()
    batch_shape = tuple(arr.shape[:-1])
    B = prod(batch_shape)
    N = arr.shape[-1]
    return arr.reshape(B, N), batch_shape


def unflatten_batch(values: np.ndarray, batch_shape: Tuple[int, ...]) -> np.ndarray:
    """Unflatten arrays with leading dimension B back to batch_shape."""
    values = np.asarray(values)
    if batch_shape == ():
        return values.reshape(())
    return values.reshape(batch_shape + values.shape[1:])


def _jittered_cholesky(cov: np.ndarray, max_tries: int = 6) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    cov = 0.5 * (cov + cov.T)
    diag = np.diag(cov)
    scale = float(np.max(diag)) if diag.size else 1.0
    scale = 1.0 if not np.isfinite(scale) or scale <= 0 else scale

    jitter = 0.0
    for i in range(max_tries):
        try:
            return np.linalg.cholesky(cov + jitter * np.eye(cov.shape[0]))
        except np.linalg.LinAlgError:
            jitter = (10.0 ** (-(max_tries - i))) * 1e-6 * scale + (jitter * 10.0 if jitter else 0.0)

    w, v = np.linalg.eigh(cov)
    w = np.clip(w, 0.0, None)
    return v @ np.diag(np.sqrt(w))


def sample_mvn(mean: np.ndarray, cov: np.ndarray, nsamples: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from MVN(mean, cov) robustly. Returns shape (nsamples, P)."""
    mean = np.asarray(mean, dtype=float)
    cov = np.asarray(cov, dtype=float)
    L = _jittered_cholesky(cov)
    z = rng.normal(size=(nsamples, mean.shape[0]))
    return mean[None, :] + z @ L.T


def is_sequence(x: Any) -> bool:
    return isinstance(x, (list, tuple))


def is_ragged_batch(x: Any, y: Any) -> bool:
    """Heuristic: x and y are sequences of equal length => ragged batch."""
    return is_sequence(x) and is_sequence(y) and len(x) == len(y)


def safe_float(x: Any) -> float:
    """Convert numpy scalar / 0-d array to python float."""
    if isinstance(x, np.ndarray) and x.shape == ():
        return float(x.item())
    return float(x)
