from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .util import flatten_batch, is_ragged_batch


@dataclass(frozen=True)
class Dataset:
    x: Any
    format: str  # "normal" | "binomial" | "beta"
    payload: Dict[str, Any]
    y_for_seed: np.ndarray


def prepare_datasets(
    x: Any, data: Any, data_format: str
) -> Tuple[List[Dataset], Tuple[int, ...]]:
    """Normalize user inputs into a list of per-dataset payloads + batch shape."""

    if data_format not in ("normal", "binomial", "beta"):
        raise NotImplementedError(
            "v1 supports data_format in {'normal','binomial','beta'}."
        )

    datasets: List[Dataset] = []
    batch_shape: Tuple[int, ...] = ()

    if is_ragged_batch(x, data):
        for xi, di in zip(x, data):
            datasets.append(_one_dataset(xi, di, data_format))
        return datasets, (len(datasets),)

    # Common-x batching: detect by array dimensionality of the primary observation.
    if data_format == "normal":
        yobs, sigma = _infer_gaussian_payload(data)
        yobs = np.asarray(yobs, dtype=float)

        if yobs.ndim == 1:
            datasets.append(
                Dataset(
                    x=x,
                    format="normal",
                    payload={"y": yobs, "sigma": sigma},
                    y_for_seed=np.asarray(yobs, dtype=float),
                )
            )
            return datasets, ()

        yflat, batch_shape = flatten_batch(yobs)  # (B,N)
        B = int(yflat.shape[0])

        # Broadcast sigma to match yobs, then flatten per batch.
        if sigma is None:
            sflat = [None] * B
        else:
            sarr = np.asarray(sigma, dtype=float)
            if sarr.shape == ():
                sflat = [float(sarr)] * B
            else:
                sb = np.broadcast_to(sarr, yobs.shape)
                sb_flat, _ = flatten_batch(sb)
                sflat = [sb_flat[i] for i in range(B)]

        for i in range(B):
            yi = np.asarray(yflat[i], dtype=float)
            datasets.append(
                Dataset(
                    x=x,
                    format="normal",
                    payload={"y": yi, "sigma": sflat[i]},
                    y_for_seed=yi,
                )
            )

        return datasets, batch_shape

    if data_format == "binomial":
        n, k = _infer_binomial_payload(data)
        k = np.asarray(k, dtype=float)

        if k.ndim == 1:
            y_seed = _safe_frac(k, n)
            datasets.append(
                Dataset(
                    x=x,
                    format="binomial",
                    payload={"n": n, "k": k},
                    y_for_seed=y_seed,
                )
            )
            return datasets, ()

        kflat, batch_shape = flatten_batch(k)
        B = int(kflat.shape[0])

        n_arr = np.asarray(n, dtype=float)
        if n_arr.shape == ():
            nflat = [float(n_arr)] * B
        else:
            nb = np.broadcast_to(n_arr, k.shape)
            nb_flat, _ = flatten_batch(nb)
            nflat = [nb_flat[i] for i in range(B)]

        for i in range(B):
            ni = np.asarray(nflat[i], dtype=float)
            ki = np.asarray(kflat[i], dtype=float)
            datasets.append(
                Dataset(
                    x=x,
                    format="binomial",
                    payload={"n": ni, "k": ki},
                    y_for_seed=_safe_frac(ki, ni),
                )
            )

        return datasets, batch_shape

    # beta
    a, b = _infer_beta_payload(data)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.ndim == 1:
        y_seed = _safe_frac(a, a + b)
        datasets.append(
            Dataset(
                x=x,
                format="beta",
                payload={"alpha": a, "beta": b},
                y_for_seed=y_seed,
            )
        )
        return datasets, ()

    aflat, batch_shape = flatten_batch(a)
    bb = np.broadcast_to(b, a.shape)
    bflat, _ = flatten_batch(bb)
    B = int(aflat.shape[0])

    for i in range(B):
        ai = np.asarray(aflat[i], dtype=float)
        bi = np.asarray(bflat[i], dtype=float)
        y_seed = _safe_frac(ai, ai + bi)
        datasets.append(
            Dataset(
                x=x,
                format="beta",
                payload={"alpha": ai, "beta": bi},
                y_for_seed=y_seed,
            )
        )

    return datasets, batch_shape


def _one_dataset(x: Any, data: Any, fmt: str) -> Dataset:
    if fmt == "normal":
        y, sigma = _infer_gaussian_payload(data)
        y = np.asarray(y, dtype=float)
        return Dataset(x=x, format="normal", payload={"y": y, "sigma": sigma}, y_for_seed=y)

    if fmt == "binomial":
        n, k = _infer_binomial_payload(data)
        y_seed = _safe_frac(k, n)
        return Dataset(x=x, format="binomial", payload={"n": n, "k": k}, y_for_seed=y_seed)

    a, b = _infer_beta_payload(data)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    y_seed = _safe_frac(a, a + b)
    return Dataset(x=x, format="beta", payload={"alpha": a, "beta": b}, y_for_seed=y_seed)


def _safe_frac(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, 0.0)


def _infer_gaussian_payload(y: Any):
    # y -> unweighted
    # (y, sigma) -> symmetric absolute errors
    # (y, sigma_low, sigma_high) -> approximate to mean sigma
    if isinstance(y, (tuple, list)) and len(y) == 2:
        yobs, yerr = y
        return np.asarray(yobs), yerr
    if isinstance(y, (tuple, list)) and len(y) == 3:
        yobs, ylo, yhi = y
        sigma = 0.5 * (np.asarray(ylo) + np.asarray(yhi))
        return np.asarray(yobs), sigma
    return np.asarray(y), None


def _infer_binomial_payload(y: Any):
    if not (isinstance(y, (tuple, list)) and len(y) == 2):
        raise TypeError("binomial data expects y=(n_samples, n_successes).")

    n, k = y
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)

    if n.shape == () and k.shape != ():
        n = np.broadcast_to(n, k.shape)

    if n.shape != k.shape:
        raise ValueError(f"n_samples shape {n.shape} != n_successes shape {k.shape}")

    if np.any(k < 0) or np.any(n < 0) or np.any(k > n):
        raise ValueError("Invalid binomial data: require 0 <= n_successes <= n_samples.")

    return n, k


def _infer_beta_payload(data: Any):
    if not (isinstance(data, (tuple, list)) and len(data) == 2):
        raise TypeError("beta data expects data=(alpha, beta).")

    a, b = data
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape == () and b.shape != ():
        a = np.broadcast_to(a, b.shape)
    if b.shape == () and a.shape != ():
        b = np.broadcast_to(b, a.shape)

    if a.shape != b.shape:
        raise ValueError(f"alpha shape {a.shape} != beta shape {b.shape}")
    if np.any(a <= 0) or np.any(b <= 0):
        raise ValueError("Invalid beta data: require alpha > 0 and beta > 0.")

    return a, b
