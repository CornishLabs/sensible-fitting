from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from warnings import warn

from .util import flatten_batch


@dataclass(frozen=True)
class Dataset:
    x: Any
    format: str  # "normal" | "binomial" | "beta"
    payload: Dict[str, Any]
    y_for_seed: np.ndarray


def prepare_datasets(
    x: Any, data: Any, data_format: str, strict: bool = False
) -> Tuple[List[Dataset], Tuple[int, ...]]:
    """Normalize user inputs into a list of per-dataset payloads + batch shape."""

    if data_format not in ("normal", "binomial", "beta"):
        raise NotImplementedError(
            "v1 supports data_format in {'normal','binomial','beta'}."
        )

    datasets: List[Dataset] = []
    batch_shape: Tuple[int, ...] = ()

    if isinstance(x, list):
        if not isinstance(data, list):
            raise TypeError(
                "Ragged batches require list inputs for both x and data."
            )
        if len(x) != len(data):
            raise ValueError("Ragged batch requires x and data lists of equal length.")
        for xi, di in zip(x, data):
            datasets.append(_one_dataset(xi, di, data_format, strict))
        return datasets, (len(datasets),)

    if isinstance(data, list):
        if len(data) == 0:
            raise ValueError("Empty data list; cannot infer batch or payload.")

        if data_format in ("binomial", "beta"):
            if _list_of_pairs(data):
                for di in data:
                    datasets.append(_one_dataset(x, di, data_format, strict))
                return datasets, (len(datasets),)
            if len(data) == 2:
                _warn_or_raise(
                    strict,
                    "Interpreting list as payload; use a tuple for (n, k)/(alpha, beta) "
                    "or a list of tuples for batching.",
                )
                data = tuple(data)
            else:
                raise TypeError(
                    "Binomial/beta data expects a 2-tuple payload or a list of 2-tuples."
                )
        else:
            arr = np.asarray(data)
            if arr.dtype != object:
                if len(data) in (2, 3) and not _list_all_scalars(data):
                    _warn_or_raise(
                        strict,
                        "Interpreting list as batch data. If you meant (y, sigma) "
                        "or (y, sigma_lo, sigma_hi), use a tuple.",
                    )
                data = arr
            else:
                if len(data) in (2, 3):
                    _warn_or_raise(
                        strict,
                        "Interpreting ragged list as batch data. If you meant "
                        "(y, sigma) or (y, sigma_lo, sigma_hi), use a tuple.",
                    )
                for di in data:
                    datasets.append(_one_dataset(x, di, data_format, strict))
                return datasets, (len(datasets),)

    # Common-x batching: detect by array dimensionality of the primary observation.
    if data_format == "normal":
        yobs, sigma = _infer_gaussian_payload(data, strict)
        yobs = np.asarray(yobs, dtype=float)

        if _x_matches_y_shape(x, yobs):
            x_flat = _flatten_grid_x(x)
            y_flat = np.asarray(yobs, dtype=float).reshape(-1)
            sigma_flat = None
            if sigma is not None:
                sarr = np.asarray(sigma, dtype=float)
                if sarr.shape == ():
                    sigma_flat = float(sarr)
                else:
                    if sarr.shape != yobs.shape:
                        sarr = np.broadcast_to(sarr, yobs.shape)
                    sigma_flat = sarr.reshape(-1)
            datasets.append(
                Dataset(
                    x=x_flat,
                    format="normal",
                    payload={"y": y_flat, "sigma": sigma_flat},
                    y_for_seed=y_flat,
                )
            )
            return datasets, ()

        if _x_is_nd(x) and yobs.ndim > 1:
            _warn_or_raise(
                strict,
                "ND x with y shape not matching x; treating leading y axes as batch. "
                "If this is a 2D/ND dataset, flatten y (and x) explicitly.",
            )

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


def _one_dataset(x: Any, data: Any, fmt: str, strict: bool) -> Dataset:
    """Build a single Dataset from x/data for a given format."""
    if fmt == "normal":
        y, sigma = _infer_gaussian_payload(data, strict)
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
    """Safely compute num/den with zero handling."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, 0.0)


def _infer_gaussian_payload(y: Any, strict: bool):
    """Parse normal data payloads into (y, sigma)."""
    # y -> unweighted
    # (y, sigma) -> symmetric absolute errors
    # (y, sigma_low, sigma_high) -> approximate to mean sigma
    if isinstance(y, tuple) and len(y) == 2:
        yobs, yerr = y
        return np.asarray(yobs), yerr
    if isinstance(y, tuple) and len(y) == 3:
        yobs, ylo, yhi = y
        sigma = 0.5 * (np.asarray(ylo) + np.asarray(yhi))
        return np.asarray(yobs), sigma
    if isinstance(y, list) and len(y) in (2, 3) and not _list_all_scalars(y):
        _warn_or_raise(
            strict,
            "Ambiguous list payload. Use a tuple for (y, sigma) or "
            "(y, sigma_lo, sigma_hi).",
        )
    return np.asarray(y), None


def _infer_binomial_payload(y: Any):
    """Parse binomial payload into (n, k) arrays."""
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
    """Parse beta payload into (alpha, beta) arrays."""
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


def _warn_or_raise(strict: bool, message: str) -> None:
    """Warn or raise based on strict mode."""
    if strict:
        raise ValueError(message)
    warn(message, UserWarning, stacklevel=2)


def _list_all_scalars(data: List[Any]) -> bool:
    """Return True if every element is scalar-like."""
    return all(_is_scalar_like(v) for v in data)


def _is_scalar_like(v: Any) -> bool:
    """Return True if v is a scalar or 0-d numpy array."""
    if isinstance(v, np.ndarray):
        return v.shape == ()
    return isinstance(v, (int, float, np.number, bool))


def _list_of_pairs(data: List[Any]) -> bool:
    """Return True if data is a list of 2-tuples/lists."""
    if not data:
        return False
    for item in data:
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            return False
    return True


def _x_matches_y_shape(x: Any, y: np.ndarray) -> bool:
    """Return True if x shape(s) match y shape exactly."""
    if isinstance(x, (tuple, list)) and x:
        shapes = [np.asarray(xi).shape for xi in x]
        return all(s == y.shape for s in shapes)
    if isinstance(x, np.ndarray):
        return x.shape == y.shape
    return False


def _x_is_nd(x: Any) -> bool:
    """Return True if x contains arrays with ndim > 1."""
    if isinstance(x, np.ndarray):
        return x.ndim > 1
    if isinstance(x, (tuple, list)) and x:
        return any(np.asarray(xi).ndim > 1 for xi in x)
    return False


def _flatten_grid_x(x: Any) -> Any:
    """Flatten grid-like x into 1D coordinate arrays."""
    if isinstance(x, (tuple, list)) and x:
        return tuple(np.asarray(xi).reshape(-1) for xi in x)
    if isinstance(x, np.ndarray):
        return x.reshape(-1)
    return x
