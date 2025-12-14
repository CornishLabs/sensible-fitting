from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.special import gammaln

from .params import ParameterSpec


def neg_loglike_binomial(p: np.ndarray, n: np.ndarray, k: np.ndarray) -> float:
    """Negative log-likelihood for Binomial(n, p) with stability clamping."""
    p = np.asarray(p, dtype=float)
    n = np.asarray(n, dtype=float)
    k = np.asarray(k, dtype=float)

    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)

    logC = gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln((n - k) + 1.0)
    ll = logC + k * np.log(p) + (n - k) * np.log(1.0 - p)
    return float(-np.sum(ll))


def neg_loglike_beta(p: np.ndarray, alpha: np.ndarray, beta_: np.ndarray) -> float:
    """Negative log-likelihood for Beta(alpha, beta) at p, with stability clamping."""
    p = np.asarray(p, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    beta_ = np.asarray(beta_, dtype=float)

    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)

    log_norm = gammaln(alpha + beta_) - gammaln(alpha) - gammaln(beta_)
    ll = (alpha - 1.0) * np.log(p) + (beta_ - 1.0) * np.log(1.0 - p) + log_norm
    return float(-np.sum(ll))


def build_gaussian_loglike(
    *,
    model: Any,
    x: Any,
    y: np.ndarray,
    sigma: Optional[np.ndarray],
    fixed_map: Dict[str, float],
    free_names: List[str],
    vectorized: bool,
) -> Any:
    """
    Gaussian log-likelihood with correct normalization for evidence:
      log L = -1/2 Σ ((y_model - y)/σ)^2 - Σ log σ - (N/2) log(2π)
    If sigma is None, uses σ=1 and keeps only the (N/2)log(2π) constant.
    """
    y = np.asarray(y, dtype=float)
    if sigma is None:
        sig = None
        log_norm = -0.5 * float(y.size) * float(np.log(2.0 * np.pi))
    else:
        sig = np.asarray(sigma, dtype=float)
        sig = np.broadcast_to(sig, y.shape)
        log_norm = -float(np.sum(np.log(sig))) - 0.5 * float(y.size) * float(np.log(2.0 * np.pi))

    def _one(theta_free: np.ndarray) -> float:
        kw = dict(fixed_map)
        for j, name in enumerate(free_names):
            kw[name] = float(theta_free[j])
        y_model = np.asarray(model.eval(x, **kw), dtype=float)
        y_model = np.broadcast_to(y_model, y.shape)
        if sig is None:
            chi2 = np.sum((y_model - y) ** 2)
        else:
            chi2 = np.sum(((y_model - y) / sig) ** 2)
        return float(-0.5 * chi2 + log_norm)

    if not vectorized:
        return _one

    def _many(thetas: np.ndarray) -> np.ndarray:
        thetas = np.asarray(thetas, dtype=float)
        if thetas.ndim == 1:
            return np.asarray(_one(thetas), dtype=float)
        out = np.empty((thetas.shape[0],), dtype=float)
        for i in range(thetas.shape[0]):
            out[i] = _one(thetas[i])
        return out

    return _many


def build_prior_transform(params: Tuple[ParameterSpec, ...], free_names: List[str]) -> Any:
    """
    Build an UltraNest-style prior transform: cube in [0,1]^P -> physical params.
    Uses ParameterSpec.prior if present; otherwise defaults to Uniform(bounds).

    Supported priors (v1):
      - ('uniform', lo, hi) or no-args + finite bounds
      - ('loguniform', lo, hi)  (lo>0, hi>0)
      - ('normal', mean, sigma) (optionally truncated if finite bounds are set)
    """
    import scipy.stats

    pmap = {p.name: p for p in params}
    transforms = []

    for name in free_names:
        spec = pmap[name]
        prior = spec.prior
        bounds = spec.bounds

        kind = None
        args: Tuple[Any, ...] = ()
        if prior is not None:
            kind, args = prior
            kind = str(kind).lower()

        def _require_finite_bounds(n: str) -> tuple[float, float]:
            if bounds is None or bounds[0] is None or bounds[1] is None:
                raise ValueError(
                    f"UltraNest needs a proper prior for {n!r}. "
                    f"Either set .prior({n}=...) or finite bounds via .bound({n}=(lo,hi))."
                )
            lo, hi = float(bounds[0]), float(bounds[1])
            if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                raise ValueError(f"Bounds for {n!r} must be finite and lo < hi for UltraNest.")
            return lo, hi

        if kind is None or kind == "uniform":
            if len(args) >= 2:
                lo, hi = float(args[0]), float(args[1])
            else:
                lo, hi = _require_finite_bounds(name)

            def t(q, lo=lo, hi=hi):
                return q * (hi - lo) + lo

        elif kind == "loguniform":
            if len(args) >= 2:
                lo, hi = float(args[0]), float(args[1])
            else:
                lo, hi = _require_finite_bounds(name)
            if lo <= 0 or hi <= 0:
                raise ValueError(f"loguniform prior for {name!r} requires lo>0 and hi>0.")
            log_lo = float(np.log(lo))
            log_hi = float(np.log(hi))

            def t(q, log_lo=log_lo, log_hi=log_hi):
                return np.exp(q * (log_hi - log_lo) + log_lo)

        elif kind == "normal":
            if len(args) < 2:
                raise TypeError(f"normal prior for {name!r} expects ('normal', mean, sigma).")
            mu = float(args[0])
            sig = float(args[1])
            if sig <= 0:
                raise ValueError(f"normal prior for {name!r} requires sigma > 0.")

            if bounds is not None and bounds[0] is not None and bounds[1] is not None:
                lo = float(bounds[0])
                hi = float(bounds[1])
                if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                    rv = scipy.stats.norm(mu, sig)
                    c_lo = rv.cdf(lo)
                    c_hi = rv.cdf(hi)

                    def t(q, rv=rv, c_lo=c_lo, c_hi=c_hi):
                        return rv.ppf(c_lo + q * (c_hi - c_lo))
                else:
                    rv = scipy.stats.norm(mu, sig)

                    def t(q, rv=rv):
                        return rv.ppf(q)
            else:
                rv = scipy.stats.norm(mu, sig)

                def t(q, rv=rv):
                    return rv.ppf(q)

        else:
            raise NotImplementedError(
                f"Unsupported prior kind {kind!r} for {name!r} (v1 supports uniform/loguniform/normal)."
            )

        transforms.append(t)

    def transform(cube: np.ndarray):
        cube = np.asarray(cube, dtype=float)
        if cube.ndim == 1:
            out = np.empty((len(transforms),), dtype=float)
            for j, fn in enumerate(transforms):
                out[j] = float(fn(float(cube[j])))
            return out
        if cube.ndim == 2:
            out = np.empty_like(cube, dtype=float)
            for j, fn in enumerate(transforms):
                out[:, j] = fn(cube[:, j])
            return out
        raise ValueError("cube must have shape (P,) or (N,P).")

    return transform
