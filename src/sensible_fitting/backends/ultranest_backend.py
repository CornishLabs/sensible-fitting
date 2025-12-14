from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class UltraNestResult:
    mean: np.ndarray                 # posterior mean (free params)
    stdev: Optional[np.ndarray]      # posterior stdev (free params)
    cov: Optional[np.ndarray]        # posterior covariance (free params)
    success: bool
    message: str = ""
    stats: Dict[str, Any] | None = None


def fit_ultranest(
    *,
    param_names: list[str],
    loglike: Callable[[np.ndarray], Any],
    transform: Callable[[np.ndarray], Any],
    wrapped_params: Optional[list[bool]] = None,
    log_dir: Optional[str] = None,
    resume: Any = "subfolder",
    sampler_kwargs: Optional[Dict[str, Any]] = None,
    run_kwargs: Optional[Dict[str, Any]] = None,
    fallback_theta: Optional[np.ndarray] = None,
) -> UltraNestResult:
    """
    Thin wrapper around ultranest.ReactiveNestedSampler for sensible-fitting.
    Returns posterior mean/stdev/cov as "fit-like" outputs, plus raw stats.
    """
    sampler_kwargs = dict(sampler_kwargs or {})
    run_kwargs = dict(run_kwargs or {})

    if fallback_theta is None:
        fallback_theta = np.full((len(param_names),), np.nan, dtype=float)
    else:
        fallback_theta = np.asarray(fallback_theta, dtype=float)

    try:
        import ultranest  # local import (optional dependency)

        sampler = ultranest.ReactiveNestedSampler(
            param_names,
            loglike,
            transform=transform,
            wrapped_params=wrapped_params,
            log_dir=log_dir,
            resume=resume,
            **sampler_kwargs,
        )

        result = sampler.run(**run_kwargs)

        samples = np.asarray(result.get("samples", []), dtype=float)
        if samples.ndim != 2 or samples.shape[1] != len(param_names) or samples.shape[0] == 0:
            # Fallback to something usable even if sampling produced no samples
            mean = fallback_theta
            stdev = None
            cov = None
        else:
            mean = samples.mean(axis=0)
            stdev = samples.std(axis=0, ddof=1) if samples.shape[0] >= 2 else None
            cov = np.cov(samples.T, ddof=1) if samples.shape[0] >= 2 else None

        stats: Dict[str, Any] = {
            "backend": "ultranest",
            "free_names": tuple(param_names),
            "logz": float(result.get("logz", np.nan)),
            "logzerr": float(result.get("logzerr", np.nan)),
            "posterior_samples": samples,
            "ultranest_result": result,
        }

        return UltraNestResult(
            mean=np.asarray(mean, dtype=float),
            stdev=None if stdev is None else np.asarray(stdev, dtype=float),
            cov=None if cov is None else np.asarray(cov, dtype=float),
            success=True,
            message="ok",
            stats=stats,
        )

    except Exception as e:
        # Be conservative: fail "softly" and return the fallback point.
        return UltraNestResult(
            mean=np.asarray(fallback_theta, dtype=float),
            stdev=None,
            cov=None,
            success=False,
            message=str(e),
            stats={"backend": "ultranest", "error": str(e), "free_names": tuple(param_names)},
        )