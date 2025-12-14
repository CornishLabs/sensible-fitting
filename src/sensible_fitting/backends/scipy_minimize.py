from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class MinimizeResult:
    xopt: np.ndarray
    cov: Optional[np.ndarray]
    success: bool
    message: str = ""


def fit_minimize(
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    *,
    method: str = "L-BFGS-B",
    options: Optional[Dict[str, Any]] = None,
) -> MinimizeResult:
    lo, hi = bounds
    scipy_bounds = [(float(lo[i]), float(hi[i])) for i in range(int(x0.shape[0]))]

    res = minimize(
        lambda v: float(objective(np.asarray(v, dtype=float))),
        np.asarray(x0, dtype=float),
        method=str(method),
        bounds=scipy_bounds,
        options=options or {},
    )

    xopt = np.asarray(res.x, dtype=float)

    cov = None
    # L-BFGS-B often provides an inverse Hessian approximation in res.hess_inv.
    if getattr(res, "hess_inv", None) is not None:
        try:
            cov = np.asarray(res.hess_inv.todense(), dtype=float)  # type: ignore[attr-defined]
        except Exception:
            try:
                cov = np.asarray(res.hess_inv, dtype=float)
            except Exception:
                cov = None

    return MinimizeResult(
        xopt=xopt,
        cov=cov,
        success=bool(res.success),
        message=str(res.message),
    )
