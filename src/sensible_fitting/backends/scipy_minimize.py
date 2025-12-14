from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import math
from scipy.optimize import minimize

from ..inference import neg_loglike_binomial, neg_loglike_beta
from .common import BackendResult

class ScipyMinimizeBackend:
    name = "scipy.minimize"

    def fit_one(
        self,
        *,
        model: Any,
        dataset: Any,
        free_names: list[str],
        fixed_map: dict[str, float],
        p0: np.ndarray,
        bounds: Tuple[np.ndarray, np.ndarray],
        options: dict[str, Any],
    ) -> BackendResult:
        fmt = getattr(dataset, "format", None)
        payload: Dict[str, Any] = dict(getattr(dataset, "payload"))
        x = getattr(dataset, "x")

        if fmt not in ("binomial", "beta"):
            raise NotImplementedError(
                "scipy.minimize backend currently supports data_format in {'binomial','beta'}."
            )

        lo, hi = bounds
        scipy_bounds = []
        for i in range(int(p0.shape[0])):
            lo_i = float(lo[i])
            hi_i = float(hi[i])
            lo_b = None if (not math.isfinite(lo_i)) else lo_i
            hi_b = None if (not math.isfinite(hi_i)) else hi_i
            scipy_bounds.append((lo_b, hi_b))

        method = str(options.get("method", "L-BFGS-B"))
        scipy_opts = options.get("options", None) or {}

        if fmt == "binomial":
            n = np.asarray(payload["n"], dtype=float)
            k = np.asarray(payload["k"], dtype=float)

            def objective(theta_free: np.ndarray) -> float:
                kw = dict(fixed_map)
                for j, name in enumerate(free_names):
                    kw[name] = float(theta_free[j])
                p = np.asarray(model.eval(x, **kw), dtype=float)
                p = np.broadcast_to(p, k.shape)
                return neg_loglike_binomial(p, n, k)

        else:  # beta
            a = np.asarray(payload["alpha"], dtype=float)
            b = np.asarray(payload["beta"], dtype=float)

            def objective(theta_free: np.ndarray) -> float:
                kw = dict(fixed_map)
                for j, name in enumerate(free_names):
                    kw[name] = float(theta_free[j])
                p = np.asarray(model.eval(x, **kw), dtype=float)
                p = np.broadcast_to(p, a.shape)
                return neg_loglike_beta(p, a, b)

        res = minimize(
            lambda v: float(objective(np.asarray(v, dtype=float))),
            np.asarray(p0, dtype=float),
            method=method,
            bounds=scipy_bounds,
            options=scipy_opts,
        )

        theta = np.asarray(res.x, dtype=float)

        cov = None
        if getattr(res, "hess_inv", None) is not None:
            try:
                cov = np.asarray(res.hess_inv.todense(), dtype=float)  # type: ignore[attr-defined]
            except Exception:
                try:
                    cov = np.asarray(res.hess_inv, dtype=float)
                except Exception:
                    cov = None

        return BackendResult(
            theta=theta,
            cov=cov,
            success=bool(res.success),
            message=str(res.message),
            stats={"backend": "scipy.minimize", "method": method},
        )
