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
        """Fit using scipy.optimize.minimize.

        Backend options:
        - method: optimizer name (default: L-BFGS-B)
        - options: dict forwarded to scipy.optimize.minimize
        - cov_method: "auto" uses res.hess_inv if available, else numeric Hessian
        - cov_step: relative step size for numeric Hessian (default: 1e-4)
        - cov_jitter: diagonal jitter before inverting Hessian (default: 1e-8)
        """
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
        cov_method = str(options.get("cov_method", "auto")).lower()
        cov_step = options.get("cov_step", None)
        cov_jitter = float(options.get("cov_jitter", 1e-8))

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

        def _numdiff_hessian(func, x0, bounds, step):
            x0 = np.asarray(x0, dtype=float)
            npar = int(x0.shape[0])
            step = 1e-4 if step is None else float(step)
            eps = step * (np.abs(x0) + 1.0)

            lo = np.array(
                [(-np.inf if b[0] is None else float(b[0])) for b in bounds],
                dtype=float,
            )
            hi = np.array(
                [(np.inf if b[1] is None else float(b[1])) for b in bounds],
                dtype=float,
            )
            for i in range(npar):
                if np.isfinite(lo[i]):
                    eps[i] = min(eps[i], 0.5 * max(0.0, x0[i] - lo[i]))
                if np.isfinite(hi[i]):
                    eps[i] = min(eps[i], 0.5 * max(0.0, hi[i] - x0[i]))
                if eps[i] <= 0.0:
                    return None

            f0 = float(func(x0))
            hess = np.zeros((npar, npar), dtype=float)
            for i in range(npar):
                ei = np.zeros(npar, dtype=float)
                ei[i] = eps[i]
                fpp = float(func(x0 + ei))
                fmm = float(func(x0 - ei))
                hess[i, i] = (fpp - 2.0 * f0 + fmm) / (eps[i] ** 2)
                for j in range(i + 1, npar):
                    ej = np.zeros(npar, dtype=float)
                    ej[j] = eps[j]
                    fpp = float(func(x0 + ei + ej))
                    fpm = float(func(x0 + ei - ej))
                    fmp = float(func(x0 - ei + ej))
                    fmm = float(func(x0 - ei - ej))
                    hij = (fpp - fpm - fmp + fmm) / (4.0 * eps[i] * eps[j])
                    hess[i, j] = hij
                    hess[j, i] = hij
            return hess

        cov = None
        if cov_method in ("auto", "hess_inv"):
            if getattr(res, "hess_inv", None) is not None:
                try:
                    cov = np.asarray(res.hess_inv.todense(), dtype=float)  # type: ignore[attr-defined]
                except Exception:
                    try:
                        cov = np.asarray(res.hess_inv, dtype=float)
                    except Exception:
                        cov = None

        if cov_method in ("auto", "numdiff") and cov is None:
            hess = _numdiff_hessian(objective, theta, scipy_bounds, cov_step)
            if hess is not None:
                if cov_jitter > 0.0:
                    hess = hess + cov_jitter * np.eye(hess.shape[0])
                try:
                    cov = np.linalg.pinv(hess)
                except Exception:
                    cov = None

        return BackendResult(
            theta=theta,
            cov=cov,
            success=bool(res.success),
            message=str(res.message),
            stats={"backend": "scipy.minimize", "method": method},
        )
