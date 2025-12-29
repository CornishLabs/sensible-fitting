from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from .common import BackendResult


class ScipyDifferentialEvolutionBackend:
    name = "scipy.differential_evolution"

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
        """Fit using scipy.optimize.differential_evolution (global optimisation).

        Notes:
        - Requires finite bounds for *all* free parameters.
        - By default, estimates covariance via a numeric Jacobian of residuals
          at the optimum (least-squares approximation).

        Backend options (subset of scipy.optimize.differential_evolution):
        - maxiter (int, default: 50)
        - popsize (int, default: 15)
        - tol (float, default: 0.01)
        - strategy (str, default: "best1bin")
        - mutation, recombination, seed, polish, disp, init, atol, updating, workers

        Covariance options:
        - cov_method: "numdiff" (default) or "none"
        - cov_step: relative step for numeric Jacobian (default: 1e-5)
        - cov_jitter: diagonal jitter added to JᵀJ before inversion (default: 0.0)
        """
        if getattr(dataset, "format", None) != "normal":
            raise NotImplementedError(
                "scipy.differential_evolution backend supports only data_format='normal'."
            )

        if len(free_names) == 0:
            return BackendResult(
                theta=np.asarray([], dtype=float),
                cov=None,
                success=True,
                message="no free parameters",
                stats={"backend": self.name},
            )

        payload: Dict[str, Any] = dict(getattr(dataset, "payload"))
        x = getattr(dataset, "x")
        y = np.asarray(payload["y"], dtype=float)
        sigma = payload.get("sigma", None)
        sigma_arr: Optional[np.ndarray]
        if sigma is None:
            sigma_arr = None
        else:
            sigma_arr = np.asarray(sigma, dtype=float)
            if sigma_arr.shape not in ((), y.shape):
                try:
                    sigma_arr = np.broadcast_to(sigma_arr, y.shape)
                except Exception as exc:
                    raise ValueError(
                        f"sigma shape {sigma_arr.shape} is not broadcastable to y shape {y.shape}."
                    ) from exc

        lo, hi = bounds
        lo = np.asarray(lo, dtype=float).reshape((-1,))
        hi = np.asarray(hi, dtype=float).reshape((-1,))
        if lo.shape != hi.shape or lo.shape != (len(free_names),):
            raise ValueError("Bounds shape mismatch for free parameters.")

        de_bounds = []
        for j in range(len(free_names)):
            lo_j = float(lo[j])
            hi_j = float(hi[j])
            if not (np.isfinite(lo_j) and np.isfinite(hi_j)):
                raise ValueError(
                    "scipy.differential_evolution requires finite bounds for all free parameters."
                )
            if hi_j <= lo_j:
                raise ValueError("Invalid bounds: require hi > lo for all parameters.")
            de_bounds.append((lo_j, hi_j))

        def _eval(theta_free: np.ndarray) -> np.ndarray:
            kw = dict(fixed_map)
            theta_free = np.asarray(theta_free, dtype=float)
            for j, name in enumerate(free_names):
                kw[name] = float(theta_free[j])
            ym = np.asarray(model.eval(x, **kw), dtype=float)
            if ym.shape != y.shape:
                try:
                    ym = np.broadcast_to(ym, y.shape)
                except Exception as exc:
                    raise ValueError(
                        f"Model output shape {ym.shape} not broadcastable to y shape {y.shape}."
                    ) from exc
            return ym

        def _residual(theta_free: np.ndarray) -> np.ndarray:
            r = _eval(theta_free) - y
            if sigma_arr is not None:
                r = r / sigma_arr
            return np.asarray(r, dtype=float).reshape(-1)

        def objective(theta_free: np.ndarray) -> float:
            try:
                r = _residual(theta_free)
            except Exception:
                return float("inf")
            if not np.all(np.isfinite(r)):
                return float("inf")
            return float(np.sum(r * r))

        de_kwargs: Dict[str, Any] = {}
        de_kwargs["maxiter"] = int(options.get("maxiter", 50))
        de_kwargs["popsize"] = int(options.get("popsize", 15))
        de_kwargs["tol"] = float(options.get("tol", 0.01))
        de_kwargs["strategy"] = str(options.get("strategy", "best1bin"))

        for k in (
            "mutation",
            "recombination",
            "seed",
            "polish",
            "disp",
            "init",
            "atol",
            "updating",
            "workers",
        ):
            if k in options:
                de_kwargs[k] = options[k]

        res = differential_evolution(objective, de_bounds, **de_kwargs)
        theta = np.asarray(res.x, dtype=float)

        cov_method = str(options.get("cov_method", "numdiff")).lower()
        cov_step = float(options.get("cov_step", 1e-5))
        cov_jitter = float(options.get("cov_jitter", 0.0))

        cov = None
        if cov_method not in ("none", "off", "false"):
            cov = _cov_from_numdiff_jacobian(
                residual=_residual,
                theta=theta,
                bounds=(lo, hi),
                step=cov_step,
                jitter=cov_jitter,
                scale=(sigma_arr is None),
            )

        stats: Dict[str, Any] = {
            "backend": self.name,
            "fun": float(res.fun) if getattr(res, "fun", None) is not None else None,
            "nfev": int(getattr(res, "nfev", 0) or 0),
            "nit": int(getattr(res, "nit", 0) or 0),
        }

        return BackendResult(
            theta=theta,
            cov=cov,
            success=bool(res.success),
            message=str(res.message),
            stats=stats,
        )


def _cov_from_numdiff_jacobian(
    *,
    residual: Any,
    theta: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    step: float,
    jitter: float,
    scale: bool,
) -> Optional[np.ndarray]:
    """Approximate covariance from numeric Jacobian of residuals at theta.

    residual(theta) must return a 1D residual vector.
    If scale=True, scales covariance by SSE/(M-P) (curve_fit-like when sigma is None).
    """
    theta = np.asarray(theta, dtype=float).reshape((-1,))
    npar = int(theta.shape[0])
    if npar == 0:
        return None

    r0 = np.asarray(residual(theta), dtype=float).reshape(-1)
    if r0.size == 0 or not np.all(np.isfinite(r0)):
        return None

    lo, hi = bounds
    lo = np.asarray(lo, dtype=float).reshape((-1,))
    hi = np.asarray(hi, dtype=float).reshape((-1,))
    if lo.shape != (npar,) or hi.shape != (npar,):
        return None

    # Central-difference Jacobian of residuals: J_{i,j} = dr_i/dtheta_j
    J = np.empty((r0.size, npar), dtype=float)
    rel = 1e-5 if not np.isfinite(step) or step <= 0.0 else float(step)

    for j in range(npar):
        eps = rel * (abs(theta[j]) + 1.0)

        # Keep within bounds to avoid invalid evaluations.
        lo_j = float(lo[j])
        hi_j = float(hi[j])
        if np.isfinite(lo_j):
            eps = min(eps, 0.5 * max(0.0, theta[j] - lo_j))
        if np.isfinite(hi_j):
            eps = min(eps, 0.5 * max(0.0, hi_j - theta[j]))

        if not np.isfinite(eps) or eps <= 0.0:
            return None

        t_plus = theta.copy()
        t_minus = theta.copy()
        t_plus[j] += eps
        t_minus[j] -= eps

        r_plus = np.asarray(residual(t_plus), dtype=float).reshape(-1)
        r_minus = np.asarray(residual(t_minus), dtype=float).reshape(-1)
        if r_plus.shape != r0.shape or r_minus.shape != r0.shape:
            return None
        if not (np.all(np.isfinite(r_plus)) and np.all(np.isfinite(r_minus))):
            return None

        J[:, j] = (r_plus - r_minus) / (2.0 * eps)

    JTJ = J.T @ J
    if jitter > 0.0:
        JTJ = JTJ + float(jitter) * np.eye(npar, dtype=float)

    try:
        cov = np.linalg.pinv(JTJ)
    except Exception:
        return None

    if not np.all(np.isfinite(cov)):
        return None

    if scale:
        dof = r0.size - npar
        if dof > 0:
            s_sq = float(np.sum(r0 * r0) / dof)
            cov = cov * s_sq

    return cov

