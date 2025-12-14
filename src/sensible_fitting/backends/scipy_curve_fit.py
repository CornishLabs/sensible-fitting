from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit

from .common import BackendResult

class ScipyCurveFitBackend:
    name = "scipy.curve_fit"

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
        if getattr(dataset, "format", None) != "normal":
            raise NotImplementedError(
                "scipy.curve_fit backend supports only data_format='normal'."
            )

        payload: Dict[str, Any] = dict(getattr(dataset, "payload"))
        x = getattr(dataset, "x")
        y = np.asarray(payload["y"], dtype=float)
        sigma = payload.get("sigma", None)
        sigma_arr = None if sigma is None else np.asarray(sigma, dtype=float)

        maxfev = options.get("maxfev", None)
        kwargs: Dict[str, Any] = {}
        if maxfev is not None:
            kwargs["maxfev"] = int(maxfev)

        def f_wrapped(xi, *theta_free):
            kw = dict(fixed_map)
            for j, n in enumerate(free_names):
                kw[n] = float(theta_free[j])
            return model.eval(xi, **kw)

        try:
            popt, pcov = curve_fit(
                f_wrapped,
                x,
                y,
                p0=np.asarray(p0, dtype=float),
                sigma=sigma_arr,
                absolute_sigma=(sigma_arr is not None),
                bounds=bounds,
                **kwargs,
            )
            return BackendResult(
                theta=np.asarray(popt, dtype=float),
                cov=None if pcov is None else np.asarray(pcov, dtype=float),
                success=True,
                message="ok",
                stats={"backend": "scipy.curve_fit"},
            )
        except Exception as e:
            # Soft fail: return seed point.
            return BackendResult(
                theta=np.asarray(p0, dtype=float),
                cov=None,
                success=False,
                message=str(e),
                stats={"backend": "scipy.curve_fit", "error": str(e)},
            )
