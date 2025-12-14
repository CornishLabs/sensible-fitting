from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class CurveFitResult:
    popt: np.ndarray
    pcov: Optional[np.ndarray]
    success: bool
    message: str = ""


def fit_curve_fit(
    f_wrapped: Callable[..., Any],
    x: Any,
    y: np.ndarray,
    *,
    sigma: Optional[np.ndarray],
    p0: np.ndarray,
    bounds: Tuple[np.ndarray, np.ndarray],
    maxfev: Optional[int] = None,
) -> CurveFitResult:
    kwargs: Dict[str, Any] = {}
    if maxfev is not None:
        kwargs["maxfev"] = int(maxfev)

    try:
        popt, pcov = curve_fit(
            f_wrapped,
            x,
            y,
            p0=p0,
            sigma=sigma,
            # v1: provided errors are absolute measurement errors
            absolute_sigma=(sigma is not None),
            bounds=bounds,
            **kwargs,
        )
        return CurveFitResult(
            popt=np.asarray(popt, float), pcov=pcov, success=True, message="ok"
        )
    except Exception as e:
        return CurveFitResult(
            popt=np.asarray(p0, float), pcov=None, success=False, message=str(e)
        )
