from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np


@dataclass(frozen=True)
class BackendResult:
    """Normalized result returned by any backend."""

    theta: np.ndarray  # free parameters, shape (P,)
    cov: Optional[np.ndarray] = None  # free-parameter covariance, (P,P)
    success: bool = True
    message: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)


class Backend(Protocol):
    """Backend protocol: fit one dataset."""

    name: str

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
    ) -> BackendResult: ...
