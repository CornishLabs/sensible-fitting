"""Backend implementations + registry."""

from __future__ import annotations

from typing import Dict

from .common import Backend
from .scipy_curve_fit import ScipyCurveFitBackend
from .scipy_differential_evolution import ScipyDifferentialEvolutionBackend
from .scipy_minimize import ScipyMinimizeBackend
from .ultranest_backend import UltraNestBackend

_BACKENDS: Dict[str, Backend] = {
    "scipy.curve_fit": ScipyCurveFitBackend(),
    "scipy.differential_evolution": ScipyDifferentialEvolutionBackend(),
    "scipy.minimize": ScipyMinimizeBackend(),
    "ultranest": UltraNestBackend(),
}


def get_backend(name: str) -> Backend:
    """Return a backend implementation by name."""
    try:
        return _BACKENDS[name]
    except KeyError as e:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {tuple(_BACKENDS.keys())}"
        ) from e


AVAILABLE_BACKENDS = tuple(_BACKENDS.keys())
