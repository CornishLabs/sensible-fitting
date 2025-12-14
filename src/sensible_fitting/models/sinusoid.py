from __future__ import annotations

import numpy as np

from ..model import Model


def sinusoid_func(x, amplitude, offset, frequency, phase):
    """Module-level sinusoid: offset + amplitude * sin(2π f x + phase)."""
    return offset + amplitude * np.sin(2 * np.pi * frequency * x + phase)


def sinusoid(*, name: str = "sinusoid") -> Model:
    """Return a sinusoid Model with sensible default seeding."""
    return Model.from_function(sinusoid_func, name=name)
