from __future__ import annotations

import numpy as np

from ..model import Model


def straight_line_func(x, m, b):
    """Module-level straight line function y = m*x + b."""
    return m * x + b


def straight_line(*, name: str = "straight line") -> Model:
    """Return a straight line Model."""
    return Model.from_function(straight_line_func, name=name)
