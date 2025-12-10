from __future__ import annotations

import numpy as np

from .model import Model


def straight_line(*, name: str = "straight line") -> Model:
    def line(x, m, b):
        return m * x + b
    return Model.from_function(line, name=name).autoguess("m", "b")


def sinusoid(*, name: str = "sinusoid") -> Model:
    def s(x, amplitude, offset, frequency, phase):
        return offset + amplitude * np.sin(2 * np.pi * frequency * x + phase)
    return Model.from_function(s, name=name).autoguess("amplitude", "offset")
