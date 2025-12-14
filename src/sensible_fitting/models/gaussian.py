from __future__ import annotations

import numpy as np

from ..model import Model


# --- Gaussian with offset, OITG-style ---------------------------------------


def gaussian_with_offset_func(x, x0, y0, a, sigma):
    """Gaussian with baseline: y = y0 + a * exp(-0.5 * ((x - x0)/sigma)^2)."""
    return y0 + a * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def gaussian_with_offset(*, name: str = "gaussian") -> Model:
    """Return a Gaussian Model with offset and OITG-like seeding.

    Parameters
    ----------
    name:
        Human-readable model name.

    Parameters in the model
    -----------------------
    x0   : center position
    y0   : baseline
    a    : amplitude (always >= 0 from the seeding heuristic)
    sigma: width (> 0)

    Derived parameters
    ------------------
    fwhm : 2.35482 * sigma
    """

    base = (
        Model.from_function(gaussian_with_offset_func, name=name)
        # sensible bounds: sigma > 0, amplitude >= 0
        .bound(a=(0.0, None), sigma=(1e-12, None))
    )

    def init_gaussian(x, y, g):
        """GuessState-based equivalent of the old parameter_initialiser.

        This sees the already-preprocessed y array (yobs from fit(...)),
        and fills g.x0, g.y0, g.a, g.sigma if they are unset.
        """
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

        if x_arr.size == 0 or y_arr.size == 0:
            # Degenerate input; don't try to guess anything.
            return

        # Baseline: mean of y
        y0 = float(np.mean(y_arr))
        if g.is_unset("y0"):
            g.y0 = y0

        # Estimate sign and amplitude of the peak relative to baseline
        dy_min = y0 - float(np.min(y_arr))
        dy_max = float(np.max(y_arr) - y0)

        if dy_max >= dy_min:
            # Peak is positive
            if g.is_unset("a"):
                g.a = dy_max
            if g.is_unset("x0"):
                g.x0 = float(x_arr[np.argmax(y_arr)])
        else:
            # Peak is negative
            # (Note: amplitude stays positive; sign is carried by y0 vs peak)
            if g.is_unset("a"):
                g.a = dy_min
            if g.is_unset("x0"):
                g.x0 = float(x_arr[np.argmin(y_arr)])

        # Crude sigma: one-fifth of the x-span, as in the original
        if g.is_unset("sigma"):
            span = float(np.max(x_arr) - np.min(x_arr))
            if span <= 0:
                # Fallback if all x are identical
                g.sigma = 1.0
            else:
                g.sigma = 0.2 * span

    model = base.with_guesser(init_gaussian).derive(
        "fwhm",
        lambda p: 2.35482 * p["sigma"],
        doc="Full-width at half maximum (2.35482 * sigma)",
    )

    return model
