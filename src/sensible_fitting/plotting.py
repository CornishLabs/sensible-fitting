from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple
from warnings import warn

import numpy as np

from .util import uncertainty_to_string


def plot_fit(
    *,
    ax: Optional[Any] = None,
    x: Any,
    y: Any,
    yerr: Optional[Any] = None,
    run: Optional[Any] = None,
    which: str = "fit",
    xg: Optional[np.ndarray] = None,
    band: bool = False,
    band_options: Optional[Mapping[str, Any]] = None,
    band_kwargs: Optional[Mapping[str, Any]] = None,
    data_kwargs: Optional[Mapping[str, Any]] = None,
    line_kwargs: Optional[Mapping[str, Any]] = None,
    show_params: bool = False,
    param_names: Optional[Sequence[str]] = None,
    param_digits: int | str | None = "auto",
    text_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[Any, Any]:
    """Plot data points and an optional fit line/band on a Matplotlib Axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        If None, a new figure/axes is created.
    x, y : array-like
        1D data to plot.
    yerr : array-like, optional
        Symmetric error bars. Scalar or array-like.
    run : Run, optional
        Run object providing predict() and band(). Required for fit line/band.
    which : {"fit", "seed"}
        Use fitted params or seed params for the line.
    xg : ndarray, optional
        Grid for plotting the fit line. Defaults to 400 points over x range.
    band : bool
        If True, draw uncertainty band using run.band().
    band_options : dict, optional
        Keyword options forwarded to run.band().
    band_kwargs, data_kwargs, line_kwargs, text_kwargs : dict, optional
        Styling kwargs for fill_between, errorbar, plot, and text.
    show_params : bool
        If True, annotate fitted parameters on the plot.
    param_names : sequence of str, optional
        Names to include in the parameter box. Defaults to free, non-derived params.
    param_digits : int | "auto"
        Significant digits for parameter uncertainty formatting.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    data_kwargs = dict(data_kwargs or {})
    line_kwargs = dict(line_kwargs or {})
    band_options = dict(band_options or {})
    band_kwargs = dict(band_kwargs or {})
    text_kwargs = dict(text_kwargs or {})

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("plot_fit requires 1D x and y arrays.")
    if x_arr.dtype == object or y_arr.dtype == object:
        raise ValueError("plot_fit requires numeric 1D arrays.")
    if x_arr.shape != y_arr.shape:
        raise ValueError("plot_fit requires x and y to have the same shape.")

    if yerr is None:
        data_kwargs.setdefault("marker", "o")
        data_kwargs.setdefault("linestyle", "none")
        ax.plot(x_arr, y_arr, **data_kwargs)
    else:
        yerr_arr = np.asarray(yerr)
        if yerr_arr.shape not in ((), x_arr.shape):
            raise ValueError("plot_fit requires yerr to be scalar or same shape as y.")
        data_kwargs.setdefault("fmt", "o")
        data_kwargs.setdefault("ms", 4)
        data_kwargs.setdefault("capsize", 2)
        ax.errorbar(x_arr, y_arr, yerr=yerr, **data_kwargs)

    if run is not None:
        if xg is None:
            xg = np.linspace(float(np.min(x_arr)), float(np.max(x_arr)), 400)
        yfit = run.predict(xg, which=which)
        line_kwargs.setdefault("label", "fit" if which == "fit" else "seed")
        ax.plot(xg, yfit, **line_kwargs)

        if band:
            try:
                band_obj = run.band(xg, **band_options)
            except Exception as exc:
                warn(f"plot_fit: could not compute band: {exc}", UserWarning)
            else:
                band_kwargs.setdefault("alpha", 0.2)
                ax.fill_between(xg, band_obj.low, band_obj.high, **band_kwargs)

        if show_params:
            params = run.results.params
            if param_names is None:
                names = [
                    n
                    for n, pv in params.items()
                    if not pv.fixed and not pv.derived
                ]
            else:
                names = list(param_names)

            lines = []
            for name in names:
                pv = params[name]
                val = float(pv.value)
                if pv.stderr is None:
                    lines.append(f"{name}={val:.{param_digits}g}")
                else:
                    err = float(pv.stderr)
                    lines.append(
                        f"{name}={uncertainty_to_string(val, err, precision=param_digits)}"
                    )

            if lines:
                text_kwargs.setdefault("ha", "left")
                text_kwargs.setdefault("va", "top")
                text_kwargs.setdefault("fontsize", 9)
                text_kwargs.setdefault("transform", ax.transAxes)
                text_kwargs.setdefault(
                    "bbox",
                    {"boxstyle": "round", "facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
                )
                ax.text(0.02, 0.98, "\n".join(lines), **text_kwargs)

    return fig, ax
