from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence, Tuple, Literal
from warnings import warn

import numpy as np

from .data import prepare_datasets
from .inputs import FitData

def uncertainty_to_string(
    x: float, err: float, precision: int | str | None = 1
) -> str:
    """Format a value with uncertainty as a compact string.

    Returns the shortest string representation of x +/- err as either
    x.xx(ee)e+xx or xxx.xx(ee). Use precision="auto" to follow the
    common 1-or-2 significant-digit rule for the uncertainty.
    """
    auto = precision is None or (
        isinstance(precision, str) and precision.lower() == "auto"
    )
    x = float(x)
    err = float(err)

    if math.isnan(x) or math.isnan(err):
        return "NaN"
    if math.isinf(x) or math.isinf(err):
        return "inf"

    err = abs(err)
    if err == 0.0:
        if auto:
            precision = 1
        precision = max(1, int(precision))  # type: ignore[arg-type]
        return f"{x:.{precision}g}(0)"

    err_exp = int(math.floor(math.log10(err)))
    if auto:
        leading = int(err / (10 ** err_exp) + 1e-12)
        precision = 2 if leading == 1 else 1
    precision = max(1, int(precision))  # type: ignore[arg-type]

    if x == 0.0 or abs(x) < err:
        x_exp = err_exp
    else:
        x_exp = int(math.floor(math.log10(abs(x))))

    un_exp = err_exp - precision + 1
    un_int = round(err * 10 ** (-un_exp))

    no_exp = un_exp
    no_int = round(x * 10 ** (-no_exp))

    fieldw = x_exp - no_exp
    fmt = f"%.{fieldw}f"
    result1 = (fmt + "(%.0f)e%d") % (no_int * 10 ** (-fieldw), un_int, x_exp)

    fieldw = max(0, -no_exp)
    fmt = f"%.{fieldw}f"
    result2 = (fmt + "(%.0f)") % (no_int * 10 ** no_exp, un_int * 10 ** max(0, un_exp))

    return result2 if len(result2) <= len(result1) else result1


def plot_fit(
    *,
    ax: Optional[Any] = None,
    x: Any,
    y: Any,
    yerr: Optional[Any] = None,
    run: Optional[Any] = None,
    which: str = "fit",
    xg: Optional[np.ndarray] = None,
    data: bool = True,
    line: bool = True,
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
    data : bool
        If True, draw the data points (and error bars if yerr is provided).
    line : bool
        If True, draw the fit/seed line using run.predict(...).
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

    if data:
        if yerr is None:
            data_kwargs.setdefault("marker", "o")
            data_kwargs.setdefault("linestyle", "none")
            ax.plot(x_arr, y_arr, **data_kwargs)
        else:
            yerr_arr = np.asarray(yerr)
            if yerr_arr.shape not in ((), x_arr.shape):
                raise ValueError(
                    "plot_fit requires yerr to be scalar or same shape as y."
                )
            data_kwargs.setdefault("fmt", "o")
            data_kwargs.setdefault("ms", 4)
            data_kwargs.setdefault("capsize", 2)
            ax.errorbar(x_arr, y_arr, yerr=yerr, **data_kwargs)

    if run is not None:
        if xg is None:
            xg = np.linspace(float(np.min(x_arr)), float(np.max(x_arr)), 400)
        if line:
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


def plot_run(
    run: Any,
    ax: Optional[Any] = None,
    axs: Optional[Any] = None,
    data: bool = True,
    errorbars: bool | Literal["auto"] = "auto",
    line: bool = True,
    band: bool | Literal["auto"] = "auto",
    which: Literal["auto", "fit", "seed"] = "auto",
    xg: Optional[np.ndarray] = None,
    band_options: Optional[Mapping[str, Any]] = None,
    band_kwargs: Optional[Mapping[str, Any]] = None,
    data_kwargs: Optional[Mapping[str, Any]] = None,
    line_kwargs: Optional[Mapping[str, Any]] = None,
    title: bool | str | None = True,
    title_names: Optional[Sequence[str]] = None,
    title_digits: int | str | None = "auto",
    title_kwargs: Optional[Mapping[str, Any]] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    hide_unused: bool = True,
) -> Tuple[Any, Any]:
    """High-level plotting helper for a Run.

    Defaults:
    - plots data points (with error bars if available),
    - plots the fitted curve (or seed curve if optimise=False),
    - plots an uncertainty band when available,
    - sets the Axes title to a compact parameter summary.

    For batched runs, pass `axs` (an array-like of Matplotlib axes) to plot each
    batch element into its own subplot.
    """
    if axs is not None and ax is not None:
        raise ValueError("Provide only one of ax= or axs=.")

    batch_shape = tuple(getattr(getattr(run, "results", None), "batch_shape", ()))
    if batch_shape != ():
        if axs is None:
            # Sensible default layout if the caller didn't provide axes.
            import matplotlib.pyplot as plt

            batch_size = int(np.prod(batch_shape))
            ncols = int(np.ceil(np.sqrt(batch_size)))
            nrows = int(np.ceil(batch_size / ncols))
            fig, axs = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True)
        else:
            fig = None

        axs_arr = np.asarray(axs, dtype=object)
        batch_size = int(np.prod(batch_shape)) if batch_shape else 1
        if axs_arr.size < batch_size:
            raise ValueError(
                f"axs has size {axs_arr.size} but run has batch_size {batch_size}."
            )

        # If axs has matching shape, index it with the same batch indices.
        if axs_arr.shape == batch_shape:
            for idx in np.ndindex(batch_shape):
                plot_run(
                    run=run[idx],
                    ax=axs_arr[idx],
                    data=data,
                    errorbars=errorbars,
                    line=line,
                    band=band,
                    which=which,
                    xg=xg,
                    band_options=band_options,
                    band_kwargs=band_kwargs,
                    data_kwargs=data_kwargs,
                    line_kwargs=line_kwargs,
                    title=title,
                    title_names=title_names,
                    title_digits=title_digits,
                    title_kwargs=title_kwargs,
                    x_label=x_label,
                    y_label=y_label,
                    hide_unused=hide_unused,
                )
            return axs_arr.ravel()[0].figure, axs

        # Otherwise plot in flattened order.
        flat = axs_arr.ravel()
        for i, idx in enumerate(np.ndindex(batch_shape)):
                plot_run(
                    run=run[idx],
                    ax=flat[i],
                    data=data,
                    errorbars=errorbars,
                    line=line,
                    band=band,
                    which=which,
                    xg=xg,
                    band_options=band_options,
                    band_kwargs=band_kwargs,
                    data_kwargs=data_kwargs,
                    line_kwargs=line_kwargs,
                    title=title,
                    title_names=title_names,
                    title_digits=title_digits,
                    title_kwargs=title_kwargs,
                    x_label=x_label,
                    y_label=y_label,
                    hide_unused=hide_unused,
                )

        if hide_unused:
            for j in range(batch_size, flat.size):
                try:
                    flat[j].set_visible(False)
                except Exception:
                    pass

        return flat[0].figure, axs

    # ---- scalar run plotting -------------------------------------------------
    x, y, yerr = _infer_xyyerr_from_run(run)
    meta = _meta_from_run(run)

    if which == "auto":
        which_use: Literal["fit", "seed"] = "seed" if _is_seed_only(run) else "fit"
    else:
        which_use = which

    # Decide whether to show error bars.
    if errorbars == "auto":
        yerr_use = yerr
    else:
        yerr_use = yerr if bool(errorbars) else None

    # Decide whether to show a band.
    if band == "auto":
        band_use = _can_compute_band(run)
    else:
        band_use = bool(band)

    # Provide sensible styling defaults but allow caller overrides.
    data_kwargs = dict(data_kwargs or {})
    data_kwargs.setdefault("label", meta.get("label", "data"))
    line_kwargs = dict(line_kwargs or {})
    band_kwargs = dict(band_kwargs or {})
    band_options = dict(band_options or {})

    # Default band options are already sensible; make them explicit for clarity.
    band_options.setdefault("method", "auto")
    band_options.setdefault("level", 2.0)
    band_options.setdefault("nsamples", 400)

    fig, ax = plot_fit(
        ax=ax,
        x=x,
        y=y,
        yerr=yerr_use,
        run=(run if (line or band_use) else None),
        which=which_use,
        xg=xg,
        data=data,
        line=line,
        band=band_use,
        band_options=band_options,
        band_kwargs=band_kwargs,
        data_kwargs=data_kwargs,
        line_kwargs=line_kwargs,
        show_params=False,
    )

    _apply_title(
        ax=ax,
        run=run,
        which=which_use,
        title=title,
        names=title_names,
        digits=title_digits,
        title_kwargs=title_kwargs,
    )

    _apply_axis_labels(
        ax=ax,
        x_label=(x_label if x_label is not None else meta.get("x_label")),
        y_label=(y_label if y_label is not None else meta.get("y_label")),
    )

    return fig, ax


def _is_seed_only(run: Any) -> bool:
    """Best-effort detection of a seed-only run."""
    msg = getattr(run, "message", "")
    if isinstance(msg, str) and "seed only" in msg:
        return True
    if isinstance(msg, str) and "optimise=False" in msg:
        return True
    return False


def _can_compute_band(run: Any) -> bool:
    """Return True if run.band(...) is likely to succeed for this scalar run."""
    res = getattr(run, "results", None)
    if res is None or getattr(res, "batch_shape", ()) != ():
        return False
    stats = getattr(res, "stats", {}) or {}
    samples = stats.get("posterior_samples", None)
    if isinstance(samples, np.ndarray) and samples.size:
        return True
    cov = getattr(res, "cov", None)
    return cov is not None


def _infer_xyyerr_from_run(run: Any) -> Tuple[np.ndarray, np.ndarray, Optional[Any]]:
    """Infer (x, y, yerr) for 1D plotting from `run.data`."""
    data = getattr(run, "data", None) or {}
    if "x" not in data or "data" not in data:
        raise ValueError("Run does not contain stored data; pass x/y explicitly to plot_fit.")

    x = np.asarray(data["x"])
    payload = data["data"]
    fmt = str(getattr(run, "data_format", "normal"))

    if fmt == "normal":
        if isinstance(payload, tuple) and len(payload) == 2:
            y, sigma = payload
            return x, np.asarray(y), sigma
        if isinstance(payload, tuple) and len(payload) == 3:
            y, lo, hi = payload
            sigma = 0.5 * (np.asarray(lo) + np.asarray(hi))
            return x, np.asarray(y), sigma
        return x, np.asarray(payload), None

    if fmt == "binomial":
        if not (isinstance(payload, tuple) and len(payload) == 2):
            raise TypeError("binomial run expects stored data payload (n, k).")
        n, k = payload
        n = np.asarray(n, dtype=float)
        k = np.asarray(k, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(n > 0, k / n, 0.0)
            yerr = np.sqrt(np.clip(p * (1.0 - p) / np.where(n > 0, n, 1.0), 0.0, np.inf))
        return x, p, yerr

    raise NotImplementedError(
        f"plot_run does not yet know how to infer x/y/yerr for data_format={fmt!r}. "
        "Use plot_fit(...) with explicit x/y/yerr."
    )


def plot_data(
    data: FitData,
    *,
    ax: Optional[Any] = None,
    axs: Optional[Any] = None,
    errorbars: bool | Literal["auto"] = "auto",
    data_kwargs: Optional[Mapping[str, Any]] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    hide_unused: bool = True,
) -> Tuple[Any, Any]:
    """Plot FitData with sensible defaults (no fit line/band)."""
    if axs is not None and ax is not None:
        raise ValueError("Provide only one of ax= or axs=.")

    fmt = data.data_format or "normal"
    datasets, batch_shape = prepare_datasets(data.x, data.data, fmt, strict=False)

    meta = dict(data.meta)
    if data.x_label is not None:
        meta["x_label"] = data.x_label
    if data.y_label is not None:
        meta["y_label"] = data.y_label
    if data.label is not None:
        meta["label"] = data.label

    if batch_shape != ():
        if axs is None:
            import matplotlib.pyplot as plt

            batch_size = int(np.prod(batch_shape))
            ncols = int(np.ceil(np.sqrt(batch_size)))
            nrows = int(np.ceil(batch_size / ncols))
            fig, axs = plt.subplots(nrows, ncols, squeeze=False, constrained_layout=True)

        axs_arr = np.asarray(axs, dtype=object)
        batch_size = int(np.prod(batch_shape))
        if axs_arr.size < batch_size:
            raise ValueError(
                f"axs has size {axs_arr.size} but data has batch_size {batch_size}."
            )

        flat = axs_arr.ravel()
        for i, ds in enumerate(datasets):
            data_kwargs_i = dict(data_kwargs or {})
            data_kwargs_i.setdefault("label", meta.get("label", "data"))

            fig_i, ax_i = plot_fit(
                ax=flat[i],
                x=_plot_x_for_dataset(ds),
                y=_plot_y_for_dataset(ds),
                yerr=_plot_yerr_for_dataset(ds, errorbars=errorbars),
                data=True,
                run=None,
                data_kwargs=data_kwargs_i,
            )
            _apply_axis_labels(
                ax=ax_i,
                x_label=(x_label if x_label is not None else meta.get("x_label")),
                y_label=(y_label if y_label is not None else meta.get("y_label")),
            )

        if hide_unused:
            for j in range(batch_size, flat.size):
                try:
                    flat[j].set_visible(False)
                except Exception:
                    pass

        return flat[0].figure, axs

    # scalar
    ds = datasets[0]
    data_kwargs = dict(data_kwargs or {})
    data_kwargs.setdefault("label", meta.get("label", "data"))

    fig, ax = plot_fit(
        ax=ax,
        x=_plot_x_for_dataset(ds),
        y=_plot_y_for_dataset(ds),
        yerr=_plot_yerr_for_dataset(ds, errorbars=errorbars),
        data=True,
        run=None,
        data_kwargs=data_kwargs,
    )

    _apply_axis_labels(
        ax=ax,
        x_label=(x_label if x_label is not None else meta.get("x_label")),
        y_label=(y_label if y_label is not None else meta.get("y_label")),
    )

    return fig, ax


def _meta_from_run(run: Any) -> dict[str, Any]:
    data = getattr(run, "data", None) or {}
    meta = data.get("meta", None)
    if isinstance(meta, dict):
        return dict(meta)
    return {}


def _apply_axis_labels(*, ax: Any, x_label: Optional[str], y_label: Optional[str]) -> None:
    if x_label:
        ax.set_xlabel(str(x_label))
    if y_label:
        ax.set_ylabel(str(y_label))


def _plot_x_for_dataset(ds: Any) -> Any:
    return getattr(ds, "x")


def _plot_y_for_dataset(ds: Any) -> np.ndarray:
    fmt = getattr(ds, "format", None)
    payload = getattr(ds, "payload", {}) or {}
    if fmt == "normal":
        return np.asarray(payload["y"], dtype=float)
    if fmt == "binomial":
        n = np.asarray(payload["n"], dtype=float)
        k = np.asarray(payload["k"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(n > 0, k / n, 0.0)
    if fmt == "beta":
        a = np.asarray(payload["alpha"], dtype=float)
        b = np.asarray(payload["beta"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((a + b) > 0, a / (a + b), 0.0)
    raise NotImplementedError(f"Unsupported dataset format for plotting: {fmt!r}")


def _plot_yerr_for_dataset(ds: Any, *, errorbars: bool | Literal["auto"]) -> Optional[Any]:
    if errorbars is False:
        return None
    fmt = getattr(ds, "format", None)
    payload = getattr(ds, "payload", {}) or {}
    if fmt == "normal":
        return payload.get("sigma", None)
    if fmt == "binomial":
        n = np.asarray(payload["n"], dtype=float)
        k = np.asarray(payload["k"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(n > 0, k / n, 0.0)
            return np.sqrt(
                np.clip(p * (1.0 - p) / np.where(n > 0, n, 1.0), 0.0, np.inf)
            )
    if fmt == "beta":
        a = np.asarray(payload["alpha"], dtype=float)
        b = np.asarray(payload["beta"], dtype=float)
        # std of Beta(alpha, beta)
        with np.errstate(divide="ignore", invalid="ignore"):
            ab = a + b
            var = (a * b) / (ab * ab * (ab + 1.0))
            return np.sqrt(np.clip(var, 0.0, np.inf))
    return None

def _apply_title(
    *,
    ax: Any,
    run: Any,
    which: Literal["fit", "seed"],
    title: bool | str | None,
    names: Optional[Sequence[str]],
    digits: int | str | None,
    title_kwargs: Optional[Mapping[str, Any]],
) -> None:
    """Apply a parameter-summary title to an Axes."""
    if title is False or title is None:
        return
    if isinstance(title, str) and title != "auto":
        ax.set_title(title, **(dict(title_kwargs or {})))
        return

    res = getattr(run, "results", None)
    if res is None:
        return

    params = None
    if which == "seed":
        params = getattr(res, "seed", None) or getattr(res, "params", None)
    else:
        params = getattr(res, "params", None)
    if params is None:
        return

    if names is None:
        try:
            names = [n for n, pv in params.items() if not pv.fixed and not pv.derived]
        except Exception:
            names = None

    if not names:
        return

    parts: list[str] = []
    for n in names:
        pv = params[n]
        v = float(pv.value)
        if pv.stderr is None:
            if isinstance(digits, int):
                parts.append(f"{n}={v:.{digits}g}")
            else:
                parts.append(f"{n}={v:.4g}")
        else:
            e = float(pv.stderr)
            parts.append(f"{n}={uncertainty_to_string(v, e, precision=digits)}")

    # Wrap long titles across lines (3 params per line).
    per_line = 3
    lines = [", ".join(parts[i : i + per_line]) for i in range(0, len(parts), per_line)]
    title_str = "\n".join(lines)

    ax.set_title(title_str, **(dict(title_kwargs or {})))
