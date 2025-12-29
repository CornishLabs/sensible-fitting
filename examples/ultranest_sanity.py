#!/usr/bin/env python3
"""
UltraNest sanity check: fit a noisy Gaussian line.

Run:
  uv run examples/ultranest_sanity.py
or:
  python examples/ultranest_sanity.py

This should create an output folder ultranest_sanity_run/run*/ with logs and plots.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

import ultranest
from ultranest.plot import cornerplot


rng = np.random.default_rng(0)

# ---- Synthetic data -----------------------------------------------------
x = np.linspace(400.0, 800.0, 120)
yerr = 1.0

# True parameters
loc_true = 500.0
amp_true = 20.0
width_true = 4.2

def model(xv: np.ndarray, location: float, amplitude: float, width: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((xv - location) / width) ** 2)

y_clean = model(x, loc_true, amp_true, width_true)
y = y_clean + rng.normal(0.0, yerr, size=x.shape)

# ---- UltraNest definitions ---------------------------------------------
param_names = ["location", "amplitude", "width"]

def prior_transform(cube: np.ndarray) -> np.ndarray:
    """
    cube in [0,1]^3 -> physical parameters.
    Mirrors UltraNest docs:
      location  ~ Uniform(400, 800)
      amplitude ~ LogUniform(0.1, 100)
      width     ~ LogNormal-ish via 10**Normal(0,1) (center ~1, 1 dex)
    """
    cube = np.asarray(cube, dtype=float)
    params = cube.copy()

    # uniform location
    lo, hi = 400.0, 800.0
    params[0] = cube[0] * (hi - lo) + lo

    # log-uniform amplitude
    lo, hi = 0.1, 100.0
    params[1] = 10 ** (cube[1] * (np.log10(hi) - np.log10(lo)) + np.log10(lo))

    # log-normal width: 10**Normal(0,1)
    params[2] = 10 ** scipy.stats.norm.ppf(cube[2], 0.0, 1.0)

    return params

def loglike(params: np.ndarray) -> float:
    location, amplitude, width = params
    ym = model(x, float(location), float(amplitude), float(width))
    r = (ym - y) / yerr
    return float(-0.5 * np.sum(r * r))

# ---- Run UltraNest ------------------------------------------------------
sampler = ultranest.ReactiveNestedSampler(
    param_names,
    loglike=loglike,
    transform=prior_transform,
    wrapped_params=[False, False, False],
    log_dir="ultranest_sanity_run",
    resume="subfolder",
)

# A couple of run() kwargs you can tweak if needed:
result = sampler.run(min_num_live_points=400)

sampler.print_results()

# ---- Plots --------------------------------------------------------------
# UltraNest diagnostics (writes files inside log_dir)
try:
    sampler.plot_run()
    sampler.plot_trace()
    sampler.plot_corner()
except Exception as e:
    print("Plotting via sampler.plot_* failed (non-fatal):", e)

# Corner plot (matplotlib figure)
try:
    cornerplot(result)
    plt.suptitle("UltraNest posterior corner plot")
    plt.show()
except Exception as e:
    print("cornerplot(...) failed (non-fatal):", e)

# Fit curve plot using posterior median (more robust for multimodal cases).
samples = np.asarray(result.get("samples", []), dtype=float)
if samples.ndim == 2 and samples.shape[0] > 0:
    center = np.quantile(samples, 0.5, axis=0, method="nearest")
else:
    # some versions provide weighted_samples["points"]
    ws = result.get("weighted_samples", {}) or {}
    pts = np.asarray(ws.get("points", []), dtype=float)
    center = (
        np.quantile(pts, 0.5, axis=0, method="nearest")
        if pts.ndim == 2 and pts.shape[0] > 0
        else None
    )

fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=yerr, fmt="x", label="data")
ax.plot(x, y_clean, "k--", lw=1, label="true")

if center is not None:
    y_fit = model(x, float(center[0]), float(center[1]), float(center[2]))
    ax.plot(x, y_fit, "-", label="posterior median")
    ax.set_title(
        f"posterior median: loc={center[0]:.3f}, amp={center[1]:.3f}, width={center[2]:.3f}"
    )
else:
    ax.set_title("No samples found in result dict (!)")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()
