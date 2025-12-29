"""
Example: fits of fits (hierarchical use).

1) For each realisation r:
   - Generate 5 sinusoids with frequencies f_i = a_true + b_true * i, i=0..4.
   - Batch-fit the 5 sinusoids to extract the frequencies f_i (with errors).
   - Fit a straight line f(i) = a + b*i to those 5 frequencies.
   - Store (a_hat, b_hat).

2) Plot the cloud of (a_hat, b_hat) over several realisations.
"""

import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import FitData, Model, models


rng = np.random.default_rng(123)

# Sinusoid model for the first-level fits
sin_model = (
    models.sinusoid(name="wave")
    .fix(offset=0.0, phase=0.0)
    .bound(amplitude=(1.0, 3.0), frequency=(0.5, 6.0))
    .weak_guess(frequency=2.8)
)

# Straight line model: f(i) = a + b * i
def line(i, a, b):
    return a + b * i

line_model = Model.from_function(line, name="linear frequency").guess(a=2.5, b=0.5)

N_REAL = 3  # number of realisations of (a_true, b_true)
N_SYSTEMS = 5  # number of sinusoids per realisation
N = 50  # points per sinusoid

x = np.linspace(0.0, 1.0, N)
sigma_y = 0.2
idx = np.arange(N_SYSTEMS, dtype=float)

a_hats = []
b_hats = []
example_run_sin = None
example_run_line = None
example_true = None  # (a_true, b_true)
example_freq_true = None

for r in range(N_REAL):
    # True linear relation for this realisation
    a_true = 2.5 + 0.2 * rng.normal()
    b_true = 0.5 + 0.05 * rng.normal()

    freq_true = a_true + b_true * idx

    # Generate 5 sinusoids with those frequencies, batched
    y_clean = []
    for i in range(N_SYSTEMS):
        y_clean.append(sin_model.eval(x, amplitude=1.5, frequency=freq_true[i]))
    y_clean = np.stack(y_clean, axis=0)

    y = y_clean + rng.normal(0, sigma_y, size=y_clean.shape)

    # First-level batch fit: get frequency per system
    sin_data = FitData.normal(
        x=x,
        y=y,
        yerr=sigma_y,
        x_label="time",
        y_label="signal",
        label="data",
    )
    # "auto" tries curve_fit first, then falls back to a global optimiser if needed.
    run_sin = sin_model.fit(
        sin_data,
        backend="auto",
        backend_options={"maxiter": 20, "popsize": 8, "seed": 0},
    )
    res_sin = run_sin.results

    freqs = res_sin["frequency"].value
    freq_err = res_sin["frequency"].stderr

    # Second-level fit: line through frequencies vs. index (fits-of-fits helper)
    # (You can also do: line_model.fit(idx, res_sin["frequency"]).)
    run_line = line_model.fit(
        res_sin["frequency"].as_fitdata(
            x=idx,
            x_label="system index i",
            y_label="frequency",
            label="batch-fit frequency ± 1σ",
        )
    ).squeeze()
    res_line = run_line.results

    a_hat = res_line["a"].value
    b_hat = res_line["b"].value

    a_hats.append(a_hat)
    b_hats.append(b_hat)

    if r == 0:
        example_run_sin = run_sin
        example_run_line = run_line
        example_true = (a_true, b_true)
        example_freq_true = freq_true

    print(
        f"realisation {r}: "
        f"true (a,b)=({a_true:.3f}, {b_true:.3f}), "
        f"fit (a,b)=({a_hat:.3f}, {b_hat:.3f})"
    )

a_hats = np.asarray(a_hats)
b_hats = np.asarray(b_hats)

    # --- Lower-level checks: first-level time-domain fits --------------------
if example_run_sin is not None and example_freq_true is not None:
    xg = np.linspace(float(x.min()), float(x.max()), 800)
    fig, axs = plt.subplots(
        2, 3, figsize=(12, 6), sharex=True, sharey=True, constrained_layout=True
    )

    def _overlay_true(ax, subrun, idx):
        i = int(idx[0])
        ax.plot(
            xg,
            sin_model.eval(xg, amplitude=1.5, frequency=float(example_freq_true[i])),
            "k--",
            lw=1,
            label="true",
        )
        ax.legend()

    example_run_sin.plot(
        axs=axs,
        xg=xg,
        errorbars=False,
        band=True,
        data_kwargs={"marker": ".", "ms": 2, "alpha": 0.7},
        title_names=["frequency", "amplitude"],
        panel_title="system {i}",
        each=_overlay_true,
    )

    # First-level extracted parameters vs system index (frequency & amplitude)
    fig, (axf, axa) = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    res = example_run_sin.results
    freq_hat = np.asarray(res["frequency"].value, dtype=float)
    freq_err = (
        None
        if res["frequency"].stderr is None
        else np.asarray(res["frequency"].stderr, dtype=float)
    )
    amp_hat = np.asarray(res["amplitude"].value, dtype=float)
    amp_err = (
        None
        if res["amplitude"].stderr is None
        else np.asarray(res["amplitude"].stderr, dtype=float)
    )

    axf.errorbar(idx, freq_hat, yerr=freq_err, fmt="o", capsize=2, label="fit")
    axf.plot(idx, example_freq_true, "k--", lw=1, label="true")
    axf.set_xlabel("system index i")
    axf.set_ylabel("frequency")
    axf.set_title("First-level extracted frequencies")
    axf.legend()

    axa.errorbar(idx, amp_hat, yerr=amp_err, fmt="o", capsize=2, label="fit")
    axa.axhline(1.5, color="k", linestyle="--", lw=1, label="true")
    axa.set_xlabel("system index i")
    axa.set_ylabel("amplitude")
    axa.set_title("First-level extracted amplitudes")
    axa.legend()

# --- Higher-level summary: second-level fit + cloud ----------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)

# 1) Example second-level fit: frequency vs index with fitted line/band
ax0 = axs[0]
if example_run_line is not None:
    example_run_line.plot(ax=ax0)
    if example_true is not None:
        a_true, b_true = example_true
        ig = np.linspace(float(idx.min()), float(idx.max()), 200)
        ax0.plot(ig, a_true + b_true * ig, "k--", lw=1, label="true")
    ax0.set_xticks(np.arange(N_SYSTEMS))
    ax0.set_title("Example realisation\n" + ax0.get_title())
    ax0.legend()

# 2) Scatter of the (a_hat, b_hat) pairs over realisations
ax1 = axs[1]
ax1.plot(a_hats, b_hats, "o")
ax1.set_xlabel("a_hat")
ax1.set_ylabel("b_hat")
ax1.set_title("Fits-of-fits: (a, b) per realisation")
plt.show()
