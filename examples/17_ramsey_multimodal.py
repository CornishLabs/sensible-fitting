import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import FitData, models


try:
    import ultranest  # noqa: F401
except Exception as exc:
    raise SystemExit(f"ultranest not installed; install it to run this example. ({exc})")

rng = np.random.default_rng(2)

# Ramsey-style oscillation: we only measure a short segment near t~0 and another
# short segment many cycles later. If the data are sparse/noisy, the "number of
# fringes skipped" in the gap becomes ambiguous, yielding a multimodal posterior
# in (frequency, phase).
model = (
    models.sinusoid(name="ramsey")
    .bound(
        amplitude=(0.0, 1.0),
        offset=(-0.5, 1.5),
        # Wide enough to include several "fringe count" modes.
        frequency=(2.6, 3.8),
        phase=(0.0, 2.0 * np.pi),
    )
    .wrap(phase=True)
    # Seed isn't used by ultranest (nested sampling), but helps curve_fit/minimize.
    .guess(amplitude=0.4, offset=0.5, frequency=3.1, phase=0.0)
)

true = dict(amplitude=0.45, offset=0.5, frequency=3.2, phase=1.2)
sigma = 0.07

t0 = np.linspace(0.0, 0.20, 12)
t1 = np.linspace(4.0, 4.20, 12)
x = np.sort(np.concatenate([t0, t1]))
y = model.eval(x, **true) + rng.normal(0.0, sigma, size=x.size)

data = FitData.normal(
    x=x,
    y=y,
    yerr=sigma,
    x_label="Ramsey time",
    y_label="signal",
    label="data",
)

run_cf = model.fit(data, backend="scipy.curve_fit").squeeze()
run_ultra = model.fit(
    data,
    backend="ultranest",
).squeeze()

print("scipy.curve_fit")
print(run_cf.results.summary(digits=5))
print()
print("ultranest")
print(run_ultra.results.summary(digits=5))

xg = np.linspace(float(np.min(x)), float(np.max(x)), 1200)

suggest = run_ultra.suggest_next_x(
    candidates=xg,
    method="posterior",
    objective="max_width",
    n=3,
    min_separation=0.05,
    return_details=True,
)
x_next = np.atleast_1d(np.asarray(suggest.x, dtype=float))

fig, axs = plt.subplots(
    1, 2, figsize=(12, 4), sharex=True, sharey=True, constrained_layout=True
)

# Show the gap explicitly.
for ax in axs:
    ax.axvspan(float(np.max(t0)), float(np.min(t1)), color="0.9", zorder=0)
    for i, xv in enumerate(x_next.tolist()):
        ax.axvline(
            float(xv),
            color="C2",
            lw=1,
            alpha=0.8,
            linestyle="--",
            label=("suggested next point" if i == 0 else "_nolegend_"),
        )

run_cf.plot(
    ax=axs[0],
    xg=xg,
    title=False,
    line_kwargs={"label": "curve_fit", "color": "C0"},
    band_kwargs={"alpha": 0.2, "color": "C0"},
)
axs[0].plot(xg, model.eval(xg, **true), "k--", lw=1, label="true")
axs[0].set_title("Local fit (single mode)")
axs[0].legend()

run_ultra.plot(
    ax=axs[1],
    xg=xg,
    title=False,
    line_kwargs={"label": "posterior median", "color": "C1"},
    band_kwargs={"alpha": 0.2, "color": "C1"},
    posterior_lines=60,
    posterior_lines_kwargs={"alpha": 0.05, "lw": 1.0},
)
axs[1].plot(xg, model.eval(xg, **true), "k--", lw=1, label="true")
axs[1].set_title("Multimodal posterior: sample curves")
axs[1].legend()

fig3, ax3 = plt.subplots()
ax3.plot(suggest.candidates, suggest.score, color="C2")
ax3.axvspan(float(np.max(t0)), float(np.min(t1)), color="0.9", zorder=0)
for xv in x_next.tolist():
    ax3.axvline(float(xv), color="C2", lw=1, linestyle="--")
ax3.set_xlabel("Ramsey time")
ax3.set_ylabel("score")
ax3.set_title("Next-point score (max predictive width)")

fig2, ax2 = plt.subplots()
stats = run_ultra.results.stats or {}
samples = np.asarray(stats.get("posterior_samples", []), dtype=float)
free_names = list(stats.get("free_names", ()))
if samples.ndim == 2 and samples.size and "frequency" in free_names:
    f = samples[:, free_names.index("frequency")]
    ax2.hist(f, bins=50, color="C1", alpha=0.75)
    ax2.axvline(true["frequency"], color="k", lw=1, linestyle="--", label="true")
    ax2.set_xlabel("frequency")
    ax2.set_ylabel("posterior sample count")
    ax2.set_title("Multimodal posterior in frequency (fringe ambiguity)")
    ax2.legend()
else:
    ax2.text(0.5, 0.5, "No posterior samples found.", ha="center", va="center")

plt.show()
