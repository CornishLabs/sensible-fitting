import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import FitData, models


rng = np.random.default_rng(4)

# A simple "spectral line": Gaussian feature on a baseline.
xmin, xmax = -0.6, 0.6
model = (
    models.gaussian_with_offset(name="spectral_line")
    .bound(
        x0=(xmin, xmax),
        y0=(-1.0, 1.0),
        a=(0.0, 2.0),
        sigma=(0.01, 0.5),
    )
    .guess(x0=0.0, y0=0.0, a=0.5, sigma=0.1)
)

true = dict(x0=0.18, y0=0.05, a=1.0, sigma=0.07)
yerr = 0.07

def measure(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(model.eval(x, **true), dtype=float)
    return y + rng.normal(0.0, yerr, size=y.shape)

# ---- initial dataset: a few coarse points across the scan -------------
x0 = np.linspace(xmin, xmax, 7)
data = FitData.normal(
    x=x0,
    y=measure(x0),
    yerr=yerr,
    x_label="detuning",
    y_label="signal",
    label="round 1",
)

# ---- sequential design loop ------------------------------------------
rounds = 6
add_per_round = 4

runs = []
added = []  # (x_new, y_new) per transition

xg = np.linspace(xmin, xmax, 800)

for r in range(rounds):
    run = model.fit(data.with_labels(label=f"round {r+1}"), backend="ultranest").squeeze()
    runs.append(run)

    if r == rounds - 1:
        break

    x_new = run.suggest_next_x(
        bounds=(xmin, xmax),
        n_candidates=800,
        n=add_per_round,
        min_separation=0.03,
        objective="auto",  # uses info_gain when yerr is available
        method="auto",
    )
    x_new = np.sort(np.atleast_1d(np.asarray(x_new, dtype=float)))
    y_new = measure(x_new)
    added.append((x_new, y_new))

    data = data.append(x=x_new, y=y_new).sorted()

# ---- plot the rounds ---------------------------------------------------
fig, axs = plt.subplots(
    2, 3, figsize=(14, 7), sharex=True, sharey=True, constrained_layout=True
)
flat = axs.ravel()

for r, run in enumerate(runs):
    ax = flat[r]
    run.plot(
        ax=ax,
        xg=xg,
        title_digits=4,
        band_kwargs={"alpha": 0.2},
    )

    if r > 0:
        x_new, y_new = added[r - 1]
        ax.plot(
            x_new,
            y_new,
            "s",
            ms=6,
            mfc="none",
            mec="C2",
            mew=1.5,
            label="new points" if r == 1 else "_nolegend_",
        )

    ax.plot(
        xg,
        model.eval(xg, **true),
        "k--",
        lw=1,
        label="true" if r == 0 else "_nolegend_",
    )

    npts = int(np.asarray(run.data["x"]).reshape((-1,)).size) if run.data is not None else 0
    ax.set_title(f"Round {r+1} (N={npts})\n" + str(ax.get_title()))
    ax.legend(loc="best", fontsize=8)

for j in range(len(runs), flat.size):
    flat[j].set_visible(False)

plt.show()
