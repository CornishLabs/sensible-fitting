import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import FitData, models


def main() -> None:
    rng = np.random.default_rng(3)

    # Change this to "ultranest" for multimodal posteriors (slower but more robust).
    backend = "ultranest"
    backend_options = {}

    if backend == "ultranest":
        try:
            import ultranest  # noqa: F401
        except Exception as exc:
            print("ultranest not installed; install it to run with backend='ultranest'.")
            print(f"Import error: {exc}")
            return
        backend_options = {
            # Keep this example reasonably quick.
            # "max_ncalls": 2500, 
        }

    # A Ramsey-style sinusoid: start with two short time windows and then adaptively
    # add new points to best reduce predictive uncertainty.
    model = (
        models.sinusoid(name="ramsey")
        .bound(
            amplitude=(0.0, 1.0),
            offset=(-0.5, 1.5),
            frequency=(2.6, 3.8),
            phase=(0.0, 2.0 * np.pi),
        )
        .wrap(phase=True)
        .guess(amplitude=0.4, offset=0.5, frequency=3.1, phase=0.0)
    )

    true = dict(amplitude=0.45, offset=0.5, frequency=3.2, phase=1.2)
    sigma = 0.07

    def measure(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(model.eval(x, **true), dtype=float)
        return y + rng.normal(0.0, sigma, size=y.shape)

    # ---- initial dataset: two small clusters ------------------------------
    t0 = np.linspace(0.0, 0.20, 4)
    t1 = np.linspace(4.0, 4.20, 4)
    x = np.sort(np.concatenate([t0, t1]))
    data = FitData.normal(
        x=x,
        y=measure(x),
        yerr=sigma,
        x_label="Ramsey time",
        y_label="signal",
        label="round 1",
    )

    # ---- sequential design loop ------------------------------------------
    rounds = 5  # number of plotted datasets/fits
    add_per_round = 4  # "next best few points"

    runs = []
    added = []  # (x_new, y_new) per transition

    xmin, xmax = 0.0, 10.0
    xg = np.linspace(xmin, xmax, 1200)

    for r in range(rounds):
        run = model.fit(
            data.with_labels(label=f"round {r+1}"),
            backend=backend,
            backend_options=backend_options,
        ).squeeze()

        runs.append(run)

        if r == rounds - 1:
            break

        x_new = run.suggest_next_x(
            bounds=(xmin, xmax),
            n_candidates=1200,
            n=add_per_round,
            min_separation=0.05,
            objective="auto",
            method="auto",
        )
        x_new = np.sort(np.atleast_1d(np.asarray(x_new, dtype=float)))
        y_new = measure(x_new)
        added.append((x_new, y_new))

        data = data.append(x=x_new, y=y_new).sorted()

    # ---- plot the five rounds --------------------------------------------
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
            posterior_lines=(40 if backend == "ultranest" else False),
            posterior_lines_kwargs={"alpha": 0.05, "lw": 1.0},
        )

        # Highlight newly added points for this round.
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

        ax.plot(xg, model.eval(xg, **true), "k--", lw=1, alpha=0.6, label="true" if r == 0 else "_nolegend_")

        npts = int(np.asarray(run.data["x"]).reshape((-1,)).size) if run.data is not None else 0
        ax.set_title(f"Round {r+1} (N={npts})\n" + str(ax.get_title()))
        ax.legend(loc="best", fontsize=8)

    # Hide the unused 6th subplot.
    for j in range(len(runs), flat.size):
        flat[j].set_visible(False)

    plt.show()


if __name__ == "__main__":
    main()
