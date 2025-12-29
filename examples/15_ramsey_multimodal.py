import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import FitData, models


def main() -> None:
    try:
        import ultranest  # noqa: F401
    except Exception as exc:
        print("ultranest not installed; install it to run this example.")
        print(f"Import error: {exc}")
        return

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
        backend="ultranest"
    ).squeeze()

    print("scipy.curve_fit")
    print(run_cf.results.summary(digits=5))
    print()
    print("ultranest")
    print(run_ultra.results.summary(digits=5))

    xg = np.linspace(float(np.min(x)), float(np.max(x)), 1200)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True, constrained_layout=True)

    # Show the gap explicitly.
    for ax in axs:
        ax.axvspan(float(np.max(t0)), float(np.min(t1)), color="0.9", zorder=0)

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


if __name__ == "__main__":
    main()
