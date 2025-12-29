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

    rng = np.random.default_rng(1)

    model = models.rabi_oscillation().bound(
        amplitude=(0.0, 0.8),
        offset=(0.1, 0.9),
        t_period=(0.5, 8.0),
        phase=(-np.pi, np.pi),
        tau=(0.5, 30.0),
        t_dead=(0.0, 2.0),
    )

    # Irregularly spaced time points.
    x = np.sort(rng.uniform(0.0, 10.0, size=60))
    true = {
        "amplitude": 0.45,
        "offset": 0.5,
        "t_period": 3.2,
        "phase": 0.6,
        "tau": 8.0,
        "t_dead": 0.4,
    }
    p_true = model.eval(x, **true)
    n = np.full(x.shape, 200, dtype=int)
    k = rng.binomial(n, p_true)

    xg = np.linspace(float(np.min(x)), float(np.max(x)), 400)

    data = FitData.binomial(
        x=x,
        n=n,
        k=k,
        x_label="time",
        y_label="population",
        label="data",
    )

    run_min = model.fit(
        data,
        backend="scipy.minimize",
        backend_options={"cov_method": "numdiff"},
    ).squeeze()

    run_ultra = model.fit(
        data,
        backend="ultranest",
        # backend_options={"max_ncalls": 4000},
    ).squeeze()

    print("scipy.minimize + numdiff")
    print(run_min.results.summary(digits=4))
    print()
    print("ultranest")
    print(run_ultra.results.summary(digits=4))

    fig, ax = plt.subplots()
    run_min.plot(
        ax=ax,
        xg=xg,
        title=False,
        line_kwargs={"label": "minimize (numdiff)", "color": "C0"},
        band_kwargs={"alpha": 0.2, "color": "C0"},
    )
    run_ultra.plot(
        ax=ax,
        xg=xg,
        data=False,
        title=False,
        line_kwargs={"label": "ultranest", "color": "C1"},
        band_kwargs={"alpha": 0.2, "color": "C1"},
    )
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Rabi oscillation: minimize vs ultranest")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
