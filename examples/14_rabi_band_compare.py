import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import models


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
    y = k / n
    yerr = np.sqrt(np.clip(y * (1.0 - y) / n, 0.0, np.inf))

    xg = np.linspace(float(np.min(x)), float(np.max(x)), 400)

    run_min = model.fit(
        x,
        (n, k),
        data_format="binomial",
        backend="scipy.minimize",
        backend_options={"cov_method": "numdiff"},
    ).squeeze()
    band_min = run_min.band(xg, level=2, method="covariance")
    y_min = run_min.predict(xg)

    run_ultra = model.fit(
        x,
        (n, k),
        data_format="binomial",
        backend="ultranest",
        # backend_options={"max_ncalls": 4000},
    ).squeeze()
    band_ultra = run_ultra.band(xg, level=2, method="posterior")
    y_ultra = run_ultra.predict(xg)

    print("scipy.minimize + numdiff")
    print(run_min.results.summary(digits=4))
    print()
    print("ultranest")
    print(run_ultra.results.summary(digits=4))

    fig, ax = plt.subplots()
    ax.errorbar(x, y, yerr=yerr, fmt=".", ms=4, alpha=0.6, label="data")
    ax.plot(xg, y_min, label="minimize (numdiff)")
    ax.fill_between(xg, band_min.low, band_min.high, alpha=0.2)
    ax.plot(xg, y_ultra, label="ultranest")
    ax.fill_between(xg, band_ultra.low, band_ultra.high, alpha=0.2)
    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
