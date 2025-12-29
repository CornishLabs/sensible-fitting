import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import models
from sensible_fitting.viz import plot_fit


def main() -> None:
    rng = np.random.default_rng(0)

    model = models.rabi_oscillation().bound(
        amplitude=(0.0, 1.0),
        offset=(0.0, 1.0),
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

    fig, ax = plt.subplots()
    # cov_method="auto" uses hess_inv if available; numdiff is steadier for binomial fits.
    runa = model.fit(
        x,
        (n, k),
        data_format="binomial",
        backend="scipy.minimize",
        backend_options={"cov_method": "numdiff"},
    ).squeeze()
    resa = runa.results

    print(resa.summary(digits=5))

    plot_fit(
        ax=ax,
        x=x,
        y=y,
        yerr=yerr,
        run=runa,
        band=True,
        band_options={"level": 2, "nsamples": 400, "method": "auto"},
        show_params=True,
    )
    runb = model.fit(
        x,
        (n, k),
        data_format="binomial",
        backend="ultranest"
    ).squeeze()
    resb = runb.results

    print(resb.summary(digits=5))

    plot_fit(
        ax=ax,
        x=x,
        y=y,
        yerr=yerr,
        run=runb,
        band=True,
        band_options={"level": 2, "nsamples": 400, "method": "auto"},
        show_params=True,
    )

    ax.set_xlabel("time")
    ax.set_ylabel("population")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
