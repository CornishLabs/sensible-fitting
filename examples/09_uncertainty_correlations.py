"""
Example: covariance-aware uncertainties.

Fit a line, then compare predictions using:
 - correlated parameters from res["m"].u / res["b"].u
 - independent parameters built from stderr only
"""

import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat

from sensible_fitting import FitData, Model


def line(x, m, b):
    return m * x + b


def main() -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 10.0, 50)
    m_true, b_true = 2.0, -1.0
    sigma = 0.5
    y = line(x, m_true, b_true) + rng.normal(0, sigma, size=x.shape)

    model = Model.from_function(line).guess(m=1.5,b=-0.5)
    data = FitData.normal(x=x, y=y, yerr=sigma, x_label="x", y_label="y", label="data")
    run = model.fit(data).squeeze()
    res = run.results

    m = res["m"]
    b = res["b"]
    if res.cov is None or m.stderr is None or b.stderr is None:
        raise RuntimeError("Example requires covariance from curve_fit.")

    # Correlated values pulled from the fit covariance.
    m_u = m.u
    b_u = b.u

    # Independent values ignore covariance.
    m_ind = ufloat(m.value, m.stderr)
    b_ind = ufloat(b.value, b.stderr)

    x0 = 10.0
    y_u = m_u * x0 + b_u
    y_ind = m_ind * x0 + b_ind

    print("m:", m_u)
    print("b:", b_u)

    corr = res.cov[0, 1] / np.sqrt(res.cov[0, 0] * res.cov[1, 1])
    print("corr(m,b):", float(corr))

    print(f"predict at x0={x0} with covariance:", y_u)
    print(f"predict at x0={x0} assuming independence:", y_ind)

    v = np.array([x0, 1.0], dtype=float)
    sigma_cov = float(np.sqrt(v @ res.cov @ v))
    print("sigma from cov formula:", sigma_cov)

    # Plot: show how covariance changes the uncertainty envelope on y(x).
    xg = np.linspace(float(x.min()), float(x.max()), 400)
    fig, ax = run.plot(xg=xg, band=False, line_kwargs={"label": "best fit", "color": "C0"})

    yfit = run.predict(xg)
    cov = np.asarray(res.cov, dtype=float)
    var_m = float(cov[0, 0])
    var_b = float(cov[1, 1])
    cov_mb = float(cov[0, 1])

    # Var(m x + b) = x^2 Var(m) + Var(b) + 2 x Cov(m,b)
    sigma_with_cov = np.sqrt(np.clip(xg * xg * var_m + var_b + 2.0 * xg * cov_mb, 0.0, np.inf))
    sigma_indep = np.sqrt(np.clip(xg * xg * (float(m.stderr) ** 2) + float(b.stderr) ** 2, 0.0, np.inf))

    ax.fill_between(
        xg,
        yfit - 2.0 * sigma_indep,
        yfit + 2.0 * sigma_indep,
        alpha=0.15,
        color="C1",
        label="~2σ assuming independence",
    )
    ax.fill_between(
        xg,
        yfit - 2.0 * sigma_with_cov,
        yfit + 2.0 * sigma_with_cov,
        alpha=0.15,
        color="C0",
        label="~2σ using covariance",
    )

    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
