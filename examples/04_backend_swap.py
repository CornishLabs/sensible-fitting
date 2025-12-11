import numpy as np
import matplotlib.pyplot as plt
from sensible_fitting import Model


def line(x, m, b):
    return m * x + b


# bounds (useful for future Bayesian backends)
model = Model.from_function(line).bound(m=(-10, 10), b=(-10, 10))

rng = np.random.default_rng(5)
x = np.linspace(0, 4, 50)
sigma = 0.3
y = line(x, 1.7, -0.4) + rng.normal(0, sigma, size=x.size)

run_cf = model.fit(x=x, y=(y, sigma), backend="scipy.curve_fit").squeeze()

fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=sigma, fmt=".", label="data")

xg = np.linspace(x.min(), x.max(), 400)
ax.plot(xg, run_cf.model.eval(xg, params=run_cf.results.params), label="curve_fit")

band_cf = run_cf.band(xg, level=2, method="covariance")
ax.fill_between(xg, band_cf.low, band_cf.high, alpha=0.2, label="curve_fit ~2σ")

ax.legend()
plt.show()
