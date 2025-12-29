import numpy as np
import matplotlib.pyplot as plt
from sensible_fitting import FitData, Model


def line(x, m, b):
    return m * x + b


# bounds (useful for future Bayesian backends)
model = Model.from_function(line).bound(m=(-10, 10), b=(-10, 10))

rng = np.random.default_rng(5)
x = np.linspace(0, 4, 50)
sigma = 0.3
y = line(x, 1.7, -0.4) + rng.normal(0, sigma, size=x.size)

data = FitData.normal(x=x, y=y, yerr=sigma, x_label="x", y_label="y", label="data")
run_cf = model.fit(data, backend="scipy.curve_fit").squeeze()

xg = np.linspace(x.min(), x.max(), 400)
fig, ax = run_cf.plot(xg=xg, line_kwargs={"label": "scipy.curve_fit"})
ax.plot(xg, line(xg, 1.7, -0.4), "k--", lw=1, label="true")

ax.legend()
plt.show()
