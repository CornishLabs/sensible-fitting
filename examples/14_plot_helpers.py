import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import FitData, Model


def line(x, m, b):
    return m * x + b


model = Model.from_function(line).bound(m=(-10, 10), b=(-10, 10))

rng = np.random.default_rng(0)
x = np.linspace(0, 10, 30)
sigma = 1.0
y = line(x, 2.0, -1.0) + rng.normal(0, sigma, size=x.size)

data = FitData.normal(x=x, y=y, yerr=sigma, x_label="x", y_label="y")
run = model.fit(data).squeeze()

xg = np.linspace(x.min(), x.max(), 400)
fig, ax = plt.subplots()

# High-level plotting: uses run.data + sensible defaults.
run.plot(ax=ax, xg=xg)
ax.plot(xg, line(xg, 2.0, -1.0), "k--", lw=1, label="true")

ax.legend()
plt.show()
