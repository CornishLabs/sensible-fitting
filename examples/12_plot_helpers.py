import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import Model, plot_fit


def line(x, m, b):
    return m * x + b


model = Model.from_function(line).bound(m=(-10, 10), b=(-10, 10))

rng = np.random.default_rng(0)
x = np.linspace(0, 10, 30)
sigma = 1.0
y = line(x, 2.0, -1.0) + rng.normal(0, sigma, size=x.size)

run = model.fit(x, (y, sigma)).squeeze()

fig, ax = plt.subplots()

plot_fit(
    ax=ax,
    x=x,
    y=y,
    yerr=sigma,
    run=run,
    band=True,
    band_options={"level": 2, "nsamples": 400},
    data_kwargs={"label": "data"},
    line_kwargs={"label": "fit"},
    show_params=True,
)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()
