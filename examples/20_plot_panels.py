import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import FitData, models


# A small example showing optional diagnostic panels:
# - next-point score (top)
# - residuals (bottom)

rng = np.random.default_rng(0)

xmin, xmax = -0.6, 0.6
model = (
    models.gaussian_with_offset(name="spectral_line")
    .bound(x0=(xmin, xmax), y0=(-1.0, 1.0), a=(0.0, 2.0), sigma=(0.01, 0.5))
    .guess(x0=0.0, y0=0.0, a=0.7, sigma=0.12)
)

true = dict(x0=0.18, y0=0.05, a=1.0, sigma=0.07)
yerr = 0.06

x = np.linspace(xmin, xmax, 17)
y = model.eval(x, **true) + rng.normal(0.0, yerr, size=x.size)

data = FitData.normal(
    x=x,
    y=y,
    yerr=yerr,
    x_label="detuning",
    y_label="signal",
    label="data",
)
run = model.fit(data).squeeze()

xg = np.linspace(xmin, xmax, 800)

fig, axes = run.plot(
    xg=xg,
    residuals=True,
    nextpoint=True,
    nextpoint_options={"n": 3, "min_separation": 0.03},
    return_axes=True,
)

axes["main"].plot(xg, model.eval(xg, **true), "k--", lw=1, label="true")
axes["main"].legend(loc="best")

plt.show()

