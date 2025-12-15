import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import models


model = (
    models.sinusoid(name="wave")
    .bound(
        amplitude=(0.0, 5.0),
        offset=(-2.0, 2.0),
        frequency=(0.5, 6.0),
        phase=(0.0, 2.0 * np.pi),
    )
    .wrap(phase=True)
    .guess(amplitude=1.0, offset=0.0, frequency=3.0, phase=0.0)
)

rng = np.random.default_rng(0)
x = np.linspace(0.0, 1.0, 12)
sigma = 0.25

true = dict(amplitude=1.8, offset=0.2, frequency=3.2, phase=1.0)
y = model.eval(x, **true) + rng.normal(0.0, sigma, size=x.size)

run = model.fit(
    x,
    (y, sigma),
    backend="ultranest",
    backend_options={
        # Helps keep logs tidy and makes it obvious where outputs went.
        "log_dir": "ultranest_wave_example",
        # Keep default resume behaviour unless you prefer otherwise.
        # "resume": "subfolder",
    },
).squeeze()

res = run.results
print(res.summary(digits=5))

xg = np.linspace(x.min(), x.max(), 800)
yg = run.predict(xg)

# With ultranest, band(method="auto") should select posterior sampling automatically.
band = run.band(xg, nsamples=800, level=2, method="auto")

fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=sigma, fmt=".", ms=3, alpha=0.7, label="data")
ax.plot(xg, model.eval(xg, **true), "k--", lw=1, label="true")
ax.plot(xg, yg, label="posterior mean (via results)")
ax.fill_between(xg, band.low, band.high, alpha=0.2, label="~2σ posterior band")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend()
plt.show()

