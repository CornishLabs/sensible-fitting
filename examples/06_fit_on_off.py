import numpy as np
import matplotlib.pyplot as plt
from sensible_fitting import models

fit_data = False  # True => fit. False => plot seed only.

model = (
    models.sinusoid(name="wave")
    .fix(offset=0.0, phase=np.pi / 3)
    .bound(amplitude=(0.2, 5.0), frequency=(1.0, 6.0))
    .guess(frequency=2.8)
    .autoguess("amplitude")
)

rng = np.random.default_rng(2)
N = 250
x = np.linspace(0, 1, N)

sigma = 0.2
y = model.eval(x, amplitude=2.0, frequency=3.0) + rng.normal(0, sigma, size=x.size)

# One code path: always produces a Run
run = model.fit(
    x=x,
    y=(y, sigma),
    backend="scipy.curve_fit",
    return_run=True,
    skip=not fit_data,           # <--- THE KEY
).squeeze()

res = run.results

# Plot
fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=sigma, fmt=".", ms=3, label="data")

xg = np.linspace(x.min(), x.max(), 500)

style = "-" if fit_data else "--"
label = "fit" if fit_data else "seed fit"
ax.plot(xg, run.model.eval(xg, params=res.params), linestyle=style, label=label)

# Band only if covariance exists (fit mode)
if res.cov is not None:
    band = run.band(xg, level=2, nsamples=400)
    ax.fill_between(xg, band.low, band.high, alpha=0.2, label="~2σ band")

ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()

# Print params either way
print(res.summary(digits=5))
