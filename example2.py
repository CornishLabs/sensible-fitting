import numpy as np
import matplotlib.pyplot as plt
from fitwrap import models

# Pre-baked model: straight line
# (still supports bounds/guessers/fixes)
model = models.straight_line().bound(m=(-10, 10))

rng = np.random.default_rng(1)
x = np.linspace(-2, 2, 60)
sigma = 0.15
y = 0.7*x + 0.2 + rng.normal(0, sigma, size=x.size)

run = model.fit(x=x, y=(y, sigma), backend="scipy.curve_fit", return_run=True).squeeze()
res = run.results

print(res.summary(digits=6))

fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=sigma, fmt=".", label="data")

xg = np.linspace(x.min(), x.max(), 400)
ax.plot(xg, run.model.eval(xg, params=res.params), label="fit")

band = run.band(xg, nsamples=500, interval=0.954)
ax.fill_between(xg, band.low, band.high, alpha=0.2, label="~2σ band")

ax.legend()
plt.show()
