import numpy as np
import matplotlib.pyplot as plt
from sensible_fitting import models

model = (
    models.sinusoid(name="wave")
    .fix(offset=0.0, phase=np.pi / 3)
    .bound(amplitude=(0.2, 5.0), frequency=(1.0, 6.0))
    .guess(frequency=2.9)
)

rng = np.random.default_rng(2)
N_SYSTEMS, N = 4, 25
x = np.linspace(0, 1, N)

A0, F0 = 2.0, 3.0
A = A0 * (1 + 0.08 * rng.normal(size=N_SYSTEMS))
F = F0 * (1 + 0.1 * rng.normal(size=N_SYSTEMS))

sigma = 0.2
y_clean = np.stack(
    [model.eval(x, amplitude=A[i], frequency=F[i]) for i in range(N_SYSTEMS)]
)
y = y_clean + rng.normal(0, sigma, size=y_clean.shape)

run = model.fit(x, (y, sigma))
res = run.results
print(res.summary(digits=4))

fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
axs = axs.ravel()
xg = np.linspace(x.min(), x.max(), 500)

for i, ax in enumerate(axs):
    ax.errorbar(x, y[i], yerr=sigma, fmt=".", ms=3, label=f"data {i}")
    sub = run[i]
    yi = sub.model.eval(xg, params=sub.results.params)
    ax.plot(xg, yi, label="fit")
    band = sub.band(xg, nsamples=300, level=2)
    ax.fill_between(xg, band.low, band.high, alpha=0.2)
    ax.set_title(f"system {i}")
    ax.legend()

plt.show()
