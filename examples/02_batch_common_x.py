import numpy as np
import matplotlib.pyplot as plt
from sensible_fitting import FitData, models

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

data = FitData.normal(
    x=x,
    y=y,
    yerr=sigma,
    x_label="time",
    y_label="signal",
    label="data",
)
run = model.fit(data)
res = run.results
print(res.summary(digits=4))

fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, constrained_layout=True)
run.plot(axs=axs)

flat = np.asarray(axs, dtype=object).ravel()
for i in range(N_SYSTEMS):
    title = flat[i].get_title()
    flat[i].set_title(f"system {i}\n{title}" if title else f"system {i}")
    flat[i].legend()

plt.show()
