import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import FitData, models

model = models.straight_line().guess(m=0.0, b=0.0)

rng = np.random.default_rng(123)
xs = []
ys = []

for i in range(3):
    n = 30 + 10 * i
    x = np.sort(rng.uniform(-2, 2, size=n))
    sigma = 0.1 + 0.05 * rng.random(size=n)
    y = 0.5 * x - 0.1 + rng.normal(0, sigma)
    xs.append(x)
    ys.append((y, sigma))

data = FitData(
    x=xs,
    data=ys,  # ragged batch payload: [(y0, sigma0), (y1, sigma1), ...]
    data_format="normal",
    x_label="x",
    y_label="y",
    label="data",
)
run = model.fit(data)
print(run.results.summary(digits=4))

fig, axs = plt.subplots(1, len(xs), figsize=(12, 3.5), sharey=True, constrained_layout=True)
run.plot(axs=axs)
for i, ax in enumerate(np.asarray(axs, dtype=object).ravel()):
    title = ax.get_title()
    ax.set_title(f"dataset {i}\n{title}" if title else f"dataset {i}")
    ax.legend()
plt.show()
