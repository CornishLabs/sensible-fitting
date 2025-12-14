import numpy as np
from sensible_fitting import models

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

run = model.fit(xs, ys)
print(run.results.summary(digits=4))
