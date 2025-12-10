import numpy as np
from fitwrap import models

model = models.straight_line()

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

run = model.fit(x=xs, y=ys, backend="scipy.curve_fit", return_run=True)
print(run.results.summary(digits=4))
