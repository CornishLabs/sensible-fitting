import numpy as np
import matplotlib.pyplot as plt
from sensible_fitting import Model

def line(x, m, b):
    return m * x + b

model = Model.from_function(line, name="straight line")

rng = np.random.default_rng(0)
x = np.linspace(0, 10, 50)
y_true = line(x, 2.0, -1.0)
sigma = 0.6
y = y_true + rng.normal(0, sigma, size=x.size)

run = model.fit(x=x, y=(y, sigma), backend="scipy.curve_fit", return_run=True).squeeze()
res = run.results

print(res.params["m"].value, "±", res.params["m"].error)
print(res.params["b"].value, "±", res.params["b"].error)
print(res.summary(digits=4))

fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=sigma, fmt="o", ms=4, capsize=2, label="data")

xg = np.linspace(x.min(), x.max(), 400)
ax.plot(xg, run.model.eval(xg, params=res.params), label="fit")

band = run.band(xg, nsamples=400, level=2)
ax.fill_between(xg, band.low, band.high, alpha=0.2, label="~2σ band")

ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()
