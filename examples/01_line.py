import numpy as np
import matplotlib.pyplot as plt
from sensible_fitting import FitData, Model


def line(x, m, b):
    return m * x + b


model = Model.from_function(line, name="straight line").bound(m=(-10, 10), b=(-10, 10))

rng = np.random.default_rng(0)
x = np.linspace(0, 10, 20)
y_true = line(x, 2.0, -1.0)
sigma = 1.2
y = y_true + rng.normal(0, sigma, size=x.size)

# Model.fit now always returns a Run; FitData stores plotting metadata too.
data = FitData.normal(x=x, y=y, yerr=sigma, x_label="x", y_label="y", label="data")
run = model.fit(data).squeeze()
res = run.results

print(res["m"].value, "±", res["m"].stderr)
print(res["b"].value, "±", res["b"].stderr)
print(res.summary(digits=4))


# Plotting (sensible defaults: data + fit + band + title)
xg = np.linspace(x.min(), x.max(), 400)
fig, ax = run.plot(xg=xg)
ax.plot(xg, line(xg, 2.0, -1.0), "k:", lw=1, label="true")
ax.legend()
plt.show()
