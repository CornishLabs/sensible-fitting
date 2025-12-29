import numpy as np
import matplotlib.pyplot as plt
from sensible_fitting import FitData, models

fit_data = False  # True => fit. False => plot seed only.

model = (
    models.sinusoid(name="wave")
    .fix(offset=0.0, phase=np.pi / 3)
    .bound(amplitude=(0.2, 5.0), frequency=(1.0, 6.0))
    .guess(frequency=2.8)
)

rng = np.random.default_rng(2)
N = 25
x = np.linspace(0, 1, N)

sigma = 0.6
y = model.eval(x, amplitude=2.0, frequency=3.0) + rng.normal(0, sigma, size=x.size)

# Always a Run; optimise=False => "seed only" mode
data = FitData.normal(
    x=x,
    y=y,
    yerr=sigma,
    x_label="time",
    y_label="signal",
    label="data",
)
run = model.fit(
    data,
    optimise=fit_data,
).squeeze()

res = run.results

xg = np.linspace(x.min(), x.max(), 500)
style = "-" if fit_data else "--"
fig, ax = run.plot(xg=xg, line_kwargs={"linestyle": style})
ax.plot(xg, model.eval(xg, amplitude=2.0, frequency=3.0), "k--", lw=1, label="true")
ax.legend()
plt.show()

print(res.summary(digits=5))
