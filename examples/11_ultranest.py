import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import FitData, models


model = (
    models.sinusoid(name="wave")
    .bound(
        amplitude=(0.0, 5.0),
        offset=(-2.0, 2.0),
        frequency=(0.5, 6.0),
        phase=(0.0, 2.0 * np.pi),
    )
    .wrap(phase=True)
    .guess(amplitude=1.0, offset=0.0, frequency=3.0, phase=0.0)
)

rng = np.random.default_rng(0)
x = np.linspace(0.0, 1.0, 12)
sigma = 0.25

true = dict(amplitude=1.8, offset=0.2, frequency=3.2, phase=1.0)
y = model.eval(x, **true) + rng.normal(0.0, sigma, size=x.size)

data = FitData.normal(x=x, y=y, yerr=sigma, x_label="time", y_label="signal", label="data")
run = model.fit(
    data,
    backend="ultranest",
    backend_options={
        # Helps keep logs tidy and makes it obvious where outputs went.
        "log_dir": "ultranest_wave_example",
        # Keep default resume behaviour unless you prefer otherwise.
        # "resume": "subfolder",
    },
).squeeze()

res = run.results
print(res.summary(digits=5))

xg = np.linspace(x.min(), x.max(), 800)
fig, ax = run.plot(xg=xg, line_kwargs={"label": "posterior median"})
ax.plot(xg, model.eval(xg, **true), "k--", lw=1, label="true")
ax.legend()
plt.show()
