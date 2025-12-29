import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import FitData, models


rng = np.random.default_rng(0)

model = models.rabi_oscillation().bound(
    amplitude=(0.0, 1.0),
    offset=(0.0, 1.0),
    t_period=(0.5, 8.0),
    phase=(-np.pi, np.pi),
    tau=(0.5, 30.0),
    t_dead=(0.0, 2.0),
)

# Irregularly spaced time points.
x = np.sort(rng.uniform(0.0, 10.0, size=60))
true = {
    "amplitude": 0.45,
    "offset": 0.5,
    "t_period": 3.2,
    "phase": 0.6,
    "tau": 8.0,
    "t_dead": 0.4,
}
p_true = model.eval(x, **true)
n = np.full(x.shape, 200, dtype=int)
k = rng.binomial(n, p_true)

data = FitData.binomial(
    x=x,
    n=n,
    k=k,
    x_label="time",
    y_label="population",
    label="data",
)

# cov_method="auto" uses hess_inv if available; numdiff is steadier for binomial fits.
runa = model.fit(
    data,
    backend="scipy.minimize",
    backend_options={"cov_method": "numdiff"},
).squeeze()
resa = runa.results

print(resa.summary(digits=5))

runb = model.fit(
    data,
    backend="ultranest",
).squeeze()
resb = runb.results

print(resb.summary(digits=5))

xg = np.linspace(float(np.min(x)), float(np.max(x)), 800)
fig, axs = plt.subplots(
    1, 2, figsize=(11, 4), sharex=True, sharey=True, constrained_layout=True
)
runa.plot(ax=axs[0], xg=xg, line_kwargs={"label": "scipy.minimize"})
runb.plot(ax=axs[1], xg=xg, line_kwargs={"label": "ultranest"})

for ax in axs:
    ax.plot(xg, model.eval(xg, **true), "k--", lw=1, label="true")

axs[0].set_title("scipy.minimize (numdiff)\n" + axs[0].get_title())
axs[1].set_title("ultranest\n" + axs[1].get_title())
for ax in axs:
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
plt.show()
