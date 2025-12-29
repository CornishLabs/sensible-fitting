import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import FitData, models


"""Showcase the Results / ParamsView indexing API."""

# Batched sinusoid fit: 4 independent datasets sharing the same x grid
model = (
    models.sinusoid(name="wave")
    .fix(offset=0.0, phase=0.0)
    .bound(amplitude=(0.2, 5.0), frequency=(1.0, 6.0))
    .guess(frequency=3.0)
)

rng = np.random.default_rng(0)
N_SYSTEMS, N = 4, 200
x = np.linspace(0, 1, N)

A0, F0 = 2.0, 3.0
A = A0 * (1 + 0.05 * rng.normal(size=N_SYSTEMS))
F = F0 * (1 + 0.02 * rng.normal(size=N_SYSTEMS))

sigma = 0.2
y_clean = np.stack(
    [model.eval(x, amplitude=A[i], frequency=F[i]) for i in range(N_SYSTEMS)]
)
y = y_clean + rng.normal(0, sigma, size=y_clean.shape)

# Tuple data payload; y has batch shape (N_SYSTEMS, N) -> common-x batch fit.
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

print("batch_shape:", res.batch_shape)
print()

# 1) Parameter by name across all batches
freq_all = res["frequency"].value  # shape (4,)
print("frequency (all batches):", freq_all)

# 2) Single batch, parameter by name
freq_0 = res[0]["frequency"].value  # scalar
print("frequency[0]:", freq_0)

# 3) Slice of batches
freq_01 = res[0:2]["frequency"].value  # shape (2,)
print("frequency[0:2]:", freq_01)

print()

# 4) Parameter by *index* via .params
# Order follows the model function signature: (amplitude, offset, frequency, phase)
first_param_all = res.params[0].value  # amplitude, shape (4,)
print("param[0] (all batches):", first_param_all)

first_param_0 = res[0].params[0].value  # amplitude, batch 0
print("param[0] (batch 0):", first_param_0)

print()

# 5) Multi-param by name -> MultiParamView
fp = res["frequency", "phase"]  # MultiParamView
print("multi names:", fp.names)
print("multi value shape:", fp.value.shape)  # (4, 2)

freq_col = fp.value[:, 0]
phase_col = fp.value[:, 1]
print("freq_col:", freq_col)
print("phase_col:", phase_col)

print()

# 6) Multi-param by index via .params
# Here: indices 1 and 2 -> (offset, frequency)
mp_idx = res.params[1, 2]
print("mp_idx names:", mp_idx.names)
print("mp_idx value shape:", mp_idx.value.shape)

print()

# 7) Using uncertainties: value ± stderr packaged as a uarray
freq_u = res["frequency"].u
print("frequency as uarray:", freq_u)

# Plot the 4 fits as a grid (also demonstrates batched plotting)
xg = np.linspace(x.min(), x.max(), 400)
fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True, constrained_layout=True)

def _overlay_true(ax, subrun, idx):
    i = int(idx[0])
    ax.plot(
        xg,
        model.eval(xg, amplitude=float(A[i]), frequency=float(F[i])),
        "k--",
        lw=1,
        label="true",
    )


run.plot(axs=axs, xg=xg, panel_title="system {i}", each=_overlay_true)
for ax in np.asarray(axs, dtype=object).ravel():
    ax.legend()
plt.show()
