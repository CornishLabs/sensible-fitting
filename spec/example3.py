import numpy as np
import matplotlib.pyplot as plt
from fitwrap import Model

def wave(x, amplitude, frequency, offset, phase):
    return offset + amplitude*np.sin(2*np.pi*frequency*x + phase)

model = (
    Model.from_function(wave, name="wave")
      .fix(offset=0.0, phase=np.pi/3)
      .bound(amplitude=(0.2, 5.0), frequency=(1.0, 6.0))
      .guess(frequency=2.8)
      .autoguess("amplitude")
)

@model.guesser
def guess_amp(x, y, g):
    g.amplitude = 0.5*(y.max() - y.min())

# make 4 datasets
rng = np.random.default_rng(2)
N_SYSTEMS, N = 4, 250
x = np.linspace(0, 1, N)

A0, F0 = 2.0, 3.0
A = A0*(1 + 0.05*rng.normal(size=N_SYSTEMS))
F = F0*(1 + 0.02*rng.normal(size=N_SYSTEMS))

sigma = 0.2
y_clean = np.stack([model.eval(x, amplitude=A[i], frequency=F[i]) for i in range(N_SYSTEMS)])
y = y_clean + rng.normal(0, sigma, size=y_clean.shape)

# Batch fit: y has shape (systems, N); sigma scalar broadcasts
run = model.fit(
    x=x,
    y=(y, sigma),
    backend="scipy.curve_fit",
    parallel="auto",
    return_run=True
)

res = run.results  # batch-shaped results

# print: nice table
print(res.summary(digits=4))

# slicing semantics:
print(res[0:2]["frequency"]["value"])  # (2,)

# plot on grid
fig, axs = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
axs = axs.ravel()

xg = np.linspace(x.min(), x.max(), 500)

for i, ax in enumerate(axs):
    ax.errorbar(x, y[i], yerr=sigma, fmt=".", ms=3, label=f"data {i}")

    # fit line for dataset i
    yi = run.model.eval(xg, params=res[i].params)
    ax.plot(xg, yi, label="fit")

    # 2σ band per dataset i
    band = run[i].band(xg, nsamples=300, interval=0.954)  # indexing run yields a sub-run
    ax.fill_between(xg, band.low, band.high, alpha=0.2)

    ax.set_title(f"system {i}")
    ax.legend()

plt.show()
