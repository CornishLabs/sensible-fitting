import numpy as np
import matplotlib.pyplot as plt
from fitwrap import models

# Start from a rich, prebuilt sinusoid model with good defaults
base = models.sinusoid(name="sinusoid")  # amplitude, offset, frequency, phase

# Specialise it: fix offset and phase (physics constraints)
restricted = base.fix(offset=0.0, phase=np.pi/3)

# Put bounds and a manual frequency seed
restricted = (
    restricted
      .bound(amplitude=(0.2, 5.0), frequency=(1.0, 10.0))
      .guess(frequency=3.1)
)

# Add a smarter guesser that can estimate frequency from data
# (could be Lomb-Scargle internally; user doesn't care)
@restricted.guesser
def smart_init(x, y, g):
    # If user supplied frequency guess, keep it. Otherwise infer.
    if g.is_unset("frequency"):
        g.frequency = restricted.estimate_frequency(x, y, method="lombscargle")

    # amplitude estimate robust-ish
    g.amplitude = np.quantile(y, 0.95) - np.quantile(y, 0.05)

# Fake data
rng = np.random.default_rng(7)
x = np.linspace(0, 1, 200)
sigma = 0.15
y = restricted.eval(x, amplitude=1.8, frequency=3.3) + rng.normal(0, sigma, size=x.size)

# Fit
run = restricted.fit(x=x, y=(y, sigma), backend="scipy.curve_fit", return_run=True).squeeze()
res = run.results
print(res.summary(digits=5))

# Plot
fig, ax = plt.subplots()
ax.errorbar(x, y, yerr=sigma, fmt=".", label="data")

xg = np.linspace(x.min(), x.max(), 400)
ax.plot(xg, run.model.eval(xg, params=res.params), label="fit")

band = run.band(xg, nsamples=600, interval=0.954)
ax.fill_between(xg, band.low, band.high, alpha=0.2, label="~2σ band")

ax.legend()
plt.show()
