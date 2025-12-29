import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

from sensible_fitting import FitData, Model


def p_step(x, x0, width):
    # Explicit probability model (0..1)
    z = (x - x0) / width
    return 1.0 / (1.0 + np.exp(-z))


# --- Model -------------------------------------------------------------------

model = (
    Model.from_function(p_step, name="step-probability")
    .bound(x0=(-2, 2), width=(1e-3, 10))
    .guess(x0=0.0, width=0.3)
)

# --- Synthetic binomial data --------------------------------------------------

rng = np.random.default_rng(0)
x = np.linspace(-2, 2, 60)

n = 80 * np.ones_like(x)  # trials per x
p_true = p_step(x, x0=0.2, width=0.35)
k = rng.binomial(n.astype(int), p_true)  # successes

# --- Fit ---------------------------------------------------------------------

data = FitData.binomial(x=x, n=n, k=k, x_label="x", y_label="p", label="data")
run = model.fit(data, backend="scipy.minimize").squeeze()

res = run.results
print(res.summary(digits=4))

# --- Plot --------------------------------------------------------------------

xg = np.linspace(x.min(), x.max(), 400)

fig, ax = plt.subplots()

x = np.asarray(x, dtype=float)
n = np.asarray(n, dtype=float)
k = np.asarray(k, dtype=float)

# Jeffreys posterior over p: Beta(k+1/2, n-k+1/2)
a = k + 0.5
b = (n - k) + 0.5

# "±1σ-ish" = Normal-equivalent 68.27% equal-tailed interval
qlo = 0.15865525393145707  # Phi(-1)
qhi = 0.8413447460685429  # Phi(+1)

median = beta.ppf(0.5, a, b)
lo = beta.ppf(qlo, a, b)
hi = beta.ppf(qhi, a, b)

# Error bars around the MEDIAN (always non-negative yerr)
yerr = np.vstack([median - lo, hi - median])

# Data summary: median ± 1σ-ish interval
ax.errorbar(
    x,
    median,
    yerr=yerr,
    fmt="o",
    ms=4,
    capsize=2,
    label="data (posterior median ±1σ, Jeffreys)",
)

# Fit curve + parameter band, using the default plotting helper.
run.plot(
    ax=ax,
    xg=xg,
    data=False,  # we drew custom (Jeffreys) error bars above
    line_kwargs={"label": "fit p(x)"},
)
ax.plot(xg, p_step(xg, x0=0.2, width=0.35), "k--", lw=1, label="true")

ax.set_ylim(-0.05, 1.05)
ax.legend()
plt.show()
