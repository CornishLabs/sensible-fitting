"""
Binomial data example (API usage only)

We measure, at each x_i, n_trials_i Bernoulli trials and observe n_success_i successes.
The model predicts a probability p(x; θ) in [0,1].
We fit by maximizing log-likelihood (i.e. minimizing negative log-likelihood).
"""

import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import Model  # imagined API

# ----------------------------
# 1) Define a probability model p(x) (must be in [0,1])
# ----------------------------
def logistic(x, a, b):
    z = a*x + b
    return 1 / (1 + np.exp(-z))

model = (
    Model.from_function(logistic, name="logistic")
      .bound(a=(-20, 20), b=(-20, 20))
      .autoguess("a", "b")
)

# ----------------------------
# 2) Synthetic binomial dataset: (n_trials, n_success)
# ----------------------------
rng = np.random.default_rng(0)
x = np.linspace(-2, 2, 60)

a_true, b_true = 1.4, -0.2
p_true = logistic(x, a_true, b_true)

n_trials = 40  # could also be an array per point
n_success = rng.binomial(n_trials, p_true, size=x.size)

# (Optional) convert to observed proportions + binomial standard error for plotting
p_obs = n_success / n_trials
p_se = np.sqrt(p_obs * (1 - p_obs) / n_trials)

# ----------------------------
# 3) Fit: specify data_format="binomial" so the library uses binomial log-likelihood
# ----------------------------
run = model.fit(
    x=x,
    y=(n_trials, n_success),
    data_format="binomial",
    backend="scipy.minimize",   # likelihood optimiser
    parallel=None,
    return_run=True,
).squeeze()

res = run.result  # (squeezed single-fit run)

print(res.summary(digits=5))

# ----------------------------
# 4) Plot: proportions + error bars + fitted probability curve + ~2σ band
# ----------------------------
fig, ax = plt.subplots()

ax.errorbar(x, p_obs, yerr=2*p_se, fmt="o", ms=4, capsize=2, label="observed (±2σ approx)")

xg = np.linspace(x.min(), x.max(), 400)
p_fit = run.model.eval(xg, params=res.params)
ax.plot(xg, p_fit, label="fit")

# Optional: uncertainty band (if covariance available from backend)
# Uses parameter covariance sampling; gives central 95.4% interval (~2σ).
try:
    band = run.band(xg, nsamples=600, interval=0.954)
    ax.fill_between(xg, band.low, band.high, alpha=0.2, label="~2σ band")
except Exception:
    pass

ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("x")
ax.set_ylabel("p(success)")
ax.legend()
plt.show()
