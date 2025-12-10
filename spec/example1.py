import numpy as np
import matplotlib.pyplot as plt
from fitwrap import Model

# 1) define model from plain function signature (params inferred from args)
def line(x, m, b):
    return m*x + b

model = Model.from_function(line, name="straight line")

# 2) make fake data
rng = np.random.default_rng(0)
x = np.linspace(0, 10, 50)
y_true = line(x, 2.0, -1.0)
sigma = 0.6
y = y_true + rng.normal(0, sigma, size=x.size)

# 3) fit (y, sigma) => Gaussian likelihood inferred
run = model.fit(
    x=x,
    y=(y, sigma),                 # symmetric errors
    backend="scipy.curve_fit",
    return_run=True,
)

run = run.squeeze()               # explicit: errors if >1 fit
res = run.results                 # now “scalar-ish” in the sense: batch dim removed

# 4) print results (slice-first, but now single-fit so param selection is scalar)
print(res.params["m"]["value"], "±", res.params["m"]["error"])
print(res.params["b"]["value"], "±", res.params["b"]["error"])
print(res.summary(digits=4))

# 5) plot data + fit line + 2σ band
fig, ax = plt.subplots()

# points + error bars
ax.errorbar(x, y, yerr=sigma, fmt="o", ms=4, capsize=2, label="data")

# fit line
xg = np.linspace(x.min(), x.max(), 400)
yg = run.model.eval(xg, params=res.params)          # params is structured scalar-ish
ax.plot(xg, yg, label="fit")

# 2σ band via covariance sampling (helper lives on run/plot utils)
# (samples theta ~ N(theta_hat, cov), eval, percentile band)
band = run.band(xg, nsamples=400, interval=0.954)   # returns (low, high)
ax.fill_between(xg, band.low, band.high, alpha=0.2, label="~2σ band")

ax.legend()
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()
