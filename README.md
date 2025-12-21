# sensible-fitting

A small, opinionated fitting framework for AMO-style workflows: quick to seed, easy to batch, and consistent across backends.

> Status: **v0.1 — lab use / API in flux**. The goal is a clean public API; contributions and feedback welcome.

## Why this exists

Most fitting code in the lab ends up re-implementing the same pieces:

- a model definition plus metadata (bounds, fixed params, priors)
- “sensible” seed parameter guessing
- repeated fitting across many datasets (batches)
- predictable access to results and uncertainties
- quick plotting helpers like prediction bands

`sensible-fitting` wraps fitting backends behind a single API and focuses on ergonomics for fast experimental iteration.

This project was born out of a desire to:
- Replace the [oitg](https://github.com/OxfordIonTrapGroup/oitg) dependency used by [ndscan](https://github.com/OxfordIonTrapGroup/ndscan) (primarily via plotting/fitting utilities)
- Replace the wrapper over lmfit that existed in our lab previously.

---

## Installation

Currently intended for development / source installs:

```bash
pip install -e .
```

Dependencies are listed in `pyproject.toml`.

---

## Quickstart

```py
import numpy as np
import matplotlib.pyplot as plt
from sensible_fitting import Model, plot_fit

def line(x, m, b):
    return m * x + b

model = (
    Model.from_function(line, name="straight line")
    .bound(m=(-10, 10), b=(-10, 10))
)

rng = np.random.default_rng(0)
x = np.linspace(0, 10, 20)
sigma = 1.2
y = line(x, 2.0, -1.0) + rng.normal(0, sigma, size=x.size)

run = model.fit(x, (y, sigma)).squeeze()
res = run.results

print(res["m"].value, "±", res["m"].stderr)
print(res["b"].value, "±", res["b"].stderr)

fig, ax = plt.subplots()
plot_fit(
    ax=ax,
    x=x,
    y=y,
    yerr=sigma,
    run=run,
    band=True,
    band_options={"level": 2, "nsamples": 400},
    show_params=True,
)
ax.legend()
plt.show()
```

---

## Core concepts

### `Model`

A `Model` wraps a function `f(x, p1, p2, ...)` plus parameter metadata:

* `.bound(...)` — set bounds
* `.fix(...)` — fix parameters
* `.guess(...)` — strong, explicit initial values
* `.weak_guess(...)` — low-precedence fallback initial values
* `.with_guesser(fn)` — attach a seed “guesser” heuristic
* `.derive(name, fn)` — post-fit derived parameters

Models are **pure builders**: each call returns a new `Model`.

### Seed vs fit

A **seed** is the initial parameter set used to start optimisation.

You can compute the seed-only result (no optimisation) via:

```py
run = model.fit(x, (y, sigma), optimise=False).squeeze()
seed_params = run.results.seed
```

You can override seeding per call:

```py
run = model.fit(x, (y, sigma), seed_override={"frequency": 3.0})
```

#### Seed precedence (per parameter)

1. `seed_override=...` (per-call)
2. strong `.guess(...)`
3. attached guessers (`with_guesser`)
4. `.weak_guess(...)` (including numeric defaults from the function signature)
5. midpoint of finite bounds (warning)
6. otherwise: error (you must provide something)

### `Run` and `Results`

* `run.results.params` — fitted parameters (`ParamsView`)
* `run.results.seed` — seed parameters actually used (`ParamsView`)
* `run.predict(x)` — evaluate with fitted parameters (or seed via `which="seed"`)
* `run.band(x)` — prediction band (currently covariance-based)

`Results` supports convenient indexing:

* `res["frequency"]` → `ParamView`
* `res["frequency", "phase"]` → `MultiParamView`
* for batches: `res[i]` slices batches, preserving the same API

---

## Data formats

v1 supports these `data_format` payloads:

### `data_format="normal"` (default)
* `y` → unweighted least squares
* `(y, sigma)` → absolute symmetric errors
* `(y, sigma_low, sigma_high)` → asymmetric errors (currently approximated to mean sigma)

### `data_format="binomial"`
* `(n_samples, n_successes)`

### `data_format="beta"`
* `(alpha, beta)`

---

## Data inference rules

Defaults are tuned for "just pass arrays," with warnings on ambiguity:

* Tuples are **payloads**: `(y, sigma)` / `(y, sigma_low, sigma_high)` for normal, `(n, k)` for binomial, `(alpha, beta)` for beta.
* Lists are **ragged batches** only when `x` is also a list (same length).
* Otherwise, lists are treated as array data when possible (e.g. common-x batches).
* Ambiguous inputs emit a `UserWarning` with a hint; pass `strict=True` to raise instead.

```py
# Ambiguous: list of two arrays could be batch data or (y, sigma).
run = model.fit(x, [y, sigma])             # warns, treated as batch data
run = model.fit(x, (y, sigma))             # explicit payload, no warning
run = model.fit(x, [y, sigma], strict=True)  # raises
```

---

## Plotting helper

For quick 1D plots, use `plot_fit` (it expects 1D `x` and `y`):

```py
fig, ax = plt.subplots()
plot_fit(
    ax=ax,
    x=x,
    y=y,
    yerr=sigma,
    run=run,
    band=True,
    band_options={"level": 2},
    show_params=True,
)
```

---

## Batching

### Common x-grid, many datasets

If `y` has shape `batch_shape + (N,)`, a single call fits all datasets:

```py
run = model.fit(x, (y, sigma))
res = run.results

print(res.batch_shape)          # e.g. (4,)
print(res["m"].value.shape)     # e.g. (4,)
```

Slice and treat any batch like a scalar fit:

```py
sub = run[0]
band0 = sub.band(xg, level=2)
```

### Ragged batches (different x per dataset)

Pass `x` and `y` as lists of datasets:

```py
xs = [x0, x1, x2]
ys = [(y0, s0), (y1, s1), (y2, s2)]
run = model.fit(xs, ys)
```

---

## Built-in AMO-ish models

```py
from sensible_fitting import models

line = models.straight_line()
sin  = models.sinusoid()
gaus = models.gaussian_with_offset()
```

(These are meant to grow over time; PRs welcome.)

---

## Backends

v1 implements:

* `backend="scipy.curve_fit"` (default for `data_format="normal"`)
* `backend="scipy.minimize"` (required for `binomial`/`beta` currently)
* `backend="ultranest"` (Bayesian nested sampling for `normal`)

UltraNest requires either explicit priors (`Model.prior(...)`) or finite bounds for all free parameters.

---

## Roadmap (short)

* Additional backends (minimise / Bayesian sampling)
* Better batching of covariance and per-batch diagnostics
* More built-in AMO models + seeders
* Clearer plotting helpers / recipes

---

## Development

```bash
pip install -e ".[dev]"
python examples/01_line.py
```
