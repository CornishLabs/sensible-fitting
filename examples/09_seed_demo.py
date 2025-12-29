import numpy as np
import matplotlib.pyplot as plt

from sensible_fitting import FitData, models


# --- Build a model with a pure guesser ---------------------------------------

base_model = models.sinusoid(name="seed-demo")


def tweak_guess(x, y, g):
    if g.is_unset("amplitude"):
        g.amplitude = 1.0
    if g.is_unset("offset"):
        g.offset = 0.0
    if g.is_unset("frequency"):
        g.frequency = 3.0
    if g.is_unset("phase"):
        g.phase = 0.0


# Pure builder: attach the guesser, get a new model
model = base_model.with_guesser(tweak_guess)

# --- Generate synthetic data --------------------------------------------------

rng = np.random.default_rng(0)
N = 150
x = np.linspace(0, 1, N)

true_params = dict(amplitude=2.0, offset=0.1, frequency=3.2, phase=0.3)
sigma = 0.25

y_clean = model.eval(x, **true_params)
y = y_clean + rng.normal(0, sigma, size=x.size)

data = FitData.normal(
    x=x,
    y=y,
    yerr=sigma,
    x_label="time",
    y_label="signal",
    label="data",
)

# --- 1) Use Model.seed(...) to look at the seed curve ------------------------

seed_params = model.seed(data)
print("Seed parameters:")
for name, pv in seed_params.items():
    print(f"  {name:>10s} = {pv.value:.4g}")

# --- 2) Do a seed-only 'fit' (optimise=False) -------------------------------------

run_seed = model.fit(
    data,
    optimise=False,  # seed-only mode
).squeeze()

# run_seed.results.params == run_seed.results.seed here (up to fixed params)

# --- 3) Do a real fit --------------------------------------------------------

run_fit = model.fit(
    data,
).squeeze()

res = run_fit.results
print("\nFitted parameters:")
print(res.summary(digits=4))

# --- 4) Per-call seed overriding everything ----------------------------------

forced_seed = {"frequency": 6.0}
run_forced_seed = model.fit(
    data,
    seed_override=forced_seed,
    optimise=False,  # use *only* this seed, no optimisation
).squeeze()

print("\nForced-seed parameters (seed={'frequency': 6.0}, optimise=False):")
for name, pv in run_forced_seed.results.seed.items():
    print(f"  {name:>10s} = {pv.value:.4g}")

# --- 5) Use Run.predict(...) for fit vs seed curves --------------------------

xg = np.linspace(x.min(), x.max(), 400)

y_seed = run_seed.predict(xg, which="seed")
y_forced = run_forced_seed.predict(xg, which="seed")  # seed-only run

# --- Plot everything ----------------------------------------------------------

fig, ax = run_fit.plot(xg=xg, line_kwargs={"label": "fit curve"})

# true underlying curve (for illustration)
ax.plot(xg, model.eval(xg, **true_params), "k--", lw=1, label="true")

# seed-only curve (from seed engine + guesser)
ax.plot(xg, y_seed, "--", label="seed curve")

# forced seed curve (frequency fixed to 6.0)
ax.plot(xg, y_forced, "-.", label="seed (frequency=6.0)")

ax.legend()
plt.show()
