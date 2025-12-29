from dataclasses import replace

import numpy as np

from sensible_fitting import FitData, Model


def line(x, m, b):
    return m * x + b


def test_suggest_next_x_posterior_max_var_prefers_extreme_x():
    model = Model.from_function(line).guess(m=0.0, b=0.0)

    # Seed-only run is fine; we'll inject posterior samples.
    x_data = np.array([0.2, 0.4, 0.6], dtype=float)
    y_data = line(x_data, 0.0, 0.0)
    run = model.fit(FitData.normal(x=x_data, y=y_data, yerr=0.1), optimise=False).squeeze()

    # Two-slope multimodal posterior: predictions differ most at largest x.
    slopes = np.array([-1.0, 1.0], dtype=float)
    samples = np.column_stack(
        [
            np.repeat(slopes, 50),          # m
            np.zeros((slopes.size * 50,)),  # b
        ]
    )
    stats = dict(run.results.stats or {})
    stats["free_names"] = ("m", "b")
    stats["posterior_samples"] = samples
    run = replace(run, results=replace(run.results, stats=stats))

    x_cand = np.linspace(0.0, 1.0, 101)
    x_next = run.suggest_next_x(
        candidates=x_cand,
        method="posterior",
        objective="max_var",
        avoid_existing=False,
    )

    assert float(x_next) == float(x_cand[-1])


def test_suggest_next_x_can_return_details():
    model = Model.from_function(line).guess(m=1.0, b=0.0)
    x_data = np.array([0.0, 1.0], dtype=float)
    y_data = line(x_data, 1.0, 0.0)
    run = model.fit(FitData.normal(x=x_data, y=y_data, yerr=0.1), optimise=False).squeeze()

    rng = np.random.default_rng(0)
    samples = rng.normal([1.0, 0.0], [0.2, 0.1], size=(200, 2))
    stats = dict(run.results.stats or {})
    stats["free_names"] = ("m", "b")
    stats["posterior_samples"] = samples
    run = replace(run, results=replace(run.results, stats=stats))

    details = run.suggest_next_x(
        candidates=np.linspace(0.0, 1.0, 50),
        method="posterior",
        objective="max_width",
        return_details=True,
        avoid_existing=False,
    )

    assert hasattr(details, "x")
    assert hasattr(details, "candidates")
    assert hasattr(details, "score")
    assert np.asarray(details.candidates).shape == np.asarray(details.score).shape

