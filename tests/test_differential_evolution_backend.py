import numpy as np

from sensible_fitting import FitData, models


def test_differential_evolution_backend_fits_sinusoid() -> None:
    rng = np.random.default_rng(0)

    model = (
        models.sinusoid(name="wave")
        .fix(offset=0.0, phase=0.0)
        .bound(amplitude=(0.5, 3.0), frequency=(0.5, 6.0))
    )

    x = np.linspace(0.0, 1.0, 200)
    true = {"amplitude": 1.5, "frequency": 4.2}
    sigma = 0.2
    y = model.eval(x, **true) + rng.normal(0.0, sigma, size=x.shape)

    data = FitData.normal(x=x, y=y, yerr=sigma)
    run = model.fit(
        data,
        backend="scipy.differential_evolution",
        backend_options={"maxiter": 20, "popsize": 8, "seed": 0},
    ).squeeze()

    assert bool(run.success)

    res = run.results
    assert res.cov is not None
    assert res["frequency"].stderr is not None
    assert res["amplitude"].stderr is not None

    assert abs(float(res["frequency"].value) - true["frequency"]) < 0.05
    assert abs(float(res["amplitude"].value) - true["amplitude"]) < 0.1


def test_backend_pipeline_runs_and_records_steps() -> None:
    rng = np.random.default_rng(0)

    model = (
        models.sinusoid(name="wave")
        .fix(offset=0.0, phase=0.0)
        .bound(amplitude=(0.5, 3.0), frequency=(0.5, 6.0))
    )

    x = np.linspace(0.0, 1.0, 200)
    sigma = 0.2
    y = model.eval(x, amplitude=1.5, frequency=4.2) + rng.normal(0.0, sigma, size=x.shape)

    data = FitData.normal(x=x, y=y, yerr=sigma)
    run = model.fit(
        data,
        backend=("scipy.differential_evolution", "scipy.curve_fit"),
        backend_options={"maxiter": 10, "popsize": 6, "seed": 0},
    ).squeeze()

    assert bool(run.success)
    stats = run.results.stats
    assert stats.get("pipeline_mode") == "pipeline"
    assert isinstance(stats.get("pipeline"), list)
    assert len(stats["pipeline"]) == 2


def test_backend_auto_falls_back_to_de() -> None:
    rng = np.random.default_rng(0)

    # Simple bounded line; force curve_fit to fail via maxfev=0 so auto must fall back.
    line = models.straight_line().bound(m=(-10, 10), b=(-10, 10)).guess(m=0.0, b=0.0)

    x = np.linspace(-1.0, 1.0, 30)
    sigma = 0.1
    y = line.eval(x, m=2.0, b=-0.5) + rng.normal(0.0, sigma, size=x.shape)

    data = FitData.normal(x=x, y=y, yerr=sigma)
    run = line.fit(
        data,
        backend="auto",
        backend_options={"maxfev": 0, "maxiter": 20, "popsize": 8, "seed": 0},
    ).squeeze()

    assert bool(run.success)
    assert run.results.backend == "scipy.differential_evolution"
    assert run.results.stats.get("pipeline_mode") == "auto"
