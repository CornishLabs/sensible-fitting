import numpy as np
import pytest

from sensible_fitting import Model


def _line_model() -> Model:
    def line(x, m, b):
        return m * x + b

    return (
        Model.from_function(line, name="line")
        .bound(m=(-10.0, 10.0), b=(-10.0, 10.0))
        .guess(m=2.0, b=0.0)
    )


def test_results_indexing_and_params_view():
    model = _line_model()
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 30)
    sigma = 0.05

    y0 = model.eval(x, m=2.0, b=-0.5) + rng.normal(0.0, sigma, size=x.size)
    y1 = model.eval(x, m=1.0, b=0.2) + rng.normal(0.0, sigma, size=x.size)
    y = np.stack([y0, y1], axis=0)

    run = model.fit(x, (y, sigma))
    res = run.results

    assert res.batch_shape == (2,)
    assert res.params["m"].value.shape == (2,)
    assert res.params[0].name == "m"
    assert res.params[1].name == "b"

    multi = res.params["m", "b"]
    assert multi.names == ("m", "b")
    assert multi.value.shape == (2, 2)

    multi_slice = res.params[0:2]
    assert multi_slice.names == ("m", "b")
    assert multi_slice.value.shape == (2, 2)

    assert res["m"] is res.params["m"]

    sub = res[0]
    assert sub.batch_shape == ()
    assert pytest.approx(res.params["m"].value[0]) == sub["m"].value
    with pytest.raises(IndexError):
        sub[0]

    run0 = run[0]
    assert run0.results.batch_shape == ()


def test_fit_recovers_line_params():
    model = _line_model()
    rng = np.random.default_rng(1)
    x = np.linspace(-1.0, 1.0, 40)
    true_m = 2.3
    true_b = -0.7
    sigma = 0.05
    y = true_m * x + true_b + rng.normal(0.0, sigma, size=x.size)

    run = model.fit(x, (y, sigma)).squeeze()
    res = run.results

    assert abs(res["m"].value - true_m) < 0.2
    assert abs(res["b"].value - true_b) < 0.2
    assert res["m"].stderr is not None
    assert res["b"].stderr is not None


def test_seed_only_mode_uses_seed_values():
    model = _line_model()
    x = np.linspace(0.0, 1.0, 10)
    y = 2.0 * x - 0.5
    seed = {"m": 1.5, "b": -0.4}

    run = model.fit(x, y, seed_override=seed, optimise=False).squeeze()
    res = run.results

    assert res.params["m"].value == seed["m"]
    assert res.params["b"].value == seed["b"]
    assert res.seed["m"].value == seed["m"]
    assert res.seed["b"].value == seed["b"]
    assert res["m"].stderr is None

    pred_fit = run.predict(x)
    pred_seed = run.predict(x, which="seed")
    assert np.allclose(pred_fit, pred_seed)


def test_binomial_fit_reasonable():
    def logistic(x, slope, offset):
        return 1.0 / (1.0 + np.exp(-(slope * x + offset)))

    model = (
        Model.from_function(logistic, name="logistic")
        .bound(slope=(-5.0, 5.0), offset=(-5.0, 5.0))
        .guess(slope=1.0, offset=0.0)
    )

    rng = np.random.default_rng(2)
    x = np.linspace(-1.0, 1.0, 25)
    true = {"slope": 2.0, "offset": -0.3}
    p = logistic(x, **true)
    n = np.full(x.shape, 200, dtype=int)
    k = rng.binomial(n, p)

    run = model.fit(x, (n, k), data_format="binomial").squeeze()
    p_fit = model.eval(x, params=run.results.params)

    assert np.mean(np.abs(p_fit - p)) < 0.1
