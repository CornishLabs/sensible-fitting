import numpy as np

from sensible_fitting import FitData, Model, models
from sensible_fitting.params import ParamView


def test_fit_accepts_paramview_payload() -> None:
    def line(i, a, b):
        return a + b * i

    model = Model.from_function(line).guess(a=0.0, b=0.0)

    x = np.arange(6, dtype=float)
    y = line(x, 2.0, 0.5)
    yerr = 0.1 * np.ones_like(x)

    pv = ParamView(name="y", value=y, stderr=yerr)
    run = model.fit(x, pv).squeeze()

    assert abs(float(run.results["a"].value) - 2.0) < 1e-2
    assert abs(float(run.results["b"].value) - 0.5) < 1e-2


def test_paramview_as_fitdata_sets_labels() -> None:
    pv = ParamView(name="frequency", value=np.array([1.0, 2.0, 3.0]), stderr=np.array([0.1, 0.1, 0.1]))
    fd = pv.as_fitdata(x=np.array([0.0, 1.0, 2.0]), x_label="i")

    assert isinstance(fd, FitData)
    assert fd.data_format == "normal"
    assert fd.x_label == "i"
    assert fd.y_label == "frequency"
    assert fd.label == "frequency"


def test_batched_stderr_is_float_when_complete() -> None:
    model = models.straight_line().bound(m=(-10, 10), b=(-10, 10)).guess(m=0.0, b=0.0)

    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 30)
    sigma = 0.1

    y0 = model.eval(x, m=2.0, b=-0.5) + rng.normal(0.0, sigma, size=x.shape)
    y1 = model.eval(x, m=1.5, b=0.2) + rng.normal(0.0, sigma, size=x.shape)
    y2 = model.eval(x, m=0.7, b=1.0) + rng.normal(0.0, sigma, size=x.shape)
    y = np.stack([y0, y1, y2], axis=0)

    run = model.fit(FitData.normal(x=x, y=y, yerr=sigma))
    stderr = run.results["m"].stderr

    assert isinstance(stderr, np.ndarray)
    assert stderr.shape == (3,)
    assert stderr.dtype != object

