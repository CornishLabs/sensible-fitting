import numpy as np
import pytest

from sensible_fitting import FitData, Model


def line(x, m, b):
    return m * x + b


def test_fitdata_metadata_is_stored_on_run():
    model = Model.from_function(line).bound(m=(-10, 10), b=(-10, 10)).guess(m=1.0, b=0.0)

    x = np.linspace(0.0, 1.0, 20)
    sigma = 0.1
    y = line(x, 2.0, -0.5) + np.random.default_rng(0).normal(0.0, sigma, size=x.size)

    fd = FitData.normal(
        x=x,
        y=y,
        yerr=sigma,
        x_label="time [s]",
        y_label="signal [arb]",
        label="data",
    )

    run = model.fit(fd).squeeze()

    assert run.data is not None
    meta = run.data.get("meta")
    assert isinstance(meta, dict)
    assert meta["x_label"] == "time [s]"
    assert meta["y_label"] == "signal [arb]"
    assert meta["label"] == "data"


def test_plot_run_uses_axis_labels_from_fitdata():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    model = Model.from_function(line).bound(m=(-10, 10), b=(-10, 10)).guess(m=1.0, b=0.0)

    x = np.linspace(0.0, 1.0, 20)
    sigma = 0.1
    y = line(x, 2.0, -0.5) + np.random.default_rng(0).normal(0.0, sigma, size=x.size)

    fd = FitData.normal(
        x=x,
        y=y,
        yerr=sigma,
        x_label="time [s]",
        y_label="signal [arb]",
    )

    run = model.fit(fd).squeeze()

    fig, ax = plt.subplots()
    run.plot(ax=ax, title=False)

    assert ax.get_xlabel() == "time [s]"
    assert ax.get_ylabel() == "signal [arb]"

    plt.close(fig)


def test_plot_data_uses_axis_labels_from_fitdata():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    x = np.linspace(0.0, 1.0, 20)
    sigma = 0.1
    y = line(x, 2.0, -0.5) + np.random.default_rng(0).normal(0.0, sigma, size=x.size)

    fd = FitData.normal(
        x=x,
        y=y,
        yerr=sigma,
        x_label="time [s]",
        y_label="signal [arb]",
    )

    fig, ax = plt.subplots()
    fd.plot(ax=ax)

    assert ax.get_xlabel() == "time [s]"
    assert ax.get_ylabel() == "signal [arb]"

    plt.close(fig)

