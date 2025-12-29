import numpy as np
import pytest

from sensible_fitting import FitData, Model
from dataclasses import replace


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


def test_plot_run_batch_panel_title_and_each_hook():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    model = Model.from_function(line).bound(m=(-10, 10), b=(-10, 10)).guess(m=1.0, b=0.0)

    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, 30)
    sigma = 0.1
    y = np.stack(
        [
            line(x, 2.0, -0.5) + rng.normal(0.0, sigma, size=x.size),
            line(x, 1.5, 0.2) + rng.normal(0.0, sigma, size=x.size),
        ],
        axis=0,
    )

    run = model.fit(FitData.normal(x=x, y=y, yerr=sigma))

    fig, axs = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    called = {"n": 0}

    def _each(ax, subrun, idx):
        called["n"] += 1

    run.plot(axs=axs, panel_title="system {i}", each=_each)

    assert called["n"] == 2
    assert "system 0" in axs[0].get_title()
    assert "system 1" in axs[1].get_title()

    plt.close(fig)


def test_plot_run_can_overlay_posterior_sample_lines():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    model = Model.from_function(line).bound(m=(-10, 10), b=(-10, 10)).guess(m=1.0, b=0.0)

    x = np.linspace(0.0, 1.0, 20)
    sigma = 0.1
    rng = np.random.default_rng(0)
    y = line(x, 2.0, -0.5) + rng.normal(0.0, sigma, size=x.size)

    run = model.fit(FitData.normal(x=x, y=y, yerr=sigma)).squeeze()

    # Inject fake posterior samples (e.g. from a Bayesian backend).
    samples = np.column_stack(
        [
            rng.normal(2.0, 0.1, size=50),   # m
            rng.normal(-0.5, 0.05, size=50), # b
        ]
    )
    stats = dict(run.results.stats or {})
    stats["free_names"] = ("m", "b")
    stats["posterior_samples"] = samples
    run = replace(run, results=replace(run.results, stats=stats))

    fig, ax = plt.subplots()
    run.plot(
        ax=ax,
        data=False,
        band=False,
        title=False,
        posterior_lines=7,
    )

    # One main fit line + N sample lines.
    assert len(ax.lines) == 1 + 7
    plt.close(fig)
