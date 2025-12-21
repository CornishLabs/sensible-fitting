import warnings

import numpy as np
import pytest

from sensible_fitting.data import prepare_datasets


def test_list_payload_warns_as_batch():
    x = np.linspace(0.0, 1.0, 5)
    y = np.linspace(0.0, 1.0, 5)
    sigma = np.ones_like(y) * 0.1

    with pytest.warns(UserWarning, match="Interpreting list as batch data"):
        datasets, batch_shape = prepare_datasets(x, [y, sigma], "normal", strict=False)

    assert batch_shape == (2,)
    assert len(datasets) == 2
    assert datasets[0].payload["sigma"] is None


def test_list_payload_strict_raises():
    x = np.linspace(0.0, 1.0, 5)
    y = np.linspace(0.0, 1.0, 5)
    sigma = np.ones_like(y) * 0.1

    with pytest.raises(ValueError, match="Interpreting list as batch data"):
        prepare_datasets(x, [y, sigma], "normal", strict=True)


def test_ragged_requires_list_x_and_data():
    x0 = np.linspace(0.0, 1.0, 5)
    y0 = np.linspace(0.0, 1.0, 5)
    with pytest.raises(TypeError, match="Ragged batches require list inputs"):
        prepare_datasets([x0], (y0, 0.1), "normal", strict=False)


def test_ragged_list_of_tuples():
    xs = [np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 6)]
    ys = [
        (np.linspace(0.0, 1.0, 5), 0.1),
        (np.linspace(0.0, 1.0, 6), 0.2),
    ]
    datasets, batch_shape = prepare_datasets(xs, ys, "normal", strict=False)
    assert batch_shape == (2,)
    assert len(datasets) == 2


def test_nd_x_matches_y_shape_single_dataset():
    x = np.linspace(-1.0, 1.0, 4)
    y = np.linspace(-1.0, 1.0, 3)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = X + Y

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        datasets, batch_shape = prepare_datasets((X, Y), Z, "normal", strict=False)

    assert batch_shape == ()
    assert len(datasets) == 1
    assert datasets[0].payload["y"].shape == (Z.size,)
