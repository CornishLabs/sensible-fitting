import numpy as np
import pytest

from sensible_fitting import Model

uncertainties = pytest.importorskip("uncertainties")


def line(x, m, b):
    return m * x + b


def test_u_uses_covariance():
    model = Model.from_function(line).bound(m=(-10, 10), b=(-10, 10))

    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 5.0, 40)
    sigma = 0.3
    y = line(x, 2.0, -1.0) + rng.normal(0, sigma, size=x.size)

    run = model.fit(x, (y, sigma)).squeeze()
    res = run.results

    assert res.cov is not None
    m_u = res["m"].u
    b_u = res["b"].u

    cov = np.array(uncertainties.covariance_matrix([m_u, b_u]), dtype=float)
    np.testing.assert_allclose(cov, res.cov, rtol=1e-6, atol=1e-6)
