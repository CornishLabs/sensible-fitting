import pytest

from sensible_fitting.util import uncertainty_to_string


@pytest.mark.parametrize(
    "x, err, precision, expected",
    [
        (0.0, 1e-4, 1, "0(1)e-4"),
        (12.34567, 0.00123, 1, "12.346(1)"),
        (12.34567, 0.00123, 2, "12.3457(12)"),
        (-0.123456, 0.000123, 2, "-0.12346(12)"),
        (0.00123456, 0.000000012345, 2, "0.001234560(12)"),
        (-0.0000123456, 0.0000001234, 1, "-1.23(1)e-5"),
        (1.0, 0.0, 2, "1(0)"),
        (float("nan"), 1.0, 1, "NaN"),
        (1.0, float("inf"), 1, "inf"),
        (1.0, -0.1, 1, "1.0(1)"),
        (1.2345, 0.067, 0, "1.23(7)"),
        (12.34567, 0.00123, "auto", "12.3457(12)"),
        (1.2345, 0.067, "auto", "1.23(7)"),
    ],
)
def test_uncertainty_to_string(x, err, precision, expected):
    assert uncertainty_to_string(x, err, precision) == expected
