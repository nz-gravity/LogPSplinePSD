"""Coordinate-only schedules must match the transform, including its tails."""

import numpy as np
import pytest

from log_psplines.preprocessing.moving_periodogram import (
    tang_moving_periodogram,
    tang_moving_periodogram_coordinates,
)


@pytest.mark.parametrize("m,thin,n", [(1, 1, 8), (4, 3, 127), (32, 2, 1001)])
def test_coordinates_match_transform_exactly(m, thin, n):
    raw = tang_moving_periodogram(np.arange(n, dtype=float), m=m, thin=thin)
    coordinates = tang_moving_periodogram_coordinates(n, m=m, thin=thin)
    for key in ("u", "omega"):
        np.testing.assert_array_equal(coordinates[key], raw[key])
    assert len(coordinates["u"]) == ((n - 2 * m) // (thin * m)) * m


@pytest.mark.parametrize(
    "n,m,thin", [(100, 0, 2), (100, 4, False), (4, 4, 1), (0, 1, 1)]
)
def test_coordinates_reject_invalid_schedules(n, m, thin):
    with pytest.raises(ValueError):
        tang_moving_periodogram_coordinates(n, m=m, thin=thin)
