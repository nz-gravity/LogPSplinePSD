import numpy as np

from log_psplines.psplines.knots_locator import knot_locator as knot_locator_mod


def test_density_knots_enforce_fixed_count_with_endpoint_tolerance(
    monkeypatch,
):
    freqs = np.linspace(10.0, 20.0, 16, dtype=np.float64)
    power = np.linspace(1.0, 2.0, 16, dtype=np.float64)
    n_knots = 6

    def _fake_quantile_based_knots(
        n_knots: int,
        periodogram,
        parametric_model=None,
    ) -> np.ndarray:
        # Intentionally include near-endpoint and near-duplicate values.
        return np.array(
            [
                10.0,
                10.0 + 1e-11,
                13.0,
                17.0,
                20.0 - 1e-11,
                20.0,
            ],
            dtype=np.float64,
        )

    monkeypatch.setattr(
        knot_locator_mod,
        "_quantile_based_knots",
        _fake_quantile_based_knots,
    )

    knots = knot_locator_mod._init_knots_from_arrays(
        n_knots=n_knots,
        freqs=freqs,
        power=power,
        method="density",
    )

    assert knots.shape == (n_knots,)
    assert np.isclose(knots[0], 0.0)
    assert np.isclose(knots[-1], 1.0)
    assert np.all(np.diff(knots) > 0.0)
