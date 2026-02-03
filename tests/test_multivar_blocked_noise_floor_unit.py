import numpy as np

from log_psplines.samplers.multivar.multivar_blocked_nuts import (
    compute_noise_floor_sq,
)


def _block_log_likelihood(
    delta_sq_value: float,
    residual_power: np.ndarray,
    freq_weights: np.ndarray,
    nu: int,
    noise_floor_sq: np.ndarray | None = None,
) -> float:
    delta_sq = np.full_like(residual_power, float(delta_sq_value))
    if noise_floor_sq is None:
        delta_eff_sq = delta_sq
    else:
        delta_eff_sq = delta_sq + noise_floor_sq
    sum_log_det = -float(nu) * np.sum(freq_weights * np.log(delta_eff_sq))
    return float(sum_log_det - np.sum(residual_power / delta_eff_sq))


def test_noise_floor_plateaus_near_null_residual_block3():
    n_freq = 64
    freqs = np.linspace(0.01, 1.0, n_freq, dtype=np.float32)
    residual_power = np.full((n_freq,), 1e-12, dtype=np.float32)
    freq_weights = np.ones((n_freq,), dtype=np.float32)
    nu = 2
    delta_sq_values = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10], dtype=float)

    ll_no_floor = np.array(
        [
            _block_log_likelihood(delta, residual_power, freq_weights, nu)
            for delta in delta_sq_values
        ]
    )

    assert np.all(np.diff(ll_no_floor) > 0.0)
    assert ll_no_floor[-1] - ll_no_floor[0] > 100.0

    noise_floor_sq = np.asarray(
        compute_noise_floor_sq(
            freqs=freqs,
            block_j=3,
            mode="constant",
            constant=1e-6,
            scale=1e-4,
            array=None,
            theory_psd=None,
        )
    )

    ll_with_floor = np.array(
        [
            _block_log_likelihood(
                delta,
                residual_power,
                freq_weights,
                nu,
                noise_floor_sq=noise_floor_sq,
            )
            for delta in delta_sq_values
        ]
    )

    assert np.all(np.diff(ll_with_floor) > 0.0)
    assert ll_with_floor[-1] - ll_with_floor[-2] < 5.0
    assert (ll_with_floor[-1] - ll_with_floor[-2]) < 0.01 * (
        ll_with_floor[1] - ll_with_floor[0]
    )
    assert ll_with_floor[-1] < ll_no_floor[-1]
