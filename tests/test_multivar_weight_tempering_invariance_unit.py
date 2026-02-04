import numpy as np


def _ll_one_freq(
    *,
    delta_sq: np.ndarray,
    residual_power_sum: float,
    nu: float,
    freq_weight: float,
    bin_count: float,
) -> np.ndarray:
    """Single-frequency blocked-factor loglik (up to constants).

    This matches the intended coarse-bin form:
      -nu * w * log(delta_sq) - w * (residual_mean / delta_sq)
    with residual_mean = residual_sum / bin_count.
    """
    delta_sq = np.asarray(delta_sq, dtype=float)
    residual_mean = float(residual_power_sum) / float(bin_count)
    return -float(nu) * float(freq_weight) * np.log(delta_sq) - float(
        freq_weight
    ) * (residual_mean / delta_sq)


def test_weight_rescaling_does_not_shift_delta_sq_scale():
    # Pick a case where the coarse-bin sufficient statistics are summed across
    # many fine-grid frequencies (bin_count > 1).
    nu = 4.0
    bin_count = 25.0
    residual_mean = 2.0e-6
    residual_sum = residual_mean * bin_count

    # Analytic optimum (per frequency) for eps2=0:
    # d/d(delta) [-nu w log(delta) - w*resid_mean/delta] = 0 => delta = resid_mean/nu
    delta_opt = residual_mean / nu

    delta_grid = np.logspace(
        np.log10(delta_opt) - 3.0, np.log10(delta_opt) + 3.0, 2000
    )

    w_raw = bin_count
    w_norm = w_raw * 0.01  # global weight normalization/tempering factor

    ll_raw = _ll_one_freq(
        delta_sq=delta_grid,
        residual_power_sum=residual_sum,
        nu=nu,
        freq_weight=w_raw,
        bin_count=bin_count,
    )
    ll_norm = _ll_one_freq(
        delta_sq=delta_grid,
        residual_power_sum=residual_sum,
        nu=nu,
        freq_weight=w_norm,
        bin_count=bin_count,
    )

    delta_hat_raw = float(delta_grid[int(np.argmax(ll_raw))])
    delta_hat_norm = float(delta_grid[int(np.argmax(ll_norm))])

    # Both should peak at the same delta scale (invariance to scaling weights).
    assert np.isclose(delta_hat_raw, delta_hat_norm, rtol=1e-2, atol=0.0)
    assert np.isclose(delta_hat_raw, delta_opt, rtol=1e-2, atol=0.0)
