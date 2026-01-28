import numpy as np

from log_psplines.psplines.multivar_psplines import MultivariateLogPSplines


def test_compute_psd_quantiles_flattens_chains_and_spreads_draws():
    # Use a dummy instance: compute_psd_quantiles uses only the passed arrays.
    spline_model = MultivariateLogPSplines.__new__(MultivariateLogPSplines)

    n_chains = 1
    n_draws = 100
    n_freq = 3
    n_channels = 1

    # Monotone increasing log(δ^2) across draws so the median depends on
    # which draws are selected. Keep values below ~80 to avoid float32 exp
    # overflow in the PSD reconstruction path.
    log_delta_sq = np.linspace(0.0, 60.0, n_draws, dtype=np.float64)[
        None, :, None, None
    ]
    log_delta_sq = np.broadcast_to(
        log_delta_sq, (n_chains, n_draws, n_freq, n_channels)
    )

    theta_re = np.zeros((n_chains, n_draws, n_freq, 0), dtype=np.float64)
    theta_im = np.zeros((n_chains, n_draws, n_freq, 0), dtype=np.float64)

    # If we incorrectly took the first 10 draws only, the median would be small
    # (around exp(2.7)). With evenly spaced draw selection it should be much larger.
    psd_real_q, _, _ = spline_model.compute_psd_quantiles(
        log_delta_sq,
        theta_re,
        theta_im,
        n_samples_max=10,
        compute_coherence=False,
    )
    median_psd = psd_real_q[1, :, 0, 0]
    assert float(np.min(median_psd)) > float(np.exp(10.0))


def test_compute_psd_quantiles_chain_flatten_matches_manual_flatten():
    spline_model = MultivariateLogPSplines.__new__(MultivariateLogPSplines)

    n_chains = 2
    n_draws = 20
    n_freq = 4
    n_channels = 2
    n_theta = 1

    rng = np.random.default_rng(0)
    log_delta_sq = rng.normal(size=(n_chains, n_draws, n_freq, n_channels))
    theta_re = rng.normal(size=(n_chains, n_draws, n_freq, n_theta))
    theta_im = rng.normal(size=(n_chains, n_draws, n_freq, n_theta))

    out_4d = spline_model.compute_psd_quantiles(
        log_delta_sq,
        theta_re,
        theta_im,
        n_samples_max=15,
        compute_coherence=False,
    )

    flat_log_delta = log_delta_sq.reshape((-1,) + log_delta_sq.shape[2:])
    flat_theta_re = theta_re.reshape((-1,) + theta_re.shape[2:])
    flat_theta_im = theta_im.reshape((-1,) + theta_im.shape[2:])

    out_3d = spline_model.compute_psd_quantiles(
        flat_log_delta,
        flat_theta_re,
        flat_theta_im,
        n_samples_max=15,
        compute_coherence=False,
    )

    for a, b in zip(out_4d[:2], out_3d[:2]):
        assert np.allclose(a, b, rtol=1e-10, atol=1e-10)
