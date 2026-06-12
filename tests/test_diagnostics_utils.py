import numpy as np
import pytest

from log_psplines.diagnostics._utils import (
    as_scalar,
    compute_ci_coverage_multivar,
    compute_ci_coverage_multivar_detailed,
    compute_coherence_coverage,
    compute_riae,
    compute_riae_errorbars,
    extract_percentile,
    find_posterior_inflation_factor,
    khat_status,
)


def _toy_psd_stack() -> tuple[np.ndarray, np.ndarray]:
    true_psd = np.asarray(
        [
            [[2.0 + 0.0j, 0.20 + 0.10j], [0.20 - 0.10j, 1.5 + 0.0j]],
            [[2.5 + 0.0j, 0.10 + 0.05j], [0.10 - 0.05j, 1.8 + 0.0j]],
            [[3.0 + 0.0j, 0.30 + 0.20j], [0.30 - 0.20j, 2.0 + 0.0j]],
        ],
        dtype=np.complex128,
    )
    q05 = true_psd - (0.5 + 0.05j)
    q50 = true_psd
    q95 = true_psd + (0.5 + 0.05j)
    return np.stack([q05, q50, q95], axis=0), true_psd


def test_scalar_and_khat_status_helpers() -> None:
    assert as_scalar(None) is None
    assert as_scalar("not-a-number") is None
    assert as_scalar(np.asarray([3.5])) == pytest.approx(3.5)

    assert khat_status(None) == (None, "unknown")
    assert khat_status(np.nan) == (None, "unknown")
    assert khat_status(0.2) == (0.0, "ok")
    assert khat_status(0.6) == (1.0, "warn")
    assert khat_status(0.8) == (2.0, "fail")


def test_riae_errorbars_and_percentile_extraction() -> None:
    freqs = np.linspace(0.0, 1.0, 5)
    true_psd = 1.0 + freqs
    samples = np.stack(
        [
            true_psd,
            true_psd * 1.1,
            true_psd * 1.2,
            true_psd * 1.3,
        ],
        axis=0,
    )

    errorbars = compute_riae_errorbars(samples, true_psd, freqs)
    expected = [compute_riae(sample, true_psd, freqs) for sample in samples]
    assert errorbars["median"] == pytest.approx(np.median(expected))
    assert errorbars["q05"] == pytest.approx(np.percentile(expected, 5))
    assert errorbars["q95"] == pytest.approx(np.percentile(expected, 95))

    values = np.asarray([[1, 2], [3, 4], [5, 6]])
    percentiles = np.asarray([5.0, 50.0, 95.0])
    np.testing.assert_array_equal(
        extract_percentile(values, percentiles, 48.0),
        np.asarray([3, 4]),
    )


def test_multivar_coverage_and_inflation_helpers() -> None:
    psd_stack, true_psd = _toy_psd_stack()

    assert compute_ci_coverage_multivar(psd_stack, true_psd) == pytest.approx(
        1.0
    )
    detail = compute_ci_coverage_multivar_detailed(psd_stack, true_psd)
    assert detail["overall"] == pytest.approx(1.0)
    assert detail["diag"] == pytest.approx(1.0)
    assert detail["offdiag_re"] == pytest.approx(1.0)
    assert detail["offdiag_im"] == pytest.approx(1.0)
    assert detail["n_diag"] == 6
    assert detail["n_offdiag_re"] == 3
    assert detail["n_offdiag_im"] == 3

    narrow = psd_stack.copy()
    narrow[0] = psd_stack[1] - 1e-3
    narrow[2] = psd_stack[1] + 1e-3
    shifted_truth = true_psd + (0.01 + 0.0005j)
    result = find_posterior_inflation_factor(
        narrow,
        shifted_truth,
        target_coverage=0.8,
        max_iter=20,
    )
    assert result["inflation_factor"] > 1.0
    assert result["achieved_coverage"] >= 0.75

    with pytest.raises(ValueError, match="shape"):
        find_posterior_inflation_factor(np.ones((2, 3, 2, 2)), true_psd)


def test_coherence_coverage_handles_quantile_shapes() -> None:
    _, true_psd = _toy_psd_stack()
    true_coh = np.zeros_like(true_psd.real)
    for i in range(true_psd.shape[-1]):
        true_coh[:, i, i] = 1.0
    true_coh[:, 0, 1] = true_coh[:, 1, 0] = np.abs(true_psd[:, 0, 1]) ** 2 / (
        true_psd[:, 0, 0].real * true_psd[:, 1, 1].real
    )
    coh_quantiles = np.stack(
        [
            np.clip(true_coh - 0.1, 0.0, 1.0),
            true_coh,
            np.clip(true_coh + 0.1, 0.0, 1.0),
        ],
        axis=0,
    )
    assert compute_coherence_coverage(
        coh_quantiles,
        true_psd,
        np.asarray([5.0, 50.0, 95.0]),
    ) == pytest.approx(1.0)

    reordered = np.stack(
        [coh_quantiles[1], coh_quantiles[2], coh_quantiles[0]]
    )
    assert compute_coherence_coverage(
        reordered,
        true_psd,
        np.asarray([50.0, 95.0, 5.0]),
    ) == pytest.approx(1.0)

    assert np.isnan(
        compute_coherence_coverage(
            np.ones((1, true_psd.shape[0], 2, 2)),
            true_psd,
            np.asarray([50.0]),
        )
    )
