import numpy as np

from log_psplines.diagnostics._utils import compute_ci_coverage_multivar


def _make_true_psd(freqs: int = 2) -> np.ndarray:
    """Return a small Hermitian PSD-like matrix stack (freq, C, C)."""
    out = np.zeros((freqs, 2, 2), dtype=np.complex128)
    for k in range(freqs):
        a = 1.0 + 0.1 * k
        d = 2.0 + 0.2 * k
        re = 0.3 + 0.05 * k
        im = 0.1 + 0.02 * k
        out[k, 0, 0] = a
        out[k, 1, 1] = d
        out[k, 0, 1] = re + 1j * im
        out[k, 1, 0] = re - 1j * im
    return out


def test_ci_coverage_multivar_complex_percentiles() -> None:
    true_psd = _make_true_psd()

    lower = true_psd - (0.5 + 0.5j)
    upper = true_psd + (0.5 + 0.5j)
    median = true_psd
    percentiles = np.stack([lower, median, upper], axis=0)

    coverage = compute_ci_coverage_multivar(percentiles, true_psd)
    assert coverage == 1.0


def test_ci_coverage_multivar_complex_draws() -> None:
    true_psd = _make_true_psd()

    rng = np.random.default_rng(0)
    draws = 200
    noise = rng.normal(scale=0.05, size=(draws,) + true_psd.shape)
    samples = true_psd[None, ...] + noise.astype(np.complex128)

    # Widen the sample spread so the 90% interval should contain the truth
    samples = np.concatenate([samples, true_psd[None, ...]], axis=0)

    coverage = compute_ci_coverage_multivar(samples, true_psd)
    assert coverage > 0.8
