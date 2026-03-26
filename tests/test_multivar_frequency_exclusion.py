import numpy as np

from log_psplines.datatypes.multivar import MultivarFFT, MultivariateTimeseries
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.preprocessing.coarse_grain import CoarseGrainConfig
from log_psplines.preprocessing.config_utils import _build_config_from_kwargs
from log_psplines.preprocessing.configs import (
    DiagnosticsConfig,
    ModelConfig,
    RunMCMCConfig,
)
from log_psplines.preprocessing.preprocessing import _preprocess_data


def test_build_config_from_kwargs_routes_excluded_frequency_bands():
    cfg = _build_config_from_kwargs(
        n_knots=4,
        exclude_freq_bands=((0.1, 0.2), (0.3, 0.4)),
    )

    assert cfg.model.exclude_freq_bands == ((0.1, 0.2), (0.3, 0.4))


def test_multivar_fft_exclude_frequency_bands_preserves_alignment():
    freq = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float64)
    u_re = np.arange(freq.size * 4, dtype=np.float64).reshape(freq.size, 2, 2)
    u_im = -u_re
    raw_psd = (
        np.arange(freq.size * 4, dtype=np.float64).reshape(freq.size, 2, 2)
        + 1j
    )

    fft = MultivarFFT(
        u_re=u_re,
        u_im=u_im,
        freq=freq,
        N=freq.size,
        p=2,
        Nb=3,
        Nh=4,
        raw_psd=raw_psd,
        raw_freq=freq,
        fs=2.0,
        duration=8.0,
    )

    filtered = fft.exclude_frequency_bands(((0.15, 0.25), (0.39, 0.41)))

    expected_idx = np.array([0, 2, 4])
    np.testing.assert_allclose(filtered.freq, freq[expected_idx])
    np.testing.assert_allclose(filtered.u_re, u_re[expected_idx])
    np.testing.assert_allclose(filtered.u_im, u_im[expected_idx])
    np.testing.assert_allclose(filtered.raw_psd, raw_psd[expected_idx])
    np.testing.assert_allclose(filtered.raw_freq, freq[expected_idx])
    assert filtered.N == expected_idx.size
    assert filtered.Nb == fft.Nb
    assert filtered.Nh == fft.Nh
    assert filtered.duration == fft.duration


def test_preprocess_multivar_applies_post_coarse_exclusion_and_aligns_outputs():
    varma = VARMAData(n_samples=256, seed=2)
    ts = MultivariateTimeseries(t=varma.time, y=varma.data)

    base_cfg = RunMCMCConfig(
        Nb=2,
        coarse_grain_config=CoarseGrainConfig(enabled=True, Nh=4, Nc=None),
        diagnostics=DiagnosticsConfig(verbose=False),
        model=ModelConfig(true_psd=(varma.freq, varma.get_true_psd())),
    )
    base_preproc = _preprocess_data(ts, config=base_cfg)
    base_freq = np.asarray(base_preproc.processed_data.freq, dtype=np.float64)
    spacing = float(np.median(np.diff(base_freq)))
    target = float(base_freq[len(base_freq) // 2])
    band = (target - 0.4 * spacing, target + 0.4 * spacing)

    cfg = RunMCMCConfig(
        Nb=2,
        coarse_grain_config=CoarseGrainConfig(enabled=True, Nh=4, Nc=None),
        diagnostics=DiagnosticsConfig(verbose=False),
        model=ModelConfig(
            true_psd=(varma.freq, varma.get_true_psd()),
            exclude_freq_bands=(band,),
        ),
    )
    preproc = _preprocess_data(ts, config=cfg)

    freq = np.asarray(preproc.processed_data.freq, dtype=np.float64)
    assert preproc.processed_data.N == base_preproc.processed_data.N - 1
    assert not np.any((freq >= band[0]) & (freq <= band[1]))
    assert preproc.scaled_true_psd is not None
    assert preproc.scaled_true_psd.shape[0] == preproc.processed_data.N
    assert preproc.processed_data.raw_psd is not None
    assert preproc.processed_data.raw_psd.shape[0] == preproc.processed_data.N

    assert preproc.extra_empirical_psd is not None
    overlay = preproc.extra_empirical_psd[0]
    assert not np.any((overlay.freq >= band[0]) & (overlay.freq <= band[1]))
    assert overlay.freq.min() >= freq.min()
    assert overlay.freq.max() <= freq.max()
