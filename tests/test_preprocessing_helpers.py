"""Stationary preprocessing and coarse-graining behaviour."""

import numpy as np
import pytest

from log_psplines.config import StationaryConfig
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.preprocessing.coarse_grain import (
    CoarseGrainConfig,
    _sum_bins_equal,
    apply_coarse_grain_multivar_fft,
    compute_binning_structure,
)
from log_psplines.preprocessing.spectral import (
    _normalize_excluded_frequency_bands,
    preprocess_to_freq_domain,
)


def _series():
    return VARMAData(n_samples=64, fs=16.0, seed=11).ts


def test_excluded_frequency_bands_are_validated_and_merged() -> None:
    assert _normalize_excluded_frequency_bands(None) == ()
    assert _normalize_excluded_frequency_bands([]) == ()
    assert _normalize_excluded_frequency_bands(
        [(0.4, 0.5), (0.2, 0.1), (0.15, 0.3)]
    ) == ((0.1, 0.3), (0.4, 0.5))
    with pytest.raises(ValueError, match="length-2"):
        _normalize_excluded_frequency_bands([(0.1, 0.2, 0.3)])
    with pytest.raises(ValueError, match="finite"):
        _normalize_excluded_frequency_bands([(0.1, np.inf)])


def test_frequency_exclusion_filters_wishart_data() -> None:
    series = _series()
    original = preprocess_to_freq_domain(series, StationaryConfig())
    band = (float(original.freq[2]), float(original.freq[4]))
    filtered = preprocess_to_freq_domain(
        series, StationaryConfig(exclude_freq_bands=[band])
    )
    mask = (original.freq < band[0]) | (original.freq > band[1])
    np.testing.assert_allclose(filtered.freq, original.freq[mask])
    np.testing.assert_allclose(filtered.U, original.U[mask])
    with pytest.raises(ValueError, match="all inference bins"):
        preprocess_to_freq_domain(
            series,
            StationaryConfig(
                exclude_freq_bands=[
                    (float(original.freq[0]), float(original.freq[-1]))
                ]
            ),
        )


def test_coarse_grain_config_and_equal_bin_helpers() -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        CoarseGrainConfig(Nc=2, Nh=2)
    with pytest.raises(TypeError, match="integer"):
        CoarseGrainConfig(Nc=True, Nh=None)
    with pytest.raises(ValueError, match="positive"):
        CoarseGrainConfig(Nc=0, Nh=None)
    np.testing.assert_array_equal(
        _sum_bins_equal(np.arange(6), Nh=2), np.asarray([1, 5, 9])
    )
    with pytest.raises(ValueError, match="divisible"):
        _sum_bins_equal(np.arange(5), Nh=2)


def test_stationary_coarse_graining_matches_direct_operation() -> None:
    series = _series()
    original = preprocess_to_freq_domain(series, StationaryConfig())
    spec = compute_binning_structure(original.freq, Nc=5)
    direct = apply_coarse_grain_multivar_fft(original, spec)
    config = StationaryConfig(
        coarse_grain_config={"enabled": True, "Nc": 5, "Nh": None}
    )
    processed = preprocess_to_freq_domain(series, config)
    np.testing.assert_allclose(processed.freq, direct.freq)
    np.testing.assert_allclose(processed.U, direct.U)
    np.testing.assert_allclose(processed.raw_psd, direct.raw_psd)
    assert processed.Nh == spec.Nh
