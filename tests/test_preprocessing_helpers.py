import numpy as np
import pytest

from log_psplines.datatypes.multivar import EmpiricalPSD
from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.pipeline.config import PipelineConfig
from log_psplines.preprocessing.coarse_grain import (
    CoarseGrainConfig,
    _sum_bins_equal,
    apply_coarse_grain_multivar_fft,
    compute_binning_structure,
)
from log_psplines.preprocessing.data_prep import (
    _apply_frequency_exclusion,
    _build_frequency_exclusion_mask,
    _coarse_grain_processed_data,
    _filter_empirical_psd,
    _normalize_coarse_grain_config,
    _normalize_excluded_frequency_bands,
)


def _small_fft():
    data = VARMAData(n_samples=64, fs=16.0, seed=11)
    return data.ts.standardise_for_psd().to_wishart_stats(Nb=1)


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


def test_frequency_exclusion_filters_fft_and_empirical_psd() -> None:
    fft = _small_fft()
    band = (float(fft.freq[2]), float(fft.freq[4]))
    mask = _build_frequency_exclusion_mask(fft.freq, [band])
    assert mask.shape == fft.freq.shape
    assert not np.all(mask)

    filtered = _apply_frequency_exclusion(fft, [band])
    assert int(np.count_nonzero(mask)) == filtered.N
    np.testing.assert_allclose(filtered.freq, fft.freq[mask])

    empirical = EmpiricalPSD(
        freq=np.asarray(fft.empirical_psd.freq),
        psd=np.asarray(fft.empirical_psd.psd),
        coherence=np.asarray(fft.empirical_psd.coherence),
        channels=fft.empirical_psd.channels,
    )
    filtered_emp = _filter_empirical_psd(empirical, [band])
    np.testing.assert_allclose(filtered_emp.freq, empirical.freq[mask])

    with pytest.raises(ValueError, match="all inference bins"):
        _apply_frequency_exclusion(
            fft,
            [(float(fft.freq[0]), float(fft.freq[-1]))],
        )
    with pytest.raises(ValueError, match="all empirical overlay bins"):
        _filter_empirical_psd(
            empirical,
            [(float(empirical.freq[0]), float(empirical.freq[-1]))],
        )


def test_coarse_grain_config_and_equal_bin_helpers() -> None:
    assert isinstance(_normalize_coarse_grain_config(None), CoarseGrainConfig)
    cfg = _normalize_coarse_grain_config(
        {"enabled": True, "Nh": 2, "Nc": None}
    )
    assert cfg.enabled is True
    assert cfg.Nh == 2

    with pytest.raises(ValueError, match="Exactly one"):
        CoarseGrainConfig(Nc=2, Nh=2)
    with pytest.raises(TypeError, match="integer"):
        CoarseGrainConfig(Nc=True, Nh=None)
    with pytest.raises(ValueError, match="positive"):
        CoarseGrainConfig(Nc=0, Nh=None)

    np.testing.assert_array_equal(
        _sum_bins_equal(np.arange(6), Nh=2),
        np.asarray([1, 5, 9]),
    )
    with pytest.raises(ValueError, match="divisible"):
        _sum_bins_equal(np.arange(5), Nh=2)


def test_coarse_grain_processed_data_trims_and_preserves_shapes() -> None:
    fft = _small_fft()
    spec = compute_binning_structure(fft.freq, Nc=5)
    coarse_direct = apply_coarse_grain_multivar_fft(fft, spec)
    assert spec.Nc == coarse_direct.N
    assert coarse_direct.Nh == spec.Nh
    assert coarse_direct.raw_psd.shape == (spec.Nc, fft.p, fft.p)

    config = CoarseGrainConfig(enabled=True, Nc=5, Nh=None)
    coarse, true_psd = _coarse_grain_processed_data(fft, config, None)
    assert true_psd is None
    assert coarse is not None
    assert spec.Nc == coarse.N
    assert coarse.freq.shape == (spec.Nc,)

    unchanged, _ = _coarse_grain_processed_data(
        fft,
        CoarseGrainConfig(enabled=False, Nc=5, Nh=None),
        None,
    )
    assert unchanged is fft

    config_obj = PipelineConfig(
        coarse_grain_config=CoarseGrainConfig(enabled=True, Nc=5, Nh=None)
    )
    assert config_obj.coarse_grain_config is not None
