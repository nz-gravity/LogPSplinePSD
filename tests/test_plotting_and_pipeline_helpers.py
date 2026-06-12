import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from log_psplines.datatypes.multivar import MultivarFFT, MultivariateTimeseries
from log_psplines.pipeline.config import PipelineConfig
from log_psplines.pipeline.preprocessing import (
    _max_n_knots,
    _unpack_true_psd,
    align_true_psd_to_freq,
    coarse_vi_freq_domain,
    preprocess_to_freq_domain,
)
from log_psplines.plotting.base import (
    PlotConfig,
    _as_matrix_quantiles,
    _quantiles_from_standard_psd_dataset,
    compute_confidence_intervals,
    extract_plotting_data,
    setup_plot_style,
)
from log_psplines.plotting.vi import (
    _compute_shift_value,
    _min_loss,
    _normalize_loss_components,
    _normalize_vi_losses,
    plot_vi_loss,
)


def _fft(n: int = 12) -> MultivarFFT:
    u_re = np.zeros((n, 2, 2))
    u_im = np.zeros_like(u_re)
    for idx in range(n):
        u_re[idx] = np.asarray(
            [[1.0 + 0.01 * idx, 0.0], [0.1, 1.2 + 0.01 * idx]]
        )
    return MultivarFFT(
        u_re=u_re,
        u_im=u_im,
        freq=np.linspace(0.1, 1.2, n),
        N=n,
        p=2,
        Nb=1,
        raw_psd=np.einsum("fkc,flc->fkl", u_re, u_re),
        raw_freq=np.linspace(0.1, 1.2, n),
    )


def test_pipeline_preprocessing_alignment_and_coarse_vi() -> None:
    fft = _fft(12)
    psd = np.stack([np.eye(2) * (1.0 + idx) for idx in range(12)]).astype(
        complex
    )

    assert _max_n_knots(5) == 5
    assert _max_n_knots({"delta": 4, "theta_re": 6, "theta_im": 5}) == 6
    assert _unpack_true_psd(None) == (None, None)
    freq, unpacked = _unpack_true_psd({"freq": fft.freq, "psd": psd})
    np.testing.assert_allclose(freq, fft.freq)
    np.testing.assert_allclose(unpacked, psd)
    freq2, unpacked2 = _unpack_true_psd((fft.freq, psd))
    np.testing.assert_allclose(freq2, fft.freq)
    np.testing.assert_allclose(unpacked2, psd)

    aligned = align_true_psd_to_freq({"freq": fft.freq, "psd": psd}, fft)
    np.testing.assert_allclose(aligned, psd)
    uniform = align_true_psd_to_freq(psd[::2], fft)
    assert uniform.shape == psd.shape
    no_data = align_true_psd_to_freq(psd, None)
    np.testing.assert_allclose(no_data, psd)

    with pytest.raises(ValueError, match="must contain"):
        _unpack_true_psd({"freq": fft.freq})
    with pytest.raises(ValueError, match="matching lengths"):
        align_true_psd_to_freq((fft.freq[:-1], psd), fft)

    explicit = coarse_vi_freq_domain(
        fft,
        PipelineConfig(coarse_grain_config_vi={"enabled": True, "Nc": 5}),
    )
    assert explicit is not None
    assert explicit.N in {4, 6}

    explicit_nh = coarse_vi_freq_domain(
        fft,
        PipelineConfig(
            coarse_grain_config_vi={"enabled": True, "Nc": None, "Nh": 5}
        ),
    )
    assert explicit_nh is not None
    assert explicit_nh.Nh in {4, 6}

    assert (
        coarse_vi_freq_domain(fft, PipelineConfig(auto_coarse_vi=False))
        is None
    )
    assert (
        coarse_vi_freq_domain(
            fft,
            PipelineConfig(
                auto_coarse_vi=True, auto_coarse_vi_min_full_nfreq=999
            ),
        )
        is None
    )
    auto = coarse_vi_freq_domain(
        _fft(60),
        PipelineConfig(
            auto_coarse_vi=True,
            n_knots=3,
            degree=1,
            auto_coarse_vi_min_full_nfreq=1,
            auto_coarse_vi_target_nfreq=10,
        ),
    )
    assert auto is not None
    assert auto.N < 60

    ts = MultivariateTimeseries(np.arange(16.0), t=np.arange(16.0))
    processed = preprocess_to_freq_domain(ts, PipelineConfig())
    assert processed.N > 0


def test_plotting_base_quantiles_and_confidence_intervals() -> None:
    spectral_density = np.ones((2, 3, 2, 2, 4), dtype=np.complex128)
    spectral_density[..., 0, 1, :] = 0.2 + 0.1j
    spectral_density[..., 1, 0, :] = 0.2 - 0.1j
    coherence = np.clip(np.abs(spectral_density) ** 2, 0.0, 1.0).real
    ds = xr.Dataset(
        {
            "spectral_density": xr.DataArray(
                spectral_density,
                dims=("chain", "draw", "channel", "channel_aux", "frequency"),
            ),
            "coherence": xr.DataArray(
                coherence,
                dims=("chain", "draw", "channel", "channel_aux", "frequency"),
            ),
        },
        coords={"frequency": np.linspace(0.1, 0.4, 4)},
    )

    quantiles = _quantiles_from_standard_psd_dataset(ds)
    assert quantiles["spectral_density"].shape == (3, 4, 2, 2)
    matrix = _as_matrix_quantiles(quantiles)
    assert matrix["coherence"].shape == (3, 4, 2, 2)

    lower, median, upper = compute_confidence_intervals(
        np.arange(12.0).reshape(3, 4)
    )
    assert lower.shape == median.shape == upper.shape == (4,)
    lower_u, median_u, upper_u = compute_confidence_intervals(
        np.arange(12.0).reshape(3, 4),
        method="uniform",
    )
    assert lower_u.shape == median_u.shape == upper_u.shape == (4,)
    with pytest.raises(ValueError, match="Unknown CI"):
        compute_confidence_intervals(np.ones((3, 4)), method="bad")

    config = setup_plot_style(PlotConfig(fontsize=9, dpi=90))
    assert config.fontsize == 9


def test_extract_plotting_data_and_vi_loss_plot(monkeypatch, tmp_path) -> None:
    import log_psplines.arviz_utils as au

    spectral_density = np.ones((1, 2, 1, 1, 3), dtype=np.complex128)
    psd_ds = xr.Dataset(
        {
            "spectral_density": xr.DataArray(
                spectral_density,
                dims=("chain", "draw", "channel", "channel_aux", "frequency"),
            )
        },
        coords={"frequency": np.asarray([0.1, 0.2, 0.3])},
    )

    monkeypatch.setattr(
        au, "get_weights", lambda *args, **kwargs: np.ones((2, 3))
    )
    monkeypatch.setattr(au, "get_psd_dataset", lambda *args, **kwargs: psd_ds)
    monkeypatch.setattr(
        au,
        "get_multivar_prior_psd_quantiles",
        lambda _: {
            "percentile": np.asarray([5.0, 50.0, 95.0]),
            "spectral_density": np.ones((3, 3, 1, 1), dtype=np.complex128),
        },
    )
    idata = xr.DataTree()
    idata.attrs.update(
        {
            "tau": 0.5,
            "design_psd": np.ones((3, 1, 1)),
            "true_psd": np.ones(3),
            "frequencies": np.asarray([1.0, 2.0, 3.0]),
        }
    )
    data = extract_plotting_data(idata, weights_key=0)
    assert data["weights"].shape == (2, 3)
    assert "posterior_psd_matrix_quantiles" in data
    assert "vi_psd_matrix_quantiles" in data
    assert "prior_psd_matrix_quantiles" in data
    assert data["frequencies"].shape == (3,)

    assert _normalize_vi_losses([]) is None
    assert set(
        _normalize_vi_losses({"losses_per_block": [[3, 2], [4, 1]]})
    ) == {
        "Factor 0",
        "Factor 1",
    }
    assert set(_normalize_vi_losses({"a": [3, 2], "b": []})) == {"a"}
    assert set(_normalize_loss_components({"x": [1, 2], "empty": []})) == {"x"}
    assert _min_loss({"a": np.asarray([])}) is None
    assert _compute_shift_value(np.asarray([0.0])) == pytest.approx(-1.0)

    fig = plot_vi_loss(
        {"losses": [5.0, 4.0, 3.0]},
        guide_name="guide",
        loss_components={
            "recon": np.asarray([4.5, 3.5, 2.5]),
            "short": np.asarray([1.0]),
        },
    )
    assert fig is not None
    fig.canvas.draw()
    plt.close(fig)

    out = tmp_path / "vi.png"
    assert plot_vi_loss([3.0, 2.0, 1.0], outfile=str(out)) is None
    assert out.exists()
    assert plot_vi_loss([]) is None
