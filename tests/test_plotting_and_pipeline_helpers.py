import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from log_psplines.data import WishartData, TimeSeries
from log_psplines.config import PipelineConfig
from log_psplines.preprocessing.spectral import (
    _unpack_true_psd,
    align_true_psd_to_freq,
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


def _fft(n: int = 12) -> WishartData:
    u_re = np.zeros((n, 2, 2))
    u_im = np.zeros_like(u_re)
    for idx in range(n):
        u_re[idx] = np.asarray(
            [[1.0 + 0.01 * idx, 0.0], [0.1, 1.2 + 0.01 * idx]]
        )
    return WishartData(
        u_re=u_re,
        u_im=u_im,
        freq=np.linspace(0.1, 1.2, n),
        N=n,
        p=2,
        Nb=1,
        raw_psd=np.einsum("fkc,flc->fkl", u_re, u_re),
        raw_freq=np.linspace(0.1, 1.2, n),
    )


def test_pipeline_preprocessing_alignment() -> None:
    fft = _fft(12)
    psd = np.stack([np.eye(2) * (1.0 + idx) for idx in range(12)]).astype(
        complex
    )

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

    ts = TimeSeries(np.arange(16.0), t=np.arange(16.0))
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


def test_extract_plotting_data_and_vi_loss_plot(tmp_path) -> None:
    from log_psplines.results import PSDResult

    posterior = xr.Dataset(
        {
            "weights": (
                ("chain", "draw", "coefficient"),
                np.ones((1, 2, 3)),
            )
        }
    )
    spectrum = xr.DataArray(
        np.ones((1, 2, 3, 1, 1), dtype=np.complex128),
        dims=("chain", "draw", "frequency", "channel", "channel_aux"),
        coords={
            "frequency": np.asarray([0.1, 0.2, 0.3]),
            "channel": [0],
            "channel_aux": [0],
        },
    )
    result = PSDResult(
        posterior=posterior,
        spectrum=spectrum,
        metadata={"true_psd": np.ones(3)},
    )
    data = extract_plotting_data(result)
    assert data["weights"].shape == (2, 3)
    assert "posterior_psd_matrix_quantiles" in data
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
