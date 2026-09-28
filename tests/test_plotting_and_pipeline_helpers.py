import matplotlib.pyplot as plt
import numpy as np
import pytest

from log_psplines.config import StationaryConfig
from log_psplines.data import TimeSeries, WishartData
from log_psplines.inference.vi import VIResult
from log_psplines.plotting.base import (
    PlotConfig,
    compute_confidence_intervals,
    setup_plot_style,
)
from log_psplines.plotting.vi import plot_vi_loss
from log_psplines.preprocessing.spectral import (
    _unpack_true_psd,
    align_true_psd_to_freq,
    preprocess_to_freq_domain,
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
    processed = preprocess_to_freq_domain(ts, StationaryConfig())
    assert processed.N > 0


def test_plotting_base_confidence_intervals() -> None:
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


def test_vi_loss_plot(tmp_path) -> None:
    vi = VIResult(
        posterior=None, losses=np.asarray([5.0, 4.0, 3.0]),
        guide_name="diag", losses_per_block=None,
    )
    fig = plot_vi_loss(vi)
    assert fig is not None
    fig.canvas.draw()
    plt.close(fig)
    out = tmp_path / "vi.png"
    assert plot_vi_loss(vi, outfile=str(out)) is None
    assert out.exists()
