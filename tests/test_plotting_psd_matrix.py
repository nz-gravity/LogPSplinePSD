import matplotlib.pyplot as plt
import numpy as np
import pytest

from log_psplines.datatypes.multivar import EmpiricalPSD
from log_psplines.plotting import PSDMatrixPlotSpec, plot_psd_matrix


def _plot_inputs():
    freq = np.linspace(0.0, 1.0, 7)
    p = 2
    q05 = np.zeros((freq.size, p, p), dtype=np.complex128)
    q50 = np.zeros_like(q05)
    q95 = np.zeros_like(q05)
    for idx, f in enumerate(freq):
        base = 1.0 + f
        q50[idx] = np.asarray(
            [[base + 1.0, 0.2 + 0.05j], [0.2 - 0.05j, base + 1.5]]
        )
        q05[idx] = q50[idx] - (0.1 + 0.01j)
        q95[idx] = q50[idx] + (0.1 + 0.01j)

    coherence = np.zeros((3, freq.size, p, p), dtype=float)
    for q_idx, matrix in enumerate((q05, q50, q95)):
        diag = np.real(np.diagonal(matrix, axis1=1, axis2=2))
        denom = diag[:, :, None] * diag[:, None, :]
        coherence[q_idx] = np.abs(matrix) ** 2 / denom

    ci_dict = {
        "psd": {
            (0, 0): (q05[:, 0, 0].real, q50[:, 0, 0].real, q95[:, 0, 0].real),
            (1, 1): (q05[:, 1, 1].real, q50[:, 1, 1].real, q95[:, 1, 1].real),
        },
        "coh": {
            (1, 0): (
                coherence[0, :, 1, 0],
                coherence[1, :, 1, 0],
                coherence[2, :, 1, 0],
            ),
        },
        "re": {
            (1, 0): (q05[:, 1, 0].real, q50[:, 1, 0].real, q95[:, 1, 0].real),
            (0, 1): (q05[:, 0, 1].real, q50[:, 0, 1].real, q95[:, 0, 1].real),
        },
        "im": {
            (1, 0): (q05[:, 1, 0].imag, q50[:, 1, 0].imag, q95[:, 1, 0].imag),
            (0, 1): (q05[:, 0, 1].imag, q50[:, 0, 1].imag, q95[:, 0, 1].imag),
        },
        "mag": {
            (1, 0): (
                np.abs(q05[:, 1, 0]),
                np.abs(q50[:, 1, 0]),
                np.abs(q95[:, 1, 0]),
            ),
        },
    }
    empirical = EmpiricalPSD(
        freq=freq,
        psd=q50,
        coherence=coherence[1],
        channels=np.asarray(["x", "y"]),
    )
    return freq, ci_dict, empirical, q50


def test_plot_psd_matrix_re_im_and_magnitude_modes(tmp_path) -> None:
    freq, ci_dict, empirical, true_psd = _plot_inputs()
    extra = EmpiricalPSD(
        freq=freq,
        psd=empirical.psd * 1.05,
        coherence=empirical.coherence,
        channels=empirical.channels,
    )

    fig, axes = plot_psd_matrix(
        PSDMatrixPlotSpec(
            ci_dict=ci_dict,
            freq=freq,
            empirical_psd=empirical,
            extra_empirical_psd=[extra],
            extra_empirical_labels=["Welch"],
            extra_empirical_styles=[{"color": "0.2"}],
            true_psd=true_psd,
            save=False,
            close=False,
            show_coherence=False,
            show_knots=True,
            channel_labels="xy",
            excluded_bands=((0.25, 0.35),),
            psd_scale=lambda f: 1.0 + f,
        )
    )
    assert axes.shape == (2, 2)
    fig.canvas.draw()
    plt.close(fig)

    out = tmp_path / "plot.png"
    fig, axes = plot_psd_matrix(
        PSDMatrixPlotSpec(
            ci_dict=ci_dict,
            freq=freq,
            empirical_psd=empirical,
            true_psd=true_psd,
            outdir=str(tmp_path),
            filename=out.name,
            save=True,
            close=False,
            show_coherence=False,
            show_csd_magnitude=True,
            show_empirical=False,
            freq_range=(0.1, 0.9),
        )
    )
    fig.canvas.draw()
    plt.close(fig)
    assert out.exists()


def test_plot_psd_matrix_validation_errors() -> None:
    freq, ci_dict, empirical, _ = _plot_inputs()

    with pytest.raises(ValueError, match="either coherence"):
        plot_psd_matrix(
            PSDMatrixPlotSpec(
                ci_dict=ci_dict,
                freq=freq,
                show_coherence=True,
                show_csd_magnitude=True,
            )
        )
    with pytest.raises(ValueError, match="Provide either"):
        plot_psd_matrix(PSDMatrixPlotSpec(freq=freq))
    with pytest.raises(ValueError, match="Frequency array"):
        plot_psd_matrix(PSDMatrixPlotSpec(ci_dict=ci_dict))
    with pytest.raises(ValueError, match="missing coherence"):
        broken = {**ci_dict, "coh": {}}
        plot_psd_matrix(
            PSDMatrixPlotSpec(
                ci_dict=broken,
                freq=freq,
                empirical_psd=empirical,
                show_coherence=True,
            )
        )
    with pytest.raises(ValueError, match="non-negative"):
        plot_psd_matrix(
            PSDMatrixPlotSpec(
                ci_dict=ci_dict,
                freq=freq,
                show_coherence=False,
                psd_scale=-1.0,
            )
        )
    with pytest.raises(ValueError, match="base_freq shape"):
        plot_psd_matrix(
            PSDMatrixPlotSpec(
                ci_dict=ci_dict,
                freq=freq,
                show_coherence=False,
                psd_scale=np.linspace(1.0, 2.0, freq.size),
            )
        )
