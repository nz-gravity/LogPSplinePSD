import numpy as np
import pytest
import xarray as xr

from log_psplines.config import PipelineConfig
from log_psplines.data.spectral import WishartData
from log_psplines.diagnostics import sampling_diagnostics, spectrum_diagnostics
from log_psplines.diagnostics.preprocessing import (
    eig_ratios,
    eigenvalue_separation_diagnostics,
    extract_component_knots,
    ordered_eigvals_hermitian,
    ratio_summary_string,
    save_eigenvalue_separation_plot,
    worst_ratio_frequencies,
)
from log_psplines.inference.components import SpectralComponents
from log_psplines.inference.initialisation import build_component
from log_psplines.models.spectrum import LogPSpline
from log_psplines.preprocessing.checks import (
    _run_preprocessing_checks,
    _save_preprocessing_plot,
)


def _psd_stack(n: int = 6, p: int = 2) -> np.ndarray:
    out = np.zeros((n, p, p), dtype=np.complex128)
    for idx in range(n):
        out[idx] = np.asarray(
            [[2.0 + 0.1 * idx, 0.9 + 0.05j], [0.9 - 0.05j, 1.8 + 0.1 * idx]]
        )
    return out


def _fft_for_checks() -> WishartData:
    raw_psd = _psd_stack()
    u_re = np.zeros_like(raw_psd.real)
    u_im = np.zeros_like(raw_psd.real)
    for idx, matrix in enumerate(raw_psd):
        chol = np.linalg.cholesky(matrix)
        u_re[idx] = chol.real
        u_im[idx] = chol.imag
    return WishartData(
        u_re=u_re,
        u_im=u_im,
        freq=np.linspace(0.1, 0.6, raw_psd.shape[0]),
        N=raw_psd.shape[0],
        p=2,
        Nb=2,
        Nh=1,
        fs=12.0,
        duration=1.0,
        raw_psd=raw_psd,
        raw_freq=np.linspace(0.1, 0.6, raw_psd.shape[0]),
    )


def _model() -> SpectralComponents:
    def component() -> LogPSpline:
        return build_component(
            knots=np.asarray([0.0, 0.5, 1.0]),
            degree=1,
            diffMatrixOrder=1,
            n=6,
            grid_points=np.linspace(0.0, 1.0, 6),
        )

    return SpectralComponents(
        degree=1,
        diffMatrixOrder=1,
        N=6,
        p=2,
        diagonal_models=[component(), component()],
        offdiag_re_models={(1, 0): component()},
        offdiag_im_models={(1, 0): component()},
    )


def test_preprocessing_diagnostics_plot_and_validation(tmp_path) -> None:
    freq = np.linspace(0.1, 0.6, 6)
    matrix = _psd_stack()
    eig = ordered_eigvals_hermitian(matrix)
    assert eig.shape == (6, 2)
    ratios = eig_ratios(eig)
    assert set(ratios) == {"r_12"}
    assert "q05/50/95" in ratio_summary_string("r_12", ratios["r_12"])
    assert "constant" in ratio_summary_string("flat", np.ones(4) * 0.5)
    assert "no finite" in ratio_summary_string("bad", np.asarray([np.nan]))

    worst = worst_ratio_frequencies(freq, ratios["r_12"], top_k=2)
    assert len(worst) == 2
    assert worst[0][1] >= worst[1][1]
    assert worst_ratio_frequencies(freq, ratios["r_12"], top_k=0) == []

    diag = eigenvalue_separation_diagnostics(
        freq=freq,
        matrix=matrix,
        min_lambda1_quantile=0.1,
    )
    assert diag.lambda1_cutoff is not None
    assert "r_12" in diag.ratio_summary()
    assert "r_12" in diag.worst_frequencies(warn_threshold=0.1)

    out = tmp_path / "eigs.png"
    knots = extract_component_knots(_model(), freq)
    assert {"LogDelta11", "LogDelta22", "Re(Theta12)", "Im(Theta21)"} <= set(
        knots
    )
    save_eigenvalue_separation_plot(
        diag,
        str(out),
        info_text="n=6",
        excluded_bands=((0.2, 0.3),),
        cholesky_matrix=matrix,
        component_knots=knots,
        dpi=80,
    )
    assert out.exists()

    with pytest.raises(ValueError, match="shape"):
        ordered_eigvals_hermitian(np.ones((2, 2)))
    with pytest.raises(ValueError, match="eigvals_desc"):
        eig_ratios(np.ones(3))
    with pytest.raises(ValueError, match="same shape"):
        worst_ratio_frequencies(freq, ratios["r_12"][:-1])
    with pytest.raises(ValueError, match="mask"):
        worst_ratio_frequencies(
            freq, ratios["r_12"], mask=np.ones(3, dtype=bool)
        )
    with pytest.raises(ValueError, match="min_lambda1"):
        eigenvalue_separation_diagnostics(
            freq=freq,
            matrix=matrix,
            min_lambda1_quantile=1.5,
        )
    with pytest.raises(ValueError, match="cholesky_matrix"):
        save_eigenvalue_separation_plot(
            diag, str(tmp_path / "bad.png"), cholesky_matrix=matrix[:-1]
        )


def test_pipeline_preprocessing_check_wrappers(tmp_path) -> None:
    fft = _fft_for_checks()
    config = PipelineConfig(
        verbose=True, outdir=str(tmp_path), exclude_freq_bands=[(0.2, 0.25)]
    )

    _run_preprocessing_checks(fft, config)
    _run_preprocessing_checks(None, config)
    no_raw = WishartData(
        u_re=fft.u_re,
        u_im=fft.u_im,
        freq=fft.freq,
        N=fft.N,
        p=fft.p,
        raw_psd=None,
        Nb=fft.Nb,
    )
    _run_preprocessing_checks(no_raw, config)

    _save_preprocessing_plot(fft, config, spline_model=_model())
    assert (
        tmp_path / "diagnostics" / "preprocessing_eigenvalue_ratios.png"
    ).exists()
    _save_preprocessing_plot(
        fft, PipelineConfig(outdir=None), spline_model=_model()
    )
    _save_preprocessing_plot(None, config)
    _save_preprocessing_plot(no_raw, config)


def _diagnostic_result():
    from log_psplines.inference.vi import VIResult
    from log_psplines.results import PSDResult

    posterior = xr.Dataset(
        {
            "weights_delta_0": xr.DataArray(
                np.ones((1, 4, 2)), dims=("chain", "draw", "weights_dim")
            )
        }
    )
    stats = xr.Dataset(
        {
            "diverging_channel_0": (("chain", "draw"), [[0, 1, 0, 1]]),
            "tree_depth_channel_0": (("chain", "draw"), [[2, 3, 5, 5]]),
            "step_size_channel_0": (("chain", "draw"), [[0.1, 0.2, 0.2, 0.3]]),
        }
    )
    spectrum = xr.DataArray(
        np.ones((1, 4, 4, 1, 1), dtype=complex),
        dims=("chain", "draw", "frequency", "channel", "channel_aux"),
        coords={"frequency": np.linspace(0.1, 0.4, 4)},
    )
    vi = VIResult(
        posterior,
        np.asarray([5.0, 4.0, 3.0]),
        "diag",
        [np.asarray([5.0, 4.0, 3.0])],
    )
    return PSDResult(
        posterior,
        spectrum,
        sample_stats=stats,
        metadata={
            "max_tree_depth": 6,
            "max_tree_depth_by_channel": [5],
        },
        vi=vi,
    )


def test_sampling_diagnostics_for_nuts_and_vi(monkeypatch) -> None:
    import pandas as pd

    monkeypatch.setattr(
        "log_psplines.diagnostics.sampling.azs.summary",
        lambda _: pd.DataFrame(
            {
                "r_hat": [1.01, 1.03],
                "ess_bulk": [100.0, 90.0],
                "ess_tail": [80.0, 60.0],
            }
        ),
    )
    result = _diagnostic_result()
    row = sampling_diagnostics(result)["nuts"][0]
    assert row["factor"] == "0"
    assert row["divergences"] == 2
    assert row["max_treedepth_hits"] == 2
    assert row["step_size"] == pytest.approx(0.2)
    assert row["rhat_max"] == pytest.approx(1.03)
    assert row["ess_bulk_min"] == pytest.approx(90.0)
    assert row["ess_tail_min"] == pytest.approx(60.0)
    assert row["n_draws"] == 4

    vi_row = sampling_diagnostics(result, elbo_window=2)["vi"][0]
    assert vi_row["factor"] == "0"
    assert vi_row["final_elbo"] == pytest.approx(-3.0)
    assert vi_row["elbo_improvement"] == pytest.approx(1.0)
    assert vi_row["n_draws"] == 4


def test_spectrum_diagnostics_for_real_multichannel_truth() -> None:
    from log_psplines.results import PSDResult

    frequency = np.linspace(0.1, 0.5, 5)
    truth = np.tile(np.asarray([[2.0, 0.2], [0.2, 1.5]]), (5, 1, 1))
    spectrum = xr.DataArray(
        np.tile(truth, (1, 3, 1, 1, 1)).astype(complex),
        dims=("chain", "draw", "frequency", "channel", "channel_aux"),
        coords={"frequency": frequency},
    )
    result = PSDResult(posterior=xr.Dataset(), spectrum=spectrum)

    metrics = spectrum_diagnostics(result, truth=truth)
    assert metrics["riae"] == pytest.approx(0.0)
    assert metrics["l2"] == pytest.approx(0.0)
    assert metrics["coverage"] == pytest.approx(1.0)
    assert metrics["channel_riae"] == pytest.approx([0.0, 0.0])
    assert metrics["coherence_mae"] == pytest.approx(0.0)


def test_power_tree_depth_and_truth_comparison(monkeypatch) -> None:
    import pandas as pd

    from log_psplines.results import PSDResult

    monkeypatch.setattr(
        "log_psplines.diagnostics.sampling.azs.summary",
        lambda _: pd.DataFrame(),
    )
    frequency = np.linspace(0.1, 0.5, 5)
    spectrum = xr.DataArray(
        np.ones((1, 4, 2, 5, 1, 1), dtype=complex),
        dims=("chain", "draw", "time", "frequency", "channel", "channel_aux"),
        coords={"time": [0.0, 1.0], "frequency": frequency},
    )
    posterior = xr.Dataset(
        {"weights": (("chain", "draw"), [[0.0, 0.0, 0.0, 0.0]])}
    )
    sample_stats = xr.Dataset(
        {
            "tree_depth": (("chain", "draw"), [[2, 3, 5, 5]]),
            "diverging": (("chain", "draw"), [[0, 0, 0, 0]]),
        }
    )
    result = PSDResult(
        posterior=posterior,
        spectrum=spectrum,
        sample_stats=sample_stats,
        metadata={"max_tree_depth": 5},
    )

    assert sampling_diagnostics(result)["nuts"][0]["max_treedepth_hits"] == 2
    assert (
        spectrum_diagnostics(result, truth=np.ones((2, 5)))["coverage"] == 1.0
    )
