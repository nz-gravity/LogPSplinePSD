import numpy as np
import pandas as pd
import pytest
import xarray as xr

from log_psplines.datatypes.multivar import MultivarFFT
from log_psplines.diagnostics import summary_tables as st
from log_psplines.diagnostics.preprocessing import (
    eig_ratios,
    eigenvalue_separation_diagnostics,
    extract_component_knots,
    ordered_eigvals_hermitian,
    ratio_summary_string,
    save_eigenvalue_separation_plot,
    worst_ratio_frequencies,
)
from log_psplines.pipeline.config import PipelineConfig
from log_psplines.preprocessing.checks import (
    _run_preprocessing_checks,
    _save_preprocessing_plot,
)
from log_psplines.psplines import MultivariateLogPSplines
from log_psplines.psplines.psplines import LogPSplines


def _psd_stack(n: int = 6, p: int = 2) -> np.ndarray:
    out = np.zeros((n, p, p), dtype=np.complex128)
    for idx in range(n):
        out[idx] = np.asarray(
            [[2.0 + 0.1 * idx, 0.9 + 0.05j], [0.9 - 0.05j, 1.8 + 0.1 * idx]]
        )
    return out


def _fft_for_checks() -> MultivarFFT:
    raw_psd = _psd_stack()
    u_re = np.zeros_like(raw_psd.real)
    u_im = np.zeros_like(raw_psd.real)
    for idx, matrix in enumerate(raw_psd):
        chol = np.linalg.cholesky(matrix)
        u_re[idx] = chol.real
        u_im[idx] = chol.imag
    return MultivarFFT(
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


def _model() -> MultivariateLogPSplines:
    def component() -> LogPSplines:
        return LogPSplines.from_knots(
            knots=np.asarray([0.0, 0.5, 1.0]),
            degree=1,
            diffMatrixOrder=1,
            n=6,
            grid_points=np.linspace(0.0, 1.0, 6),
        )

    return MultivariateLogPSplines(
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
    no_raw = MultivarFFT(
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


def _factor_tree() -> xr.DataTree:
    posterior = xr.Dataset(
        {
            "weights_delta_0": xr.DataArray(
                np.ones((1, 4, 2)),
                dims=("chain", "draw", "weights_dim"),
            )
        }
    )
    sample_stats = xr.Dataset(
        {
            "diverging": xr.DataArray([[0, 1, 0, 1]], dims=("chain", "draw")),
            "tree_depth": xr.DataArray([[2, 3, 5, 5]], dims=("chain", "draw")),
            "step_size": xr.DataArray(
                [[0.1, 0.2, 0.2, 0.3]], dims=("chain", "draw")
            ),
        }
    )
    tree = xr.DataTree(
        children={
            "posterior": xr.DataTree(dataset=posterior),
            "sample_stats": xr.DataTree(dataset=sample_stats),
        }
    )
    tree.attrs["max_tree_depth"] = 5
    return tree


def test_summary_table_helpers_and_builders(monkeypatch) -> None:
    monkeypatch.setattr(
        st.azs,
        "summary",
        lambda _: pd.DataFrame(
            {
                "r_hat": [1.01, np.nan, 1.03],
                "ess_bulk": [100.0, 90.0, np.nan],
                "ess_tail": [80.0, 70.0, 60.0],
            }
        ),
    )

    table = st.build_nuts_summary_table({"1": _factor_tree()})
    row = table.iloc[0]
    assert row["factor"] == "1"
    assert row["divergences"] == 2
    assert row["max_treedepth_hits"] == 2
    assert row["step_size"] == pytest.approx(0.2)
    assert row["rhat_max"] == pytest.approx(1.03)
    assert row["ess_bulk_min"] == pytest.approx(90.0)
    assert row["ess_tail_min"] == pytest.approx(60.0)
    assert row["n_draws"] == 4

    assert np.isnan(
        st._summary_reduction(pd.DataFrame({"x": [np.nan]}), "x", "max")
    )
    with pytest.raises(ValueError, match="Unsupported reducer"):
        st._summary_reduction(pd.DataFrame({"x": [1.0]}), "x", "mean")

    mapping_table = st.build_vi_summary_table(
        {
            "losses_per_block": np.asarray([[5.0, 4.0, 3.0], [6.0, 5.5, 5.0]]),
            "pareto_k_per_block": np.asarray([0.4, 0.9]),
            "riae": 0.1,
            "l2": 0.2,
            "coverage": 0.3,
        },
        elbo_window=2,
    )
    assert list(mapping_table["factor"]) == ["0", "1"]
    assert mapping_table.loc[0, "final_elbo"] == pytest.approx(3.0)
    assert mapping_table.loc[
        0, "elbo_improvement_last_window"
    ] == pytest.approx(1.0)
    assert mapping_table.loc[1, "pareto_k_max"] == pytest.approx(0.9)
    assert bool(mapping_table.loc[1, "loo_warning"]) is True
    assert mapping_table.loc[0, "riae"] == pytest.approx(0.1)

    single = st.build_vi_summary_table(
        {"losses": [3.0], "pareto_k": [np.nan]}, elbo_window=5
    )
    assert single.loc[0, "factor"] == "0"
    assert np.isnan(single.loc[0, "elbo_improvement_last_window"])

    with pytest.raises(TypeError, match="Expected"):
        st._split_vi_inputs(object())
