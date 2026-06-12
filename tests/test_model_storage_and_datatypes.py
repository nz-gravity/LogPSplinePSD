import numpy as np
import pytest
import xarray as xr

from log_psplines.datatypes.multivar import (
    EmpiricalPSD,
    MultivarFFT,
    MultivariateTimeseries,
)
from log_psplines.datatypes.multivar_utils import (
    U_to_Y,
    Y_to_S,
    Y_to_U,
    _get_coherence,
    interp_matrix,
    psd_to_cholesky_components,
    u_re_im_to_U,
    wishart_u_to_psd,
)
from log_psplines.psplines import MultivariateLogPSplines
from log_psplines.psplines.multivar_psplines import MultivarComponentKey
from log_psplines.psplines.psplines import LogPSplines, build_spline


def _simple_log_pspline(n: int = 6) -> LogPSplines:
    return LogPSplines.from_knots(
        knots=np.asarray([0.0, 0.4, 0.7, 1.0]),
        degree=2,
        diffMatrixOrder=1,
        n=n,
        grid_points=np.linspace(0.0, 1.0, n),
    )


def _simple_multivar_model() -> MultivariateLogPSplines:
    diag = [_simple_log_pspline(), _simple_log_pspline()]
    return MultivariateLogPSplines(
        degree=2,
        diffMatrixOrder=1,
        N=6,
        p=2,
        diagonal_models=diag,
        offdiag_re_models={(1, 0): _simple_log_pspline()},
        offdiag_im_models={(1, 0): _simple_log_pspline()},
    )


def _simple_fft() -> MultivarFFT:
    u_re = np.zeros((4, 2, 2))
    u_im = np.zeros_like(u_re)
    for idx in range(4):
        u_re[idx] = np.asarray([[1.0 + idx, 0.0], [0.2, 1.5 + idx]])
    raw_psd = U_to_Y(u_re + 1j * u_im)
    return MultivarFFT(
        u_re=u_re,
        u_im=u_im,
        freq=np.linspace(0.1, 0.4, 4),
        N=4,
        p=2,
        Nb=2,
        raw_psd=raw_psd,
        raw_freq=np.linspace(0.1, 0.4, 4),
        channel_stds=np.asarray([2.0, 3.0]),
        scaling_factor=2.0,
    )


def test_log_pspline_storage_round_trip_and_validation(tmp_path) -> None:
    model = _simple_log_pspline()
    payload, coords = model.to_storage_payload(
        prefix="diag_0", include_linear_operators=True
    )
    ds = xr.Dataset(
        {
            "degree": xr.DataArray(model.degree),
            "diffMatrixOrder": xr.DataArray(model.diffMatrixOrder),
            "N": xr.DataArray(model.n),
            **{
                key: xr.DataArray(value, dims=dims)
                for key, (dims, value) in payload.items()
            },
        },
        coords=coords,
    )

    loaded = LogPSplines.from_storage_dataset(ds, prefix="diag_0")
    np.testing.assert_allclose(loaded.knots, model.knots)
    np.testing.assert_allclose(loaded.grid_points, model.grid_points)
    np.testing.assert_allclose(
        np.asarray(loaded.basis), np.asarray(model.basis)
    )
    np.testing.assert_allclose(
        np.asarray(loaded.penalty_matrix), np.asarray(model.penalty_matrix)
    )

    values = model(np.ones(model.n_basis))
    np.testing.assert_allclose(
        values, build_spline(model.basis, np.ones(model.n_basis))
    )

    model.plot_basis(outdir=str(tmp_path))
    assert (tmp_path / "basis_plot.png").exists()

    with pytest.raises(KeyError, match="Missing required spline key"):
        LogPSplines.from_storage_dataset({})
    with pytest.raises(ValueError, match="weights length"):
        LogPSplines(
            degree=1,
            diffMatrixOrder=1,
            n=4,
            knots=np.asarray([0.0, 1.0]),
            weights=np.ones(99),
        )
    with pytest.raises(ValueError, match="log_target"):
        LogPSplines.from_knots(
            knots=np.asarray([0.0, 1.0]),
            degree=1,
            diffMatrixOrder=1,
            n=4,
            log_target=np.ones(3),
        )


def test_multivar_fft_timeseries_and_conversion_helpers() -> None:
    fft = _simple_fft()
    assert repr(fft) == "MultivarFFT(N=4, p=2)"
    np.testing.assert_allclose(fft.Y, U_to_Y(fft.U))

    masked = fft.apply_mask(np.asarray([True, False, True, False]))
    assert masked.N == 2
    np.testing.assert_allclose(masked.freq, [0.1, 0.3])
    same = fft.exclude_frequency_bands([])
    assert same is fft
    cut = fft.cut(0.15, 0.35)
    np.testing.assert_allclose(cut.freq, [0.2, 0.3])

    empirical = fft.empirical_psd
    assert empirical.psd.shape == (4, 2, 2)
    assert np.all((empirical.coherence >= 0.0) & (empirical.coherence <= 1.0))

    ts = MultivariateTimeseries(np.arange(8.0), t=np.arange(8.0) / 4.0)
    assert ts.p == 1
    assert ts.fs == pytest.approx(4.0)
    std = ts.standardise_for_psd()
    assert std.original_stds.shape == (1,)
    wishart = std.to_wishart_stats(Nb=2, window="hann", detrend=False)
    assert wishart.Nb == 2
    emp = std.get_empirical_psd(nperseg=4)
    assert isinstance(emp, EmpiricalPSD)
    assert emp.freq.ndim == 1

    with pytest.raises(ValueError, match="mask"):
        fft.apply_mask(np.asarray([True, False]))
    with pytest.raises(ValueError, match="removed all"):
        fft.apply_mask(np.zeros(4, dtype=bool))
    with pytest.raises(ValueError, match="Invalid frequency bounds"):
        fft.cut(0.4, 0.1)
    with pytest.raises(ValueError, match="duration"):
        MultivarFFT(
            u_re=np.zeros((1, 1, 1)),
            u_im=np.zeros((1, 1, 1)),
            freq=np.asarray([0.1]),
            N=1,
            p=1,
            duration=0.0,
        )
    with pytest.raises(ValueError, match="same length"):
        MultivariateTimeseries(np.ones((4, 2)), t=np.arange(3.0))
    with pytest.raises(ValueError, match="NaN"):
        MultivariateTimeseries(np.asarray([1.0, np.nan]))
    with pytest.raises(ValueError, match="divisible"):
        MultivariateTimeseries(np.ones((7, 2))).to_wishart_stats(Nb=3)


def test_multivar_fft_validation_and_wishart_edge_cases() -> None:
    base = dict(
        u_re=np.zeros((2, 2, 2)),
        u_im=np.zeros((2, 2, 2)),
        freq=np.asarray([0.1, 0.2]),
        N=2,
        p=2,
    )
    with pytest.raises(ValueError, match="u_re"):
        MultivarFFT(**{**base, "u_re": np.zeros((1, 2, 2))})
    with pytest.raises(ValueError, match="u_im"):
        MultivarFFT(**{**base, "u_im": np.zeros((1, 2, 2))})
    with pytest.raises(ValueError, match="freq"):
        MultivarFFT(**{**base, "freq": np.asarray([0.1])})
    with pytest.raises(ValueError, match="raw_psd"):
        MultivarFFT(**{**base, "raw_psd": np.zeros((1, 2, 2))})
    with pytest.raises(ValueError, match="raw_freq"):
        MultivarFFT(**{**base, "raw_freq": np.asarray([0.1])})
    with pytest.raises(TypeError, match="Nb"):
        MultivarFFT(**{**base, "Nb": True})
    with pytest.raises(ValueError, match="Nb"):
        MultivarFFT(**{**base, "Nb": 0})
    with pytest.raises(TypeError, match="Nh"):
        MultivarFFT(**{**base, "Nh": 1.5})
    with pytest.raises(ValueError, match="Nh"):
        MultivarFFT(**{**base, "Nh": 0})
    with pytest.raises(ValueError, match="enbw"):
        MultivarFFT(**{**base, "enbw": np.nan})
    with pytest.raises(ValueError, match="channel_stds"):
        MultivarFFT(**{**base, "channel_stds": np.ones(3)})

    data = np.arange(12.0).reshape(6, 2)
    with pytest.raises(TypeError, match="Nb"):
        MultivarFFT.compute_wishart(data, fs=1.0, Nb=True)
    with pytest.raises(ValueError, match="positive"):
        MultivarFFT.compute_wishart(data, fs=1.0, Nb=0)
    with pytest.raises(ValueError, match="divisible"):
        MultivarFFT.compute_wishart(data, fs=1.0, Nb=4)
    with pytest.raises(ValueError, match="Block length"):
        MultivarFFT.compute_wishart(np.ones((3, 3)), fs=1.0, Nb=1)
    with pytest.raises(ValueError, match="detrend"):
        MultivarFFT.compute_wishart(data, fs=1.0, Nb=1, detrend="bad")

    floored = MultivarFFT.compute_wishart(
        np.column_stack([np.arange(16.0), np.arange(16.0)]),
        fs=4.0,
        Nb=1,
        detrend=False,
        wishart_floor_fraction=1e-6,
    )
    assert floored.raw_psd is not None
    assert np.all(np.linalg.eigvalsh(floored.raw_psd).real >= -1e-10)

    long = np.column_stack(
        [np.sin(np.arange(600.0)), np.cos(np.arange(600.0))]
    )
    emp = EmpiricalPSD.from_timeseries_data(long, fs=10.0)
    assert emp.freq.size > 0
    assert emp.psd.shape[1:] == (2, 2)

    ts = MultivariateTimeseries(np.arange(8.0), t=np.arange(8.0))
    standardized = ts.standardise()
    assert standardized.std.shape == (1,)
    csd = standardized.to_cross_spectral_density(fmin=0.1, fmax=0.4)
    assert csd.N > 0


def test_multivar_utils_interpolation_scaling_and_cholesky_errors() -> None:
    freq_src = np.asarray([0.3, 0.1, 0.1])
    freq_tgt = np.asarray([0.1, 0.2, 0.3])
    values = np.asarray([3.0 + 1j, 1.0 + 2j, 2.0 + 3j])
    interp = interp_matrix(freq_src, values[:, None, None], freq_tgt)
    assert interp.shape == (3, 1, 1)
    assert np.iscomplexobj(interp)

    U = np.asarray([[[1.0, 0.0], [0.5, 1.0]]], dtype=np.complex128)
    Y = U_to_Y(U)
    np.testing.assert_allclose(
        Y_to_U(Y) @ np.swapaxes(Y_to_U(Y).conj(), -1, -2), Y
    )
    np.testing.assert_allclose(
        wishart_u_to_psd(U, Nb=2, Nh=2), Y_to_S(Y, 2, Nh=2)
    )

    psd = np.asarray([[[2.0, 0.2 + 0.1j], [0.2 - 0.1j, 1.5]]])
    log_delta, theta = psd_to_cholesky_components(psd)
    assert log_delta.shape == (1, 2)
    assert theta.shape == (1, 2, 2)
    coh = _get_coherence(psd)
    assert coh[0, 0, 0] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="matching shapes"):
        u_re_im_to_U(np.zeros((1, 1, 1)), np.zeros((2, 1, 1)))
    with pytest.raises(ValueError, match="Hermitian"):
        Y_to_U(np.asarray([[[1.0, 2.0], [3.0, 1.0]]]))
    with pytest.raises(ValueError, match="duration"):
        Y_to_S(Y, 1, duration=0.0)
    with pytest.raises(TypeError, match="positive integer"):
        Y_to_S(Y, True)
    with pytest.raises(ValueError, match="square"):
        _get_coherence(np.ones((2, 2, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        psd_to_cholesky_components(psd, cholesky_jitter=-1.0)


def test_multivariate_model_registry_design_weights_and_psd_reconstruction() -> (
    None
):
    model = _simple_multivar_model()
    assert model.n_theta == 1
    assert model.theta_pairs == [(1, 0)]
    assert model.total_components == 4
    assert model.theta_index(1, 0) == 0
    assert model.theta_pair_from_index(0) == (1, 0)
    assert len(model.iter_component_specs()) == 4
    assert model.n_knots == 4
    assert model.n_basis == model.diagonal_models[0].n_basis

    bases, penalties = model.get_all_bases_and_penalties()
    assert len(bases) == len(penalties) == 4

    design_psd = np.zeros((model.N, model.p, model.p), dtype=np.complex128)
    for idx in range(model.N):
        design_psd[idx] = np.asarray(
            [[2.0 + 0.1 * idx, 0.1 + 0.05j], [0.1 - 0.05j, 1.5 + 0.1 * idx]]
        )
    weights = model.compute_design_weights(design_psd)
    assert set(weights) == {
        "delta_0",
        "delta_1",
        "theta_re_1_0",
        "theta_im_1_0",
    }

    n_draws = 3
    n_basis = model.diagonal_models[0].n_basis
    log_delta_sq = np.zeros((n_draws, model.N, model.p))
    theta_re = np.zeros((n_draws, model.N, model.n_theta))
    theta_im = np.zeros_like(theta_re)
    psd = model.reconstruct_psd_matrix(log_delta_sq, theta_re, theta_im)
    assert psd.shape == (n_draws, model.N, model.p, model.p)
    np.testing.assert_allclose(psd, np.swapaxes(psd.conj(), -1, -2))

    real_q, imag_q, coh_q = model.compute_psd_quantiles(
        log_delta_sq,
        theta_re,
        theta_im,
        percentiles=(5.0, 50.0, 95.0),
        compute_coherence=True,
    )
    assert real_q.shape == (3, model.N, model.p, model.p)
    assert imag_q.shape == real_q.shape
    assert coh_q.shape == real_q.shape
    assert np.all((coh_q >= 0.0) & (coh_q <= 1.0))

    with pytest.raises(ValueError, match="Unknown theta part"):
        model.theta_key("bad", 1, 0)
    with pytest.raises(IndexError, match="out of range"):
        model.theta_pair_from_index(99)
    with pytest.raises(ValueError, match="Invalid theta pair"):
        model.theta_index(0, 1)
    with pytest.raises(ValueError, match="design_psd"):
        model.compute_design_weights(np.eye(2)[None])
    with pytest.raises(ValueError, match="j must be"):
        MultivarComponentKey("delta", -1)
    with pytest.raises(ValueError, match="delta components"):
        MultivarComponentKey("delta", 0, l=0)
    with pytest.raises(ValueError, match="Unknown component"):
        MultivarComponentKey("bad", 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="theta components require l"):
        MultivarComponentKey("theta", 1)

    incomplete = [_simple_log_pspline(), _simple_log_pspline()]
    with pytest.raises(ValueError, match="incomplete"):
        MultivariateLogPSplines(
            degree=2,
            diffMatrixOrder=1,
            N=6,
            p=2,
            diagonal_models=incomplete,
        )


def test_log_pspline_validation_branches() -> None:
    with pytest.raises(ValueError, match="Degree"):
        LogPSplines(
            degree=1, diffMatrixOrder=2, n=4, knots=np.asarray([0.0, 1.0])
        )
    with pytest.raises(ValueError, match="between 0 and 5"):
        LogPSplines(
            degree=6, diffMatrixOrder=1, n=4, knots=np.asarray([0.0, 1.0])
        )
    with pytest.raises(ValueError, match="diffMatrixOrder"):
        LogPSplines(
            degree=5, diffMatrixOrder=5, n=4, knots=np.asarray([0.0, 1.0])
        )
    with pytest.raises(ValueError, match="Number of knots"):
        LogPSplines(
            degree=3, diffMatrixOrder=1, n=4, knots=np.asarray([0.0, 1.0])
        )
    with pytest.raises(ValueError, match="knots must be 1-D"):
        LogPSplines(
            degree=1, diffMatrixOrder=1, n=4, knots=np.asarray([[0.0, 1.0]])
        )
    with pytest.raises(ValueError, match="non-empty"):
        LogPSplines(degree=0, diffMatrixOrder=0, n=4, knots=np.asarray([]))
    with pytest.raises(ValueError, match="finite"):
        LogPSplines(
            degree=1, diffMatrixOrder=1, n=4, knots=np.asarray([0.0, np.nan])
        )
    with pytest.raises(ValueError, match="sorted"):
        LogPSplines(
            degree=1, diffMatrixOrder=1, n=4, knots=np.asarray([1.0, 0.0])
        )
    with pytest.raises(ValueError, match="grid_points must be"):
        LogPSplines(
            degree=1,
            diffMatrixOrder=1,
            n=4,
            knots=np.asarray([0.0, 1.0]),
            grid_points=np.ones((2, 2)),
        )
    with pytest.raises(ValueError, match="grid_points length"):
        LogPSplines(
            degree=1,
            diffMatrixOrder=1,
            n=4,
            knots=np.asarray([0.0, 1.0]),
            grid_points=np.ones(3),
        )
    with pytest.raises(ValueError, match="grid_points must be finite"):
        LogPSplines(
            degree=1,
            diffMatrixOrder=1,
            n=4,
            knots=np.asarray([0.0, 1.0]),
            grid_points=np.asarray([0.0, 0.5, np.nan, 1.0]),
        )
    with pytest.raises(ValueError, match="grid_points must be sorted"):
        LogPSplines(
            degree=1,
            diffMatrixOrder=1,
            n=4,
            knots=np.asarray([0.0, 1.0]),
            grid_points=np.asarray([0.0, 0.5, 0.4, 1.0]),
        )
    with pytest.raises(ValueError, match="basis must be 2-D"):
        LogPSplines(
            degree=1,
            diffMatrixOrder=1,
            n=4,
            knots=np.asarray([0.0, 1.0]),
            basis=np.ones(4),
            penalty_matrix=np.eye(2),
        )
    with pytest.raises(ValueError, match="basis first dimension"):
        LogPSplines(
            degree=1,
            diffMatrixOrder=1,
            n=4,
            knots=np.asarray([0.0, 1.0]),
            basis=np.ones((3, 2)),
            penalty_matrix=np.eye(2),
        )
    with pytest.raises(ValueError, match="penalty_matrix must be 2-D"):
        LogPSplines(
            degree=1,
            diffMatrixOrder=1,
            n=4,
            knots=np.asarray([0.0, 1.0]),
            basis=np.ones((4, 2)),
            penalty_matrix=np.ones(2),
        )
    with pytest.raises(ValueError, match="penalty_matrix must be square"):
        LogPSplines(
            degree=1,
            diffMatrixOrder=1,
            n=4,
            knots=np.asarray([0.0, 1.0]),
            basis=np.ones((4, 2)),
            penalty_matrix=np.ones((2, 3)),
        )
    with pytest.raises(ValueError, match="penalty_matrix dimension"):
        LogPSplines(
            degree=1,
            diffMatrixOrder=1,
            n=4,
            knots=np.asarray([0.0, 1.0]),
            basis=np.ones((4, 2)),
            penalty_matrix=np.eye(3),
        )
    with pytest.raises(ValueError, match="weights must be 1-D"):
        LogPSplines(
            degree=1,
            diffMatrixOrder=1,
            n=4,
            knots=np.asarray([0.0, 1.0]),
            basis=np.ones((4, 2)),
            penalty_matrix=np.eye(2),
            weights=np.ones((1, 2)),
        )


def test_multivar_factory_with_analytical_guides_and_validation() -> None:
    fft = _simple_fft()
    freq_ana = np.linspace(float(fft.freq[0]), float(fft.freq[-1]), fft.N)
    analytical = np.zeros((fft.N, fft.p, fft.p), dtype=np.complex128)
    for idx in range(fft.N):
        analytical[idx] = np.asarray(
            [[2.0 + 0.1 * idx, 0.1 + 0.02j], [0.1 - 0.02j, 1.5 + 0.1 * idx]]
        )

    model = MultivariateLogPSplines.from_multivar_fft(
        fft,
        n_knots={"delta": 3, "theta_re": 4, "theta_im": 5},
        degree=1,
        diffMatrixOrder=1,
        knot_kwargs={"method": "uniform", "scoring": "legacy"},
        analytical_psd=(freq_ana, analytical),
    )
    assert model.p == fft.p
    assert model.total_components == 4
    assert isinstance(model.n_knots, list)

    with pytest.raises(ValueError, match="analytical_psd"):
        MultivariateLogPSplines.from_multivar_fft(
            fft,
            n_knots=3,
            degree=1,
            diffMatrixOrder=1,
            analytical_psd=np.ones((fft.N, fft.p, fft.p + 1)),
        )
    with pytest.raises(ValueError, match="Unsupported"):
        MultivariateLogPSplines.from_multivar_fft(
            fft,
            n_knots=3,
            degree=1,
            diffMatrixOrder=1,
            knot_kwargs={"method": "bad"},
        )

    one_channel = MultivarFFT(
        u_re=np.ones((3, 1, 1)),
        u_im=np.zeros((3, 1, 1)),
        freq=np.asarray([0.1, 0.2, 0.3]),
        N=3,
        p=1,
    )
    model_p1 = MultivariateLogPSplines.from_multivar_fft(
        one_channel,
        n_knots=3,
        degree=1,
        diffMatrixOrder=1,
    )
    assert model_p1.n_theta == 0
    assert model_p1.total_components == 1
