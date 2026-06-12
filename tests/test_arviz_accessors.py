from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

from log_psplines.arviz_utils._datatree import (
    _netcdf_safe_attr,
    _sanitize_attrs_for_netcdf,
    require_dataset,
)
from log_psplines.arviz_utils.from_arviz import (
    _compute_coherence_from_spectral_density,
    _normalize_chain_draw_array,
    get_multivar_ci_summary,
    get_multivar_posterior_psd_quantiles,
    get_multivar_prior_psd_quantiles,
    get_multivar_spline_model,
    get_posterior_psd,
    get_psd_dataset,
    get_sample_dataset,
    get_weights,
)
from log_psplines.arviz_utils.to_arviz import (
    _compute_prior_predictive_multivar,
    _flatten_posterior_draws,
    _pack_spline_model_multivar,
    _reconstruct_log_delta_sq,
    _reconstruct_theta_params,
    _select_evenly_spaced_indices,
)
from log_psplines.psplines import MultivariateLogPSplines
from log_psplines.psplines.psplines import LogPSplines


def _simple_log_pspline(n: int = 5) -> LogPSplines:
    return LogPSplines.from_knots(
        knots=np.asarray([0.0, 0.5, 1.0]),
        degree=1,
        diffMatrixOrder=1,
        n=n,
        grid_points=np.linspace(0.0, 1.0, n),
    )


def _simple_multivar_model() -> MultivariateLogPSplines:
    return MultivariateLogPSplines(
        degree=1,
        diffMatrixOrder=1,
        N=5,
        p=2,
        diagonal_models=[_simple_log_pspline(), _simple_log_pspline()],
        offdiag_re_models={(1, 0): _simple_log_pspline()},
        offdiag_im_models={(1, 0): _simple_log_pspline()},
    )


def _posterior_dataset(
    model: MultivariateLogPSplines, chains: int = 2, draws: int = 3
) -> xr.Dataset:
    coords = {"chain": np.arange(chains), "draw": np.arange(draws)}
    data = {}
    for key in model.expected_component_order:
        n_basis = model.get_component_spec(key).model.n_basis
        name = f"weights_{key.name}"
        dim = f"{name}_dim_0"
        coords[dim] = np.arange(n_basis)
        values = np.zeros((chains, draws, n_basis), dtype=float)
        values[..., 0] = 0.05
        data[name] = xr.DataArray(values, dims=("chain", "draw", dim))
    return xr.Dataset(data, coords=coords)


def _idata() -> xr.DataTree:
    model = _simple_multivar_model()
    freq = np.linspace(0.1, 0.5, model.N)
    periodogram = xr.DataArray(
        np.ones((model.N, model.p, model.p), dtype=np.complex128),
        dims=("freq", "channel", "channel_aux"),
        coords={
            "freq": freq,
            "channel": np.arange(model.p),
            "channel_aux": np.arange(model.p),
        },
    )
    truth = np.zeros((model.N, model.p, model.p), dtype=np.complex128)
    for idx in range(model.N):
        truth[idx] = np.asarray(
            [[1.0 + idx * 0.1, 0.05 + 0.02j], [0.05 - 0.02j, 1.2 + idx * 0.1]]
        )
    tree = xr.DataTree(
        children={
            "spline_model": xr.DataTree(
                dataset=_pack_spline_model_multivar(model)
            ),
            "posterior": xr.DataTree(dataset=_posterior_dataset(model)),
            "vi_posterior": xr.DataTree(
                dataset=_posterior_dataset(model, chains=1, draws=2)
            ),
            "observed_data": xr.DataTree(
                dataset=xr.Dataset({"periodogram": periodogram})
            ),
            "truth_psd": xr.DataTree(
                dataset=xr.Dataset(
                    {
                        "spectral_density": xr.DataArray(
                            truth,
                            dims=("frequency", "channel", "channel_aux"),
                            coords={
                                "frequency": freq,
                                "channel": np.arange(model.p),
                                "channel_aux": np.arange(model.p),
                            },
                        )
                    }
                )
            ),
        }
    )
    tree.attrs.update(
        {
            "posterior_psd_max_draws": 4,
            "vi_psd_max_draws": None,
            "channel_stds": np.asarray([2.0, 3.0]),
            "scaling_factor": 1.0,
            "tau": 0.5,
            "design_psd": truth,
            "alpha_phi": 2.0,
            "beta_phi": 1.0,
            "alpha_delta": 2.0,
            "beta_delta": 1.0,
        }
    )
    return tree


def test_datatree_helpers_sanitize_attrs_and_errors() -> None:
    tree = xr.DataTree(
        dataset=xr.Dataset(attrs={"drop": None, "complex": 1.0 + 2.0j}),
        children={
            "posterior": xr.DataTree(
                dataset=xr.Dataset({"x": xr.DataArray([1])})
            )
        },
    )
    tree.attrs.update(
        {"flag": True, "arr": np.asarray([True, False]), "obj": {"a": 1}}
    )

    assert require_dataset(tree, "posterior")["x"].item() == 1
    assert _netcdf_safe_attr(True) == 1
    assert _netcdf_safe_attr(np.asarray([True, False])).dtype == np.int8
    assert _netcdf_safe_attr(1.0 + 2.0j) == str(1.0 + 2.0j)
    safe = _sanitize_attrs_for_netcdf(tree)
    assert "drop" not in safe.dataset.attrs
    assert safe.attrs["flag"] == 1

    with pytest.raises(KeyError, match="missing"):
        require_dataset(tree, "missing")


def test_arviz_sample_and_psd_accessors() -> None:
    idata = _idata()
    assert get_sample_dataset(idata, "primary").identical(
        get_sample_dataset(idata, "posterior")
    )
    assert get_sample_dataset(idata, "best").identical(
        get_sample_dataset(idata, "posterior")
    )
    assert get_sample_dataset(idata, "vi").sizes["draw"] == 2

    model = get_multivar_spline_model(idata)
    assert model.p == 2
    assert model.N == 5

    psd_ds = get_psd_dataset(idata, "posterior")
    assert psd_ds["spectral_density"].dims == (
        "chain",
        "draw",
        "channel",
        "channel_aux",
        "frequency",
    )
    assert psd_ds.sizes["frequency"] == 5
    assert np.all(
        (psd_ds["coherence"].values >= 0.0)
        & (psd_ds["coherence"].values <= 1.0)
    )

    quantiles = get_multivar_posterior_psd_quantiles(
        idata,
        n_keep=3,
        percentiles=(10.0, 50.0, 90.0),
        freq_idx=[0, 2, 4],
    )
    assert quantiles["spectral_density"].shape == (3, 3, 2, 2)
    assert quantiles["coherence"].shape == (3, 3, 2, 2)

    summary = get_multivar_ci_summary(idata)
    assert summary["psd_real_q50"].shape == (5, 2, 2)
    assert summary["true_psd_real"].shape == (5, 2, 2)

    freqs, median, lower, upper = get_posterior_psd(idata)
    assert freqs.shape == median.shape == lower.shape == upper.shape

    weights = get_weights(idata, thin=2)
    assert weights.ndim == 2

    with pytest.raises(ValueError, match="Unsupported PSD source"):
        get_psd_dataset(idata, "bad")  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="prior group"):
        get_sample_dataset(xr.DataTree(), "prior")
    with pytest.raises(KeyError, match="any supported"):
        get_sample_dataset(xr.DataTree(), "best")


def test_reconstruction_helpers_and_prior_predictive() -> None:
    idata = _idata()
    model = get_multivar_spline_model(idata)
    posterior = get_sample_dataset(idata, "posterior")
    flat = {
        name: _flatten_posterior_draws(var.values)
        for name, var in posterior.data_vars.items()
    }
    fft_stub = SimpleNamespace(
        N=model.N, p=model.p, freq=np.linspace(0.1, 0.5, model.N)
    )

    log_delta = _reconstruct_log_delta_sq(flat, model, fft_stub)
    theta_re = _reconstruct_theta_params(flat, model, fft_stub, "re")
    theta_im = _reconstruct_theta_params(flat, model, fft_stub, "im")
    assert log_delta.shape == (6, model.N, model.p)
    assert theta_re.shape == theta_im.shape == (6, model.N, model.n_theta)

    fallback = {
        "weights_delta_0": flat["weights_delta_0"],
        "weights_theta_re": flat["weights_theta_re_1_0"],
    }
    tiled = _reconstruct_theta_params(fallback, model, fft_stub, "re")
    assert tiled.shape == (6, model.N, model.n_theta)

    assert _select_evenly_spaced_indices(3, 5) is None
    np.testing.assert_array_equal(
        _select_evenly_spaced_indices(10, 4), [0, 3, 6, 9]
    )
    np.testing.assert_array_equal(
        _normalize_chain_draw_array(np.asarray(1.0)).shape, [1, 1]
    )
    np.testing.assert_array_equal(
        _normalize_chain_draw_array(np.ones(3)).shape, [1, 3]
    )
    np.testing.assert_array_equal(
        _normalize_chain_draw_array(np.ones((2, 4))).shape, [1, 2, 4]
    )

    spectral_density = np.ones((1, 2, 2, 2, 3), dtype=np.complex128)
    coherence = _compute_coherence_from_spectral_density(spectral_density)
    assert coherence.shape == spectral_density.shape
    assert np.all(coherence[:, :, 0, 0, :] == 1.0)

    prior = get_multivar_prior_psd_quantiles(idata, n_prior_draws=3, seed=1)
    assert prior["spectral_density"].shape == (3, model.N, model.p, model.p)
    assert prior["coherence"] is None

    config = SimpleNamespace(
        tau=0.5,
        design_psd=(
            np.linspace(0.1, 0.5, model.N),
            np.asarray(idata.attrs["design_psd"]),
        ),
        channel_stds=np.asarray([2.0, 3.0]),
        alpha_phi=2.0,
        beta_phi=1.0,
        alpha_delta=2.0,
        beta_delta=1.0,
    )
    real_q, imag_q = _compute_prior_predictive_multivar(
        model,
        fft_stub,
        config,
        n_prior_draws=2,
        seed=2,
    )
    assert real_q.shape == imag_q.shape == (3, model.N, model.p, model.p)
