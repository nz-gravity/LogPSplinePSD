import os

import numpy as np
import pytest
import xarray as xr

from log_psplines.arviz_utils.from_arviz import (
    get_multivar_cholesky_params,
    get_multivar_posterior_psd_quantiles,
)
from log_psplines.arviz_utils.to_arviz import (
    _prepare_attributes_and_dims,
    _reconstruct_log_delta_sq,
    _reconstruct_theta_params,
    _subset_weight_samples_for_psd,
)
from log_psplines.datatypes.multivar import EmpiricalPSD, _get_coherence
from log_psplines.samplers.base_sampler import SamplerConfig


class _DummyThetaModel:
    def __init__(self, basis: np.ndarray):
        self.basis = basis


class _DummySplineModel:
    def __init__(self, basis: np.ndarray):
        self._basis = basis
        self.n_theta = 1
        self.theta_pairs = [(1, 0)]

    def get_all_bases_and_penalties(self):
        return [self._basis, self._basis], None

    def get_theta_model(self, param_type: str, j: int, l: int):
        assert param_type in {"re", "im"}
        assert (j, l) == (1, 0)
        return _DummyThetaModel(self._basis)

    def theta_pair_from_index(self, index: int):
        assert index == 0
        return (1, 0)


class _DummyFFTData:
    def __init__(self, n_freq: int, p: int):
        self.N = n_freq
        self.p = p


class _DummyIDATA:
    def __init__(self, posterior: xr.Dataset, spline_model):
        self.posterior = posterior
        self._groups = {"spline_model": spline_model}

    def __getitem__(self, key: str):
        return self._groups[key]


def test_prepare_attributes_filters_nonserialisable(tmp_path):
    psd = np.ones((3, 2, 2), dtype=np.complex128)
    emp = EmpiricalPSD(
        freq=np.array([0.1, 0.2, 0.3]),
        psd=psd,
        coherence=_get_coherence(psd),
        channels=np.array([0, 1]),
    )
    cfg = SamplerConfig(
        outdir=None,
        extra_empirical_psd=[emp],
        extra_empirical_labels=["Welch"],
        extra_empirical_styles=[{"zorder": -10}],
    )

    attributes = {}
    coords = {}
    dims = {}
    samples = {"x": np.zeros((1, 1))}
    sample_stats = {}

    _prepare_attributes_and_dims(
        cfg,
        attributes,
        samples,
        coords,
        dims,
        sample_stats,
        data=None,
        model=None,
    )

    assert "extra_empirical_psd" not in attributes
    assert "extra_empirical_styles" not in attributes

    ds = xr.Dataset(attrs=attributes)
    out = tmp_path / "attrs.nc"
    ds.to_netcdf(out)
    assert os.path.exists(out)


def test_subset_weight_samples_for_psd_flattens_and_caps_draws():
    samples = {
        "weights_delta_0": np.arange(1, 5, dtype=float).reshape(2, 2, 1),
        "weights_theta_re_1_0": np.arange(5, 9, dtype=float).reshape(2, 2, 1),
        "phi_delta_0": np.ones((2, 2), dtype=float),
    }

    subset = _subset_weight_samples_for_psd(samples, 3)

    assert set(subset) == {"weights_delta_0", "weights_theta_re_1_0"}
    assert subset["weights_delta_0"].shape == (3, 1)
    np.testing.assert_allclose(
        np.asarray(subset["weights_delta_0"]).ravel(),
        np.array([1.0, 2.0, 4.0]),
    )


def test_multivar_reconstruction_uses_all_chains():
    basis = np.ones((3, 1), dtype=float)
    spline_model = _DummySplineModel(basis)
    fft_data = _DummyFFTData(n_freq=3, p=2)
    samples = {
        "weights_delta_0": np.arange(1, 5, dtype=float).reshape(2, 2, 1),
        "weights_delta_1": np.arange(11, 15, dtype=float).reshape(2, 2, 1),
        "weights_theta_re_1_0": np.arange(21, 25, dtype=float).reshape(
            2, 2, 1
        ),
        "weights_theta_im_1_0": np.arange(31, 35, dtype=float).reshape(
            2, 2, 1
        ),
    }

    log_delta_sq = _reconstruct_log_delta_sq(samples, spline_model, fft_data)
    theta_re = _reconstruct_theta_params(samples, spline_model, fft_data, "re")
    theta_im = _reconstruct_theta_params(samples, spline_model, fft_data, "im")

    assert log_delta_sq.shape == (4, 3, 2)
    assert theta_re.shape == (4, 3, 1)
    assert theta_im.shape == (4, 3, 1)

    np.testing.assert_allclose(log_delta_sq[:, 0, 0], np.array([1, 2, 3, 4]))
    np.testing.assert_allclose(
        log_delta_sq[:, 0, 1], np.array([11, 12, 13, 14])
    )
    np.testing.assert_allclose(theta_re[:, 0, 0], np.array([21, 22, 23, 24]))
    np.testing.assert_allclose(theta_im[:, 0, 0], np.array([31, 32, 33, 34]))


def test_get_multivar_cholesky_params_reconstructs_from_idata():
    posterior = {
        "weights_delta_0": np.arange(1, 5, dtype=float).reshape(2, 2, 1),
        "weights_delta_1": np.arange(11, 15, dtype=float).reshape(2, 2, 1),
        "weights_theta_re_1_0": np.arange(21, 25, dtype=float).reshape(
            2, 2, 1
        ),
        "weights_theta_im_1_0": np.arange(31, 35, dtype=float).reshape(
            2, 2, 1
        ),
    }
    posterior_ds = xr.Dataset(
        {
            key: xr.DataArray(
                value,
                dims=("chain", "draw", f"{key}_basis_dim"),
            )
            for key, value in posterior.items()
        }
    )
    spline_dataset = xr.Dataset(
        {
            "degree": xr.DataArray(1),
            "diffMatrixOrder": xr.DataArray(1),
            "N": xr.DataArray(3),
            "p": xr.DataArray(2),
            "n_theta": xr.DataArray(1),
            "diag_0_knots": (["diag_0_knots_dim"], np.array([0.0, 1.0])),
            "diag_0_grid_points": (
                ["diag_0_freq"],
                np.array([0.0, 0.5, 1.0]),
            ),
            "diag_0_parametric_model": (
                ["diag_0_freq"],
                np.ones(3, dtype=float),
            ),
            "diag_1_knots": (["diag_1_knots_dim"], np.array([0.0, 1.0])),
            "diag_1_grid_points": (
                ["diag_1_freq"],
                np.array([0.0, 0.5, 1.0]),
            ),
            "diag_1_parametric_model": (
                ["diag_1_freq"],
                np.ones(3, dtype=float),
            ),
            "theta_re_1_0_knots": (
                ["theta_re_1_0_knots_dim"],
                np.array([0.0, 1.0]),
            ),
            "theta_re_1_0_grid_points": (
                ["theta_re_1_0_freq"],
                np.array([0.0, 0.5, 1.0]),
            ),
            "theta_re_1_0_parametric_model": (
                ["theta_re_1_0_freq"],
                np.ones(3, dtype=float),
            ),
            "theta_im_1_0_knots": (
                ["theta_im_1_0_knots_dim"],
                np.array([0.0, 1.0]),
            ),
            "theta_im_1_0_grid_points": (
                ["theta_im_1_0_freq"],
                np.array([0.0, 0.5, 1.0]),
            ),
            "theta_im_1_0_parametric_model": (
                ["theta_im_1_0_freq"],
                np.ones(3, dtype=float),
            ),
        }
    )
    idata = _DummyIDATA(
        posterior=posterior_ds,
        spline_model=xr.DataTree(dataset=spline_dataset),
    )

    params = get_multivar_cholesky_params(idata)

    assert params["log_delta_sq"].shape == (4, 3, 2)
    assert params["theta_re"].shape == (4, 3, 1)
    assert params["theta_im"].shape == (4, 3, 1)
    np.testing.assert_allclose(
        params["log_delta_sq"][:, 0, 0], np.array([1, 2, 3, 4])
    )
    np.testing.assert_allclose(
        params["theta_re"][:, 0, 0], np.array([21, 22, 23, 24])
    )
    np.testing.assert_allclose(
        params["theta_im"][:, 0, 0], np.array([31, 32, 33, 34])
    )


def test_get_multivar_posterior_psd_quantiles_freq_subset_matches_full():
    posterior = {
        "weights_delta_0": np.arange(1, 5, dtype=float).reshape(2, 2, 1),
        "weights_delta_1": np.arange(11, 15, dtype=float).reshape(2, 2, 1),
        "weights_theta_re_1_0": np.arange(21, 25, dtype=float).reshape(
            2, 2, 1
        ),
        "weights_theta_im_1_0": np.arange(31, 35, dtype=float).reshape(
            2, 2, 1
        ),
    }
    posterior_ds = xr.Dataset(
        {
            key: xr.DataArray(
                value,
                dims=("chain", "draw", f"{key}_basis_dim"),
            )
            for key, value in posterior.items()
        }
    )
    periodogram = xr.DataArray(
        np.ones((3, 2, 2), dtype=np.complex128),
        dims=("freq", "channels", "channels2"),
        coords={"freq": np.array([0.1, 0.2, 0.3])},
    )
    spline_dataset = xr.Dataset(
        {
            "degree": xr.DataArray(1),
            "diffMatrixOrder": xr.DataArray(1),
            "N": xr.DataArray(3),
            "p": xr.DataArray(2),
            "n_theta": xr.DataArray(1),
            "diag_0_knots": (["diag_0_knots_dim"], np.array([0.0, 1.0])),
            "diag_0_grid_points": (
                ["diag_0_freq"],
                np.array([0.0, 0.5, 1.0]),
            ),
            "diag_0_parametric_model": (
                ["diag_0_freq"],
                np.ones(3, dtype=float),
            ),
            "diag_1_knots": (["diag_1_knots_dim"], np.array([0.0, 1.0])),
            "diag_1_grid_points": (
                ["diag_1_freq"],
                np.array([0.0, 0.5, 1.0]),
            ),
            "diag_1_parametric_model": (
                ["diag_1_freq"],
                np.ones(3, dtype=float),
            ),
            "theta_re_1_0_knots": (
                ["theta_re_1_0_knots_dim"],
                np.array([0.0, 1.0]),
            ),
            "theta_re_1_0_grid_points": (
                ["theta_re_1_0_freq"],
                np.array([0.0, 0.5, 1.0]),
            ),
            "theta_re_1_0_parametric_model": (
                ["theta_re_1_0_freq"],
                np.ones(3, dtype=float),
            ),
            "theta_im_1_0_knots": (
                ["theta_im_1_0_knots_dim"],
                np.array([0.0, 1.0]),
            ),
            "theta_im_1_0_grid_points": (
                ["theta_im_1_0_freq"],
                np.array([0.0, 0.5, 1.0]),
            ),
            "theta_im_1_0_parametric_model": (
                ["theta_im_1_0_freq"],
                np.ones(3, dtype=float),
            ),
        }
    )
    idata = _DummyIDATA(
        posterior=posterior_ds,
        spline_model=xr.DataTree(dataset=spline_dataset),
    )
    idata._groups["observed_data"] = xr.DataTree(
        dataset=xr.Dataset({"periodogram": periodogram})
    )

    full = get_multivar_posterior_psd_quantiles(
        idata,
        n_keep=3,
        compute_coherence=True,
    )
    subset = get_multivar_posterior_psd_quantiles(
        idata,
        n_keep=3,
        compute_coherence=True,
        freq_idx=np.array([0, 2]),
    )

    assert np.asarray(full["real"]).shape == (3, 3, 2, 2)
    assert np.asarray(full["imag"]).shape == (3, 3, 2, 2)
    assert np.asarray(full["coherence"]).shape == (3, 3, 2, 2)
    np.testing.assert_allclose(
        np.asarray(subset["freq"]), np.asarray(full["freq"])[[0, 2]]
    )
    np.testing.assert_allclose(
        np.asarray(subset["real"]), np.asarray(full["real"])[:, [0, 2], ...]
    )
    np.testing.assert_allclose(
        np.asarray(subset["imag"]), np.asarray(full["imag"])[:, [0, 2], ...]
    )
    np.testing.assert_allclose(
        np.asarray(subset["coherence"]),
        np.asarray(full["coherence"])[:, [0, 2], ...],
    )
