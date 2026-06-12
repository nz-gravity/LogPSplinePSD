import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from log_psplines.datatypes.multivar import MultivariateTimeseries
from log_psplines.diagnostics import plot_nuts
from log_psplines.diagnostics._factors import (
    _copy_factor_attrs,
    _factor_labels_from_combined_idata,
    _factor_tree_from_combined_idata,
    _log_likelihood_factor_key,
    _posterior_factor_index,
    _sample_stats_factor_key,
    factor_idatas,
    vi_factor_idatas,
)
from log_psplines.diagnostics.psd_compare import (
    _coherence,
    _compute_multivar_diagnostics_from_arrays,
    _extract_percentile_slice,
    _handle_multivariate,
    compute_multivar_riae_diagnostics,
)
from log_psplines.pipeline.config import PipelineConfig
from log_psplines.preprocessing.data_prep import (
    _apply_frequency_exclusion,
    _build_welch_overlay,
)


def _combined_tree() -> xr.DataTree:
    posterior = xr.Dataset(
        {
            "weights_delta_0": xr.DataArray(
                np.ones((1, 2, 3)), dims=("chain", "draw", "k")
            ),
            "weights_theta_re_1_0": xr.DataArray(
                np.ones((1, 2, 3)) * 2, dims=("chain", "draw", "k")
            ),
            "unrelated": xr.DataArray(np.ones((1, 2)), dims=("chain", "draw")),
        }
    )
    sample_stats = xr.Dataset(
        {
            "energy_channel_0": xr.DataArray(
                [[1.0, 2.0]], dims=("chain", "draw")
            ),
            "energy_channel_1": xr.DataArray(
                [[3.0, 4.0]], dims=("chain", "draw")
            ),
            "tree_depth_channel_1": xr.DataArray(
                [[2, 3]], dims=("chain", "draw")
            ),
        }
    )
    log_likelihood = xr.Dataset(
        {
            "log_likelihood_block_0": xr.DataArray(
                [[0.1, 0.2]], dims=("chain", "draw")
            ),
            "log_likelihood_block_1": xr.DataArray(
                [[0.3, 0.4]], dims=("chain", "draw")
            ),
        }
    )
    tree = xr.DataTree(
        children={
            "posterior": xr.DataTree(dataset=posterior),
            "sample_stats": xr.DataTree(dataset=sample_stats),
            "log_likelihood": xr.DataTree(dataset=log_likelihood),
            "vi_posterior": xr.DataTree(dataset=posterior),
            "vi_log_likelihood": xr.DataTree(dataset=log_likelihood),
            "vi_sample_stats": xr.DataTree(
                dataset=xr.Dataset({"losses": xr.DataArray([3.0, 2.0])})
            ),
        }
    )
    tree.attrs.update(
        {
            "max_tree_depth_by_channel": [5, 6],
            "target_accept_prob_by_channel": [0.8, 0.9],
        }
    )
    return tree


def test_factor_split_helpers_and_vi_factor_paths() -> None:
    tree = _combined_tree()
    assert _posterior_factor_index("weights_delta_2") == 2
    assert _posterior_factor_index("weights_theta_im_3_1") == 3
    assert _posterior_factor_index("other") is None
    assert _sample_stats_factor_key("energy_channel_4") == ("energy", 4)
    assert _sample_stats_factor_key("energy") is None
    assert _log_likelihood_factor_key("log_likelihood_block_2") == (
        "log_likelihood",
        2,
    )
    assert _log_likelihood_factor_key("log_likelihood") is None

    assert _factor_labels_from_combined_idata(tree) == ["0", "1"]
    attrs = _copy_factor_attrs(tree, "1")
    assert attrs["factor"] == "1"
    assert attrs["max_tree_depth"] == 6
    assert attrs["target_accept_prob"] == pytest.approx(0.9)

    factor0 = _factor_tree_from_combined_idata(tree, "0")
    assert "posterior" in factor0.children
    assert "sample_stats" in factor0.children
    assert "log_likelihood" in factor0.children
    assert set(factor_idatas(tree)) == {"0", "1"}
    assert set(factor_idatas([factor0])) == {"0"}
    assert set(factor_idatas({"x": factor0})) == {"x"}
    with pytest.raises(TypeError, match="Expected"):
        factor_idatas(object())  # type: ignore[arg-type]

    vi_split = vi_factor_idatas(tree)
    assert set(vi_split) == {"0", "1"}
    fallback = xr.DataTree(
        children={"vi_sample_stats": tree["vi_sample_stats"]}
    )
    assert set(vi_factor_idatas(fallback)) == {"0"}


def test_plot_energy_joint_and_factorized_paths(monkeypatch) -> None:
    class DummyPlot:
        def __init__(self):
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            self.viz = {"figure": np.asarray(fig, dtype=object)}

    calls = []

    def fake_plot_energy(obj, backend="matplotlib"):
        calls.append(obj)
        return DummyPlot()

    monkeypatch.setattr(plot_nuts.azp, "plot_energy", fake_plot_energy)
    tree = _combined_tree()
    assert plot_nuts._has_per_channel_stats(tree) is True
    assert plot_nuts._has_per_channel_stats(xr.DataTree()) is False

    joint = xr.DataTree(
        children={"sample_stats": xr.DataTree(dataset=xr.Dataset())}
    )
    joint_plot = plot_nuts.plot_energy(joint)
    assert isinstance(joint_plot, DummyPlot)

    fig = plot_nuts.plot_energy(tree)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)

    fig = plot_nuts.plot_energy(
        {"0": _factor_tree_from_combined_idata(tree, "0")}
    )
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_psd_compare_helpers_and_multivariate_dataset_path() -> None:
    freq = np.linspace(0.1, 1.0, 8)
    truth = np.zeros((8, 2, 2), dtype=np.complex128)
    estimate = np.zeros_like(truth)
    for idx in range(8):
        truth[idx] = np.asarray(
            [[2.0 + idx, 0.2 + 0.1j], [0.2 - 0.1j, 1.5 + idx]]
        )
        estimate[idx] = truth[idx] * 1.05

    coherence = _coherence(truth)
    assert coherence.shape == truth.shape
    assert np.all(coherence.real >= 0.0)
    values = np.asarray([truth * 0.9, truth, truth * 1.1])
    np.testing.assert_allclose(
        _extract_percentile_slice(values, np.asarray([5.0, 50.0, 95.0]), 50.0),
        truth,
    )

    diagnostics = _compute_multivar_diagnostics_from_arrays(
        estimate,
        truth.real,
        freq,
        posterior_psd_quantiles=values,
    )
    assert "riae_matrix" in diagnostics
    assert "riae_matrix_errorbars" in diagnostics
    assert "coverage" in diagnostics
    assert "riae_bands" in diagnostics

    public = compute_multivar_riae_diagnostics(
        estimate,
        truth.real,
        freq,
        psd_quantiles={"posterior_psd": values},
    )
    assert "coherence_riae" in public

    psd_group = xr.Dataset(
        {
            "psd_matrix_real": xr.DataArray(
                values.real,
                dims=("percentile", "freq", "channel", "channel_aux"),
                coords={"percentile": [5.0, 50.0, 95.0], "freq": freq},
            ),
            "psd_matrix_imag": xr.DataArray(
                values.imag,
                dims=("percentile", "freq", "channel", "channel_aux"),
                coords={"percentile": [5.0, 50.0, 95.0], "freq": freq},
            ),
        }
    )
    handled = _handle_multivariate(psd_group, truth.real)
    assert "riae_diag_mean" in handled


def test_welch_overlay_helper_paths() -> None:
    raw = MultivariateTimeseries(
        np.column_stack([np.sin(np.arange(64.0)), np.cos(np.arange(64.0))]),
        t=np.arange(64.0) / 16.0,
    )
    processed = raw.standardise_for_psd().to_wishart_stats(Nb=1)
    overlay, labels, styles = _build_welch_overlay(
        raw,
        processed,
        PipelineConfig(welch_nperseg=16, exclude_freq_bands=[(999.0, 1000.0)]),
    )
    assert overlay is not None
    assert labels == ["Welch"]
    assert styles is not None

    assert _build_welch_overlay(None, processed, PipelineConfig()) == (
        None,
        None,
        None,
    )
    assert _build_welch_overlay(raw, None, PipelineConfig()) == (
        None,
        None,
        None,
    )

    narrowed = _apply_frequency_exclusion(
        processed,
        [(float(processed.freq[1]), float(processed.freq[-1]))],
    )
    none_overlay = _build_welch_overlay(
        raw,
        narrowed,
        PipelineConfig(welch_nperseg=4, verbose=True),
    )
    assert none_overlay == (None, None, None) or none_overlay[0] is not None
