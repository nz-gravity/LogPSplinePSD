import arviz as az
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from log_psplines.plotting.diagnostics import (
    DiagnosticsConfig,
    _get_channel_indices,
    _plot_acceptance_diagnostics_blockaware,
    _plot_log_posterior,
    _plot_nuts_diagnostics_blockaware,
    _plot_single_nuts_block,
    plot_trace,
)


def _make_minimal_idata() -> az.InferenceData:
    rng = np.random.default_rng(0)
    posterior = {
        "delta": rng.lognormal(mean=0.0, sigma=0.1, size=(1, 20, 2)),
        "phi": rng.lognormal(mean=0.0, sigma=0.1, size=(1, 20, 2)),
        "weights": rng.normal(size=(1, 20, 3)),
    }
    sample_stats = {
        "lp": rng.normal(size=(1, 20)),
        "accept_prob": rng.uniform(0.6, 0.9, size=(1, 20)),
        "energy": rng.normal(size=(1, 20)),
        "potential_energy": rng.normal(size=(1, 20)),
        "num_steps": rng.integers(1, 5, size=(1, 20)),
        "accept_prob_channel_0": rng.uniform(0.6, 0.9, size=(1, 20)),
        "energy_channel_0": rng.normal(size=(1, 20)),
        "potential_energy_channel_0": rng.normal(size=(1, 20)),
        "num_steps_channel_0": rng.integers(1, 5, size=(1, 20)),
    }
    dims = {"delta": ["param"], "phi": ["param"], "weights": ["weight"]}
    idata = az.from_dict(
        posterior=posterior, sample_stats=sample_stats, dims=dims
    )
    idata.attrs["sampler_type"] = "nuts"
    idata.attrs["target_accept_rate"] = 0.8
    return idata


def _latest_figure(before):
    after = set(plt.get_fignums())
    new_figs = after - before
    assert new_figs
    return plt.figure(max(new_figs))


def test_plot_trace_smoke():
    idata = _make_minimal_idata()
    fig = plot_trace(idata, compact=True)
    assert fig is not None
    assert len(fig.axes) > 0
    plt.close(fig)


def test_plot_log_posterior_smoke():
    idata = _make_minimal_idata()
    config = DiagnosticsConfig(figsize=(6, 4))
    before = set(plt.get_fignums())
    _plot_log_posterior(idata, config)
    fig = _latest_figure(before)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_acceptance_blockaware_smoke():
    idata = _make_minimal_idata()
    config = DiagnosticsConfig(figsize=(6, 4))
    before = set(plt.get_fignums())
    _plot_acceptance_diagnostics_blockaware(idata, config)
    fig = _latest_figure(before)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_nuts_blockaware_smoke():
    idata = _make_minimal_idata()
    config = DiagnosticsConfig(figsize=(6, 4))
    before = set(plt.get_fignums())
    _plot_nuts_diagnostics_blockaware(idata, config)
    fig = _latest_figure(before)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_single_nuts_block_smoke():
    idata = _make_minimal_idata()
    config = DiagnosticsConfig(figsize=(6, 4))
    before = set(plt.get_fignums())
    _plot_single_nuts_block(idata, config, channel_idx=0)
    fig = _latest_figure(before)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_get_channel_indices_finds_channels():
    idata = _make_minimal_idata()
    indices = _get_channel_indices(idata.sample_stats, "accept_prob")
    assert indices == {0}
