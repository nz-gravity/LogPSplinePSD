import arviz as az
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from log_psplines.plotting.diagnostics import (
    DiagnosticsConfig,
    _create_divergences_diagnostics,
    _create_sampler_diagnostics,
    _plot_acceptance_diagnostics,
    _plot_log_posterior,
    _plot_mh_step_sizes,
    _plot_nuts_diagnostics,
)


def _make_nuts_idata():
    rng = np.random.default_rng(1)
    posterior = {
        "delta": rng.lognormal(mean=0.0, sigma=0.1, size=(1, 12, 1)),
        "phi": rng.lognormal(mean=0.0, sigma=0.1, size=(1, 12, 1)),
        "weights": rng.normal(size=(1, 12, 2)),
    }
    sample_stats = {
        "lp": rng.normal(size=(1, 12)),
        "energy": rng.normal(size=(1, 12)),
        "potential_energy": rng.normal(size=(1, 12)),
        "num_steps": rng.integers(1, 8, size=(1, 12)),
        "accept_prob": rng.uniform(0.6, 0.9, size=(1, 12)),
        "diverging": rng.integers(0, 2, size=(1, 12)),
        "tree_depth": rng.integers(4, 6, size=(1, 12)),
        "energy_error": rng.normal(scale=0.1, size=(1, 12)),
        "accept_prob_channel_0": rng.uniform(0.6, 0.9, size=(1, 12)),
        "energy_channel_0": rng.normal(size=(1, 12)),
        "potential_energy_channel_0": rng.normal(size=(1, 12)),
        "num_steps_channel_0": rng.integers(1, 8, size=(1, 12)),
        "diverging_channel_0": rng.integers(0, 2, size=(1, 12)),
    }
    idata = az.from_dict(posterior=posterior, sample_stats=sample_stats)
    idata.attrs["sampler_type"] = "nuts"
    idata.attrs["target_accept_rate"] = 0.8
    return idata


def _make_mh_idata():
    rng = np.random.default_rng(2)
    posterior = {"weights": rng.normal(size=(1, 10, 2))}
    sample_stats = {
        "step_size_mean": rng.uniform(0.1, 0.2, size=(1, 10)),
        "step_size_std": rng.uniform(0.01, 0.02, size=(1, 10)),
    }
    idata = az.from_dict(posterior=posterior, sample_stats=sample_stats)
    idata.attrs["sampler_type"] = "mh"
    return idata


def _make_acceptance_rate_idata():
    rng = np.random.default_rng(3)
    posterior = {"weights": rng.normal(size=(1, 12, 1))}
    sample_stats = {
        "acceptance_rate": rng.uniform(0.1, 0.4, size=(1, 12)),
    }
    idata = az.from_dict(posterior=posterior, sample_stats=sample_stats)
    idata.attrs["sampler_type"] = "mh"
    idata.attrs["target_accept_rate"] = 0.3
    return idata


def _latest_figure(before):
    after = set(plt.get_fignums())
    new_figs = after - before
    assert new_figs
    return plt.figure(max(new_figs))


def test_plot_log_posterior_fallback():
    idata = az.from_dict(
        posterior={"weights": np.zeros((1, 5, 1))},
        sample_stats={"dummy_stat": np.zeros((1, 1))},
    )
    config = DiagnosticsConfig(figsize=(6, 4))
    before = set(plt.get_fignums())
    _plot_log_posterior(idata, config)
    fig = _latest_figure(before)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_create_sampler_diagnostics_nuts_writes_files(tmp_path):
    idata = _make_nuts_idata()
    config = DiagnosticsConfig(figsize=(6, 4))
    diag_dir = tmp_path / "diagnostics"
    diag_dir.mkdir()
    _create_sampler_diagnostics(idata, str(diag_dir), config)
    assert (diag_dir / "nuts_diagnostics.png").exists()
    assert (diag_dir / "nuts_block_0_diagnostics.png").exists()


def test_create_sampler_diagnostics_mh_writes_files(tmp_path):
    idata = _make_mh_idata()
    config = DiagnosticsConfig(figsize=(6, 4))
    diag_dir = tmp_path / "diagnostics"
    diag_dir.mkdir()
    _create_sampler_diagnostics(idata, str(diag_dir), config)
    assert (diag_dir / "mh_step_sizes.png").exists()


def test_create_divergences_diagnostics_writes_file(tmp_path):
    idata = _make_nuts_idata()
    config = DiagnosticsConfig(figsize=(6, 4))
    diag_dir = tmp_path / "diagnostics"
    diag_dir.mkdir()
    _create_divergences_diagnostics(idata, str(diag_dir), config)
    assert (diag_dir / "divergences.png").exists()


def test_plot_nuts_diagnostics_smoke():
    idata = _make_nuts_idata()
    config = DiagnosticsConfig(figsize=(6, 4))
    before = set(plt.get_fignums())
    _plot_nuts_diagnostics(idata, config)
    fig = _latest_figure(before)
    assert len(fig.axes) >= 4
    plt.close(fig)


def test_plot_acceptance_acceptance_rate_branch():
    idata = _make_acceptance_rate_idata()
    config = DiagnosticsConfig(figsize=(6, 4))
    before = set(plt.get_fignums())
    _plot_acceptance_diagnostics(idata, config)
    fig = _latest_figure(before)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_mh_step_sizes_smoke():
    idata = _make_mh_idata()
    config = DiagnosticsConfig(figsize=(6, 4))
    before = set(plt.get_fignums())
    _plot_mh_step_sizes(idata, config)
    fig = _latest_figure(before)
    assert len(fig.axes) == 4
    plt.close(fig)
