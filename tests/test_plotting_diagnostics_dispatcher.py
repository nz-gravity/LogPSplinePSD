import arviz as az
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from log_psplines.diagnostics.plotting import (
    DiagnosticsConfig,
    _create_divergences_diagnostics,
    _create_pair_diagnostics,
    _create_rank_diagnostics,
    _create_sampler_diagnostics,
    _plot_acceptance_diagnostics,
    _plot_nuts_diagnostics_blockaware,
    _select_pair_plot_vars,
    _select_rank_plot_vars,
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
    idata.attrs["target_accept_prob"] = 0.8
    return idata


def _make_acceptance_rate_idata():
    rng = np.random.default_rng(3)
    posterior = {"weights": rng.normal(size=(1, 12, 1))}
    sample_stats = {
        "acceptance_rate": rng.uniform(0.1, 0.4, size=(1, 12)),
    }
    idata = az.from_dict(posterior=posterior, sample_stats=sample_stats)
    idata.attrs["sampler_type"] = "nuts"
    idata.attrs["target_accept_prob"] = 0.8
    return idata


def _make_high_dim_weights_idata():
    rng = np.random.default_rng(11)
    posterior = {
        "delta": rng.lognormal(mean=0.0, sigma=0.1, size=(1, 12, 1)),
        "phi": rng.lognormal(mean=0.0, sigma=0.1, size=(1, 12, 1)),
        "weights": rng.normal(size=(1, 12, 64)),
    }
    sample_stats = {"accept_prob": rng.uniform(0.6, 0.9, size=(1, 12))}
    idata = az.from_dict(posterior=posterior, sample_stats=sample_stats)
    idata.attrs["sampler_type"] = "nuts"
    return idata


def _latest_figure(before):
    after = set(plt.get_fignums())
    new_figs = after - before
    assert new_figs
    return plt.figure(max(new_figs))


def test_create_sampler_diagnostics_nuts_writes_files(tmp_path):
    idata = _make_nuts_idata()
    config = DiagnosticsConfig(figsize=(6, 4))
    diag_dir = tmp_path / "diagnostics"
    diag_dir.mkdir()
    _create_sampler_diagnostics(idata, str(diag_dir), config)
    assert (diag_dir / "nuts_diagnostics.png").exists()
    assert (diag_dir / "nuts_block_0_diagnostics.png").exists()


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
    _plot_nuts_diagnostics_blockaware(idata, config)
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


def test_select_rank_plot_vars_skips_high_dim_weights():
    idata = _make_high_dim_weights_idata()
    config = DiagnosticsConfig(rank_max_dims_per_var=4, rank_max_vars=6)
    selected = _select_rank_plot_vars(idata, config)
    assert "weights" not in selected
    assert "delta" in selected
    assert "phi" in selected


def test_create_rank_diagnostics_writes_file(tmp_path):
    idata = _make_nuts_idata()
    config = DiagnosticsConfig(
        figsize=(6, 4), save_rank_plots=True, rank_max_vars=4
    )
    diag_dir = tmp_path / "diagnostics"
    diag_dir.mkdir()
    _create_rank_diagnostics(idata, str(diag_dir), config)
    assert (diag_dir / "rank_plots.png").exists()


def test_pair_var_selection_and_plot(tmp_path):
    idata = _make_nuts_idata()
    config = DiagnosticsConfig(
        figsize=(6, 4), save_pair_plots=True, pair_max_vars=3
    )
    selected = _select_pair_plot_vars(idata, config)
    assert len(selected) >= 2
    diag_dir = tmp_path / "diagnostics"
    diag_dir.mkdir()
    _create_pair_diagnostics(idata, str(diag_dir), config)
    assert (diag_dir / "pair_plot.png").exists()
