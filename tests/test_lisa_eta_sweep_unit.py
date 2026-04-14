from __future__ import annotations

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pytest


def _load_eta_sweep_module():
    lisa_root = (
        Path(__file__).resolve().parents[1] / "docs" / "studies" / "lisa"
    )
    if str(lisa_root) not in sys.path:
        sys.path.insert(0, str(lisa_root))
    import eta_sweep as module

    return module


def test_extract_channel_diagnostics_collects_block_metrics():
    module = _load_eta_sweep_module()
    idata = az.from_dict(
        {
            "sample_stats": {
                "step_size_channel_0": np.array([[0.2, 0.2]], dtype=float),
                "accept_prob_channel_0": np.array([[0.8, 0.9]], dtype=float),
                "num_steps_channel_0": np.array([[5, 7]], dtype=int),
                "diverging_channel_0": np.array([[0, 1]], dtype=int),
                "step_size_channel_1": np.array([[0.05, 0.04]], dtype=float),
            }
        }
    )
    idata.attrs["p"] = 2
    idata.attrs["sampling_eta_channel_0"] = 0.125
    idata.attrs["sampling_eta_channel_1"] = 0.125

    metrics = module._extract_channel_diagnostics(idata)

    assert metrics["sampling_eta_channel_0"] == 0.125
    assert metrics["step_size_channel_0_median"] == 0.2
    assert metrics["accept_prob_channel_0_mean"] == pytest.approx(0.85)
    assert metrics["num_steps_channel_0_max"] == 7.0
    assert metrics["n_divergences_channel_0"] == 1
    assert metrics["step_size_channel_1_min"] == 0.04


def test_eta_slug_is_filesystem_friendly():
    module = _load_eta_sweep_module()

    assert module._eta_slug(0.125) == "0p125"
    assert module._eta_slug(1.0) == "1"


def test_default_labels_continue_after_ap():
    module = _load_eta_sweep_module()

    assert module._default_labels(6) == ["AQ", "AR", "AS", "AT", "AU", "AV"]


def test_build_run_dir_name_uses_letter_label_prefix():
    module = _load_eta_sweep_module()

    class _Args:
        duration_days = 365.0
        knot_method = "density"
        exclude_transfer_nulls = True
        exclude_bins_per_side = 3
        vi = True

    run_dir = module._build_run_dir_name(
        label="AQ",
        args=_Args(),
        eta=0.08,
        knot_counts=20,
    )

    assert run_dir.startswith("run_AQ_")
    assert "_eta0p08_" in run_dir


def test_resolve_vi_coarse_cfg_uses_raw_grid_divisor():
    module = _load_eta_sweep_module()

    cfg = module._resolve_vi_coarse_cfg(
        target_nfreq=6_024,
        explicit_nc=0,
        auto_enabled=True,
        auto_target_nfreq=192,
    )

    assert cfg is not None
    assert cfg.enabled is True
    assert cfg.Nc == 192


def test_resolve_vi_coarse_cfg_uses_explicit_nc_when_provided():
    module = _load_eta_sweep_module()

    cfg = module._resolve_vi_coarse_cfg(
        target_nfreq=6_024,
        explicit_nc=190,
        auto_enabled=True,
        auto_target_nfreq=192,
    )

    assert cfg is not None
    assert cfg.Nc == 190
