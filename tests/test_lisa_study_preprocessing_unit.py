from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_lisa_preprocessing_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "studies"
        / "lisa"
        / "utils"
        / "preprocessing.py"
    )
    spec = importlib.util.spec_from_file_location(
        "lisa_utils_preprocessing", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_analysis_frequencies_matches_requested_coarse_grid():
    module = _load_lisa_preprocessing_module()

    coarse_cfg = module.setup_coarse_grain(
        module.compute_Nl_analysis(
            Lb=120_960,
            dt=5.0,
            fmin=1e-4,
            fmax=0.10809474686819218,
        ),
        8192,
    )
    freq = module.compute_analysis_frequencies(
        Lb=120_960,
        dt=5.0,
        fmin=1e-4,
        fmax=0.10809474686819218,
        coarse_cfg=coarse_cfg,
    )

    assert freq.ndim == 1
    assert freq.size == coarse_cfg.Nc
    assert np.all(np.diff(freq) > 0.0)


def test_transfer_null_bands_built_on_analysis_grid_capture_null_bin():
    module = _load_lisa_preprocessing_module()

    coarse_cfg = module.setup_coarse_grain(
        module.compute_Nl_analysis(
            Lb=120_960,
            dt=5.0,
            fmin=1e-4,
            fmax=0.10809474686819218,
        ),
        8192,
    )
    analysis_freq = module.compute_analysis_frequencies(
        Lb=120_960,
        dt=5.0,
        fmin=1e-4,
        fmax=0.10809474686819218,
        coarse_cfg=coarse_cfg,
    )
    bands = module.build_transfer_null_exclusion_bands(
        analysis_freq,
        bins_per_side=1,
        fmin=1e-4,
        fmax=0.10809474686819218,
    )

    second_null = module.compute_transfer_null_frequencies(
        fmin=1e-4,
        fmax=0.10809474686819218,
    )[1]
    analysis_hits = [
        int(np.count_nonzero((analysis_freq >= low) & (analysis_freq <= high)))
        for low, high in bands
    ]

    assert any(hit > 0 for hit in analysis_hits)
    assert any(low <= second_null <= high for low, high in bands)

    total_duration = 365.0 * 86_400.0
    freq_true = np.fft.rfftfreq(int(round(total_duration / 5.0)), d=5.0)[1:]
    fine_bands = module.build_transfer_null_exclusion_bands(
        freq_true[(freq_true >= 1e-4) & (freq_true <= 0.10809474686819218)],
        bins_per_side=1,
        fmin=1e-4,
        fmax=0.10809474686819218,
    )
    fine_hits = [
        int(np.count_nonzero((analysis_freq >= low) & (analysis_freq <= high)))
        for low, high in fine_bands
    ]

    assert any(hit > 0 for hit in analysis_hits)
    assert all(hit == 0 for hit in fine_hits)
