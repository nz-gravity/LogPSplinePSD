from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")


def _load_lisa_plotting_module():
    lisa_root = (
        Path(__file__).resolve().parents[1] / "docs" / "studies" / "lisa"
    )
    if str(lisa_root) not in sys.path:
        sys.path.insert(0, str(lisa_root))
    import utils.plotting as module

    return module


def test_make_preprocessing_psd_plot_smoke(tmp_path):
    module = _load_lisa_plotting_module()

    fs = 0.2
    n = 1024
    t = np.arange(n, dtype=float) / fs
    data = np.stack(
        [
            np.sin(2.0 * np.pi * 0.01 * t),
            0.7 * np.sin(2.0 * np.pi * 0.01 * t + 0.2),
            0.5 * np.sin(2.0 * np.pi * 0.03 * t - 0.4),
        ],
        axis=1,
    )

    freq_true = np.fft.rfftfreq(n, d=1.0 / fs)[1:]
    base = 1e-10 + 1e-9 * np.exp(-((freq_true - 0.02) ** 2) / 1e-4)
    s_true = np.zeros((freq_true.size, 3, 3), dtype=np.complex128)
    for idx in range(3):
        s_true[:, idx, idx] = base * (1.0 + 0.2 * idx)
    s_true[:, 1, 0] = s_true[:, 0, 1] = 0.15 * base
    s_true[:, 2, 0] = s_true[:, 0, 2] = 0.05 * base
    s_true[:, 2, 1] = s_true[:, 1, 2] = 0.08 * base

    module.make_preprocessing_psd_plot(
        y_full=data,
        fs=fs,
        Lb=256,
        freq_true=freq_true,
        S_true=s_true,
        outdir=str(tmp_path),
        fmin=1e-3,
        fmax=0.08,
    )

    assert (tmp_path / "preprocessing_psd_matrix.png").exists()
