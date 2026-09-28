"""Pre-fit eigenvalue checks and their figure boundary."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from log_psplines.config import StationaryConfig
from log_psplines.data.spectral import WishartData
from log_psplines.plotting.preprocessing import plot_eigenvalue_separation
from log_psplines.preprocessing.checks import (
    _run_preprocessing_checks,
    _save_preprocessing_plot,
)
from log_psplines.preprocessing.diagnostics import (
    eigenvalue_separation_diagnostics,
    extract_component_knots,
    model_component_curves,
)


def test_eigenvalue_diagnostics_components_and_plot():
    freq = np.linspace(0.1, 0.6, 6)
    matrix = np.repeat(np.diag([2.0, 1.8])[None, :, :], freq.size, axis=0)
    diag = eigenvalue_separation_diagnostics(freq=freq, matrix=matrix)
    np.testing.assert_allclose(diag.ratios["r_12"], 0.9)
    assert diag.worst_frequencies(top_k=2)["r_12"][0][1] == 0.9

    component = SimpleNamespace(knots=np.array([0.0, 0.5, 1.0]))
    model = SimpleNamespace(
        p=2,
        diagonal_models=[component, component],
        get_theta_model=lambda kind, row, col: component,
    )
    knots = extract_component_knots(model, freq)
    np.testing.assert_allclose(knots["Re(Theta12)"], [0.1, 0.35, 0.6])
    curves = model_component_curves(freq, matrix)
    assert set(curves) == set(knots)
    fig = plot_eigenvalue_separation(
        diag, component_curves=curves, component_knots=knots
    )
    assert len(fig.axes) == 6
    plt.close(fig)


def test_preprocessing_check_saves_plot_and_warns(tmp_path, monkeypatch):
    freq = np.linspace(0.1, 0.6, 6)
    matrix = np.array([np.diag([scale, 0.9 * scale]) for scale in range(2, 8)])
    data = WishartData(
        u_re=np.zeros((6, 2, 2)),
        u_im=np.zeros((6, 2, 2)),
        freq=freq,
        N=6,
        p=2,
        Nb=2,
        fs=12.0,
        raw_psd=matrix,
    )
    warnings = []
    monkeypatch.setattr(
        "log_psplines.preprocessing.checks.logger.warning", warnings.append
    )
    config = StationaryConfig(outdir=str(tmp_path))
    _run_preprocessing_checks(data, config)
    assert any("Eigenvalue separation r_12" in message for message in warnings)
    _save_preprocessing_plot(data, config)
    assert (tmp_path / "diagnostics" / "preprocessing_eigenvalue_ratios.png").exists()
