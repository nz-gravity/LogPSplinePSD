"""Pre-fit eigenvalue checks and their figure boundary."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from log_psplines import LogPSpline, SplineBasis
from log_psplines.config import StationaryConfig
from log_psplines.data.spectral import WishartData
from log_psplines.plotting import (
    compute_confidence_intervals,
    plot_basis_diagnostics,
)
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
from log_psplines.preprocessing.knot_locator import quantile_knots


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
    assert (
        tmp_path / "diagnostics" / "preprocessing_eigenvalue_ratios.png"
    ).exists()


@pytest.mark.parametrize("placement", ["uniform", "quantile", "supplied"])
def test_basis_diagnostics_inspect_actual_model_basis(placement, tmp_path):
    grid = np.linspace(0.1, 1.0, 41)
    if placement == "uniform":
        basis = SplineBasis.from_grid(grid, n_interior_knots=3)
    else:
        interior = (
            quantile_knots(
                grid,
                np.exp(-(((grid - 0.4) / 0.15) ** 2)),
                3,
                min_spacing=0.02,
            )
            if placement == "quantile"
            else np.array([0.2, 0.5, 0.8])
        )
        basis = SplineBasis.from_grid(grid, interior_knots=interior)
    model = LogPSpline(basis)
    fig, axes = plot_basis_diagnostics(
        model.frequency, path=tmp_path / "basis.png"
    )
    assert model.frequency is basis
    assert len(axes[0].lines) == basis.basis.shape[1]
    np.testing.assert_allclose(axes[0].lines[0].get_xdata(), basis.grid)
    collections = {item.get_label(): item for item in axes[0].collections}
    assert len(collections["Interior knots"].get_segments()) == 3
    assert len(collections["Boundary knots"].get_segments()) == 2
    np.testing.assert_allclose(
        axes[1].collections[0].get_array().reshape(basis.penalty.shape),
        basis.penalty,
    )
    assert f"order {basis.penalty_order}" in axes[0].get_title()
    assert (tmp_path / "basis.png").stat().st_size > 0
    plt.close(fig)


@pytest.mark.parametrize("method", ["percentile", "uniform", "invalid"])
def test_supported_interval_modes(method):
    draws = np.column_stack([np.linspace(-1, 1, 101), np.full(101, 3.0)])
    if method == "invalid":
        with pytest.raises(ValueError, match="Unknown CI method"):
            compute_confidence_intervals(draws, method=method)
        return
    low, median, high = compute_confidence_intervals(draws, method=method)
    assert np.isfinite([low, median, high]).all()
    np.testing.assert_allclose(median, np.median(draws, axis=0))
    if method == "percentile":
        np.testing.assert_allclose(
            [low, median, high], np.percentile(draws, [16, 50, 84], axis=0)
        )
    else:
        assert np.mean(np.all((draws >= low) & (draws <= high), axis=1)) >= 0.9
    np.testing.assert_allclose([low[1], median[1], high[1]], 3.0)
