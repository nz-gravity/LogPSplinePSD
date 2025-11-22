import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr


def _install_skfda_stub() -> None:
    """Provide a minimal stub for scikit-fda so imports succeed in tests."""
    if "skfda" in sys.modules:
        return

    def _dummy(*args, **kwargs):
        return None

    skfda = types.ModuleType("skfda")

    misc = types.ModuleType("skfda.misc")
    operators = types.ModuleType("skfda.misc.operators")
    operators.LinearDifferentialOperator = _dummy
    reg = types.ModuleType("skfda.misc.regularization")
    reg.L2Regularization = _dummy
    misc.operators = operators
    misc.regularization = reg

    representation = types.ModuleType("skfda.representation")
    basis = types.ModuleType("skfda.representation.basis")

    class _Basis:
        def __init__(self, *args, **kwargs):
            pass

    basis.BSplineBasis = _Basis
    representation.basis = basis

    skfda.misc = misc
    skfda.representation = representation

    sys.modules["skfda"] = skfda
    sys.modules["skfda.misc"] = misc
    sys.modules["skfda.misc.operators"] = operators
    sys.modules["skfda.misc.regularization"] = reg
    sys.modules["skfda.representation"] = representation
    sys.modules["skfda.representation.basis"] = basis


_install_skfda_stub()

import log_psplines.arviz_utils as arviz_utils  # noqa: E402
from log_psplines.plotting import base as plotting_base  # noqa: E402
from log_psplines.plotting.psd_matrix import plot_psd_matrix  # noqa: E402


@pytest.fixture(autouse=True)
def _stub_arviz_helpers(monkeypatch):
    """Stub ArviZ helper accessors that are irrelevant for these tests."""

    def _none(*args, **kwargs):
        return None

    monkeypatch.setattr(arviz_utils, "get_periodogram", _none)
    monkeypatch.setattr(arviz_utils, "get_spline_model", _none)
    monkeypatch.setattr(arviz_utils, "get_weights", _none)


class DummyIdata:
    """Simple stand-in for an ArviZ InferenceData object."""

    def __init__(self, posterior_psd=None, vi_posterior_psd=None):
        self.posterior_psd = posterior_psd
        self.vi_posterior_psd = vi_posterior_psd
        self.attrs = {}
        self.observed_data = None

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def __getitem__(self, item: str):
        return getattr(self, item)


def _make_psd_dataset(
    freq: np.ndarray,
    real_vals: np.ndarray,
    imag_vals: np.ndarray,
    coherence: np.ndarray | None = None,
) -> xr.Dataset:
    percentile = np.array([5.0, 50.0, 95.0], dtype=float)
    channels = np.arange(real_vals.shape[-1])
    dims = ["percentile", "freq", "channels", "channels2"]
    coords = dict(
        percentile=percentile, freq=freq, channels=channels, channels2=channels
    )
    data_vars = {
        "psd_matrix_real": (dims, real_vals),
        "psd_matrix_imag": (dims, imag_vals),
    }
    if coherence is not None:
        data_vars["coherence"] = (dims, coherence)
    return xr.Dataset(data_vars=data_vars, coords=coords)


def test_extract_plotting_data_uses_vi_fallback(monkeypatch):
    freq = np.array([0.1, 0.2], dtype=float)
    real = np.ones((3, freq.size, 2, 2), dtype=float)
    imag = np.zeros_like(real)
    coherence = np.linspace(0.1, 0.3, 3, dtype=float)[:, None, None, None]
    coherence = np.broadcast_to(coherence, real.shape)

    vi_ds = _make_psd_dataset(freq, real, imag, coherence)
    posterior_ds = _make_psd_dataset(
        freq, real, imag, np.zeros_like(coherence)
    )
    idata = DummyIdata(
        posterior_psd=posterior_ds,
        vi_posterior_psd=vi_ds,
    )
    idata.attrs["only_vi"] = True

    results = plotting_base.extract_plotting_data(idata)

    assert "posterior_psd_matrix_quantiles" in results
    np.testing.assert_allclose(results["frequencies"], freq)
    np.testing.assert_allclose(
        results["posterior_psd_matrix_quantiles"]["coherence"], coherence
    )


def test_plot_psd_matrix_overlays_vi_coherence(tmp_path):
    freq = np.array([0.1, 0.2, 0.4], dtype=float)
    real = np.ones((3, freq.size, 2, 2), dtype=float)
    imag = np.zeros_like(real)

    posterior_coh = np.full_like(real, 0.2)
    vi_coh = np.full_like(real, 0.6)

    posterior_ds = _make_psd_dataset(freq, real, imag, posterior_coh)
    vi_ds = _make_psd_dataset(freq, real, imag, vi_coh)
    idata = DummyIdata(
        posterior_psd=posterior_ds,
        vi_posterior_psd=vi_ds,
    )

    fig, axes = plot_psd_matrix(
        idata=idata,
        outdir=str(tmp_path),
        filename="vi_overlay.png",
        show_coherence=True,
        overlay_vi=True,
        save=False,
    )

    # Coherence axis (channel 2 vs 1) should include a VI overlay line
    coh_ax = axes[1, 0]
    vi_lines = [
        line for line in coh_ax.lines if line.get_color() == "tab:orange"
    ]
    assert vi_lines, "Expected VI coherence overlay line to be drawn"
    np.testing.assert_allclose(vi_lines[-1].get_ydata(), vi_coh[1, :, 1, 0])

    plt.close(fig)
