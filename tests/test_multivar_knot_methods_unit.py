import jax.numpy as jnp
import numpy as np
import pytest

from log_psplines.datatypes import MultivarFFT
from log_psplines.datatypes.multivar_utils import U_to_Y
from log_psplines.psplines import MultivariateLogPSplines
from log_psplines.psplines.multivar_psplines import MultivarComponentKey


@pytest.fixture(autouse=True)
def _fast_init_weights(monkeypatch):
    def _zero_weights(
        log_pdgrm: jnp.ndarray,
        log_psplines,
        init_weights: jnp.ndarray | None = None,
        num_steps: int = 5000,
    ) -> jnp.ndarray:
        return jnp.zeros(log_psplines.n_basis)

    monkeypatch.setattr(
        "log_psplines.psplines.multivar_psplines.init_weights",
        _zero_weights,
    )


def _make_fft_data(
    n_freq: int = 40,
    peak_index: int | None = None,
    peak_scale: float = 0.0,
) -> MultivarFFT:
    """Construct deterministic multivariate FFT stats for knot tests."""
    p = 2
    freq = np.linspace(0.05, 1.0, n_freq, dtype=np.float64)
    u_re = np.full((n_freq, p, p), 0.05, dtype=np.float64)
    u_im = np.zeros((n_freq, p, p), dtype=np.float64)

    # Keep a non-trivial baseline over all frequencies
    for k in range(n_freq):
        scale = 1.0 + 0.1 * np.sin(2.0 * np.pi * k / max(n_freq - 1, 1))
        u_re[k, 0, 0] = 0.15 * scale
        u_re[k, 1, 1] = 0.12 * scale
        u_re[k, 0, 1] = 0.03 * scale
        u_re[k, 1, 0] = 0.02 * scale
        u_im[k, 0, 1] = 0.01 * scale
        u_im[k, 1, 0] = -0.01 * scale

    if peak_index is not None and peak_scale > 0.0:
        peak_index = int(np.clip(peak_index, 0, n_freq - 1))
        u_re[peak_index] *= peak_scale
        u_im[peak_index] *= peak_scale

    return MultivarFFT(
        u_re=u_re,
        u_im=u_im,
        freq=freq,
        N=n_freq,
        p=p,
        Nb=1,
    )


def _make_fft_data_channel_specific_peaks(
    n_freq: int = 64,
    peak_a: int = 16,
    peak_b: int = 46,
    scale: float = 20.0,
) -> MultivarFFT:
    fft_data = _make_fft_data(n_freq=n_freq)
    peak_a = int(np.clip(peak_a, 0, n_freq - 1))
    peak_b = int(np.clip(peak_b, 0, n_freq - 1))

    # Emphasize row-0 and row-1 energy at different frequencies so
    # density-based knot placement can differ by diagonal component.
    fft_data.u_re[peak_a, 0, :] *= scale
    fft_data.u_im[peak_a, 0, :] *= scale
    fft_data.u_re[peak_b, 1, :] *= scale
    fft_data.u_im[peak_b, 1, :] *= scale
    return fft_data


@pytest.mark.parametrize("method", ["uniform", "log", "density"])
def test_multivar_knot_methods_supported(method: str):
    fft_data = _make_fft_data()
    model = MultivariateLogPSplines.from_multivar_fft(
        fft_data=fft_data,
        n_knots=8,
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs={"method": method},
    )
    knots = np.asarray(model.diagonal_models[0].knots, dtype=np.float64)
    assert knots.ndim == 1
    assert np.all(knots >= 0.0)
    assert np.all(knots <= 1.0)
    assert np.all(np.diff(knots) >= 0.0)


def test_multivar_knot_method_default_matches_density():
    fft_data = _make_fft_data()
    model_default = MultivariateLogPSplines.from_multivar_fft(
        fft_data=fft_data,
        n_knots=8,
        degree=3,
        diffMatrixOrder=2,
    )
    model_density = MultivariateLogPSplines.from_multivar_fft(
        fft_data=fft_data,
        n_knots=8,
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs={"method": "density"},
    )
    np.testing.assert_allclose(
        model_default.diagonal_models[0].knots,
        model_density.diagonal_models[0].knots,
    )


def test_multivar_density_knots_adapt_to_u_energy_peak():
    n_freq = 60
    peak_index = 36
    fft_data = _make_fft_data(
        n_freq=n_freq,
        peak_index=peak_index,
        peak_scale=25.0,
    )
    model_uniform = MultivariateLogPSplines.from_multivar_fft(
        fft_data=fft_data,
        n_knots=10,
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs={"method": "uniform"},
    )
    model_density = MultivariateLogPSplines.from_multivar_fft(
        fft_data=fft_data,
        n_knots=10,
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs={"method": "density"},
    )

    uniform_knots = np.asarray(model_uniform.diagonal_models[0].knots)
    density_knots = np.asarray(model_density.diagonal_models[0].knots)

    rounded_uniform = np.round(uniform_knots, 12)
    rounded_density = np.round(density_knots, 12)
    assert not np.array_equal(rounded_uniform, rounded_density)

    peak_freq = fft_data.freq[peak_index]
    fmin = float(fft_data.freq[0])
    fmax = float(fft_data.freq[-1])
    peak_norm = (peak_freq - fmin) / (fmax - fmin)
    window = 0.15

    n_uniform_near_peak = int(
        np.sum(np.abs(uniform_knots - peak_norm) <= window)
    )
    n_density_near_peak = int(
        np.sum(np.abs(density_knots - peak_norm) <= window)
    )
    assert n_density_near_peak >= n_uniform_near_peak


@pytest.mark.parametrize("method", ["linear", "quantile", "lvk", "unknown"])
def test_multivar_invalid_knot_methods_raise_clear_error(method: str):
    fft_data = _make_fft_data()
    with pytest.raises(
        ValueError, match="Unsupported multivariate knot method"
    ):
        MultivariateLogPSplines.from_multivar_fft(
            fft_data=fft_data,
            n_knots=8,
            degree=3,
            diffMatrixOrder=2,
            knot_kwargs={"method": method},
        )


def test_multivar_density_allows_component_specific_diagonal_knots():
    fft_data = _make_fft_data_channel_specific_peaks()
    model = MultivariateLogPSplines.from_multivar_fft(
        fft_data=fft_data,
        n_knots=10,
        degree=3,
        diffMatrixOrder=2,
        knot_kwargs={"method": "density"},
    )
    knots_0 = np.round(np.asarray(model.diagonal_models[0].knots), 12)
    knots_1 = np.round(np.asarray(model.diagonal_models[1].knots), 12)
    assert not np.array_equal(knots_0, knots_1)


def test_multivar_density_uses_fixed_basis_count_per_component():
    n_freq = 64
    p = 3
    rng = np.random.default_rng(0)
    freq = np.linspace(0.1, 1.0, n_freq, dtype=np.float64)
    u_re = rng.normal(size=(n_freq, p, p)).astype(np.float64) * 0.1
    u_im = rng.normal(size=(n_freq, p, p)).astype(np.float64) * 0.1
    fft_data = MultivarFFT(
        u_re=u_re,
        u_im=u_im,
        freq=freq,
        N=n_freq,
        p=p,
        Nb=2,
    )

    n_knots = 12
    degree = 2
    model = MultivariateLogPSplines.from_multivar_fft(
        fft_data=fft_data,
        n_knots=n_knots,
        degree=degree,
        diffMatrixOrder=2,
        knot_kwargs={"method": "density"},
    )

    component_models = (
        model.diagonal_models
        + [
            model.offdiag_re_models[pair]
            for pair in model.theta_pairs
        ]
        + [
            model.offdiag_im_models[pair]
            for pair in model.theta_pairs
        ]
    )
    knot_sizes = {
        int(component.knots.shape[0]) for component in component_models
    }
    basis_sizes = {
        int(component.basis.shape[1]) for component in component_models
    }

    assert knot_sizes == {n_knots}
    assert basis_sizes == {n_knots + degree - 1}


def test_multivar_density_scoring_uses_channel_space_wishart(
    monkeypatch,
):
    n_freq = 8
    p = 3
    rng = np.random.default_rng(123)
    freq = np.linspace(0.05, 0.5, n_freq, dtype=np.float64)
    u_re = rng.normal(size=(n_freq, p, p)).astype(np.float64)
    u_im = rng.normal(size=(n_freq, p, p)).astype(np.float64)
    fft_data = MultivarFFT(
        u_re=u_re,
        u_im=u_im,
        freq=freq,
        N=n_freq,
        p=p,
        Nb=2,
    )

    captured: dict[str, np.ndarray] = {}

    def _capture_scores(
        Y_np: np.ndarray,
        Nb: int,
        p: int,
        *,
        scoring: str = "cholesky",
        u_re: np.ndarray | None = None,
        u_im: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        captured["Y_np"] = np.asarray(Y_np, dtype=np.complex128)
        diag_scores = [np.ones(n_freq, dtype=np.float64) for _ in range(p)]
        n_theta = p * (p - 1) // 2
        offdiag_re = [np.ones(n_freq, dtype=np.float64) for _ in range(n_theta)]
        offdiag_im = [np.ones(n_freq, dtype=np.float64) for _ in range(n_theta)]
        return diag_scores, offdiag_re, offdiag_im

    monkeypatch.setattr(
        "log_psplines.psplines.multivar_psplines.multivar_psd_knot_scores",
        _capture_scores,
    )

    MultivariateLogPSplines.from_multivar_fft(
        fft_data=fft_data,
        n_knots=6,
        degree=2,
        diffMatrixOrder=2,
        knot_kwargs={"method": "density", "scoring": "cholesky"},
    )

    assert (
        "Y_np" in captured
    ), "Expected knot scoring to receive a Wishart matrix."
    expected = U_to_Y(
        np.asarray(u_re, dtype=np.float64)
        + 1j * np.asarray(u_im, dtype=np.float64)
    )
    np.testing.assert_allclose(
        captured["Y_np"], expected, rtol=1e-10, atol=1e-10
    )


def test_multivar_density_assigns_distinct_re_im_knot_vectors(monkeypatch):
    n_freq = 64
    p = 2
    freq = np.linspace(0.05, 0.5, n_freq, dtype=np.float64)
    u_re = np.ones((n_freq, p, p), dtype=np.float64) * 0.1
    u_im = np.zeros((n_freq, p, p), dtype=np.float64)
    fft_data = MultivarFFT(
        u_re=u_re,
        u_im=u_im,
        freq=freq,
        N=n_freq,
        p=p,
        Nb=2,
    )

    def _separate_scores(
        Y_np: np.ndarray,
        Nb: int,
        p: int,
        *,
        scoring: str = "cholesky",
        u_re: np.ndarray | None = None,
        u_im: np.ndarray | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        diag_scores = [np.ones(n_freq, dtype=np.float64) for _ in range(p)]

        # Force different density concentration so re/im knot vectors should differ.
        re_score = np.ones(n_freq, dtype=np.float64)
        im_score = np.ones(n_freq, dtype=np.float64)
        re_score[8] = 100.0
        im_score[52] = 100.0
        return diag_scores, [re_score], [im_score]

    monkeypatch.setattr(
        "log_psplines.psplines.multivar_psplines.multivar_psd_knot_scores",
        _separate_scores,
    )

    model = MultivariateLogPSplines.from_multivar_fft(
        fft_data=fft_data,
        n_knots=10,
        degree=2,
        diffMatrixOrder=2,
        knot_kwargs={"method": "density", "scoring": "cholesky"},
    )

    knots_re = np.round(np.asarray(model.offdiag_re_models[(1, 0)].knots), 12)
    knots_im = np.round(np.asarray(model.offdiag_im_models[(1, 0)].knots), 12)
    assert not np.array_equal(knots_re, knots_im)


def test_multivar_component_registry_tracks_all_components():
    n_freq = 32
    p = 3
    rng = np.random.default_rng(7)
    freq = np.linspace(0.05, 0.5, n_freq, dtype=np.float64)
    u_re = rng.normal(scale=0.2, size=(n_freq, p, p)).astype(np.float64)
    u_im = rng.normal(scale=0.2, size=(n_freq, p, p)).astype(np.float64)
    fft_data = MultivarFFT(
        u_re=u_re,
        u_im=u_im,
        freq=freq,
        N=n_freq,
        p=p,
        Nb=2,
    )

    model = MultivariateLogPSplines.from_multivar_fft(
        fft_data=fft_data,
        n_knots=8,
        degree=2,
        diffMatrixOrder=2,
        knot_kwargs={"method": "density"},
    )

    expected_order = [
        "delta_0",
        "delta_1",
        "delta_2",
        "theta_re_1_0",
        "theta_re_2_0",
        "theta_re_2_1",
        "theta_im_1_0",
        "theta_im_2_0",
        "theta_im_2_1",
    ]
    order_names = [key.name for key in model.component_order]
    assert order_names == expected_order
    assert len(model.component_specs) == len(expected_order)

    key = MultivarComponentKey("theta", 2, l=1, part="im")
    spec = model.get_component_spec(key)
    assert spec.key == key
    assert spec.model is model.get_theta_model("im", 2, 1)
