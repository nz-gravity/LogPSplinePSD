"""Lightweight checks for example datasets used by researchers."""

import os

import numpy as np
from pytest import fixture, mark, raises

from log_psplines.example_datasets import LS2Data, TVData, TVVARData, VARMAData


@fixture
def plotdir(outdir):
    """Fixture for temporary output directory."""
    d = os.path.join(outdir, "example_datasets")
    os.makedirs(d, exist_ok=True)
    return d


def test_varma_example(plotdir):
    data = VARMAData.ar(order=2, n_samples=128, fs=32.0, seed=42)
    assert data.ts.data.shape == (128, 1)
    assert data.get_true_psd().shape == (64, 1, 1)
    assert np.isfinite(data.ts.data).all()
    assert np.isfinite(data.get_true_psd()).all()
    assert np.all(data.get_true_psd()[:, 0, 0].real > 0)
    save_path = os.path.join(plotdir, "varma_example.png")
    data.plot(fname=save_path)
    assert os.path.exists(save_path)


def test_ls2_example(plotdir):
    data = LS2Data(n_samples=256, fs=32.0, seed=42)
    assert data.data.shape == (256, 1)
    assert data.get_true_psd().shape == (256, len(data.freq))
    assert np.isfinite(data.data).all()
    assert np.isfinite(data.get_true_psd()).all()
    assert np.all(data.freq > 0)
    assert np.isclose(data.freq.max(), data.fs / 2)
    save_path = os.path.join(plotdir, "ls2_example.png")
    data.plot(fname=save_path)
    assert os.path.exists(save_path)


def test_tvvar_example(plotdir):
    data = TVData(n_samples=256, fs=32.0, seed=42)
    assert data.data.shape == (256, 1)
    assert data.get_true_psd().shape == (256, len(data.freq))
    assert np.isfinite(data.data).all()
    assert np.isfinite(data.get_true_psd()).all()
    assert np.all(data.freq > 0)
    assert np.isclose(data.freq.max(), data.fs / 2)
    save_path = os.path.join(plotdir, "tvvar_example.png")
    data.plot(fname=save_path)
    assert os.path.exists(save_path)


def test_tvvar_dataset(plotdir):
    data = TVVARData(n_samples=256, fs=32.0, seed=42)
    assert data.data.shape == (256, 3)
    assert data.ts.data.shape == (256, 3)
    assert np.isfinite(data.data).all()
    assert data.p == 3
    assert data.dt == 1 / 32.0
    assert data.duration == 8.0
    np.testing.assert_allclose(data.time, np.arange(256) / 32.0)
    np.testing.assert_allclose(data.rescaled_time, np.arange(256) / 256)
    np.testing.assert_allclose(data.freq, np.fft.rfftfreq(256, d=data.dt)[1:])
    assert data.is_locally_stable
    assert data.max_local_spectral_radius < 1
    assert data.a2_at(0.1).shape == (3, 3)
    assert data.var_coeffs_at(0.1).shape == (2, 3, 3)

    u = np.array([0.1, 0.5, 0.9])
    assert data.a2_at(u).shape == (3, 3, 3)
    spectrum = data.get_true_psd(time_grid=u, freq_grid=data.freq[:10])
    assert spectrum.shape == (3, 10, 3, 3)
    assert np.iscomplexobj(spectrum)
    assert np.isfinite(spectrum).all()
    assert np.allclose(spectrum, spectrum.swapaxes(-1, -2).conj())
    diagonal = np.diagonal(spectrum, axis1=-2, axis2=-1).real
    assert np.all(diagonal > 0)
    assert np.all(np.linalg.eigvalsh(spectrum) > 0)
    for i in range(3):
        assert not np.allclose(diagonal[0, :, i], diagonal[1, :, i])
        for j in range(i + 1, 3):
            cross = spectrum[..., i, j]
            assert np.any(np.abs(cross.real) > 1e-10)
            assert np.any(np.abs(cross.imag) > 1e-10)
            assert not np.allclose(cross[0], cross[1])
    coherence = spectrum / np.sqrt(
        diagonal[..., :, None] * diagonal[..., None, :]
    )
    assert np.all(np.abs(coherence) ** 2 <= 1.0 + 1e-12)
    original = data.data.copy()
    np.testing.assert_array_equal(data.simulate(seed=42), original)
    data.resimulate(seed=7)
    assert not np.array_equal(data.data, original)
    np.testing.assert_array_equal(data.resimulate(seed=42), original)
    np.testing.assert_array_equal(data.data, original)

    save_path = os.path.join(plotdir, "tvvar_matrix.png")
    data.plot(fname=save_path)
    assert os.path.exists(save_path)


def test_tvvar_stationary_limit():
    data = TVVARData(n_samples=256, fs=32.0, seed=42, alpha=0)
    spectrum = data.get_true_psd(
        time_grid=np.array([0.0, 0.1, 0.5, 0.9, 1.0]),
        freq_grid=data.freq[:10],
    )
    assert data.is_locally_stable
    assert np.allclose(spectrum, spectrum[:1], rtol=1e-12, atol=1e-12)


def test_tvvar_white_noise_psd_normalization():
    """A known constant spectrum checks units and endpoint doubling."""
    sigma = np.array([[2.0, 0.5], [0.5, 1.0]])
    data = TVVARData(
        n_samples=16,
        fs=32.0,
        seed=42,
        burn_in=0,
        stability_grid_size=11,
        a1=np.zeros((2, 2)),
        a2_base=np.zeros((2, 2)),
        sigma=sigma,
    )
    # An interior frequency close to Nyquist must still be doubled.
    freq = np.array([16.0, 0.0, 8.0, 16.0 - 1e-5])
    spectrum = data.get_true_psd(
        time_grid=np.array([0.1, 0.5, 0.9]), freq_grid=freq
    )
    expected = np.array([1.0, 1.0, 2.0, 2.0])[:, None, None] * sigma / data.fs
    np.testing.assert_allclose(
        spectrum, np.broadcast_to(expected, spectrum.shape)
    )


@mark.parametrize(
    "options, message",
    [
        ({"n_samples": 1}, "n_samples"),
        ({"fs": 0.0}, "fs"),
        ({"fs": np.nan}, "fs"),
        ({"burn_in": -1}, "burn_in"),
        ({"stability_grid_size": 1}, "stability_grid_size"),
        ({"a1": np.zeros((2, 3))}, "square"),
        ({"a2_base": np.zeros((2, 2))}, "same shape"),
        ({"sigma": np.eye(2)}, "shape"),
        ({"sigma": np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])}, "symmetric"),
        ({"sigma": np.diag([1, 1, 0])}, "positive definite"),
        ({"sigma": np.diag([1, 1, np.nan])}, "finite"),
        (
            {"a1": 1.1 * np.eye(3), "a2_base": np.zeros((3, 3))},
            "locally stable",
        ),
    ],
)
def test_tvvar_invalid_inputs(options, message):
    config = {"n_samples": 16, "burn_in": 0, "stability_grid_size": 11}
    config.update(options)
    with raises(ValueError, match=message):
        TVVARData(**config)


@mark.parametrize(
    "freq", [np.array([-0.1]), np.array([16.1]), np.array([np.nan])]
)
def test_tvvar_invalid_frequency_grid(freq):
    data = TVVARData(n_samples=16, fs=32.0, seed=42, burn_in=0)
    with raises(ValueError, match="freq_grid"):
        data.get_true_psd(time_grid=np.array([0.5]), freq_grid=freq)
