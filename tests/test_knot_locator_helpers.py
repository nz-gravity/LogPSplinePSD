import numpy as np
import pytest

from log_psplines.psplines.knots_locator.knot_locator import (
    _adaptive_denoise,
    _dedup_sorted_with_tol,
    _enforce_exact_knot_count,
    _quantile_based_knots,
    denoise_score,
    init_knots,
    multivar_psd_knot_scores,
)


def test_knot_count_enforcement_and_deduplication() -> None:
    np.testing.assert_allclose(_dedup_sorted_with_tol(np.asarray([])), [])
    np.testing.assert_allclose(
        _dedup_sorted_with_tol(
            np.asarray([0.0, 1e-14, 0.5, 1.0 - 1e-14, 1.0])
        ),
        [0.0, 0.5, 1.0],
    )

    np.testing.assert_allclose(
        _enforce_exact_knot_count(np.asarray([]), target_count=4),
        np.linspace(0.0, 1.0, 4),
    )
    reduced = _enforce_exact_knot_count(
        np.asarray([0.0, 0.2, 0.21, 0.8, 1.0]),
        target_count=4,
    )
    assert reduced.size == 4
    assert reduced[0] == 0.0
    assert reduced[-1] == 1.0
    expanded = _enforce_exact_knot_count(
        np.asarray([0.0, 1.0]), target_count=5
    )
    assert expanded.size == 5
    assert expanded[0] == 0.0
    assert expanded[-1] == 1.0

    with pytest.raises(ValueError, match="target_count"):
        _enforce_exact_knot_count(np.asarray([0.0, 1.0]), target_count=1)
    with pytest.raises(ValueError, match="include endpoints"):
        _enforce_exact_knot_count(np.asarray([0.2, 1.0]), target_count=3)


def test_init_knots_methods_and_validation() -> None:
    freqs = np.linspace(0.1, 1.0, 12)
    power = 1.0 + freqs**2

    np.testing.assert_allclose(init_knots(2, freqs, power), [0.0, 1.0])
    assert init_knots(5, freqs, power, method="uniform").shape == (5,)
    assert init_knots(5, freqs, power, method="log").shape == (5,)
    density = init_knots(5, freqs, power, method="density")
    assert density.shape == (5,)
    assert density[0] == 0.0
    assert density[-1] == 1.0

    custom = init_knots(4, freqs, power, knots=np.asarray([0.1, 0.55, 1.0]))
    assert custom.shape == (4,)
    assert custom[0] == 0.0
    assert custom[-1] == 1.0

    with pytest.warns(UserWarning, match="NaN"):
        nan_knots = init_knots(
            4,
            freqs,
            power,
            knots=np.asarray([0.1, np.nan, 1.0]),
        )
    assert np.all(np.isfinite(nan_knots))

    with pytest.raises(ValueError, match="freqs must be"):
        init_knots(4, freqs[:, None], power)
    with pytest.raises(ValueError, match="same length"):
        init_knots(4, freqs, power[:-1])
    with pytest.raises(ValueError, match="non-empty"):
        init_knots(4, np.asarray([]), np.asarray([]))
    with pytest.raises(ValueError, match="Unknown knot"):
        init_knots(4, freqs, power, method="unknown")


def test_density_knots_denoising_and_multivar_scores() -> None:
    short = np.asarray([1.0, 2.0, 1.0])
    np.testing.assert_allclose(_adaptive_denoise(short), short)
    np.testing.assert_allclose(denoise_score(short, np.arange(3.0)), short)

    freqs = np.linspace(0.0, 1.0, 20)
    power = np.sin(freqs * np.pi) ** 2 + 1.0
    guide = np.cos(freqs * np.pi) ** 2 + 1.0
    knots = _quantile_based_knots(
        6, freqs, power, guide_power=guide, guide_strength=0.5
    )
    assert knots.shape == (6,)
    assert np.all(np.diff(knots) >= 0.0)

    with pytest.raises(ValueError, match="guide_power"):
        _quantile_based_knots(5, freqs, power, guide_power=guide[:-1])

    Y = np.zeros((3, 2, 2), dtype=np.complex128)
    for idx in range(3):
        Y[idx] = np.asarray([[2.0 + idx, 0.2 + 0.1j], [0.2 - 0.1j, 1.5 + idx]])
    diagonal, off_re, off_im = multivar_psd_knot_scores(Y, Nb=2, p=2)
    assert len(diagonal) == 2
    assert len(off_re) == len(off_im) == 1
    assert diagonal[0].shape == (3,)
