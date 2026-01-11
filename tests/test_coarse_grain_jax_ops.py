import jax.numpy as jnp
import numpy as np

from log_psplines.coarse_grain.jax_ops import coarse_grain_univar


def test_coarse_grain_univar_low_only():
    power = jnp.array([1.0, 2.0, 3.0, 4.0])
    mask_low = jnp.array([True, False, False, True])
    mask_high = jnp.array([False, True, True, False])
    power_coarse, weights = coarse_grain_univar(
        power,
        mask_low=mask_low,
        mask_high=mask_high,
        sort_indices=jnp.array([], dtype=int),
        bin_indices_sorted=jnp.array([], dtype=int),
        bin_counts=jnp.array([], dtype=int),
        n_low=2,
        n_bins_high=0,
        freqs=jnp.array([]),
        f_coarse=jnp.array([]),
        bin_widths=jnp.array([]),
        fine_spacing=1.0,
    )
    np.testing.assert_allclose(np.asarray(power_coarse), np.array([1.0, 4.0]))
    np.testing.assert_allclose(np.asarray(weights), np.array([1.0, 1.0]))


def test_coarse_grain_univar_with_high_bins():
    power = jnp.array([1.0, 2.0, 3.0, 4.0])
    mask_low = jnp.array([True, False, False, False])
    mask_high = jnp.array([False, True, True, True])
    power_coarse, weights = coarse_grain_univar(
        power,
        mask_low=mask_low,
        mask_high=mask_high,
        sort_indices=jnp.array([0, 1, 2]),
        bin_indices_sorted=jnp.array([0, 0, 1]),
        bin_counts=jnp.array([2, 1]),
        n_low=1,
        n_bins_high=2,
        freqs=jnp.array([]),
        f_coarse=jnp.array([]),
        bin_widths=jnp.array([0.5, 1.0]),
        fine_spacing=0.5,
    )
    np.testing.assert_allclose(
        np.asarray(power_coarse), np.array([1.0, 2.5, 4.0])
    )
    np.testing.assert_allclose(np.asarray(weights), np.array([1.0, 1.0, 2.0]))
