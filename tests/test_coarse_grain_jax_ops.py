import jax.numpy as jnp
import numpy as np

from log_psplines.coarse_grain.jax_ops import coarse_grain_univar


def test_coarse_grain_univar_full_band_sum():
    power = jnp.array([1.0, 2.0, 3.0, 4.0])
    mask_low = jnp.array([False, False, False, False])
    mask_high = jnp.array([True, True, True, True])
    power_coarse, weights = coarse_grain_univar(
        power,
        mask_low=mask_low,
        mask_high=mask_high,
        sort_indices=jnp.array([0, 1, 2, 3]),
        bin_indices_sorted=jnp.array([0, 0, 0, 1]),
        bin_counts=jnp.array([3, 1]),
        n_low=0,
        n_bins_high=2,
        freqs=jnp.array([]),
        f_coarse=jnp.array([]),
        bin_widths=jnp.array([]),
        fine_spacing=1.0,
    )
    np.testing.assert_allclose(np.asarray(power_coarse), np.array([6.0, 4.0]))
    np.testing.assert_allclose(np.asarray(weights), np.array([3.0, 1.0]))
