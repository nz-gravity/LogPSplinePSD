import numpy as np

from log_psplines.datatypes.multivar_utils import (
    sum_wishart_outer_products,
    U_to_Y,
    Y_to_S,
    wishart_u_to_psd,
)


def test_U_to_Y_matches_manual():
    rng = np.random.default_rng(0)
    u = rng.standard_normal((5, 3, 3)) + 1j * rng.standard_normal((5, 3, 3))
    Y_expected = np.empty((5, 3, 3), dtype=np.complex128)
    for idx in range(5):
        U = u[idx]
        Y_expected[idx] = U @ U.conj().T

    Y = U_to_Y(u)
    np.testing.assert_allclose(Y, Y_expected)


def test_sum_wishart_outer_products():
    rng = np.random.default_rng(1)
    u_stack = rng.standard_normal((4, 2, 2)) + 1j * rng.standard_normal(
        (4, 2, 2)
    )
    expected = np.zeros((2, 2), dtype=np.complex128)
    for U in u_stack:
        expected += U @ U.conj().T

    result = sum_wishart_outer_products(u_stack)
    np.testing.assert_allclose(result, expected)


def test_Y_to_S_scaling_and_normalisation():
    rng = np.random.default_rng(2)
    Y = rng.standard_normal((6, 2, 2)) + 1j * rng.standard_normal((6, 2, 2))
    Y = Y + np.conj(np.transpose(Y, axes=(0, 2, 1)))  # force Hermitian
    Y *= 0.5
    Nb = 4
    Nh = 3
    psd = Y_to_S(Y, Nb, scaling_factor=2.5, Nh=Nh)
    manual = (Y / (Nb * Nh)) * 2.5
    np.testing.assert_allclose(psd, manual)


def test_wishart_u_to_psd_matches_two_step():
    rng = np.random.default_rng(3)
    u = rng.standard_normal((4, 3, 3)) + 1j * rng.standard_normal((4, 3, 3))
    Nb = 5
    Nh = 2
    scaling = 1.7

    psd_direct = wishart_u_to_psd(u, Nb, scaling_factor=scaling, Nh=Nh)
    Y = U_to_Y(u)
    psd_two_step = Y_to_S(Y, Nb, scaling_factor=scaling, Nh=Nh)
    np.testing.assert_allclose(psd_direct, psd_two_step)
