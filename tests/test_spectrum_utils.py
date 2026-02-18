import numpy as np
import pytest

from log_psplines.datatypes.multivar_utils import (
    sum_wishart_outer_products,
    U_to_Y,
    Y_to_S,
    wishart_u_to_psd,
)


def test_U_to_Y_matches_manual_product():
    u = np.array(
        [
            [[1.0 + 1.0j, 2.0 - 1.0j], [0.5 + 0.0j, -1.0 + 2.0j]],
            [[-1.0 + 0.5j, 0.0 + 1.0j], [2.0 - 2.0j, 1.0 + 0.0j]],
        ],
        dtype=np.complex128,
    )
    expected = np.stack([u_f @ u_f.conj().T for u_f in u], axis=0)
    result = U_to_Y(u)
    np.testing.assert_allclose(result, expected)


def test_U_to_Y_rejects_wrong_shape():
    with pytest.raises(ValueError):
        U_to_Y(np.zeros((2, 2)))


def test_sum_wishart_outer_products_matches_manual_sum():
    u_stack = np.array(
        [
            [[1.0 + 0.0j, 2.0 + 0.0j], [0.0 + 1.0j, 1.0 - 1.0j]],
            [[-1.0 + 0.5j, 0.0 + 0.0j], [2.0 + 0.0j, 0.5 + 0.0j]],
            [[0.0 + 1.0j, 0.0 - 1.0j], [1.0 + 0.0j, -2.0 + 0.0j]],
        ],
        dtype=np.complex128,
    )
    expected = sum(u @ u.conj().T for u in u_stack)
    result = sum_wishart_outer_products(u_stack)
    np.testing.assert_allclose(result, expected)


def test_sum_wishart_outer_products_rejects_wrong_shape():
    with pytest.raises(ValueError):
        sum_wishart_outer_products(np.zeros((2, 2)))


def test_Y_to_S_scales_and_broadcasts():
    Y = np.arange(8.0).reshape(2, 2, 2)
    result = Y_to_S(Y, 4, scaling_factor=2.5)
    expected = Y / 4.0 * 2.5
    np.testing.assert_allclose(result, expected)


def test_Y_to_S_with_Nh():
    Y = np.arange(8.0).reshape(2, 2, 2)
    result = Y_to_S(Y, 2, Nh=2)
    expected = Y / (2.0 * 2.0)
    np.testing.assert_allclose(result, expected)


def test_Y_to_S_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Y_to_S(np.zeros((2, 2)), 2)

    with pytest.raises((TypeError, ValueError)):
        Y_to_S(np.zeros((2, 2, 2)), np.array([1.0, 2.0, 3.0]))

    with pytest.raises(TypeError):
        Y_to_S(np.zeros((2, 2, 2)), 2, Nh=0.0)

    with pytest.raises(ValueError):
        Y_to_S(np.zeros((2, 2, 2)), 2, Nh=0)


def test_wishart_u_to_psd_matches_explicit_path():
    u = np.array(
        [
            [[1.0 + 0.5j, 0.0 + 1.0j], [2.0 + 0.0j, 1.0 + 0.0j]],
            [[0.0 + 0.0j, 1.0 + 1.0j], [1.5 + 0.0j, -0.5 + 0.0j]],
        ],
        dtype=np.complex128,
    )
    expected = Y_to_S(
        U_to_Y(u),
        2,
        scaling_factor=1.5,
        Nh=1,
    )
    result = wishart_u_to_psd(u, 2, scaling_factor=1.5, Nh=1)
    np.testing.assert_allclose(result, expected)


def test_wishart_u_to_psd_rejects_non_integer_Nh():
    u = np.ones((1, 1, 1), dtype=np.complex128)
    with pytest.raises(TypeError):
        wishart_u_to_psd(u, 2, Nh=0.5)
