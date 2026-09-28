"""Pytest's jaxtyping hook enforces array contracts across function arguments."""

import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import TypeCheckError

from log_psplines.data.spectral_utils import u_re_im_to_U
from log_psplines.models.spectrum import build_spline


def test_valid_array_contracts() -> None:
    basis = jnp.ones((3, 2))
    assert build_spline(basis, jnp.ones(2)).shape == (3,)
    assert u_re_im_to_U(np.zeros((3, 2, 2)), np.zeros((3, 2, 2))).shape == (
        3,
        2,
        2,
    )


def test_mismatched_basis_dimension_fails() -> None:
    with pytest.raises(TypeCheckError, match="weights"):
        build_spline(jnp.ones((3, 2)), jnp.ones(4))


def test_incorrect_rank_fails() -> None:
    with pytest.raises(TypeCheckError, match="basis"):
        build_spline(jnp.ones(3), jnp.ones(3))


def test_repeated_channel_dimension_fails() -> None:
    with pytest.raises(TypeCheckError, match="u_im"):
        u_re_im_to_U(np.zeros((3, 2, 2)), np.zeros((3, 2, 3)))
