import numpy as np

from log_psplines.datatypes import MultivarFFT
from log_psplines.datatypes.multivar_utils import u_re_im_to_U


def test_u_re_im_to_U_and_property():
    u_re = np.array(
        [
            [[1.0, 0.5], [0.5, 2.0]],
            [[0.1, 0.2], [0.3, 0.4]],
        ],
        dtype=np.float64,
    )
    u_im = np.array(
        [
            [[0.0, 0.25], [-0.25, 0.0]],
            [[0.0, -0.1], [0.1, 0.0]],
        ],
        dtype=np.float64,
    )
    expected = (u_re + 1j * u_im).astype(np.complex128)

    U = u_re_im_to_U(u_re, u_im)
    assert U.dtype == np.complex128
    np.testing.assert_allclose(U, expected)

    freq = np.array([1.0, 2.0], dtype=np.float64)
    fft = MultivarFFT(u_re=u_re, u_im=u_im, freq=freq, N=2, p=2, Nb=1)
    np.testing.assert_allclose(fft.U, expected)
