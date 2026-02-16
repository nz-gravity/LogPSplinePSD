import numpy as np
import pytest

from log_psplines.datatypes.multivar_utils import _get_coherence


def test_get_coherence_clips_values_to_unit_interval():
    psd = np.zeros((1, 2, 2), dtype=np.complex128)
    psd[0, 0, 0] = 1.0
    psd[0, 1, 1] = 1.0
    psd[0, 0, 1] = 2.0
    psd[0, 1, 0] = 2.0

    coherence = _get_coherence(psd)

    assert coherence[0, 0, 1] == pytest.approx(1.0)
    assert coherence[0, 1, 0] == pytest.approx(1.0)
    assert np.min(coherence) >= 0.0
    assert np.max(coherence) <= 1.0


def test_get_coherence_rejects_nonsquare_input():
    bad_psd = np.ones((4, 2, 3), dtype=np.complex128)

    with pytest.raises(ValueError, match="square channel dimensions"):
        _get_coherence(bad_psd)
