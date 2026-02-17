from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .._jaxtypes import Complex, Float
from .._typecheck import runtime_typecheck
from ..datatypes.multivar_utils import _interp_frequency_indexed_array
from ..logger import logger
from .configs import TruePSDInput


def _unpack_true_psd(
    true_psd: TruePSDInput,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (freq, psd) pair from accepted true_psd formats."""
    if true_psd is None:
        return None, None

    if isinstance(true_psd, dict):
        freq = true_psd.get("freq")
        psd = true_psd.get("psd")
        if psd is None:
            raise ValueError(
                "true_psd dict must contain a 'psd' entry (optional 'freq')."
            )
        freq_arr = np.asarray(freq) if freq is not None else None
        return freq_arr, np.asarray(psd)

    if isinstance(true_psd, (tuple, list)) and len(true_psd) == 2:
        freq = np.asarray(true_psd[0]) if true_psd[0] is not None else None
        return freq, np.asarray(true_psd[1])

    return None, np.asarray(true_psd)


@runtime_typecheck
def _interp_psd_array(
    psd: Complex[np.ndarray, "f_src ..."] | Float[np.ndarray, "f_src ..."],
    freq_src: Float[np.ndarray, "f_src"],
    freq_tgt: Float[np.ndarray, "f_tgt"],
) -> Complex[np.ndarray, "f_tgt ..."] | Float[np.ndarray, "f_tgt ..."]:
    """Interpolate PSD arrays (real or complex) onto target frequencies."""
    return _interp_frequency_indexed_array(
        freq_src,
        freq_tgt,
        psd,
        sort_and_dedup=True,
    )


def _prepare_true_psd_for_freq(
    true_psd: TruePSDInput,
    freq_target: Optional[Float[np.ndarray, "f_tgt"]],
) -> Optional[np.ndarray]:
    """Resample supplied true PSD onto the target frequency grid."""
    if true_psd is None:
        return None

    freq_src, psd_array = _unpack_true_psd(true_psd)
    if psd_array is None:
        return None
    if freq_target is None:
        return np.asarray(psd_array)

    psd_array = np.asarray(psd_array)
    freq_target = np.asarray(freq_target)

    if freq_src is None:
        if psd_array.shape[0] == freq_target.size:
            return psd_array
        logger.warning(
            "true_psd length {} does not match target frequencies {}; assuming uniform spacing for interpolation.",
            psd_array.shape[0],
            freq_target.size,
        )
        freq_src = np.linspace(
            freq_target[0], freq_target[-1], psd_array.shape[0]
        )
    else:
        freq_src = np.asarray(freq_src)
        if freq_src.ndim != 1:
            raise ValueError(
                "true_psd frequency grid must be one-dimensional."
            )
        if freq_src.shape[0] != psd_array.shape[0]:
            raise ValueError(
                "true_psd frequency and value arrays must have matching lengths."
            )

    return _interp_psd_array(psd_array, freq_src, freq_target)


def _align_true_psd_to_freq(
    true_psd: TruePSDInput,
    processed_data,
) -> Optional[np.ndarray]:
    if true_psd is None:
        return None
    if processed_data is None:
        _, psd = _unpack_true_psd(true_psd)
        return None if psd is None else np.asarray(psd)

    from ..datatypes import Periodogram
    from ..datatypes.multivar import MultivarFFT

    if isinstance(processed_data, Periodogram):
        freq_target = np.asarray(processed_data.freqs)
    elif isinstance(processed_data, MultivarFFT):
        freq_target = np.asarray(processed_data.freq)
    else:
        _, psd = _unpack_true_psd(true_psd)
        return None if psd is None else np.asarray(psd)

    aligned = _prepare_true_psd_for_freq(true_psd, freq_target)
    if aligned is not None:
        return aligned
    _, psd = _unpack_true_psd(true_psd)
    return None if psd is None else np.asarray(psd)
