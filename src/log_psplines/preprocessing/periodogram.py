"""Stationary periodograms and Wishart sufficient statistics.

Time-frequency power preprocessing lives in ``moving_periodogram.py`` and
``wdm.py``; its scalar likelihood and inference live in ``inference.power``.
"""

import numpy as np
from scipy.signal import csd, welch, windows
from scipy.signal import detrend as signal_detrend

from log_psplines.data.spectral import EmpiricalPSD, WishartData
from log_psplines.data.spectral_utils import (
    U_to_Y,
    Y_to_S,
    Y_to_U,
    _get_coherence,
)


def compute_fft(
    x: np.ndarray,
    fs: float = 1.0,
    fmin: float | None = None,
    fmax: float | None = None,
    scaling_factor: float | None = 1.0,
    channel_stds: np.ndarray | None = None,
    window: str | tuple | None = None,
) -> "WishartData":
    """Compute FFT and Wishart replicates with a single (full-length) block."""
    return compute_wishart(
        x,
        fs=fs,
        Nb=1,
        fmin=fmin,
        fmax=fmax,
        scaling_factor=scaling_factor,
        channel_stds=channel_stds,
        window=window,
    )


def compute_wishart(
    x: np.ndarray,
    fs: float,
    Nb: int,
    fmin: float | None = None,
    fmax: float | None = None,
    scaling_factor: float | None = 1.0,
    channel_stds: np.ndarray | None = None,
    window: str | tuple | None = None,
    detrend: str | bool = "constant",
    wishart_floor_fraction: float | None = None,
) -> "WishartData":
    """
    Compute block-averaged (Wishart) FFT statistics for multivariate series.

    Parameters
    ----------
    x : np.ndarray
        Input time series (n, p)
    fs : float
        Sampling frequency
    Nb : int
        Number of non-overlapping blocks to average. Must divide n.
    fmin, fmax : float, optional
        Optional frequency truncation applied after blocking.
    scaling_factor : float, optional
        PSD scaling factor carried through sampling.
    window : str or tuple, optional
        Taper applied to each block before the FFT. Defaults to
        ``None`` (rectangular window).
    detrend : {"constant", "linear"} or bool, optional
        Per-block detrending applied before tapering. ``"constant"``
        subtracts the block mean, ``"linear"`` removes a linear trend,
        and ``False`` disables detrending.
    wishart_floor_fraction : float, optional
        If set, floor the eigenvalues of each Wishart matrix at this
        fraction of the median trace.  Useful for spectra with
        deterministic nulls (e.g. LISA TDI transfer functions) where
        the spectral matrix becomes near-singular.  A value of 1e-6
        is recommended.
    """
    if isinstance(Nb, bool) or not isinstance(Nb, (int, np.integer)):
        raise TypeError("Nb must be a positive integer.")
    Nb = int(Nb)
    if Nb < 1:
        raise ValueError("Nb must be positive.")

    n, p = x.shape
    if n % Nb != 0:
        raise ValueError(f"n={n} must be divisible by Nb={Nb}.")

    Lb = n // Nb
    if Lb <= p:
        raise ValueError(
            "Block length must exceed number of channels for FFT stability."
        )

    blocks = x.reshape(Nb, Lb, p)
    if detrend in (False, None):
        pass
    elif detrend == "constant":
        blocks = blocks - np.mean(blocks, axis=1, keepdims=True)
    elif detrend == "linear":
        blocks = signal_detrend(blocks, axis=1, type="linear")
    else:
        raise ValueError(
            "detrend must be one of False, None, 'constant', or 'linear'."
        )

    if window is None:
        taper = np.ones(Lb, dtype=np.float64)
    else:
        taper = windows.get_window(window, Lb, fftbins=True)
        taper = np.asarray(taper, dtype=np.float64)
    taper_energy = float(np.sum(taper**2))
    if taper_energy <= 0.0:
        raise ValueError("Window energy must be positive.")
    taper_sum = float(np.sum(taper))
    # NENBW = Lb * Σw² / (Σw)²  (Heinzel et al. 2002, eq. 21)
    # Rect → 1.0;  Hann → 1.5;  Tukey(0.1) → 1.04.
    # The sampler divides the Whittle log-likelihood by this factor to
    # account for the reduced effective DOF per frequency bin.
    enbw = float(Lb) * taper_energy / (taper_sum**2)
    blocks = blocks * taper[None, :, None]

    block_ffts = np.fft.rfft(blocks, axis=1)
    freq = np.fft.rfftfreq(Lb, 1 / fs)
    # Drop the zero-frequency bin for numerical stability
    block_ffts = block_ffts[:, 1:, :]
    freq = freq[1:]
    if freq.size == 0:
        raise ValueError(
            "Block length too small to retain positive frequencies."
        )

    scale = np.full(freq.shape, 2.0 / (taper_energy * fs), dtype=np.float64)
    if Lb % 2 == 0 and scale.size > 0:
        scale[-1] = 1.0 / (taper_energy * fs)
    sqrt_scale = np.sqrt(scale, dtype=np.float64)[None, :, None]
    block_ffts = block_ffts * sqrt_scale
    # Convert from a "per-block-periodogram" normalisation to the Whittle
    # convention that keeps an explicit 1/T in the likelihood. This ensures
    # the likelihood uses the observation duration explicitly while PSD
    # conversions remain unchanged (see Y_to_S(duration=...)).
    duration = float(Lb) / float(fs)
    sqrt_duration = float(np.sqrt(np.asarray(duration, dtype=np.float64)))
    block_ffts = block_ffts * sqrt_duration

    if fmin is not None or fmax is not None:
        freq_min = float(freq[0])
        freq_max = float(freq[-1])
        fmin_eff = freq_min if fmin is None else float(fmin)
        fmax_eff = freq_max if fmax is None else float(fmax)

        fmin_eff = min(max(fmin_eff, freq_min), freq_max)
        fmax_eff = min(max(fmax_eff, freq_min), freq_max)
        if fmax_eff < fmin_eff:
            fmax_eff = fmin_eff

        freq_mask = (freq >= fmin_eff) & (freq <= fmax_eff)
        if not np.any(freq_mask):
            raise ValueError(
                "Frequency truncation removed all bins; check fmin/fmax."
            )
        freq = freq[freq_mask]
        block_ffts = block_ffts[:, freq_mask, :]

    Y = np.einsum("bnc,bnd->ncd", block_ffts, np.conj(block_ffts))

    # Regularize near-singular Wishart matrices (e.g. LISA TDI transfer
    # nulls where the spectral matrix drops to near-zero).  We floor the
    # eigenvalues of Y at a small fraction of each bin's OWN trace so that
    # downstream Cholesky / likelihood computations remain stable without
    # discarding any frequency bins or corrupting the off-diagonal
    # structure at low-power frequencies.
    #
    # Using per-frequency trace (not global median) is critical: the PSD
    # can span 6+ orders of magnitude, so a global floor derived from
    # the median trace destroys the eigenvalue structure (and hence
    # coherence) at low-power frequencies far from the nulls.
    if wishart_floor_fraction is not None:
        trace_per_bin = np.real(np.trace(Y, axis1=-2, axis2=-1))
        wishart_floor = float(wishart_floor_fraction) * trace_per_bin
        lam, v = np.linalg.eigh(Y)
        lam_real = lam.real
        # Floor each bin's eigenvalues against its own trace-based threshold
        needs_clip = lam_real < wishart_floor[:, np.newaxis]
        if np.any(needs_clip):
            lam_real = np.where(
                needs_clip,
                wishart_floor[:, np.newaxis],
                lam_real,
            )
            Y = (v * lam_real[:, np.newaxis, :]) @ np.conj(
                np.swapaxes(v, -2, -1)
            )

    U = Y_to_U(Y)
    u_re = U.real
    u_im = U.imag
    raw_psd = Y_to_S(
        U_to_Y(U),
        Nb=Nb,
        duration=duration,
        scaling_factor=float(scaling_factor or 1.0),
    )

    return WishartData(
        freq=freq,
        N=len(freq),
        p=p,
        u_re=u_re,
        u_im=u_im,
        raw_psd=raw_psd,
        raw_freq=freq,
        Nb=Nb,
        scaling_factor=scaling_factor,
        fs=fs,
        duration=duration,
        enbw=enbw,
        channel_stds=(
            None
            if channel_stds is None
            else np.asarray(channel_stds, dtype=np.float64)
        ),
    )


def empirical_spectrum(
    data: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
    window: str = "hann",
    detrend: str | bool = "constant",
) -> "EmpiricalPSD":
    p = data.shape[1]

    if nperseg is None:
        # Use half or full data length depending on total size
        n = data.shape[0]
        nperseg = n if n <= 512 else n // 2
    if noverlap is None:
        noverlap = nperseg // 2

    # --- auto spectra ---
    psds: list[np.ndarray] = []
    f_ref: np.ndarray | None = None
    for i in range(p):
        f, Pxx = welch(
            data[:, i],
            fs=fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=True,
            detrend=detrend,
            scaling="density",
        )
        psds.append(Pxx)
        if f_ref is None:
            f_ref = f
    if f_ref is None:
        raise ValueError("Failed to compute Welch frequencies.")
    psd_auto = np.stack(psds, axis=1)  # (N, p)

    # --- full CSD matrix ---
    S = np.zeros((len(f_ref), p, p), dtype=complex)
    for i in range(p):
        S[:, i, i] = psd_auto[:, i]
        for j in range(i + 1, p):
            _, Sij = csd(
                data[:, i],
                data[:, j],
                fs=fs,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                return_onesided=True,
                detrend=detrend,
                scaling="density",
            )
            S[:, i, j] = Sij
            S[:, j, i] = np.conj(Sij)

    coh = _get_coherence(S)
    return EmpiricalPSD(freq=f_ref, psd=S, coherence=coh)
