import numpy as np
from scipy.signal import medfilt


def estimate_psd_with_lines(
    freq,
    Pxx,
    window_width_hz,
    threshold=10,
    fmin=20,
    fmax=2048,
):
    """
    Estimate a PSD model from a periodogram by identifying narrow lines within a given frequency range.

    Parameters
    ----------
    freq : 1D array_like
        Frequencies corresponding to each bin of the periodogram (Hz), sorted and roughly uniformly spaced.
    Pxx : 1D array_like
        Periodogram (power) values at each frequency in `freq`.
    window_width_hz : float
        Width (in Hz) of the running-median window—should exceed any expected line width but be
        smaller than the scale on which broadband noise changes (e.g. 4–16 Hz).
    threshold : float, optional
        A bin is marked as “line” if Pxx[i] > threshold * running_median[i]. Default is 10.
    fmin : float or None, optional
        Minimum frequency (Hz) in which to look for lines. If None, no lower bound is applied.
    fmax : float or None, optional
        Maximum frequency (Hz) in which to look for lines. If None, no upper bound is applied.

    Returns
    -------
    running_median : 1D ndarray
        The running-median PSD estimate (same length as `Pxx`).
    is_line_bin : 1D boolean ndarray
        Mask (length = len(freq)) that is True where a “line” is detected *within* [fmin, fmax].
    psd_model : 1D ndarray
        The final PSD model: equals `running_median` where no line is detected, and equals `Pxx` where lines are detected (within the specified band).
    line_details : list of tuples
        For each contiguous line interval within [fmin, fmax], a tuple:
          (f_start, f_end, f0, bandwidth, max_ratio)
        where
          • f_start    = frequency of the first bin in that interval
          • f_end      = frequency of the last bin in that interval
          • f0         = frequency at which (Pxx / running_median) is maximal inside the interval
          • bandwidth  = f_end − f_start
          • max_ratio  = maximum of (Pxx / running_median) over that interval

    Notes
    -----
    1. We first median‐filter `Pxx` over a fixed‐width window (in bins ≈ window_width_hz/df).
    2. We compute the ratio `Pxx / running_median`. Any bin outside [fmin, fmax] is forced to False.
    3. Contiguous runs of True in that mask yield “line intervals.”
    4. Within each interval, we record start/end frequencies, peak frequency, approximate width, and peak ratio.
    """
    freq = np.asarray(freq)
    Pxx = np.asarray(Pxx)
    N = len(freq)
    if len(Pxx) != N:
        raise ValueError("`freq` and `Pxx` must have the same length.")

    # Estimate bin width (assume approx. uniform)
    df = np.median(np.diff(freq))
    if not np.allclose(np.diff(freq), df, rtol=1e-3, atol=1e-6 * df):
        import warnings

        warnings.warn(
            "freq spacing not uniform. "
            f"Using median(df) = {df:.3e} Hz to size window."
        )

    # Determine how many bins correspond to window_width_hz
    half_bins = int(np.round((window_width_hz / df) / 2))
    kernel_size = max(1, 2 * half_bins + 1)  # force odd

    # Compute running-median via median filter
    running_median = medfilt(Pxx, kernel_size=kernel_size)

    # Safeguard against division by zero
    eps = np.finfo(float).tiny
    ratio = Pxx / (running_median + eps)

    # Build a mask for the specified frequency range
    in_range = np.ones(N, dtype=bool)
    if fmin is not None:
        in_range &= freq >= fmin
    if fmax is not None:
        in_range &= freq <= fmax

    # Identify “line” bins only if they lie within [fmin, fmax]
    is_line_bin = (ratio > threshold) & in_range

    # Construct final PSD model: Pxx where is_line_bin is True, else running_median
    psd_model = np.where(is_line_bin, Pxx, running_median)

    # Now find contiguous runs of True in is_line_bin
    line_details = []
    if is_line_bin.any():
        # Pad with False at both ends so transitions at the boundaries are captured
        padded = np.concatenate([[False], is_line_bin, [False]])
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == +1)[
            0
        ]  # index in padded where a line begins
        ends = np.where(diffs == -1)[0]  # index in padded where a line ends

        for s, e in zip(starts, ends):
            i0 = max(0, s - 1)
            i1 = min(N - 1, e - 1)
            f_start = freq[i0]
            f_end = freq[i1]
            bandwidth = f_end - f_start

            # Within [i0:i1], find the bin where ratio is maximal
            idx_peak = np.argmax(ratio[i0 : i1 + 1])  # relative to slice
            idx_peak_global = i0 + idx_peak
            f0 = freq[idx_peak_global]
            max_ratio = float(ratio[idx_peak_global])

            line_details.append((f_start, f_end, f0, bandwidth, max_ratio))

    return running_median, is_line_bin, psd_model, line_details
