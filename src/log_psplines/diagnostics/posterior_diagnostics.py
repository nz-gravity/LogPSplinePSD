"""Utilities for summarizing and visualizing MCMC diagnostics for PSD posteriors."""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def summarize_existing_mcmc_metrics(
    idata: az.InferenceData,
) -> Tuple[float, float, float]:
    """Summarize global convergence diagnostics already computed by ArviZ.

    Parameters
    ----------
    idata
        An :class:`arviz.InferenceData` object containing posterior draws.

    Returns
    -------
    tuple of float
        ``(max_rhat, min_ess, median_ess)`` across all parameters.
    """

    rhat = az.rhat(idata)
    ess = az.ess(idata)

    rhat_values = np.asarray(rhat.to_array())
    ess_values = np.asarray(ess.to_array())

    return (
        float(np.nanmax(rhat_values)),
        float(np.nanmin(ess_values)),
        float(np.nanmedian(ess_values)),
    )


def _prepare_psd_matrix(psd_samples: np.ndarray, p: int) -> np.ndarray:
    """Ensure PSD samples are represented as full matrices.

    Accepts either diagonal-only PSD samples with shape ``(n_samples, p, N)``
    or full cross-spectral matrices with shape ``(n_samples, p, p, N)``.
    Returns an array of shape ``(n_samples, p, p, N)``.
    """

    if psd_samples.ndim == 3:
        n_samples, _, N = psd_samples.shape
        psd_matrix = np.zeros(
            (n_samples, p, p, N),
            dtype=psd_samples.dtype,
        )
        for idx in range(p):
            psd_matrix[:, idx, idx, :] = psd_samples[:, idx, :]
        return psd_matrix

    if psd_samples.ndim == 4:
        # Already in (n_samples, p, p, N) layout
        if psd_samples.shape[1] == p and psd_samples.shape[2] == p:
            return psd_samples

        # Frequency axis immediately after samples: (n_samples, N, p, p)
        if psd_samples.shape[-1] == p and psd_samples.shape[-2] == p:
            return np.transpose(psd_samples, (0, 2, 3, 1))

    raise ValueError(
        "psd_samples must have shape (n_samples, p, N) or "
        "(n_samples, p, p, N)."
    )


def _band_mask(freqs: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    low, high = band
    return (freqs >= low) & (freqs <= high)


def compute_psd_functionals(
    psd_samples: np.ndarray,
    freqs: np.ndarray,
    bands: Sequence[Tuple[float, float]],
    channel_pairs: Sequence[Tuple[int, int]],
) -> Dict[str, np.ndarray]:
    """Compute variance, band power, and band-averaged coherence samples.

    Parameters
    ----------
    psd_samples
        PSD posterior samples with shape ``(n_samples, p, N)`` or
        ``(n_samples, p, p, N)``.
    freqs
        Frequency grid corresponding to the PSD evaluations.
    bands
        Iterable of ``(fmin, fmax)`` tuples describing frequency bands.
    channel_pairs
        Iterable of ``(i, j)`` channel index pairs for coherence calculations.

    Returns
    -------
    dict
        Dictionary containing posterior samples for ``variance`` (shape ``(n_samples, p)``),
        ``band_powers`` (shape ``(n_samples, p, n_bands)``), ``coherence``
        (shape ``(n_samples, n_pairs, n_bands)``), and coordinate metadata.
    """

    psd_samples = np.asarray(psd_samples)
    freqs = np.asarray(freqs)
    n_samples, p = psd_samples.shape[0], psd_samples.shape[1]
    psd_matrix = _prepare_psd_matrix(psd_samples, p)

    # Total variance per channel
    variance = np.trapezoid(
        psd_matrix[:, np.arange(p), np.arange(p), :],
        freqs,
        axis=-1,
    )

    # Band powers
    n_bands = len(bands)
    band_powers = np.zeros((n_samples, p, n_bands), dtype=np.float64)
    for band_idx, band in enumerate(bands):
        mask = _band_mask(freqs, band)
        band_width = (
            float(np.max(freqs[mask]) - np.min(freqs[mask]))
            if np.any(mask)
            else 0.0
        )
        if not np.any(mask) or band_width <= 0:
            continue
        band_powers[:, :, band_idx] = np.trapezoid(
            psd_matrix[:, np.arange(p), np.arange(p), :][:, :, mask],
            freqs[mask],
            axis=-1,
        )

    # Band-averaged coherence
    pair_labels: List[str] = []
    coherence = np.zeros(
        (n_samples, len(channel_pairs), n_bands), dtype=np.float64
    )
    for pair_idx, (i, j) in enumerate(channel_pairs):
        pair_labels.append(f"{i}-{j}")
        denom = psd_matrix[:, i, i, :] * psd_matrix[:, j, j, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            coh_spectrum = np.abs(psd_matrix[:, i, j, :]) ** 2 / denom
            coh_spectrum = np.nan_to_num(
                coh_spectrum, nan=0.0, posinf=0.0, neginf=0.0
            )

        for band_idx, band in enumerate(bands):
            mask = _band_mask(freqs, band)
            if not np.any(mask):
                continue
            band_span = freqs[mask]
            band_width = float(band_span[-1] - band_span[0])
            if band_width <= 0:
                continue
            avg = (
                np.trapezoid(coh_spectrum[:, mask], band_span, axis=-1)
                / band_width
            )
            coherence[:, pair_idx, band_idx] = avg

    return {
        "variance": variance,
        "band_powers": band_powers,
        "coherence": coherence,
        "band_labels": np.asarray([f"{low}-{high}" for low, high in bands]),
        "coherence_pairs": np.asarray(pair_labels),
    }


def _select_weight_vars(
    posterior: xr.Dataset, weight_subset: Optional[Sequence[str]]
) -> List[str]:
    if weight_subset is not None:
        return [var for var in weight_subset if var in posterior.data_vars]

    candidate_vars = [
        var
        for var in posterior.data_vars
        if "weight" in str(var).lower() or "spline" in str(var).lower()
    ]
    if not candidate_vars:
        return []

    max_vars = min(10, len(candidate_vars))
    rng = np.random.default_rng()
    selected = rng.choice(
        np.asarray(candidate_vars, dtype=object),
        size=max_vars,
        replace=False,
    )
    return [str(v) for v in selected]


def _collect_functional_idata(
    psd_functionals: Mapping[str, np.ndarray],
) -> Optional[az.InferenceData]:
    if not psd_functionals:
        return None

    variance = psd_functionals.get("variance")
    band_powers = psd_functionals.get("band_powers")
    coherence = psd_functionals.get("coherence")
    band_labels = psd_functionals.get("band_labels")
    pair_labels = psd_functionals.get("coherence_pairs")

    posterior_dict: Dict[str, np.ndarray] = {}
    coords: Dict[str, Iterable] = {}
    dims: Dict[str, List[str]] = {}

    if variance is not None:
        posterior_dict["variance"] = variance[np.newaxis, ...]
        dims["variance"] = ["channel"]
        coords["channel"] = np.arange(variance.shape[1])

    if band_powers is not None:
        posterior_dict["band_power"] = band_powers[np.newaxis, ...]
        dims["band_power"] = ["channel", "band"]
        coords.setdefault("channel", np.arange(band_powers.shape[1]))
        coords["band"] = (
            band_labels
            if band_labels is not None
            else np.arange(band_powers.shape[2])
        )

    if coherence is not None and coherence.size > 0:
        posterior_dict["band_coherence"] = coherence[np.newaxis, ...]
        dims["band_coherence"] = ["pair", "band"]
        coords["pair"] = (
            pair_labels
            if pair_labels is not None
            else np.arange(coherence.shape[1])
        )
        coords.setdefault(
            "band",
            (
                band_labels
                if band_labels is not None
                else np.arange(coherence.shape[2])
            ),
        )

    if not posterior_dict:
        return None

    return az.from_dict(posterior=posterior_dict, coords=coords, dims=dims)


def plot_subset_traces_and_ranks(
    idata: az.InferenceData,
    psd_functionals: Mapping[str, np.ndarray],
    weight_subset: Optional[Sequence[str]] = None,
) -> List[plt.Figure]:
    """Plot trace and rank summaries for selected variables and PSD functionals.

    Parameters
    ----------
    idata
        ArviZ inference data containing the sampled parameters.
    psd_functionals
        Output dictionary from :func:`compute_psd_functionals`.
    weight_subset
        Optional explicit list of weight variable names to plot. When ``None``, a random
        subset (up to 10) of detected spline/weight variables is used.

    Returns
    -------
    list of :class:`matplotlib.figure.Figure`
        Generated trace and rank plot figures.
    """

    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return []
    weight_vars = _select_weight_vars(posterior, weight_subset)

    hyper_vars = [
        var
        for var in posterior.data_vars
        if var not in weight_vars and posterior[var].ndim <= 1
    ]

    var_names = weight_vars + hyper_vars
    figs: List[plt.Figure] = []

    if var_names:
        figs.append(az.plot_trace(idata, var_names=var_names))
        figs.append(az.plot_rank(idata, var_names=var_names))

    func_idata = _collect_functional_idata(psd_functionals)
    if func_idata is not None:
        figs.append(az.plot_trace(func_idata))
        figs.append(az.plot_rank(func_idata))

    return figs
