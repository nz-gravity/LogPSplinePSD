"""Overplot spectral vs cholesky posterior PSDs in 3x3 matrix layout."""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-cache"))

import arviz as az
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_FS = 1.0

A1 = np.diag([0.4, 0.3, 0.2])
A2 = np.array(
    [
        [-0.2, 0.5, 0.0],
        [0.4, -0.1, 0.0],
        [0.0, 0.0, -0.1],
    ],
    dtype=np.float64,
)
VAR_COEFFS = np.array([A1, A2], dtype=np.float64)

SIGMA_VAL = 0.25
OFF_DIAG = 0.08
SIGMA = np.array(
    [
        [SIGMA_VAL, 0.0, OFF_DIAG],
        [0.0, SIGMA_VAL, OFF_DIAG],
        [OFF_DIAG, OFF_DIAG, SIGMA_VAL],
    ],
    dtype=np.float64,
)


def _extract_psd_quantiles(
    idata_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idata = az.from_netcdf(idata_path)
    if not hasattr(idata, "posterior_psd"):
        raise ValueError(f"{idata_path} has no posterior_psd.")
    group = idata.posterior_psd
    if "psd_matrix_real" not in group:
        raise ValueError(f"{idata_path} missing psd_matrix_real.")

    pcts = np.asarray(
        group["psd_matrix_real"].coords["percentile"].values, dtype=float
    )
    freq = np.asarray(
        group["psd_matrix_real"].coords["freq"].values, dtype=float
    )
    real = np.asarray(group["psd_matrix_real"].values, dtype=np.float64)
    imag = (
        np.asarray(group["psd_matrix_imag"].values, dtype=np.float64)
        if "psd_matrix_imag" in group
        else np.zeros_like(real)
    )

    def _grab(target: float) -> np.ndarray:
        idx = int(np.argmin(np.abs(pcts - target)))
        return real[idx] + 1j * imag[idx]

    return freq, _grab(5.0), _grab(50.0), _grab(95.0)


def _calculate_true_var_psd_hz(
    freqs_hz: np.ndarray,
    var_coeffs: np.ndarray,
    sigma: np.ndarray,
    *,
    fs: float = DEFAULT_FS,
) -> np.ndarray:
    freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
    ar_order, n_channels, _ = var_coeffs.shape
    omega = 2.0 * np.pi * freqs_hz / float(fs)
    psd = np.empty(
        (freqs_hz.shape[0], n_channels, n_channels), dtype=np.complex128
    )
    ident = np.eye(n_channels, dtype=np.complex128)

    for idx, w in enumerate(omega):
        a_f = ident.copy()
        for lag in range(1, ar_order + 1):
            a_f = a_f - var_coeffs[lag - 1] * np.exp(-1j * w * lag)
        h_f = np.linalg.inv(a_f)
        s_f = h_f @ sigma @ h_f.conj().T
        psd[idx] = (2.0 / float(fs)) * s_f

    if freqs_hz.size and np.isclose(freqs_hz[-1], fs / 2.0):
        psd[-1] = 0.5 * psd[-1]
    return 0.5 * (psd + np.swapaxes(psd.conj(), -1, -2))


def _panel_component(matrix: np.ndarray, i: int, j: int) -> np.ndarray:
    if i == j:
        return np.real(matrix[:, i, j])
    if i < j:
        return np.real(matrix[:, i, j])
    return np.imag(matrix[:, i, j])


def _panel_label(i: int, j: int) -> str:
    if i == j:
        return f"S_{i+1}{j+1}"
    if i < j:
        return f"Re S_{i+1}{j+1}"
    return f"Im S_{i+1}{j+1}"


def _plot(
    *,
    freq: np.ndarray,
    truth: np.ndarray | None,
    spectral_q05: np.ndarray,
    spectral_q50: np.ndarray,
    spectral_q95: np.ndarray,
    cholesky_q05: np.ndarray,
    cholesky_q50: np.ndarray,
    cholesky_q95: np.ndarray,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)

    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            y_s05 = _panel_component(spectral_q05, i, j)
            y_s50 = _panel_component(spectral_q50, i, j)
            y_s95 = _panel_component(spectral_q95, i, j)
            y_c05 = _panel_component(cholesky_q05, i, j)
            y_c50 = _panel_component(cholesky_q50, i, j)
            y_c95 = _panel_component(cholesky_q95, i, j)

            ax.fill_between(freq, y_s05, y_s95, color="#1f77b4", alpha=0.20)
            ax.plot(freq, y_s50, color="#1f77b4", lw=1.2)
            ax.fill_between(freq, y_c05, y_c95, color="#ff7f0e", alpha=0.18)
            ax.plot(freq, y_c50, color="#ff7f0e", lw=1.2)
            if truth is not None:
                y_true = _panel_component(truth, i, j)
                ax.plot(freq, y_true, color="black", lw=1.0)

            if i != j:
                ax.axhline(0.0, color="0.75", lw=0.8, ls="--")
            ax.set_title(_panel_label(i, j))
            ax.grid(alpha=0.15)
            if j == 0:
                ax.set_ylabel("Value")
            if i == 2:
                ax.set_xlabel("Frequency (Hz)")

            if i == 0 and j == 0:
                handles = [
                    plt.Line2D(
                        [0], [0], color="#1f77b4", lw=1.5, label="spectral q50"
                    ),
                    plt.Line2D(
                        [0], [0], color="#ff7f0e", lw=1.5, label="cholesky q50"
                    ),
                ]
                if truth is not None:
                    handles.append(
                        plt.Line2D(
                            [0], [0], color="black", lw=1.2, label="true PSD"
                        )
                    )
                ax.legend(handles=handles, fontsize=8, loc="upper right")

    fig.suptitle("Posterior PSD Overplot in 3x3 Matrix Layout", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overplot spectral/cholesky posterior PSD quantiles in 3x3 layout."
    )
    parser.add_argument("--spectral-idata", type=str, required=True)
    parser.add_argument("--cholesky-idata", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument(
        "--no-truth",
        action="store_true",
        help="Skip true PSD line (default includes true VAR(2) PSD).",
    )
    args = parser.parse_args()

    spectral_path = os.path.abspath(args.spectral_idata)
    cholesky_path = os.path.abspath(args.cholesky_idata)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    freq_s, s_q05, s_q50, s_q95 = _extract_psd_quantiles(spectral_path)
    freq_c, c_q05, c_q50, c_q95 = _extract_psd_quantiles(cholesky_path)
    if freq_s.shape != freq_c.shape or not np.allclose(freq_s, freq_c):
        raise ValueError(
            "Frequency grids differ between spectral/cholesky idata."
        )

    truth = None
    if not args.no_truth:
        truth = _calculate_true_var_psd_hz(
            freq_s, VAR_COEFFS, SIGMA, fs=DEFAULT_FS
        )

    _plot(
        freq=freq_s,
        truth=truth,
        spectral_q05=s_q05,
        spectral_q50=s_q50,
        spectral_q95=s_q95,
        cholesky_q05=c_q05,
        cholesky_q50=c_q50,
        cholesky_q95=c_q95,
        out_path=out_path,
    )
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
