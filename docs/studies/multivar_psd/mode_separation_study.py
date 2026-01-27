"""Mode-separation experiments for diagnosing multivariate blocked NUTS.

This script creates three synthetic 3-channel datasets that differ mainly in
the separation of the 2nd/3rd eigenvalues of the true spectral matrix S(f).

Goal
----
Turn the eigenvalue separation plot into a predictive diagnostic:
  - well-separated (r23 << 1): block 3 should behave well
  - bunched (r23 ~ 1): block 3 should become pathological

Outputs (per dataset)
---------------------
- `diagnostics_mode_sep.json`: small machine-readable summary
- `mode_sep_summary.csv`: one-line-per-dataset summary
- standard `run_mcmc` outputs (InferenceData, plots, etc.)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import arviz as az
import jax
import numpy as np

from log_psplines.logger import logger, set_level
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc


@dataclass(frozen=True)
class StudyConfig:
    n_time: int = 1024
    fs: float = 1.0
    seed: int = 0
    n_knots: int = 7
    n_samples: int = 400
    n_warmup: int = 400
    num_chains: int = 4
    n_time_blocks: int = 8
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    dense_mass: bool = True
    alpha_phi: float = 1.0
    beta_phi: float = 1.0
    alpha_delta: float = 1e-4
    beta_delta: float = 1e-4
    alpha_phi_theta: float | None = None
    beta_phi_theta: float | None = None
    init_from_vi: bool = True
    vi_steps: int = 5000
    vi_lr: float = 1e-2
    vi_guide: str | None = None


def _hermitianize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.complex128)
    return 0.5 * (mat + np.swapaxes(mat.conj(), -1, -2))


def _make_orthogonal(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(dim, dim))
    q, _ = np.linalg.qr(a)
    # Fix sign to make deterministic-ish
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def _freq_grid(n_time: int, fs: float) -> np.ndarray:
    freq = np.fft.rfftfreq(n_time, d=1.0 / fs)[1:]
    if freq.size == 0:
        raise ValueError("n_time must be >= 4 to retain positive frequencies.")
    return freq


def _eig_ratio_r23(psd: np.ndarray) -> np.ndarray:
    """Compute r23(f) = λ3/λ2 for Hermitian PSD matrices, sorted descending."""
    psd = _hermitianize(psd)
    eigvals = np.linalg.eigvalsh(psd).real  # ascending
    eigvals = np.maximum(eigvals, 0.0)
    eigvals_desc = eigvals[:, ::-1]
    lam2 = eigvals_desc[:, 1]
    lam3 = eigvals_desc[:, 2]
    return lam3 / (lam2 + 1e-12)


def _summarize_ratio(ratio: np.ndarray, *, threshold: float = 0.8) -> Dict[str, float]:
    ratio = np.asarray(ratio, dtype=float)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return {"r23_p50": float("nan"), "r23_frac_gt": float("nan")}
    return {
        "r23_p50": float(np.median(ratio)),
        "r23_frac_gt": float(np.mean(ratio > float(threshold))),
    }


def _log_bump(freq: np.ndarray, *, f0: float, sigma_log: float) -> np.ndarray:
    freq = np.asarray(freq, dtype=float)
    logf = np.log(np.maximum(freq, 1e-12))
    return np.exp(-0.5 * ((logf - np.log(f0)) / float(sigma_log)) ** 2)


def _build_psd_from_eigs(
    freq: np.ndarray, q: np.ndarray, eigvals: np.ndarray
) -> np.ndarray:
    """Return PSD S(f) = Q diag(eigvals(f)) Q^T with Q orthogonal (real)."""
    freq = np.asarray(freq, dtype=float)
    q = np.asarray(q, dtype=float)
    eigvals = np.asarray(eigvals, dtype=float)
    if eigvals.shape != (freq.size, q.shape[0]):
        raise ValueError(
            f"eigvals must have shape (n_freq, dim)={(freq.size, q.shape[0])}, got {eigvals.shape}."
        )
    s = np.zeros((freq.size, q.shape[0], q.shape[0]), dtype=np.complex128)
    for k in range(freq.size):
        d = np.diag(eigvals[k])
        s[k] = q @ d @ q.T
    return s


def simulate_from_one_sided_psd(
    *,
    psd: np.ndarray,  # (n_freq, dim, dim), one-sided, excludes DC
    fs: float,
    n_time: int,
    seed: int,
) -> np.ndarray:
    """Simulate a real multivariate Gaussian series matching the PSD convention.

    The PSD convention matches `VARMAData.get_periodogram`:
      periodogram(f_k) = scale_k * FFT_k FFT_k^H, with scale_k = 2/(N*fs)
      except the Nyquist bin where scale_k = 1/(N*fs).
    """
    psd = _hermitianize(psd)
    n_freq, dim, _ = psd.shape
    expected_n_freq = n_time // 2
    if n_freq != expected_n_freq:
        raise ValueError(
            f"psd must have n_freq=n_time//2={expected_n_freq}, got {n_freq}."
        )
    rng = np.random.default_rng(seed)

    # rfft bins: [0..N/2], complex
    spec = np.zeros((n_time // 2 + 1, dim), dtype=np.complex128)
    spec[0] = 0.0

    n = float(n_time)
    for k in range(1, n_time // 2 + 1):
        idx = k - 1  # psd excludes DC
        cov = np.asarray(psd[idx], dtype=np.complex128)
        cov = _hermitianize(cov)
        # Numeric jitter in case of tiny negative eigenvalues
        w, v = np.linalg.eigh(cov)
        w = np.clip(w.real, a_min=0.0, a_max=None)
        cov = (v * w[None, :]) @ v.conj().T

        if k == n_time // 2 and n_time % 2 == 0:
            # Nyquist bin: real-valued coefficient for real time series.
            scale = n * float(fs)
            cov_fft = scale * cov.real
            cov_fft = 0.5 * (cov_fft + cov_fft.T)
            vec = rng.multivariate_normal(np.zeros(dim), cov_fft)
            spec[k] = vec.astype(np.float64)
            continue

        # Non-Nyquist positive frequencies (one-sided doubling)
        scale = n * float(fs) / 2.0
        cov_fft = scale * cov
        chol = np.linalg.cholesky(cov_fft + 1e-12 * np.eye(dim))
        z = (rng.normal(size=dim) + 1j * rng.normal(size=dim)) / np.sqrt(2.0)
        spec[k] = chol @ z

    x = np.fft.irfft(spec, n=n_time, axis=0)
    x = np.asarray(x, dtype=np.float64)
    x -= np.mean(x, axis=0, keepdims=True)
    return x


def _block3_var_names(posterior_vars: Iterable[str]) -> list[str]:
    out = []
    for name in posterior_vars:
        if name in {"delta_2", "phi_delta_2"}:
            out.append(name)
            continue
        if name.startswith("weights_delta_2"):
            out.append(name)
            continue
        if name.startswith("delta_theta_re_2_") or name.startswith("delta_theta_im_2_"):
            out.append(name)
            continue
        if name.startswith("phi_theta_re_2_") or name.startswith("phi_theta_im_2_"):
            out.append(name)
            continue
        if name.startswith("weights_theta_re_2_") or name.startswith("weights_theta_im_2_"):
            out.append(name)
            continue
    return sorted(set(out))


def _summarize_block3_sampling(
    idata: az.InferenceData, *, max_tree_depth: int
) -> Dict[str, float]:
    if not hasattr(idata, "sample_stats"):
        return {}
    stats = idata.sample_stats
    out: Dict[str, float] = {}

    max_steps = (2**int(max_tree_depth)) - 1
    steps_key = "num_steps_channel_2"
    if steps_key in stats:
        steps = np.asarray(stats[steps_key].values, dtype=float)
        out["block3_frac_max_tree_depth"] = float(np.mean(steps == max_steps))
        out["block3_num_steps_median"] = float(np.median(steps))

    step_key = "step_size_channel_2"
    if step_key in stats:
        step = np.asarray(stats[step_key].values, dtype=float)
        step = step[np.isfinite(step)]
        out["block3_step_size_median"] = float(np.median(step)) if step.size else float("nan")

    div_key = "diverging_channel_2"
    if div_key in stats:
        div = np.asarray(stats[div_key].values, dtype=float)
        div = div[np.isfinite(div)]
        out["block3_divergence_frac"] = float(np.mean(div > 0.0)) if div.size else float("nan")

    if not hasattr(idata, "posterior"):
        return out

    var_names = _block3_var_names(idata.posterior.data_vars.keys())
    out["block3_n_params"] = float(len(var_names))
    if not var_names:
        return out

    try:
        summ = az.summary(idata, var_names=var_names, round_to=None)
        if "r_hat" in summ:
            r = np.asarray(summ["r_hat"].values, dtype=float)
            r = r[np.isfinite(r)]
            out["block3_rhat_max"] = float(np.max(r)) if r.size else float("nan")
        if "ess_bulk" in summ:
            e = np.asarray(summ["ess_bulk"].values, dtype=float)
            e = e[np.isfinite(e)]
            out["block3_ess_bulk_min"] = float(np.min(e)) if e.size else float("nan")
    except Exception as exc:
        logger.warning(f"Could not compute block3 summary via ArviZ: {exc}")

    return out


def _write_csv(path: Path, rows: list[Dict[str, float]]) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row.get(k, "")) for k in keys))
    path.write_text("\n".join(lines) + "\n")


def _case_psd_eigs(freq: np.ndarray, case: str) -> np.ndarray:
    """Return eigenvalues (n_freq, 3) for the requested case."""
    freq = np.asarray(freq, dtype=float)
    f0 = float(freq[len(freq) // 6])
    f1 = float(freq[len(freq) // 2])

    low = 40.0 / (1.0 + (freq / max(f0, 1e-6)) ** 4) + 2.0
    bump = 12.0 * _log_bump(freq, f0=f1, sigma_log=0.45) + 4.0
    high = 1.0 + 1.6 * (freq / max(f1, 1e-6)) ** 4 / (1.0 + (freq / max(f1, 1e-6)) ** 4)

    if case == "separated":
        eig2 = bump
        eig3 = high
    elif case == "bunched":
        # Make eig2 and eig3 nearly identical (r23 ~ 1).
        eig2 = bump
        eig3 = 0.92 * bump + 0.08 * high
    elif case == "force_separation":
        # Start from bunched and then deflate the 3rd eigenvalue away from the 2nd.
        eig2 = bump
        eig3 = 0.3 * bump
    else:
        raise ValueError(f"Unknown case {case!r}")

    eig1 = low + 0.2 * bump
    # Enforce ordering eig1 >= eig2 >= eig3 to keep r23 interpretable.
    eig2 = np.minimum(eig2, eig1 - 1e-6)
    eig3 = np.minimum(eig3, eig2 - 1e-6)
    eigvals = np.stack([eig1, eig2, eig3], axis=1)
    eigvals = np.maximum(eigvals, 1e-8)
    return eigvals


def run_case(
    *,
    cfg: StudyConfig,
    outdir: Path,
    case: str,
) -> Dict[str, float]:
    outdir.mkdir(parents=True, exist_ok=True)
    freq = _freq_grid(cfg.n_time, cfg.fs)
    q = _make_orthogonal(3, seed=cfg.seed + 123)
    eigvals = _case_psd_eigs(freq, case)
    true_psd = _build_psd_from_eigs(freq, q, eigvals)

    ratio = _eig_ratio_r23(true_psd)
    ratio_summary = _summarize_ratio(ratio, threshold=0.8)
    logger.info(
        f"{case}: r23 p50={ratio_summary['r23_p50']:.3f}, frac(r23>0.8)={ratio_summary['r23_frac_gt']:.3f}"
    )

    if cfg.n_time_blocks < 3:
        logger.warning(
            f"n_time_blocks={cfg.n_time_blocks} < n_channels=3; Wishart matrices are rank-deficient "
            "and eigenvalue ratio diagnostics (and geometry) will be misleading."
        )

    y = simulate_from_one_sided_psd(psd=true_psd, fs=cfg.fs, n_time=cfg.n_time, seed=cfg.seed)
    t = np.arange(cfg.n_time, dtype=float) / float(cfg.fs)
    ts = MultivariateTimeseries(t=t, y=y)

    # Provide true_psd on the block-frequency grid to avoid the interpolation
    # fallback warning and to keep preprocessing diagnostics consistent.
    block_len = int(cfg.n_time // cfg.n_time_blocks)
    freq_block = _freq_grid(block_len, cfg.fs)
    eigvals_block = _case_psd_eigs(freq_block, case)
    true_psd_block = _build_psd_from_eigs(freq_block, q, eigvals_block)

    idata = run_mcmc(
        data=ts,
        sampler="multivar_blocked_nuts",
        n_knots=cfg.n_knots,
        degree=2,
        diffMatrixOrder=2,
        n_samples=cfg.n_samples,
        n_warmup=cfg.n_warmup,
        num_chains=cfg.num_chains,
        outdir=str(outdir),
        verbose=True,
        n_time_blocks=cfg.n_time_blocks,
        target_accept_prob=cfg.target_accept_prob,
        max_tree_depth=cfg.max_tree_depth,
        dense_mass=cfg.dense_mass,
        alpha_phi=cfg.alpha_phi,
        beta_phi=cfg.beta_phi,
        alpha_delta=cfg.alpha_delta,
        beta_delta=cfg.beta_delta,
        alpha_phi_theta=cfg.alpha_phi_theta,
        beta_phi_theta=cfg.beta_phi_theta,
        init_from_vi=cfg.init_from_vi,
        vi_steps=cfg.vi_steps,
        vi_lr=cfg.vi_lr,
        vi_guide=cfg.vi_guide,
        compute_psis=False,
        vi_psd_max_draws=32,
        posterior_psd_max_draws=32,
        compute_coherence_quantiles=True,
        true_psd=(freq_block, true_psd_block),
        save_preprocessing_plots=True,
    )

    block3 = _summarize_block3_sampling(idata, max_tree_depth=cfg.max_tree_depth)
    row: Dict[str, float] = {
        "case": case,
        "n_time_blocks": float(cfg.n_time_blocks),
        "alpha_phi": float(cfg.alpha_phi),
        "beta_phi": float(cfg.beta_phi),
        "alpha_delta": float(cfg.alpha_delta),
        "beta_delta": float(cfg.beta_delta),
        "init_from_vi": float(int(cfg.init_from_vi)),
        "vi_steps": float(cfg.vi_steps),
        "vi_lr": float(cfg.vi_lr),
        **ratio_summary,
        **block3,
    }

    (outdir / "diagnostics_mode_sep.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mode-separation experiments for multivariate blocked NUTS"
    )
    parser.add_argument("--out", type=str, default="out_mode_sep")
    parser.add_argument("--n-time", type=int, default=1024)
    parser.add_argument("--fs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--knots", type=int, default=7)
    parser.add_argument("--samples", type=int, default=400)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--time-blocks", type=int, default=2)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--target-accept", type=float, default=0.8)
    parser.add_argument("--alpha-phi", type=float, default=1.0)
    parser.add_argument("--beta-phi", type=float, default=1.0)
    parser.add_argument("--alpha-delta", type=float, default=1e-4)
    parser.add_argument("--beta-delta", type=float, default=1e-4)
    parser.add_argument("--alpha-phi-theta", type=float, default=None)
    parser.add_argument("--beta-phi-theta", type=float, default=None)
    parser.add_argument("--init-from-vi", action="store_true", help="Use VI to initialise NUTS (default).")
    parser.add_argument("--no-init-from-vi", action="store_true", help="Disable VI init and use default NUTS init.")
    parser.add_argument("--vi-steps", type=int, default=5000)
    parser.add_argument("--vi-lr", type=float, default=1e-2)
    parser.add_argument("--vi-guide", type=str, default=None)
    args = parser.parse_args()

    set_level("DEBUG")
    logger.info(f"JAX devices: {jax.devices()}")

    if args.init_from_vi and args.no_init_from_vi:
        raise ValueError("Choose at most one of --init-from-vi / --no-init-from-vi.")
    init_from_vi = True
    if args.no_init_from_vi:
        init_from_vi = False

    cfg = StudyConfig(
        n_time=int(args.n_time),
        fs=float(args.fs),
        seed=int(args.seed),
        n_knots=int(args.knots),
        n_samples=int(args.samples),
        n_warmup=int(args.warmup),
        num_chains=int(args.chains),
        n_time_blocks=int(args.time_blocks),
        target_accept_prob=float(args.target_accept),
        max_tree_depth=int(args.max_tree_depth),
        alpha_phi=float(args.alpha_phi),
        beta_phi=float(args.beta_phi),
        alpha_delta=float(args.alpha_delta),
        beta_delta=float(args.beta_delta),
        alpha_phi_theta=(None if args.alpha_phi_theta is None else float(args.alpha_phi_theta)),
        beta_phi_theta=(None if args.beta_phi_theta is None else float(args.beta_phi_theta)),
        init_from_vi=init_from_vi,
        vi_steps=int(args.vi_steps),
        vi_lr=float(args.vi_lr),
        vi_guide=(None if args.vi_guide is None or args.vi_guide.strip() == "" else str(args.vi_guide)),
    )

    here = Path(__file__).resolve().parent
    root_out = here / str(args.out) / (
        f"seed_{cfg.seed}_N{cfg.n_time}_K{cfg.n_knots}_B{cfg.n_time_blocks}"
    )
    root_out.mkdir(parents=True, exist_ok=True)
    (root_out / "study_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n"
    )

    rows = []
    for case in ("separated", "force_separation", "bunched"):
        rows.append(
            run_case(cfg=cfg, outdir=root_out / case, case=case)
        )
    _write_csv(root_out / "mode_sep_summary.csv", rows)
    logger.info(f"Wrote summary to {root_out / 'mode_sep_summary.csv'}")


if __name__ == "__main__":
    main()
