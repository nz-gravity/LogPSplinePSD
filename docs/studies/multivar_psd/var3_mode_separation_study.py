"""VAR-based mode-separation study for blocked multivariate NUTS.

This is the VAR/VARMA analogue of `mode_separation_study.py`. Instead of
specifying a target PSD directly, we generate a 3D VAR(2) process whose
transfer function is diagonal in an orthogonal eigenbasis, then rotate back to
the observation basis. This lets us build datasets with controlled spectral
eigenvalue separation while keeping a realistic time-domain simulator and an
analytic true PSD via `VARMAData`.

Cases
-----
- separated: distinct modes (low / mid / high-ish)
- bunched: modes 2 and 3 identical (λ2 ≈ λ3 across most of the band)
- force_separation: start from bunched, then deflate mode 3 variance
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

from log_psplines.example_datasets.varma_data import VARMAData
from log_psplines.logger import logger, set_level
from log_psplines.mcmc import MultivariateTimeseries, run_mcmc


@dataclass(frozen=True)
class StudyConfig:
    n: int = 1024
    fs: float = 1.0
    seed: int = 0
    n_knots: int = 7
    n_samples: int = 500
    n_warmup: int = 500
    num_chains: int = 4
    Nb: int = 8
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
    vi_steps: int = 8000
    vi_lr: float = 1e-2
    vi_guide: str | None = None


def _hermitianize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.complex128)
    return 0.5 * (mat + np.swapaxes(mat.conj(), -1, -2))


def _make_orthogonal(dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(dim, dim))
    q, _ = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    return q


def _diag_var_coeffs(
    *,
    ar1: np.ndarray,  # (3,)
    ar2: np.ndarray,  # (3,)
    q: np.ndarray,  # (3,3)
) -> np.ndarray:
    """Return VAR(2) coeffs A1,A2 in observation basis: A_k = Q diag(ar_k) Q^T."""
    ar1 = np.asarray(ar1, dtype=float)
    ar2 = np.asarray(ar2, dtype=float)
    q = np.asarray(q, dtype=float)
    if ar1.shape != (3,) or ar2.shape != (3,) or q.shape != (3, 3):
        raise ValueError("Expected shapes ar1=(3,), ar2=(3,), q=(3,3).")
    a1 = q @ np.diag(ar1) @ q.T
    a2 = q @ np.diag(ar2) @ q.T
    return np.stack([a1, a2], axis=0)


def _sigma_from_eigvars(q: np.ndarray, eig_vars: np.ndarray) -> np.ndarray:
    """Return innovation covariance Σ = Q diag(eig_vars) Q^T."""
    q = np.asarray(q, dtype=float)
    eig_vars = np.asarray(eig_vars, dtype=float)
    if q.shape != (3, 3) or eig_vars.shape != (3,):
        raise ValueError("Expected shapes q=(3,3), eig_vars=(3,).")
    if np.any(eig_vars <= 0):
        raise ValueError("eig_vars must be positive.")
    return q @ np.diag(eig_vars) @ q.T


def _eig_ratio_r23(psd: np.ndarray) -> np.ndarray:
    psd = _hermitianize(psd)
    eigvals = np.linalg.eigvalsh(psd).real
    eigvals = np.maximum(eigvals, 0.0)
    eigvals_desc = eigvals[:, ::-1]
    lam2 = eigvals_desc[:, 1]
    lam3 = eigvals_desc[:, 2]
    return lam3 / (lam2 + 1e-12)


def _summarize_ratio(
    ratio: np.ndarray, *, threshold: float = 0.8
) -> Dict[str, float]:
    ratio = np.asarray(ratio, dtype=float)
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return {"r23_p50": float("nan"), "r23_frac_gt": float("nan")}
    return {
        "r23_p50": float(np.median(ratio)),
        "r23_frac_gt": float(np.mean(ratio > float(threshold))),
    }


def _block3_var_names(posterior_vars: Iterable[str]) -> list[str]:
    out = []
    for name in posterior_vars:
        if name in {"delta_2", "phi_delta_2"}:
            out.append(name)
            continue
        if name.startswith("weights_delta_2"):
            out.append(name)
            continue
        if name.startswith("delta_theta_re_2_") or name.startswith(
            "delta_theta_im_2_"
        ):
            out.append(name)
            continue
        if name.startswith("phi_theta_re_2_") or name.startswith(
            "phi_theta_im_2_"
        ):
            out.append(name)
            continue
        if name.startswith("weights_theta_re_2_") or name.startswith(
            "weights_theta_im_2_"
        ):
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

    max_steps = (2 ** int(max_tree_depth)) - 1
    steps_key = "num_steps_channel_2"
    if steps_key in stats:
        steps = np.asarray(stats[steps_key].values, dtype=float)
        out["block3_frac_max_tree_depth"] = float(np.mean(steps == max_steps))
        out["block3_num_steps_median"] = float(np.median(steps))

    step_key = "step_size_channel_2"
    if step_key in stats:
        step = np.asarray(stats[step_key].values, dtype=float)
        step = step[np.isfinite(step)]
        out["block3_step_size_median"] = (
            float(np.median(step)) if step.size else float("nan")
        )

    div_key = "diverging_channel_2"
    if div_key in stats:
        div = np.asarray(stats[div_key].values, dtype=float)
        div = div[np.isfinite(div)]
        out["block3_divergence_frac"] = (
            float(np.mean(div > 0.0)) if div.size else float("nan")
        )

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
            out["block3_rhat_max"] = (
                float(np.max(r)) if r.size else float("nan")
            )
        if "ess_bulk" in summ:
            e = np.asarray(summ["ess_bulk"].values, dtype=float)
            e = e[np.isfinite(e)]
            out["block3_ess_bulk_min"] = (
                float(np.min(e)) if e.size else float("nan")
            )
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


def _build_case_varma(cfg: StudyConfig, case: str) -> VARMAData:
    q = _make_orthogonal(3, seed=cfg.seed + 321)

    # Mode definitions in eigenbasis:
    # - mode 1: AR(1) close to 1 -> strong low-frequency power
    # - mode 2: AR(2) with resonant peak -> mid-band bump-ish
    # - mode 3: AR(1) negative -> relatively higher power near Nyquist
    ar1 = np.array([0.98, 0.0, -0.75], dtype=float)
    ar2 = np.array([0.0, 0.0, 0.0], dtype=float)

    # AR(2) resonance for mode 2: roots r e^{±iw0} -> a1=2 r cos(w0), a2=-r^2
    r = 0.92
    w0 = np.pi / 3.0
    ar1[1] = 2.0 * r * np.cos(w0)
    ar2[1] = -(r**2)

    # Scale innovations so mode-1 is clearly dominant across the band, making
    # the r23 diagnostic reflect separation between modes 2 and 3.
    mode1_var = 6.0
    mode23_var = 1.0

    if case == "separated":
        eig_vars = np.array(
            [mode1_var, mode23_var, 0.6 * mode23_var], dtype=float
        )
        ar1_case = ar1
        ar2_case = ar2
    elif case == "bunched":
        # Make modes 2 and 3 identical (λ2≈λ3, eigenvectors poorly identified).
        eig_vars = np.array([mode1_var, mode23_var, mode23_var], dtype=float)
        ar1_case = ar1.copy()
        ar2_case = ar2.copy()
        ar1_case[2] = ar1_case[1]
        ar2_case[2] = ar2_case[1]
    elif case == "force_separation":
        # Same dynamics as bunched, but deflate mode 3 variance.
        eig_vars = np.array(
            [mode1_var, mode23_var, 0.3 * mode23_var], dtype=float
        )
        ar1_case = ar1.copy()
        ar2_case = ar2.copy()
        ar1_case[2] = ar1_case[1]
        ar2_case[2] = ar2_case[1]
    else:
        raise ValueError(f"Unknown case {case!r}")

    var_coeffs = _diag_var_coeffs(ar1=ar1_case, ar2=ar2_case, q=q)
    vma_coeffs = np.array([np.eye(3)], dtype=float)
    sigma = _sigma_from_eigvars(q, eig_vars)

    return VARMAData(
        n_samples=cfg.n,
        seed=cfg.seed,
        fs=cfg.fs,
        var_coeffs=var_coeffs,
        vma_coeffs=vma_coeffs,
        sigma=sigma,
    )


def run_case(
    *,
    cfg: StudyConfig,
    outdir: Path,
    case: str,
    skip_plot_diagnostics: bool = False,
    diagnostics_summary_mode: str = "light",
    diagnostics_summary_position: str = "end",
    save_preprocessing_plots: bool = True,
    compute_coherence_quantiles: bool = True,
) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    varma = _build_case_varma(cfg, case)
    true_psd = varma.get_true_psd()

    # Report eigenvalue separation on the *base* VARMA PSD before any
    # channel-wise scaling corrections applied by VARMAData.
    sf = float(getattr(varma, "psd_scaling", 1.0) or 1.0)
    channel_stds = np.asarray(
        getattr(varma, "channel_stds", None), dtype=float
    )
    if channel_stds.shape != (3,):
        raise ValueError("VARMAData.channel_stds must have shape (3,).")
    scale_matrix = np.outer(channel_stds, channel_stds) / sf
    base_psd = true_psd / scale_matrix[None, :, :]

    ratio_summary = _summarize_ratio(_eig_ratio_r23(base_psd), threshold=0.8)
    logger.info(
        f"{case}: r23 p50={ratio_summary['r23_p50']:.3f}, frac(r23>0.8)={ratio_summary['r23_frac_gt']:.3f}"
    )

    ts = MultivariateTimeseries(t=varma.time, y=varma.data)
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
        Nb=cfg.Nb,
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
        skip_plot_diagnostics=skip_plot_diagnostics,
        diagnostics_summary_mode=diagnostics_summary_mode,
        diagnostics_summary_position=diagnostics_summary_position,
        vi_psd_max_draws=32,
        posterior_psd_max_draws=32,
        compute_coherence_quantiles=compute_coherence_quantiles,
        true_psd=(varma.freq, true_psd),
        save_preprocessing_plots=save_preprocessing_plots,
    )

    block3 = _summarize_block3_sampling(
        idata, max_tree_depth=cfg.max_tree_depth
    )
    runtime = float(idata.attrs.get("runtime", float("nan")))
    row: Dict[str, object] = {
        "case": case,
        "output_dir": str(outdir),
        "Nb": float(cfg.Nb),
        "n": float(cfg.n),
        "fs": float(cfg.fs),
        "seed": float(cfg.seed),
        "n_knots": float(cfg.n_knots),
        "n_samples": float(cfg.n_samples),
        "n_warmup": float(cfg.n_warmup),
        "num_chains": float(cfg.num_chains),
        "target_accept_prob": float(cfg.target_accept_prob),
        "max_tree_depth": float(cfg.max_tree_depth),
        "alpha_phi": float(cfg.alpha_phi),
        "beta_phi": float(cfg.beta_phi),
        "alpha_delta": float(cfg.alpha_delta),
        "beta_delta": float(cfg.beta_delta),
        "init_from_vi": float(int(cfg.init_from_vi)),
        "vi_steps": float(cfg.vi_steps),
        "vi_lr": float(cfg.vi_lr),
        "vi_guide": "" if cfg.vi_guide is None else str(cfg.vi_guide),
        "runtime_seconds": runtime,
        **ratio_summary,
        **block3,
    }
    (outdir / "diagnostics_var3_mode_sep.json").write_text(
        json.dumps(row, indent=2, sort_keys=True) + "\n"
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VAR-based mode-separation study for multivariate blocked NUTS"
    )
    parser.add_argument("--out", type=str, default="out_var3_mode_sep")
    parser.add_argument("--n-time", type=int, default=1024)
    parser.add_argument("--fs", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--knots", type=int, default=7)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--time-blocks", type=int, default=8)
    parser.add_argument("--max-tree-depth", type=int, default=10)
    parser.add_argument("--target-accept", type=float, default=0.8)
    parser.add_argument("--alpha-phi", type=float, default=1.0)
    parser.add_argument("--beta-phi", type=float, default=1.0)
    parser.add_argument("--alpha-delta", type=float, default=1e-4)
    parser.add_argument("--beta-delta", type=float, default=1e-4)
    parser.add_argument("--alpha-phi-theta", type=float, default=None)
    parser.add_argument("--beta-phi-theta", type=float, default=None)
    parser.add_argument(
        "--init-from-vi",
        action="store_true",
        help="Use VI to initialise NUTS (default).",
    )
    parser.add_argument(
        "--no-init-from-vi",
        action="store_true",
        help="Disable VI init and use default NUTS init.",
    )
    parser.add_argument("--vi-steps", type=int, default=8000)
    parser.add_argument("--vi-lr", type=float, default=1e-2)
    parser.add_argument("--vi-guide", type=str, default=None)
    args = parser.parse_args()

    set_level("DEBUG")
    logger.info(f"JAX devices: {jax.devices()}")

    if args.init_from_vi and args.no_init_from_vi:
        raise ValueError(
            "Choose at most one of --init-from-vi / --no-init-from-vi."
        )
    init_from_vi = True
    if args.no_init_from_vi:
        init_from_vi = False

    cfg = StudyConfig(
        n=int(args.n),
        fs=float(args.fs),
        seed=int(args.seed),
        n_knots=int(args.knots),
        n_samples=int(args.samples),
        n_warmup=int(args.warmup),
        num_chains=int(args.chains),
        Nb=int(args.time_blocks),
        target_accept_prob=float(args.target_accept),
        max_tree_depth=int(args.max_tree_depth),
        alpha_phi=float(args.alpha_phi),
        beta_phi=float(args.beta_phi),
        alpha_delta=float(args.alpha_delta),
        beta_delta=float(args.beta_delta),
        alpha_phi_theta=(
            None
            if args.alpha_phi_theta is None
            else float(args.alpha_phi_theta)
        ),
        beta_phi_theta=(
            None if args.beta_phi_theta is None else float(args.beta_phi_theta)
        ),
        init_from_vi=init_from_vi,
        vi_steps=int(args.vi_steps),
        vi_lr=float(args.vi_lr),
        vi_guide=(
            None
            if args.vi_guide is None or args.vi_guide.strip() == ""
            else str(args.vi_guide)
        ),
    )

    if cfg.Nb < 3:
        logger.warning(
            f"Nb={cfg.Nb} < p=3; Wishart matrices are rank-deficient "
            "and the blocked likelihood geometry will be pathological."
        )

    here = Path(__file__).resolve().parent
    root_out = (
        here
        / str(args.out)
        / (f"seed_{cfg.seed}_N{cfg.n}_K{cfg.n_knots}_B{cfg.Nb}")
    )
    root_out.mkdir(parents=True, exist_ok=True)
    (root_out / "study_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, sort_keys=True) + "\n"
    )

    rows = []
    for case in ("separated", "force_separation", "bunched"):
        rows.append(run_case(cfg=cfg, outdir=root_out / case, case=case))
    _write_csv(root_out / "var3_mode_sep_summary.csv", rows)
    logger.info(f"Wrote summary to {root_out / 'var3_mode_sep_summary.csv'}")


if __name__ == "__main__":
    main()
