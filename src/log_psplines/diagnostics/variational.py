"""Opt-in VI audits: density contracts, weights and posterior comparisons.

These diagnostics answer separate questions. No function returns a universal
VI quality score. Density evaluation uses packed unconstrained latent values
and NumPyro's potential_energy (including support Jacobians and factors).
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import xarray as xr
from arviz_stats.base import array_stats
from numpyro.distributions.transforms import biject_to
from numpyro.infer.util import potential_energy
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance


@dataclass(frozen=True)
class VIDiagnosticConfig:
    """Diagnostic RNG streams are independent of training/posterior streams."""

    seeds: tuple[int, ...] = (8101, 8102, 8103)
    num_particles: int = 4096
    chunk_size: int = 128
    evaluation_seeds: tuple[int, ...] = (8201, 8202, 8203, 8204)
    evaluation_particles: int = 32
    checkpoint_steps: tuple[int, ...] = ()
    target_fingerprint: str | None = None
    parameterization: str = "unspecified"
    stopping_rule: str = "legacy"

    def __post_init__(self):
        if self.stopping_rule not in ("legacy", "noise_aware"):
            raise ValueError("stopping_rule must be legacy or noise_aware")
        for name in ("num_particles", "chunk_size", "evaluation_particles"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("seeds", "evaluation_seeds"):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain distinct seeds")
            if any(not isinstance(x, int) or x < 0 for x in values):
                raise ValueError(f"invalid {name}")
        if len(self.evaluation_seeds) < 2:
            raise ValueError("at least two evaluation seeds are required")
        if any(not isinstance(x, int) or x < 1 for x in self.checkpoint_steps):
            raise ValueError("checkpoint steps must be positive integers")


def fingerprint(*values: Any) -> str:
    """Hash explicit numerical target inputs, configuration and source identity.

    The caller must include closed-over observations/model data. A callable's
    name alone cannot identify its target; absent caller fingerprints are
    explicitly marked unavailable and cannot pass a same-target comparison.
    """
    digest = hashlib.sha256()

    def update(value):
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, Mapping):
            digest.update(b"mapping")
            for key in sorted(value):
                update(str(key))
                update(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(b"sequence")
            for item in value:
                update(item)
        elif isinstance(value, (np.ndarray, jax.Array)):
            array = np.asarray(value)
            if array.dtype.hasobject:
                raise ValueError(
                    "object arrays cannot identify numerical targets"
                )
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        else:
            digest.update(
                json.dumps(value, sort_keys=True, allow_nan=False).encode()
            )
        digest.update(b"\0")

    for value in values:
        update(value)
    return digest.hexdigest()


def runtime_provenance() -> dict[str, Any]:
    """Source/dirty-content hashes, installed versions and actual backend."""
    root = Path(__file__).resolve().parents[3]

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args])

    try:
        sha = git("rev-parse", "HEAD").decode().strip()
        dirty = hashlib.sha256(git("diff", "HEAD", "--binary"))
        for name in sorted(
            git("ls-files", "--others", "--exclude-standard")
            .decode()
            .splitlines()
        ):
            path = root / name
            if path.is_file():
                dirty.update(name.encode())
                dirty.update(path.read_bytes())
        diff_hash = dirty.hexdigest()
        status = git("status", "--short").decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        sha, diff_hash, status = None, None, "unavailable"
    versions = {}
    for package in (
        "numpyro",
        "jax",
        "jaxlib",
        "numpy",
        "scipy",
        "optax",
        "arviz-base",
        "arviz-stats",
        "xarray",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "utc_started": datetime.now(UTC).isoformat(),
        "repository_sha": sha,
        "dirty_diff_sha256": diff_hash,
        "dirty_status": status,
        "versions": versions,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "x64_enabled": jax.config.x64_enabled,
    }


def require_same_target(first: str | None, second: str | None) -> None:
    if not first or not second:
        raise ValueError("unavailable target fingerprint")
    if first != second:
        raise ValueError("mismatched target fingerprints")


def _encode_tree(value, arrays, prefix):
    """A JSON tree recipe plus non-object numerical leaves; never pickle."""
    if isinstance(value, dict):
        return {
            "dict": [
                [key, _encode_tree(item, arrays, prefix)]
                for key, item in value.items()
            ]
        }
    if isinstance(value, (tuple, list)):
        kind = "tuple" if isinstance(value, tuple) else "list"
        return {kind: [_encode_tree(item, arrays, prefix) for item in value]}
    array = np.asarray(value)
    if array.dtype.kind not in "bifuc":
        raise ValueError("only numerical checkpoint leaves are supported")
    key = f"{prefix}_{len(arrays)}"
    arrays[key] = array
    return {"leaf": key, "shape": list(array.shape), "dtype": str(array.dtype)}


def _decode_tree(recipe, arrays):
    if "dict" in recipe:
        return {
            key: _decode_tree(item, arrays) for key, item in recipe["dict"]
        }
    for kind in ("tuple", "list"):
        if kind in recipe:
            values = [_decode_tree(item, arrays) for item in recipe[kind]]
            return tuple(values) if kind == "tuple" else values
    array = np.asarray(arrays[recipe["leaf"]])
    if (
        list(array.shape) != recipe["shape"]
        or str(array.dtype) != recipe["dtype"]
    ):
        raise ValueError("checkpoint leaf shape/dtype mismatch")
    return array.copy()


@dataclass
class VIDiagnosticState:
    """Small guide checkpoint plus optional joint diagnostic draws/densities.

    Model and guide executables are deliberately absent. Rebuilding requires
    the identical prepared model and its fingerprint. Optimizer resumption is
    not supported by this guide-only checkpoint.
    """

    params: dict[str, Any]
    metadata: dict[str, Any]
    checkpoints: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    def _archive(self):
        leaves = {}
        recipe = {
            "version": 1,
            "metadata": self.metadata,
            "params": _encode_tree(self.params, leaves, "param"),
            "checkpoints": _encode_tree(
                self.checkpoints, leaves, "checkpoint"
            ),
            "arrays": _encode_tree(self.arrays, leaves, "diagnostic"),
        }
        return recipe, leaves

    @classmethod
    def _restore(cls, recipe, leaves):
        if recipe["version"] != 1:
            raise ValueError("unsupported VI checkpoint version")
        return cls(
            params=_decode_tree(recipe["params"], leaves),
            metadata=recipe["metadata"],
            checkpoints=_decode_tree(recipe["checkpoints"], leaves),
            arrays=_decode_tree(recipe["arrays"], leaves),
        )

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        recipe, leaves = self._archive()
        (directory / "guide.json").write_text(
            json.dumps(recipe, indent=2, allow_nan=False) + "\n"
        )
        np.savez_compressed(directory / "guide.npz", **leaves)

    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory)
        recipe = json.loads((directory / "guide.json").read_text())
        with np.load(directory / "guide.npz", allow_pickle=False) as leaves:
            return cls._restore(recipe, leaves)

    def to_dataset(self) -> xr.Dataset:
        recipe, leaves = self._archive()
        dataset = xr.Dataset(
            {
                key: xr.DataArray(
                    value,
                    dims=tuple(f"{key}_dim_{i}" for i in range(value.ndim)),
                )
                for key, value in leaves.items()
            }
        )
        dataset["recipe"] = xr.DataArray(json.dumps(recipe, allow_nan=False))
        return dataset

    @classmethod
    def from_dataset(cls, dataset: xr.Dataset):
        return cls._restore(json.loads(dataset["recipe"].item()), dataset)


def latent_schema(guide) -> list[dict[str, Any]]:
    """Ordering is the autoguide's actual pack order, excluding deterministic sites."""
    schema = []
    offset = 0
    for name, value in guide._init_locs.items():
        site = guide.prototype_trace[name]
        transform = biject_to(site["fn"].support)
        size = int(np.size(value))
        schema.append(
            {
                "name": name,
                "shape": list(np.shape(value)),
                "dtype": str(np.asarray(value).dtype),
                "offset": offset,
                "size": size,
                "support": str(site["fn"].support),
                "transform": type(transform).__name__,
            }
        )
        offset += size
    return schema


def rebuild_guide(
    state, model, *, target_fingerprint, model_args=(), model_kwargs=None
):
    """Recreate built-in Gaussian guides against a verified identical target."""
    from log_psplines.inference.vi import resolve_guide

    require_same_target(
        state.metadata.get("target_fingerprint"), target_fingerprint
    )
    if state.metadata.get("reconstruction_status") != "supported":
        raise ValueError("unsupported guide reconstruction recipe")
    installed = importlib.metadata.version("numpyro")
    if installed != state.metadata["numpyro_version"]:
        raise ValueError(
            "NumPyro version mismatch; checkpoint recipe must be revalidated"
        )
    guide, _ = resolve_guide(state.metadata["guide_name"], model)
    numpyro.handlers.seed(
        numpyro.handlers.substitute(guide, data=state.params), 0
    )(*tuple(model_args), **(model_kwargs or {}))
    if latent_schema(guide) != state.metadata["latent_schema"]:
        raise ValueError("checkpoint latent schema mismatch")
    return guide


def packed_log_densities(
    model, guide, params, packed, *, model_args=(), model_kwargs=None
):
    """log p_u and log q_u on the same Lebesgue measure.

    potential_energy transforms each latent site, includes the Jacobian exactly
    once, and evaluates all priors, hyperpriors, factors and scaled likelihoods.
    No deterministic spectrum pixel is a latent density dimension.
    """
    model_args = tuple(model_args)
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    conditioned = numpyro.handlers.substitute(model, data=params)
    joint = -potential_energy(
        conditioned, model_args, model_kwargs, guide._unpack_latent(packed)
    )
    log_q = jnp.sum(guide.get_posterior(params).log_prob(packed))
    return joint, log_q


@lru_cache(maxsize=1)
def _psis_input_sign():
    probe = np.linspace(-2.0, 0.0, 100)
    smoothed, _ = array_stats.psislw(probe, r_eff=1, axis=-1)
    lower_change = float(smoothed[10] - smoothed[0])
    upper_change = float(smoothed[-1] - smoothed[-11])
    if np.isclose(lower_change, probe[10] - probe[0], rtol=1e-10):
        return 1
    if np.isclose(upper_change, -(probe[-1] - probe[-11]), rtol=1e-10):
        return -1
    raise ValueError("unrecognized installed PSIS log-weight convention")


def weight_diagnostics(log_ratios) -> dict[str, Any]:
    """PSIS and normalized weight ESS (not MCMC ESS), with degeneracy statuses."""
    values = np.asarray(log_ratios, dtype=float)
    if values.ndim != 1:
        raise ValueError("log ratios must have one joint-particle axis")
    n = len(values)
    output = {
        "num_particles": n,
        "backend": "arviz_stats.base.array_stats.psislw",
        "k": None,
    }
    if n < 2:
        return {
            **output,
            "status": "insufficient_samples",
            "tail_status": "insufficient_samples",
        }
    if not np.isfinite(values).all():
        return {
            **output,
            "status": "invalid_density",
            "tail_status": "not_evaluated",
            "nonfinite_count": int(np.sum(~np.isfinite(values))),
        }
    raw = np.exp(values - logsumexp(values))
    raw_ess = float(1 / np.sum(raw**2))
    output.update(
        raw_weight_ess=raw_ess,
        raw_ess_fraction=raw_ess / n,
        raw_max_weight=float(np.max(raw)),
        reference_k=min(0.7, 1 - 1 / np.log10(n)),
    )
    # Compare differences, not the absolute magnitude of arbitrary log normalizers.
    if np.ptp(values) <= 1e-10:
        return {
            **output,
            "status": "constant_weights",
            "tail_status": "degenerate_constant",
            "smoothed_weight_ess": raw_ess,
            "smoothed_ess_fraction": raw_ess / n,
            "smoothed_max_weight": float(np.max(raw)),
        }
    shifted = values - np.max(values)
    tail_count = min(int(np.ceil(n / 5)), int(np.ceil(3 * np.sqrt(n))))
    cutoff = np.sort(shifted)[max(0, n - tail_count - 1)]
    distinct_tail = int(np.sum(shifted > cutoff))
    output["tail_samples"] = distinct_tail
    if distinct_tail < 5:
        return {
            **output,
            "status": "insufficient_tail",
            "tail_status": "insufficient_tail",
        }
    try:
        sign = _psis_input_sign()
        output["backend_input_convention"] = (
            "log_weights" if sign == 1 else "negative_log_weights"
        )
        output["backend_version"] = importlib.metadata.version("arviz-stats")
        smoothed, khat = array_stats.psislw(sign * values, r_eff=1, axis=-1)
    except (AttributeError, TypeError, ValueError) as error:
        return {
            **output,
            "status": "unsupported_psis_api",
            "tail_status": "not_evaluated",
            "reason": str(error),
        }
    smoothed = np.asarray(smoothed)
    k = float(khat)
    if not np.isfinite(smoothed).all() or not np.isfinite(k):
        return {
            **output,
            "status": "nonfinite_psis",
            "tail_status": "fit_failed",
        }
    weights = np.exp(smoothed - logsumexp(smoothed))
    ess = float(1 / np.sum(weights**2))
    interpretation = (
        "encouraging"
        if k < 0.5
        else "caution"
        if k < 0.7
        else "reliability_warning"
    )
    return {
        **output,
        "status": "ok",
        "tail_status": "fitted",
        "k": k,
        "interpretation": interpretation,
        "smoothed_weight_ess": ess,
        "smoothed_ess_fraction": ess / n,
        "smoothed_max_weight": float(np.max(weights)),
    }


def combine_factor_ratios(ratios, *, factorization_verified: bool):
    """Sum aligned independent factor ratios before estimating JOINT k."""
    if not factorization_verified:
        raise ValueError("independent factorization must be verified")
    values = [np.asarray(x) for x in ratios]
    if not values or any(
        x.ndim != 1 or x.shape != values[0].shape for x in values
    ):
        raise ValueError("factors require aligned independent joint particles")
    joint = np.sum(values, axis=0)
    return joint, weight_diagnostics(joint)


def evaluate_objective(
    model,
    guide,
    params,
    config,
    *,
    model_args=(),
    model_kwargs=None,
    seeds=None,
):
    """Fixed-seed multi-particle negative ELBO, separate from training losses."""
    from numpyro.infer import Trace_ELBO

    loss = Trace_ELBO(num_particles=config.evaluation_particles)
    evaluate = jax.jit(
        lambda key: loss.loss(
            key, params, model, guide, *model_args, **(model_kwargs or {})
        )
    )
    values = np.array(
        [
            float(evaluate(jax.random.PRNGKey(seed)))
            for seed in (seeds or config.evaluation_seeds)
        ]
    )
    return {
        "negative_elbo": float(values.mean()),
        "replicate_sd": float(values.std(ddof=1)),
        "mcse": float(values.std(ddof=1) / np.sqrt(len(values))),
        "values": values.tolist(),
    }


def diagnose_guide(
    model,
    guide,
    params,
    guide_name,
    config,
    *,
    model_args=(),
    model_kwargs=None,
    checkpoints=None,
    optimization=None,
) -> VIDiagnosticState:
    """Compute independent repeated-seed weight diagnostics while model is live."""
    model_args = tuple(model_args)
    model_kwargs = {} if model_kwargs is None else dict(model_kwargs)
    metadata = {
        "provenance": runtime_provenance(),
        "parameter_dtypes": jax.tree.map(
            lambda value: str(np.asarray(value).dtype), params
        ),
        "recipe_version": 1,
        "numpyro_version": numpyro.__version__,
        "guide_name": guide_name,
        "reconstruction_status": "supported"
        if guide_name == "diag"
        or guide_name == "mvn"
        or guide_name.startswith("lowrank:")
        else "unsupported_guide_recipe",
        "coordinate_system": "packed_unconstrained",
        "parameterization": config.parameterization,
        "target_fingerprint": config.target_fingerprint,
        "fingerprint_status": "available"
        if config.target_fingerprint
        else "unavailable",
        "diagnostic_seeds": list(config.seeds),
        "evaluation_seeds": list(config.evaluation_seeds),
        "evaluation_particles": config.evaluation_particles,
        "num_particles": config.num_particles,
        "chunk_size": config.chunk_size,
        "optimization": optimization or {},
    }
    state = VIDiagnosticState(
        params=jax.tree.map(np.asarray, params),
        metadata=metadata,
        checkpoints=checkpoints or {},
    )
    if not all(
        hasattr(guide, attr)
        for attr in ("_unpack_latent", "_init_locs", "get_posterior")
    ):
        metadata["density_status"] = "unsupported_guide"
        metadata["native_psis"] = {"status": "unsupported_guide"}
        return state
    metadata["latent_schema"] = latent_schema(guide)
    # AutoContinuous forbids local subsampled latent plates; also reject data
    # subsampling: diagnostics must evaluate the actual full-data target.
    if any(
        site.get("type") == "plate"
        and np.size(site["value"]) != site["args"][0]
        for site in guide.prototype_trace.values()
    ):
        metadata["density_status"] = "unsupported_subsampled_model"
        return state
    native = getattr(numpyro.infer, "psis_diagnostic", None)
    metadata["native_psis"] = (
        {"status": "unsupported_api"}
        if native is None
        else {"status": "available", "runs": []}
    )
    records = []
    posterior = guide.get_posterior(params)
    evaluate = jax.jit(
        jax.vmap(
            lambda u: packed_log_densities(
                model,
                guide,
                params,
                u,
                model_args=model_args,
                model_kwargs=model_kwargs,
            )
        )
    )
    for seed in config.seeds:
        key = jax.random.PRNGKey(seed)
        draws = np.asarray(
            posterior.sample(key, sample_shape=(config.num_particles,))
        )
        log_p, log_q = [], []
        for start in range(0, config.num_particles, config.chunk_size):
            p, q = evaluate(
                jnp.asarray(draws[start : start + config.chunk_size])
            )
            log_p.append(np.asarray(p))
            log_q.append(np.asarray(q))
        p, q = np.concatenate(log_p), np.concatenate(log_q)
        ratios = p - q
        prefix = f"seed_{seed}"
        state.arrays.update(
            {
                f"{prefix}_packed": draws,
                f"{prefix}_log_joint": p,
                f"{prefix}_log_guide": q,
                f"{prefix}_log_ratios": ratios,
            }
        )
        records.append({"seed": seed, **weight_diagnostics(ratios)})
        if native is not None:
            try:
                k = float(
                    native(
                        key,
                        params,
                        model,
                        guide,
                        *model_args,
                        num_particles=config.num_particles,
                        chunk_size=config.chunk_size,
                        **model_kwargs,
                    )
                )
                metadata["native_psis"]["runs"].append(
                    {
                        "seed": seed,
                        "status": "constant_weights"
                        if records[-1]["status"] == "constant_weights"
                        else "ok"
                        if np.isfinite(k)
                        else "nonfinite_native_k",
                        "k": k
                        if np.isfinite(k)
                        and records[-1]["status"] != "constant_weights"
                        else None,
                    }
                )
            except (TypeError, ValueError, RuntimeError) as error:
                metadata["native_psis"]["runs"].append(
                    {"seed": seed, "status": "failed", "reason": str(error)}
                )
    metadata["weights"] = records
    metadata["density_status"] = (
        "ok"
        if all(r["status"] != "invalid_density" for r in records)
        else "invalid_density"
    )
    metadata["objective"] = evaluate_objective(
        model,
        guide,
        params,
        config,
        model_args=model_args,
        model_kwargs=model_kwargs,
    )
    metadata["independent_objective"] = evaluate_objective(
        model,
        guide,
        params,
        config,
        model_args=model_args,
        model_kwargs=model_kwargs,
        seeds=config.seeds,
    )
    return state


def mmd_squared(
    first, second, *, bandwidths=(0.5, 1.0, 2.0), chunk_size=128, unbiased=True
):
    """Exact multi-bandwidth RBF MMD^2, with bounded kernel-block memory.

    Inputs must already share ONE reference-derived feature transform.
    Unbiased estimates exclude within-sample diagonals and may be negative.
    Bandwidths are absolute in the supplied transformed coordinates.
    """
    x, y = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    bandwidths = np.asarray(bandwidths, dtype=float)
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[1] != y.shape[1]
        or min(len(x), len(y)) < (2 if unbiased else 1)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or bandwidths.ndim != 1
        or not len(bandwidths)
        or np.any(bandwidths <= 0)
        or not np.isfinite(bandwidths).all()
        or chunk_size < 1
    ):
        raise ValueError("invalid MMD inputs")

    def total(a, b, diagonal=False):
        value = 0.0
        for i in range(0, len(a), chunk_size):
            for j in range(0, len(b), chunk_size):
                left, right = a[i : i + chunk_size], b[j : j + chunk_size]
                distance = np.maximum(
                    np.sum(left**2, axis=1)[:, None]
                    + np.sum(right**2, axis=1)[None, :]
                    - 2 * left @ right.T,
                    0,
                )
                kernel = np.mean(
                    np.exp(-distance[..., None] / (2 * bandwidths**2)), axis=-1
                )
                if diagonal and i == j:
                    np.fill_diagonal(kernel, 0.0)
                value += float(kernel.sum())
        return value

    nx, ny = len(x), len(y)
    return (
        total(x, x, unbiased) / (nx * (nx - int(unbiased)))
        + total(y, y, unbiased) / (ny * (ny - int(unbiased)))
        - 2 * total(x, y) / (nx * ny)
    )


def reference_transform(training, *, ridge=1e-6):
    """Learn regularized whitening on a separate NUTS training subset."""
    training = np.asarray(training, dtype=float)
    if (
        training.ndim != 2
        or len(training) < 2
        or not np.isfinite(training).all()
        or ridge <= 0
    ):
        raise ValueError("invalid transform training data")
    mean = training.mean(axis=0)
    covariance = np.atleast_2d(np.cov(training, rowvar=False))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    scale = float(np.max(eigenvalues))
    if scale <= 0:
        raise ValueError("zero reference variance")
    matrix = (
        eigenvectors / np.sqrt(np.maximum(eigenvalues, ridge * scale))
    ) @ eigenvectors.T
    return {
        "mean": mean,
        "matrix": matrix,
        "ridge_relative": ridge,
        "raw_covariance_eigenvalues": eigenvalues,
    }


def nuts_reference_health(
    values, sample_stats, *, max_tree_depth=10, mean_mcse_sd=0.05
):
    """Four-chain reference screens, preserving every failed/unavailable check."""
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 3
        or values.shape[0] < 4
        or values.shape[1] < 4
        or not np.isfinite(values).all()
    ):
        return {
            "status": "failed_reference",
            "reason": "four finite chains required",
        }
    rhat = np.asarray(
        array_stats.rhat(values, chain_axis=0, draw_axis=1, method="rank")
    )
    bulk = np.asarray(
        array_stats.ess(values, chain_axis=0, draw_axis=1, method="bulk")
    )
    tail = np.asarray(
        array_stats.ess(
            values, chain_axis=0, draw_axis=1, method="tail", prob=(0.05, 0.95)
        )
    )
    mcse = np.asarray(
        array_stats.mcse(values, chain_axis=0, draw_axis=1, method="mean")
    )
    sd = values.reshape(-1, values.shape[-1]).std(axis=0, ddof=1)
    required = ("diverging", "n_steps", "energy")
    if sample_stats is None or any(
        name not in sample_stats for name in required
    ):
        return {
            "status": "failed_reference",
            "reason": "unavailable sampler statistics",
        }
    energy = np.asarray(sample_stats["energy"])
    bfmi = np.asarray(array_stats.bfmi(energy, chain_axis=0, draw_axis=1))
    divergences = int(np.sum(sample_stats["diverging"]))
    hits = int(
        np.sum(np.asarray(sample_stats["n_steps"]) >= 2**max_tree_depth - 1)
    )
    checked = np.concatenate(
        [rhat.ravel(), bulk.ravel(), tail.ravel(), mcse.ravel(), bfmi.ravel()]
    )
    valid = (
        np.isfinite(checked).all()
        and np.all(sd > 0)
        and divergences == 0
        and hits == 0
        and np.all(rhat < 1.01)
        and np.all(bulk >= 400)
        and np.all(tail >= 400)
        and np.all(mcse / sd <= mean_mcse_sd)
        and np.all(bfmi >= 0.3)
    )
    return {
        "status": "accepted_screen" if valid else "failed_reference",
        "rhat": rhat.tolist(),
        "ess_bulk": bulk.tolist(),
        "ess_tail": tail.tolist(),
        "mcse_mean": mcse.tolist(),
        "mcse_mean_sd_units": (mcse / sd).tolist(),
        "bfmi": bfmi.tolist(),
        "divergences": divergences,
        "depth_saturation": hits,
        "gates": {
            "rhat": 1.01,
            "ess": 400,
            "mcse_mean_sd": mean_mcse_sd,
            "bfmi": 0.3,
        },
    }


def compare_features(
    vi,
    reference,
    *,
    names,
    vi_fingerprint,
    reference_fingerprint,
    reference_status,
    events=None,
):
    """Moments, intervals, Wasserstein and events; retain reference limitations."""
    require_same_target(vi_fingerprint, reference_fingerprint)
    q, p = np.asarray(vi, dtype=float), np.asarray(reference, dtype=float)
    if (
        q.ndim != 3
        or p.ndim != 3
        or q.shape[-1] != p.shape[-1]
        or len(names) != p.shape[-1]
    ):
        raise ValueError(
            "features must preserve (chain, draw, feature) coordinates"
        )
    if (
        min(q.shape[1], p.shape[1]) < 2
        or not np.isfinite(q).all()
        or not np.isfinite(p).all()
    ):
        return {"status": "invalid_features"}
    flat_q, flat_p = q.reshape(-1, q.shape[-1]), p.reshape(-1, p.shape[-1])
    rows = []
    for index, name in enumerate(names):
        a, b = flat_q[:, index], flat_p[:, index]
        sd = b.std(ddof=1)
        if (
            sd
            <= np.finfo(float).eps
            * max(np.max(np.abs(b)), np.finfo(float).tiny)
            * 100
        ):
            rows.append({"name": name, "status": "zero_reference_sd"})
            continue
        row = {
            "name": name,
            "status": "ok",
            "standardized_mean_difference": float((a.mean() - b.mean()) / sd),
            "sd_ratio": float(a.std(ddof=1) / sd),
            "wasserstein_sd": float(wasserstein_distance(a, b) / sd),
        }
        for level in (0.5, 0.9, 0.95):
            percentiles = [(1 - level) / 2, (1 + level) / 2]
            width = np.diff(np.quantile(b, percentiles))[0]
            row[f"interval_width_ratio_{int(level * 100)}"] = (
                float(np.diff(np.quantile(a, percentiles))[0] / width)
                if width > 0
                else None
            )
        rows.append(row)
    probability_rows = []
    for name, index, threshold in events or []:
        iq, ip = (
            (q[..., index] <= threshold).astype(float),
            (p[..., index] <= threshold).astype(float),
        )
        pq, pp = float(iq.mean()), float(ip.mean())
        if 0 < pp < 1 and np.all(np.var(ip, axis=1) > 0):
            se_p = float(array_stats.mcse(ip, method="mean"))
            ess_p = float(array_stats.ess(ip, method="mean"))
            status = "estimated"
        else:
            se_p, ess_p, status = (
                None,
                None,
                "constant_indicator_unresolved_tail",
            )
        se_q = float(np.sqrt(pq * (1 - pq) / iq.size)) if 0 < pq < 1 else None
        probability_rows.append(
            {
                "name": name,
                "threshold": threshold,
                "vi_probability": pq,
                "nuts_probability": pp,
                "difference": pq - pp,
                "vi_mcse": se_q,
                "nuts_mcse": se_p,
                "nuts_event_ess": ess_p,
                "status": status,
                "difference_mcse": float(np.sqrt(se_q**2 + se_p**2))
                if se_q is not None and se_p is not None
                else None,
            }
        )
    sd = flat_p.std(axis=0, ddof=1)
    retained = (
        sd
        > np.finfo(float).eps
        * np.maximum(np.max(np.abs(flat_p), axis=0), np.finfo(float).tiny)
        * 100
    )
    if np.any(retained):
        standardized_q, standardized_p = (
            flat_q[:, retained] / sd[retained],
            flat_p[:, retained] / sd[retained],
        )
        covariance_error = float(
            np.linalg.norm(
                np.atleast_2d(np.cov(standardized_q, rowvar=False))
                - np.atleast_2d(np.cov(standardized_p, rowvar=False)),
                ord="fro",
            )
        )
    else:
        covariance_error = None
    return {
        "status": "ok"
        if reference_status == "accepted_screen"
        else "unresolved_reference",
        "features": rows,
        "events": probability_rows,
        "covariance_discrepancy_frobenius": covariance_error,
        "vi_correlation": np.corrcoef(
            flat_q[:, retained], rowvar=False
        ).tolist()
        if retained.sum() > 1
        else None,
        "nuts_correlation": np.corrcoef(
            flat_p[:, retained], rowvar=False
        ).tolist()
        if retained.sum() > 1
        else None,
        "vi_count": len(flat_q),
        "nuts_count": len(flat_p),
    }


def whiten_coefficients(
    coefficients, covariance, *, complex_coefficients=False, mask=None
):
    """Conditional coefficient whitening; never infer phases from power data.

    This primitive is for density-contract tests. Predictive adequacy and
    dependence-aware residual testing are separate follow-on work.
    """
    from scipy.linalg import solve_triangular

    if coefficients is None:
        return {"status": "unavailable_coefficient_phases"}
    values = np.asarray(coefficients)
    covariance = np.asarray(covariance)
    if values.ndim != 2 or covariance.shape != (
        values.shape[1],
        values.shape[1],
    ):
        raise ValueError(
            "expected independent records by coefficient channels"
        )
    if not np.allclose(covariance, covariance.conj().T):
        raise ValueError("covariance must be Hermitian")
    selected = (
        np.ones(len(values), dtype=bool)
        if mask is None
        else np.asarray(mask, dtype=bool)
    )
    if selected.shape != (len(values),):
        raise ValueError("mask must match records")
    values = values[selected]
    if not len(values) or not np.isfinite(values).all():
        return {"status": "invalid_coefficients"}
    if not complex_coefficients and np.iscomplexobj(values):
        raise ValueError(
            "complex coefficients require the proper-complex convention"
        )
    lower = np.linalg.cholesky(covariance)
    residuals = solve_triangular(lower, values.T, lower=True).T
    return {
        "status": "ok",
        "residuals": residuals,
        "covariance": residuals.T @ residuals.conj() / len(values),
        "pseudo_covariance": residuals.T @ residuals / len(values)
        if complex_coefficients
        else None,
        "component_scale": np.sqrt(2.0) if complex_coefficients else 1.0,
    }
