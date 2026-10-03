"""Shared numerical helpers for active VI validation studies.

Historical experiment drivers live in archive/ and are never imported here.
The extracted numerical functions retain the original implementations.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import optax
import xarray as xr
from numpyro.distributions.transforms import biject_to
from scipy.stats import kurtosis, skew

from log_psplines.diagnostics.variational import (
    mmd_squared,
    reference_transform,
)


@dataclass
class Target:
    model: object
    spline: object
    data: object
    config: object
    pair: object
    init: dict | None
    identity: str
    descriptor: dict


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def safe(x):
        if isinstance(x, dict):
            return {k: safe(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [safe(v) for v in x]
        if isinstance(x, np.ndarray):
            return safe(x.tolist())
        if isinstance(x, np.generic):
            return safe(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x

    path.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n")


def quadrature(grid):
    return np.r_[
        np.diff(grid)[0] / 2, (grid[2:] - grid[:-2]) / 2, np.diff(grid)[-1] / 2
    ]


def packed_reference(guide, posterior):
    nc, nd = posterior.sizes["chain"], posterior.sizes["draw"]
    values = []
    for name in guide._init_locs:
        site = guide.prototype_trace[name]
        unconstrained = biject_to(site["fn"].support).inv(
            jnp.asarray(posterior[name].values)
        )
        values.append(np.asarray(unconstrained).reshape(nc, nd, -1))
    return np.concatenate(values, axis=-1)


def posterior_geometry(guide, params, reference_packed):
    distribution = guide.get_posterior(params)
    mean = np.asarray(distribution.mean)
    if hasattr(distribution, "covariance_matrix"):
        covariance = np.asarray(distribution.covariance_matrix)
    else:
        covariance = np.diag(np.asarray(distribution.variance))
    whitening = np.linalg.inv(np.linalg.cholesky(covariance)).T
    transformed = (reference_packed.reshape(-1, len(mean)) - mean) @ whitening
    return {
        "nuts_mean_in_q_coordinates": transformed.mean(axis=0),
        "nuts_covariance_eigenvalues_in_q_coordinates": np.linalg.eigvalsh(
            np.cov(transformed, rowvar=False)
        ),
        "nuts_skew_in_q_coordinates": skew(transformed, axis=0),
        "nuts_excess_kurtosis_in_q_coordinates": kurtosis(transformed, axis=0),
    }


def mmd_comparison(q, p, cfg):
    """Reference transform trained on first 1/4; disjoint chain baselines."""
    nd = p.shape[1]
    train = p[:, : nd // 4].reshape(-1, p.shape[-1])
    remaining = p[:, nd // 4 :]
    transform = reference_transform(train)

    def project(x):
        return (x - transform["mean"]) @ transform["matrix"]

    n = min(
        cfg["mmd_count"], remaining[:2].size // p.shape[-1], q.shape[1] // 2
    )
    # No thinning/IID claim: consecutive chain segments yield descriptive baselines.
    p1 = project(remaining[:2].reshape(-1, p.shape[-1])[:n])
    p2 = project(remaining[2:].reshape(-1, p.shape[-1])[:n])
    q1 = project(q.reshape(-1, q.shape[-1])[:n])
    q2 = project(q.reshape(-1, q.shape[-1])[n : 2 * n])
    bandwidths = np.sqrt(p.shape[-1]) * np.asarray(
        cfg["mmd_bandwidth_multipliers"]
    )
    return {
        "status": "descriptive_autocorrelated_reference",
        "estimator": "unbiased_MMD_squared_diagonals_excluded",
        "bandwidths": bandwidths,
        "reference_transform": transform,
        "transform_training_count": len(train),
        "matched_count": n,
        "vi_vs_nuts": mmd_squared(q1, p1, bandwidths=bandwidths),
        "nuts_vs_nuts": mmd_squared(p1, p2, bandwidths=bandwidths),
        "vi_vs_vi": mmd_squared(q1, q2, bandwidths=bandwidths),
        "permutation_p_value": None,
        "sampling_design": "disjoint early transform-training subset; separate chains for NUTS baseline; no IID claim",
    }


def learning_rate(schedule, steps):
    if schedule["type"] == "constant":
        return schedule["peak_lr"]
    if schedule["type"] == "warmup_cosine":
        return optax.warmup_cosine_decay_schedule(
            init_value=schedule["initial_lr"],
            peak_value=schedule["peak_lr"],
            warmup_steps=schedule["warmup_steps"],
            decay_steps=schedule.get("decay_steps", steps),
            end_value=schedule["end_lr"],
        )
    raise ValueError(f"unknown schedule: {schedule}")


def draw_dataset(guide, params, key, count):
    packed = guide.get_posterior(params).sample(key, (count,))
    samples = guide._unpack_and_constrain(packed, params)
    dataset = xr.Dataset(
        {
            name: xr.DataArray(
                np.asarray(value)[None],
                dims=(
                    "chain",
                    "draw",
                    *[f"{name}_dim{i}" for i in range(np.ndim(value) - 1)],
                ),
            )
            for name, value in samples.items()
            if not name.startswith("log_likelihood")
        }
    )
    return np.asarray(packed), dataset
