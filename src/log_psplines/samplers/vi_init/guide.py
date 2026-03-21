"""Helper utilities for selecting VI guide configurations."""

from __future__ import annotations

from typing import Tuple


def _suggest_guide(
    n_latents: int,
    *,
    thresholds: Tuple[int, int] = (40, 160),
    rank_range: Tuple[int, int] = (8, 32),
    rank_divisor: int = 4,
) -> str:
    """Return a reasonable autoguide choice based on problem dimension.

    Parameters
    ----------
    n_latents:
        Total number of latent variables.
    thresholds:
        ``(flow_max, mvn_max)`` — use flow below the first, MVN below the
        second, low-rank above.
    rank_range:
        ``(min_rank, max_rank)`` clamp for the low-rank guide.
    rank_divisor:
        Divides ``n_latents`` to set the target rank.
    """
    flow_max, mvn_max = thresholds
    if n_latents <= flow_max:
        return "flow:1"
    if n_latents <= mvn_max:
        return "mvn"

    rank = max(rank_range[0], min(rank_range[1], n_latents // rank_divisor))
    rank = min(rank, max(2, n_latents - 1))
    if rank < 2:
        return "diag"
    return f"lowrank:{rank}"


def suggest_guide_univar(n_latents: int) -> str:
    """Return a reasonable autoguide choice for univariate models."""
    return _suggest_guide(n_latents, thresholds=(40, 160))


def suggest_guide_multivar(total_latents: int) -> str:
    """Return an autoguide choice for fully coupled multivariate models."""
    return _suggest_guide(
        total_latents,
        thresholds=(80, 220),
        rank_range=(16, 64),
        rank_divisor=6,
    )


def suggest_guide_block(
    delta_basis_cols: int, theta_count: int, theta_basis_cols: int
) -> str:
    """Return an autoguide choice for blocked multivariate models."""
    total_latents = delta_basis_cols + 2
    if theta_count > 0:
        total_latents += theta_count * (theta_basis_cols + 2)
    return _suggest_guide(
        total_latents,
        thresholds=(80, 160),
        rank_range=(8, 32),
        rank_divisor=5,
    )
