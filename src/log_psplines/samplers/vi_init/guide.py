"""Helper utilities for selecting VI guide configurations."""

from __future__ import annotations


def suggest_guide_univar(n_latents: int) -> str:
    """Return a reasonable autoguide choice for univariate models."""

    # Prefer flow for tiny problems, MVN for modest, low-rank for big.
    if n_latents <= 40:
        return "flow:1"
    if n_latents <= 160:
        return "mvn"

    rank = max(8, min(32, n_latents // 4))
    rank = min(rank, max(2, n_latents - 1))
    if rank < 2:
        return "diag"
    return f"lowrank:{rank}"


def suggest_guide_multivar(total_latents: int) -> str:
    """Return an autoguide choice for fully coupled multivariate models."""

    # Prefer flow for small, MVN for medium, low-rank for large.
    if total_latents <= 80:
        return "flow:1"
    if total_latents <= 220:
        return "mvn"

    rank = max(16, min(64, total_latents // 6))
    rank = min(rank, max(2, total_latents - 1))
    if rank < 2:
        return "diag"
    return f"lowrank:{rank}"


def suggest_guide_block(
    delta_basis_cols: int, theta_count: int, theta_basis_cols: int
) -> str:
    """Return an autoguide choice for blocked multivariate models."""

    total_latents = delta_basis_cols + 2
    if theta_count > 0:
        total_latents += theta_count * (theta_basis_cols + 2)

    if total_latents <= 80:
        return "flow:1"
    if total_latents <= 160:
        return "mvn"

    rank = max(8, min(32, total_latents // 5))
    rank = min(rank, max(2, total_latents - 1))
    if rank < 2:
        return "diag"
    return f"lowrank:{rank}"
