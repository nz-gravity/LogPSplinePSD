"""Helpers for coarse-to-fine VI transfer orchestration metadata."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .plan import VIWarmStartPlan


def coarse_vi_metadata(
    warm_start_plan: VIWarmStartPlan | None = None,
) -> Dict[str, Any]:
    """Return normalized metadata for coarse VI runs."""
    metadata = dict(
        (warm_start_plan.metadata if warm_start_plan else {}) or {}
    )
    metadata.setdefault("coarse_vi_attempted", 0)
    metadata.setdefault("coarse_vi_success", 0)
    return metadata


def mark_coarse_vi(
    diagnostics: Optional[Dict[str, Any]],
    metadata: Dict[str, Any],
    *,
    attempted: bool,
    success: bool,
) -> Dict[str, Any]:
    """Attach coarse-VI attempt/success flags to diagnostics."""
    out = dict(diagnostics or {})
    out.update(metadata)
    out["coarse_vi_attempted"] = int(bool(attempted))
    out["coarse_vi_success"] = int(bool(success))
    return out
