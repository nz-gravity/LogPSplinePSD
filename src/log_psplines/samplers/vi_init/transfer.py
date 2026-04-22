"""Helpers for coarse-to-fine VI transfer orchestration metadata."""

from __future__ import annotations

from typing import Any, Dict, Optional


def coarse_vi_metadata(sampler) -> Dict[str, Any]:
    """Return normalized metadata stored on samplers for coarse VI runs."""
    metadata = dict(getattr(sampler, "_coarse_vi_metadata", {}) or {})
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
