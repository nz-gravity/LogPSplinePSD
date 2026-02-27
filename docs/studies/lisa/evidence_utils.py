"""Utilities for comparing full vs diagonal LISA hypothesis evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


def parse_hypothesis_mode(value: str) -> str:
    """Validate the hypothesis mode string."""
    mode = str(value).strip().lower()
    allowed = {"full", "diag", "both"}
    if mode not in allowed:
        raise ValueError(
            f"Unsupported LISA_HYPOTHESIS_MODE='{value}'. "
            f"Allowed values: {sorted(allowed)}."
        )
    return mode


def parse_channel_names(raw: str) -> list[str]:
    """Parse comma-separated channel labels for diagonal runs."""
    labels = [token.strip() for token in str(raw).split(",")]
    labels = [token for token in labels if token]
    if len(labels) != 3:
        return ["X", "Y", "Z"]
    if len(set(labels)) != 3:
        return ["X", "Y", "Z"]
    return labels


def extract_lnz(idata: Any) -> tuple[float, float, bool]:
    """Return ``(lnz, lnz_err, valid)`` from an ArviZ InferenceData object."""
    attrs = getattr(idata, "attrs", {})
    lnz = attrs.get("lnz", np.nan)
    lnz_err = attrs.get("lnz_err", np.nan)
    try:
        lnz_val = float(lnz)
        lnz_err_val = float(lnz_err)
    except Exception:
        return np.nan, np.nan, False
    valid = bool(np.isfinite(lnz_val) and np.isfinite(lnz_err_val))
    return lnz_val, lnz_err_val, valid


def combine_diag_lnz(
    channel_results: Sequence[Mapping[str, Any]],
) -> tuple[float, float, bool]:
    """Combine per-channel diagonal evidence estimates."""
    if not channel_results:
        return np.nan, np.nan, False
    if not all(bool(entry.get("valid", False)) for entry in channel_results):
        return np.nan, np.nan, False
    lnz_values = np.asarray(
        [float(entry["lnz"]) for entry in channel_results], dtype=np.float64
    )
    err_values = np.asarray(
        [float(entry["lnz_err"]) for entry in channel_results],
        dtype=np.float64,
    )
    lnz_diag = float(np.sum(lnz_values))
    lnz_diag_err = float(np.sqrt(np.sum(err_values**2)))
    return lnz_diag, lnz_diag_err, True


def compare_full_vs_diag(
    full_lnz: float,
    full_err: float,
    diag_lnz: float,
    diag_err: float,
    *,
    full_valid: bool,
    diag_valid: bool,
) -> tuple[float, float, bool]:
    """Return ``(log_bf, log_bf_err, valid)`` for full vs diagonal models."""
    if not (full_valid and diag_valid):
        return np.nan, np.nan, False
    log_bf = float(full_lnz - diag_lnz)
    log_bf_err = float(np.sqrt(full_err**2 + diag_err**2))
    valid = bool(np.isfinite(log_bf) and np.isfinite(log_bf_err))
    return log_bf, log_bf_err, valid


def _to_json_value(value: Any) -> Any:
    """Convert numbers containing NaN/inf to JSON null."""
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (int, np.integer, str, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_to_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_json_value(v) for k, v in value.items()}
    return value


def write_evidence_summary(
    *,
    txt_path: Path,
    json_path: Path,
    run_mode: str,
    full: Mapping[str, Any] | None,
    diag_channels: Iterable[Mapping[str, Any]],
    diag_combined: Mapping[str, Any] | None,
    comparison: Mapping[str, Any] | None,
) -> None:
    """Write evidence comparison summaries in text and JSON format."""
    diag_channels_list = list(diag_channels)
    lines = [f"run_mode: {run_mode}", ""]
    if full is not None:
        lines.append("full hypothesis")
        lines.append(
            f"  valid: {bool(full.get('valid', False))} | lnz: {full.get('lnz')} | lnz_err: {full.get('lnz_err')}"
        )
        lines.append("")
    if diag_channels_list:
        lines.append("diag hypothesis (per channel)")
        for entry in diag_channels_list:
            lines.append(
                f"  {entry.get('name')}: valid={bool(entry.get('valid', False))} | lnz={entry.get('lnz')} | lnz_err={entry.get('lnz_err')}"
            )
        lines.append("")
    if diag_combined is not None:
        lines.append("diag hypothesis (combined)")
        lines.append(
            f"  valid: {bool(diag_combined.get('valid', False))} | lnz: {diag_combined.get('lnz')} | lnz_err: {diag_combined.get('lnz_err')}"
        )
        lines.append("")
    if comparison is not None:
        lines.append("comparison: logBF(full - diag)")
        lines.append(
            f"  valid: {bool(comparison.get('valid', False))} | log_bf: {comparison.get('log_bf')} | log_bf_err: {comparison.get('log_bf_err')}"
        )
        lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "run_mode": run_mode,
        "full": full,
        "diag_channels": diag_channels_list,
        "diag_combined": diag_combined,
        "comparison": comparison,
    }
    json_payload = _to_json_value(payload)
    json_path.write_text(
        json.dumps(json_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
