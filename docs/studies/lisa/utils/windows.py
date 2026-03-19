"""Window specification helpers for LISA study.

Adapted from lisa_multivar.py _window_spec / _window_slug / _welch_window_arg.
"""

from __future__ import annotations


def window_spec(
    name: str, *, tukey_alpha: float = 0.1
) -> str | tuple[str, float] | None:
    """Parse a window name into a run_mcmc-compatible spec.

    Returns
    -------
    None            : rectangular / no taper
    "hann"          : Hann taper
    ("tukey", alpha): Tukey taper with the given alpha
    """
    key = str(name).strip().lower()
    if key in ("none", "rect", "rectangular"):
        return None
    if key == "hann":
        return "hann"
    if key == "tukey":
        if not (0.0 <= float(tukey_alpha) <= 1.0):
            raise ValueError(
                f"Tukey alpha must be in [0, 1], got {tukey_alpha}."
            )
        return ("tukey", float(tukey_alpha))
    raise ValueError(
        f"Unsupported window name '{name}'. "
        "Choices: none, rect, rectangular, hann, tukey."
    )


def window_slug(spec: str | tuple[str, float] | None) -> str:
    """Return a filesystem-safe slug for a window spec."""
    if spec is None:
        return "rect"
    if isinstance(spec, tuple):
        name, alpha = spec
        alpha_slug = f"{float(alpha):g}".replace(".", "p")
        return f"{str(name)}{alpha_slug}"
    return str(spec)


def welch_window_arg(
    spec: str | tuple[str, float] | None,
) -> str | tuple[str, float]:
    """Convert a window spec to a scipy.signal.welch-compatible argument.

    scipy does not accept None; use "boxcar" for rectangular.
    """
    if spec is None:
        return "boxcar"
    return spec
