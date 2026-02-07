from __future__ import annotations

from typing import TYPE_CHECKING, Any


class _AnyJaxtype:
    """Fallback jaxtyping annotation placeholder."""

    def __class_getitem__(cls, _: object) -> Any:
        return Any


if TYPE_CHECKING:
    from jaxtyping import Bool, Complex, Float, Int
else:
    try:
        from jaxtyping import Bool, Complex, Float, Int
    except Exception:
        Bool = Complex = Float = Int = _AnyJaxtype

    # Keep a runtime fallback if optional dependency import fails.
    if "Bool" not in globals():
        Bool = Complex = Float = Int = _AnyJaxtype


__all__ = ["Bool", "Complex", "Float", "Int"]
