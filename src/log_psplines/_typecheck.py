from __future__ import annotations

import importlib
import os
from functools import lru_cache, wraps
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def _typecheck_enabled() -> bool:
    flag = os.getenv("LOG_PSPLINES_RUNTIME_TYPECHECK", "1").strip().lower()
    return flag not in {"0", "false", "off", "no"}


@lru_cache(maxsize=1)
def _load_checkers() -> (
    tuple[Callable[..., Any] | None, Callable[..., Any] | None]
):
    """Load runtime typechecker callables without hard import requirements."""
    try:
        beartype_module = importlib.import_module("beartype")
        jaxtyping_module = importlib.import_module("jaxtyping")
    except Exception:  # pragma: no cover - optional dev dependency path
        return None, None

    beartype_callable = getattr(beartype_module, "beartype", None)
    jaxtyped_callable = getattr(jaxtyping_module, "jaxtyped", None)
    return jaxtyped_callable, beartype_callable


def runtime_typecheck(func: F) -> F:
    """Apply jaxtyping+beartype checks when available."""
    jaxtyped_callable, beartype_callable = _load_checkers()
    if jaxtyped_callable is None or beartype_callable is None:
        return func

    checked = jaxtyped_callable(typechecker=beartype_callable)(func)

    @wraps(func)
    def _wrapped(*args: Any, **kwargs: Any):
        if _typecheck_enabled():
            return checked(*args, **kwargs)
        return func(*args, **kwargs)

    return cast(F, _wrapped)
