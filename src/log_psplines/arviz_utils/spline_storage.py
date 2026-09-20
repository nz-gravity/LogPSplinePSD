from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import jax.numpy as jnp
import numpy as np

from log_psplines.basis import SplineBasis
from log_psplines.models.spectrum import LogPSpline


def _storage_name(field: str, prefix: str | None = None) -> str:
    return f"{prefix}_{field}" if prefix else field


def _storage_dim(dim: str, prefix: str | None = None) -> str:
    return f"{prefix}_{dim}" if prefix else dim


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "values"):
        return np.asarray(value.values)
    return np.asarray(value)


def _dataset_scalar(dataset: Mapping[str, Any], key: str) -> float | int:
    value = _as_numpy(dataset[key])
    if value.ndim == 0:
        return value.item()
    if value.size == 1:
        return value.reshape(()).item()
    raise ValueError(
        f"Expected scalar at key '{key}', got shape {value.shape}"
    )


def to_storage_payload(
    model: LogPSpline,
    *,
    prefix: str | None = None,
    include_linear_operators: bool = False,
) -> tuple[dict[str, tuple[list[str], np.ndarray]], dict[str, np.ndarray]]:
    """Return dataset-ready payload for a stationary spline component."""
    if model.time is not None:
        raise NotImplementedError(
            "Time-frequency model storage is not yet supported"
        )
    knots_key = _storage_name("knots", prefix)
    grid_key = _storage_name("grid_points", prefix)
    knots_dim = _storage_dim("knots_dim", prefix)
    freq_dim = _storage_dim("freq", prefix)

    data: dict[str, tuple[list[str], np.ndarray]] = {
        knots_key: ([knots_dim], np.asarray(model.knots, dtype=np.float64)),
        grid_key: (
            [freq_dim],
            np.asarray(model.grid_points, dtype=np.float64),
        ),
    }
    coords: dict[str, np.ndarray] = {
        knots_dim: np.arange(len(model.knots)),
        freq_dim: np.arange(int(model.n)),
    }

    # Clamped knots are a different representation from historical breakpoints.
    # Store the exact operators so loading never guesses a backend or prior.
    include_linear_operators |= model.frequency.knot_convention == "clamped"
    for name in ("penalty_normalization", "penalty_ridge", "knot_convention"):
        data[_storage_name(name, prefix)] = (
            [],
            np.asarray(getattr(model.frequency, name)),
        )

    if include_linear_operators:
        basis_key = _storage_name("basis", prefix)
        penalty_key = _storage_name("penalty_matrix", prefix)
        weights_dim = _storage_dim("weights_dim", prefix)
        weights_dim_row = _storage_dim("weights_dim_row", prefix)
        weights_dim_col = _storage_dim("weights_dim_col", prefix)
        data[basis_key] = (
            [freq_dim, weights_dim],
            np.asarray(model.basis),
        )
        data[penalty_key] = (
            [weights_dim_row, weights_dim_col],
            np.asarray(model.penalty_matrix),
        )
        coords[weights_dim] = np.arange(int(model.basis.shape[1]))
        coords[weights_dim_row] = np.arange(int(model.penalty_matrix.shape[0]))
        coords[weights_dim_col] = np.arange(int(model.penalty_matrix.shape[1]))

    return data, coords


def from_storage_dataset(
    dataset: Mapping[str, Any],
    *,
    prefix: str | None = None,
    degree: int | None = None,
    diffMatrixOrder: int | None = None,
    n: int | None = None,
) -> "LogPSpline":
    """Rehydrate a spline model component from a saved dataset mapping."""
    knots_key = _storage_name("knots", prefix)
    if knots_key not in dataset:
        raise KeyError(f"Missing required spline key '{knots_key}'.")
    knots = np.asarray(_as_numpy(dataset[knots_key]), dtype=np.float64)

    if degree is None:
        if "degree" not in dataset:
            raise KeyError("Missing required scalar 'degree'.")
        degree = int(_dataset_scalar(dataset, "degree"))
    if diffMatrixOrder is None:
        if "diffMatrixOrder" not in dataset:
            raise KeyError("Missing required scalar 'diffMatrixOrder'.")
        diffMatrixOrder = int(_dataset_scalar(dataset, "diffMatrixOrder"))

    grid_key = _storage_name("grid_points", prefix)
    basis_key = _storage_name("basis", prefix)
    penalty_key = _storage_name("penalty_matrix", prefix)

    grid_points = (
        np.asarray(_as_numpy(dataset[grid_key]), dtype=np.float64)
        if grid_key in dataset
        else None
    )
    basis = (
        jnp.asarray(_as_numpy(dataset[basis_key]))
        if basis_key in dataset
        else None
    )
    penalty_matrix = (
        jnp.asarray(_as_numpy(dataset[penalty_key]))
        if penalty_key in dataset
        else None
    )

    if n is None:
        if "n" in dataset:
            n = int(_dataset_scalar(dataset, "n"))
        elif "N" in dataset:
            n = int(_dataset_scalar(dataset, "N"))
        elif grid_points is not None:
            n = int(grid_points.shape[0])
        elif basis is not None:
            n = int(basis.shape[0])
        else:
            raise KeyError(
                "Could not infer n while loading spline component. "
                "Provide n explicitly or store one of "
                "{n, N, *_grid_points,  *_basis}."
            )

    frequency = SplineBasis.create(
        degree=int(degree),
        penalty_order=int(diffMatrixOrder),
        n=int(n),
        knots=knots,
        basis=basis,
        penalty=penalty_matrix,
        grid=grid_points,
        normalization=(
            str(
                _as_numpy(
                    dataset[_storage_name("penalty_normalization", prefix)]
                ).item()
            )
            if _storage_name("penalty_normalization", prefix) in dataset
            else "max"
        ),
        ridge=(
            float(
                _as_numpy(
                    dataset[_storage_name("penalty_ridge", prefix)]
                ).item()
            )
            if _storage_name("penalty_ridge", prefix) in dataset
            else 1e-6
        ),
    )
    key = _storage_name("knot_convention", prefix)
    if key in dataset:
        frequency = replace(
            frequency, knot_convention=str(_as_numpy(dataset[key]).item())
        )
    return LogPSpline(frequency)
