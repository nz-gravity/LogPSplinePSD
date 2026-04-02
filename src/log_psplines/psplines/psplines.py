from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp

from ..datatypes import Periodogram
from .initialisation import init_basis_and_penalty, init_knots, init_weights
from .plot_basis import plot_basis, plot_penalty


@dataclass
class LogPSplines:
    """
    Bayesian log-power spectral density model using penalized B-splines.

    This class implements a flexible non-parametric model for log power spectral
    densities using B-spline basis functions with roughness penalties. The approach
    is particularly well-suited for gravitational wave data analysis, enabling
    smooth spectral reconstruction while preserving sharp spectral features and
    automatically selecting appropriate smoothness levels.

    The model represents the log power spectral density as:

    .. math::
        \\log S(f) = \\sum_{j=1}^{K} w_j B_j(f) + \\log S_{\\text{param}}(f)

    where :math:`B_j(f)` are B-spline basis functions, :math:`w_j` are spline
    coefficients, and :math:`S_{\\text{param}}(f)` is an optional parametric component.
    Smoothness is enforced via a penalty on the :math:`k`-th order differences of
    the coefficients.

    Parameters
    ----------
    degree : int
        Polynomial degree of B-spline basis functions (0-5). Higher degrees
        provide smoother basis functions but require more knots for flexibility.
        Common choices: 1 (linear), 2 (quadratic), 3 (cubic)
    diffMatrixOrder : int
        Order of finite difference penalty matrix (0-4). Controls the type of
        smoothness enforced:

        - 0: Penalizes coefficient magnitude (ridge-like)
        - 1: Penalizes first differences (encourages constant slopes)
        - 2: Penalizes second differences (encourages straight lines)
        - 3+: Higher-order smoothness penalties

    n : int
        Number of frequency bins in the periodogram/data
    basis : jnp.ndarray, shape (n, n_basis)
        B-spline basis matrix evaluated at periodogram frequencies.
        Each column represents one basis function
    penalty_matrix : jnp.ndarray, shape (n_basis, n_basis)
        Roughness penalty matrix for enforcing smoothness. Typically sparse
        with structure determined by `diffMatrixOrder`
    knots : np.ndarray, shape (n_knots,)
        Interior knot positions defining the B-spline basis. Knot placement
        affects model flexibility and computational efficiency
    weights : jnp.ndarray, shape (n_basis,), optional
        Spline coefficients/weights. Initialized to zeros and typically
        estimated during MCMC sampling. Default is None
    parametric_model : jnp.ndarray, shape (n,), optional
        Known parametric component of the power spectrum (e.g., instrumental
        lines, astrophysical templates). Default is None (uniform spectrum)

    Attributes
    ----------
    log_parametric_model : jnp.ndarray
        Cached logarithm of the parametric model component
    order : int
        B-spline order (degree + 1)
    n_knots : int
        Number of interior knots
    n_basis : int
        Number of basis functions (n_knots + degree - 1)

    Methods
    -------
    from_periodogram(periodogram, n_knots, degree, ...)
        Class method to construct model from periodogram data
    __call__(weights, use_parametric_model)
        Evaluate the log power spectrum given spline weights
    plot_basis(outdir)
        Visualize the basis functions and penalty matrix

    Examples
    --------
    Create a log P-spline model from gravitational wave periodogram data:

    >>> from mypackage import Periodogram
    >>> # Assume 'pdgrm' is a Periodogram object from GW strain data
    >>> model = LogPSplines.from_periodogram(
    ...     pdgrm,
    ...     n_knots=15,
    ...     degree=3,
    ...     diffMatrixOrder=2
    ... )
    >>> print(f"Model has {model.n_basis} basis functions")

    Evaluate the model with specific weights:

    >>> import jax.numpy as jnp
    >>> weights = jnp.ones(model.n_basis) * 0.1  # Small positive weights
    >>> log_psd = model(weights)
    >>> psd = jnp.exp(log_psd)  # Convert back to linear scale

    Create model with known parametric component (e.g., 60 Hz line):

    >>> # Create template for 60 Hz line and harmonics
    >>> line_template = create_powerline_template(pdgrm.frequencies)
    >>> model = LogPSplines.from_periodogram(
    ...     pdgrm,
    ...     n_knots=12,
    ...     degree=3,
    ...     parametric_model=line_template
    ... )

    Visualize the basis functions:

    >>> model.plot_basis(outdir="./plots")  # Saves basis_plot.png

    Notes
    -----
    **Mathematical Foundation:**

    The penalized B-spline approach balances model flexibility with smoothness
    by minimizing an objective function of the form:

    .. math::
        L(w) = ||y - Bw||^2 + \\lambda w^T P w

    where :math:`y` is the log periodogram, :math:`B` is the basis matrix,
    :math:`w` are coefficients, :math:`P` is the penalty matrix, and
    :math:`\\lambda` is the smoothing parameter.

    **Computational Considerations:**

    - Basis and penalty matrices are typically sparse, enabling efficient
      computation for large datasets
    - Knot placement affects both model flexibility and numerical stability
    - Higher-degree bases provide smoother interpolation but may be less
      stable with few knots

    **Gravitational Wave Applications:**

    This model excels at:

    - Detector noise characterization and PSD estimation
    - Non-parametric background modeling for burst searches
    - Spectral line detection and characterization
    - Model-independent reconstruction of astrophysical spectra
    - Handling both smooth continuum and sharp spectral features

    **Validation Rules:**

    The class enforces several consistency checks:

    - Degree must be ≥ diffMatrixOrder for mathematical validity
    - Degree limited to 0-5 for numerical stability
    - Number of knots must exceed degree for well-defined basis
    - Parametric model must be specified (defaults to uniform if None)

    Warnings
    --------
    - Very high degrees (>3) may lead to numerical instability
    - Too few knots relative to data resolution can cause underfitting
    - Parametric model should match the frequency grid of the periodogram

    See Also
    --------
    run_mcmc : MCMC sampling with LogPSplines models
    Periodogram : Input data structure for spectral analysis
    init_knots : Knot placement algorithms
    init_basis_and_penalty : Basis and penalty matrix construction
    """

    degree: int
    diffMatrixOrder: int
    n: int
    knots: np.ndarray
    basis: Optional[jnp.ndarray] = None
    penalty_matrix: Optional[jnp.ndarray] = None
    weights: Optional[jnp.ndarray] = None
    parametric_model: Union[jnp.ndarray, None] = None
    grid_points: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate model parameters and check mathematical consistency."""
        if self.degree < self.diffMatrixOrder:
            raise ValueError(
                f"Degree ({self.degree}) must be ≥ diffMatrixOrder ({self.diffMatrixOrder}) "
                "for mathematically well-defined penalty matrix."
            )
        if self.degree not in [0, 1, 2, 3, 4, 5]:
            raise ValueError(
                f"Degree must be between 0 and 5, got {self.degree}. "
                "Higher degrees may cause numerical instability."
            )
        if self.diffMatrixOrder not in [0, 1, 2, 3, 4]:
            raise ValueError(
                f"diffMatrixOrder must be between 0 and 4, got {self.diffMatrixOrder}."
            )
        if len(self.knots) < self.degree:
            raise ValueError(
                f"Number of knots ({len(self.knots)}) must be ≥ degree ({self.degree}) "
                "for well-defined B-spline basis."
            )

        self.knots = np.asarray(self.knots, dtype=np.float64)
        if self.knots.ndim != 1:
            raise ValueError(f"knots must be 1-D, got shape {self.knots.shape}")
        if self.knots.size == 0:
            raise ValueError("knots must be non-empty")
        if not np.all(np.isfinite(self.knots)):
            raise ValueError("knots must be finite")
        if np.any(np.diff(self.knots) < 0):
            raise ValueError("knots must be sorted ascending")

        if self.grid_points is None:
            self.grid_points = np.linspace(
                0.0, 1.0, int(self.n), dtype=np.float64
            )
        else:
            self.grid_points = np.asarray(self.grid_points, dtype=np.float64)
            if self.grid_points.ndim != 1:
                raise ValueError(
                    "grid_points must be 1-D with length n, "
                    f"got shape {self.grid_points.shape}"
                )
            if self.grid_points.shape[0] != int(self.n):
                raise ValueError(
                    f"grid_points length must match n={self.n}, "
                    f"got {self.grid_points.shape[0]}"
                )
            if not np.all(np.isfinite(self.grid_points)):
                raise ValueError("grid_points must be finite")
            if np.any(np.diff(self.grid_points) < 0):
                raise ValueError("grid_points must be sorted ascending")

        if self.basis is None or self.penalty_matrix is None:
            basis, penalty_matrix = init_basis_and_penalty(
                self.knots,
                self.degree,
                self.n,
                self.diffMatrixOrder,
                grid_points=self.grid_points,
            )
            self.basis = basis
            self.penalty_matrix = penalty_matrix

        self.basis = jnp.asarray(self.basis)
        self.penalty_matrix = jnp.asarray(self.penalty_matrix)
        if self.basis.ndim != 2:
            raise ValueError(
                f"basis must be 2-D (n, n_basis), got shape {self.basis.shape}"
            )
        if self.basis.shape[0] != int(self.n):
            raise ValueError(
                f"basis first dimension must match n={self.n}, got {self.basis.shape[0]}"
            )
        if self.penalty_matrix.ndim != 2:
            raise ValueError(
                f"penalty_matrix must be 2-D, got shape {self.penalty_matrix.shape}"
            )
        if self.penalty_matrix.shape[0] != self.penalty_matrix.shape[1]:
            raise ValueError("penalty_matrix must be square")
        if self.penalty_matrix.shape[0] != self.basis.shape[1]:
            raise ValueError(
                "penalty_matrix dimension must match basis n_basis: "
                f"{self.penalty_matrix.shape[0]} vs {self.basis.shape[1]}"
            )

        if self.weights is None:
            self.weights = jnp.zeros(self.basis.shape[1], dtype=self.basis.dtype)
        else:
            self.weights = jnp.asarray(self.weights)
            if self.weights.ndim != 1:
                raise ValueError(
                    f"weights must be 1-D, got shape {self.weights.shape}"
                )
            if self.weights.shape[0] != self.basis.shape[1]:
                raise ValueError(
                    "weights length must match basis n_basis: "
                    f"{self.weights.shape[0]} vs {self.basis.shape[1]}"
                )

        if self.parametric_model is None:
            self.parametric_model = jnp.ones(self.n, dtype=self.basis.dtype)
        else:
            self.parametric_model = jnp.asarray(self.parametric_model)
            if self.parametric_model.ndim != 1:
                raise ValueError(
                    "parametric_model must be 1-D with length n, "
                    f"got shape {self.parametric_model.shape}"
                )
            if self.parametric_model.shape[0] != int(self.n):
                raise ValueError(
                    f"parametric_model length must match n={self.n}, "
                    f"got {self.parametric_model.shape[0]}"
                )
            self.parametric_model = jnp.maximum(self.parametric_model, 1e-24)
        self._log_parametric_model = jnp.log(self.parametric_model)
        assert (
            self.log_parametric_model is not None
        ), "parametric_model must be provided or initialized."

    def __repr__(self):
        return f"LogPSplines(knots={self.n_knots}, degree={self.degree}, penaltyOrder={self.diffMatrixOrder}, n={self.n})"  # , sparsity={self.basis_sparsity:.2f}, penalty_sparsity={self.penalty_sparsity:.2f})"

    @staticmethod
    def _storage_name(field: str, prefix: str | None = None) -> str:
        return f"{prefix}_{field}" if prefix else field

    @staticmethod
    def _storage_dim(dim: str, prefix: str | None = None) -> str:
        return f"{prefix}_{dim}" if prefix else dim

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "values"):
            return np.asarray(value.values)
        return np.asarray(value)

    @classmethod
    def _dataset_scalar(
        cls, dataset: Mapping[str, Any], key: str
    ) -> float | int:
        value = cls._as_numpy(dataset[key])
        if value.ndim == 0:
            return value.item()
        if value.size == 1:
            return value.reshape(()).item()
        raise ValueError(f"Expected scalar at key '{key}', got shape {value.shape}")

    def to_storage_payload(
        self,
        *,
        prefix: str | None = None,
        include_linear_operators: bool = False,
    ) -> tuple[dict[str, tuple[list[str], np.ndarray]], dict[str, np.ndarray]]:
        """Return dataset-ready payload for this spline component."""
        knots_key = self._storage_name("knots", prefix)
        grid_key = self._storage_name("grid_points", prefix)
        param_key = self._storage_name("parametric_model", prefix)
        knots_dim = self._storage_dim("knots_dim", prefix)
        freq_dim = self._storage_dim("freq", prefix)

        data: dict[str, tuple[list[str], np.ndarray]] = {
            knots_key: ([knots_dim], np.asarray(self.knots, dtype=np.float64)),
            grid_key: ([freq_dim], np.asarray(self.grid_points, dtype=np.float64)),
            param_key: ([freq_dim], np.asarray(self.parametric_model)),
        }
        coords: dict[str, np.ndarray] = {
            knots_dim: np.arange(len(self.knots)),
            freq_dim: np.arange(int(self.n)),
        }

        if include_linear_operators:
            basis_key = self._storage_name("basis", prefix)
            penalty_key = self._storage_name("penalty_matrix", prefix)
            weights_dim = self._storage_dim("weights_dim", prefix)
            weights_dim_row = self._storage_dim("weights_dim_row", prefix)
            weights_dim_col = self._storage_dim("weights_dim_col", prefix)
            data[basis_key] = (
                [freq_dim, weights_dim],
                np.asarray(self.basis),
            )
            data[penalty_key] = (
                [weights_dim_row, weights_dim_col],
                np.asarray(self.penalty_matrix),
            )
            coords[weights_dim] = np.arange(int(self.basis.shape[1]))
            coords[weights_dim_row] = np.arange(
                int(self.penalty_matrix.shape[0])
            )
            coords[weights_dim_col] = np.arange(
                int(self.penalty_matrix.shape[1])
            )

        return data, coords

    @classmethod
    def from_storage_dataset(
        cls,
        dataset: Mapping[str, Any],
        *,
        prefix: str | None = None,
        degree: int | None = None,
        diffMatrixOrder: int | None = None,
        n: int | None = None,
    ) -> "LogPSplines":
        """Rehydrate a spline model component from a saved dataset mapping."""
        knots_key = cls._storage_name("knots", prefix)
        if knots_key not in dataset:
            raise KeyError(f"Missing required spline key '{knots_key}'.")
        knots = np.asarray(cls._as_numpy(dataset[knots_key]), dtype=np.float64)

        if degree is None:
            if "degree" not in dataset:
                raise KeyError("Missing required scalar 'degree'.")
            degree = int(cls._dataset_scalar(dataset, "degree"))
        if diffMatrixOrder is None:
            if "diffMatrixOrder" not in dataset:
                raise KeyError("Missing required scalar 'diffMatrixOrder'.")
            diffMatrixOrder = int(cls._dataset_scalar(dataset, "diffMatrixOrder"))

        grid_key = cls._storage_name("grid_points", prefix)
        param_key = cls._storage_name("parametric_model", prefix)
        basis_key = cls._storage_name("basis", prefix)
        penalty_key = cls._storage_name("penalty_matrix", prefix)

        grid_points = (
            np.asarray(cls._as_numpy(dataset[grid_key]), dtype=np.float64)
            if grid_key in dataset
            else None
        )
        parametric_model = (
            jnp.asarray(cls._as_numpy(dataset[param_key]))
            if param_key in dataset
            else None
        )
        basis = (
            jnp.asarray(cls._as_numpy(dataset[basis_key]))
            if basis_key in dataset
            else None
        )
        penalty_matrix = (
            jnp.asarray(cls._as_numpy(dataset[penalty_key]))
            if penalty_key in dataset
            else None
        )

        if n is None:
            if "n" in dataset:
                n = int(cls._dataset_scalar(dataset, "n"))
            elif "N" in dataset:
                n = int(cls._dataset_scalar(dataset, "N"))
            elif grid_points is not None:
                n = int(grid_points.shape[0])
            elif parametric_model is not None:
                n = int(parametric_model.shape[0])
            elif basis is not None:
                n = int(basis.shape[0])
            else:
                raise KeyError(
                    "Could not infer n while loading spline component. "
                    "Provide n explicitly or store one of "
                    "{n, N, *_grid_points, *_parametric_model, *_basis}."
                )

        return cls(
            degree=int(degree),
            diffMatrixOrder=int(diffMatrixOrder),
            n=int(n),
            knots=knots,
            basis=basis,
            penalty_matrix=penalty_matrix,
            weights=None,
            parametric_model=parametric_model,
            grid_points=grid_points,
        )

    @classmethod
    def from_knots(
        cls,
        *,
        knots: np.ndarray,
        degree: int,
        diffMatrixOrder: int,
        n: int,
        grid_points: np.ndarray | None = None,
        parametric_model: jnp.ndarray | None = None,
        log_target: jnp.ndarray | np.ndarray | None = None,
        init_num_steps: int = 5000,
    ) -> "LogPSplines":
        """Build from knots/grid and optionally initialize weights from log-target.

        This is the high-level constructor for research workflows:
        it computes basis + penalty internally and can fit initial spline
        weights directly from a supplied log-spectrum target.
        """
        model = cls(
            knots=np.asarray(knots, dtype=np.float64),
            degree=degree,
            diffMatrixOrder=diffMatrixOrder,
            n=int(n),
            basis=None,
            penalty_matrix=None,
            weights=None,
            parametric_model=parametric_model,
            grid_points=grid_points,
        )
        if log_target is not None:
            target = jnp.asarray(log_target)
            if target.ndim != 1 or target.shape[0] != int(n):
                raise ValueError(
                    "log_target must be 1-D with length n, "
                    f"got shape {target.shape} for n={n}"
                )
            model.weights = init_weights(
                target,
                model,
                init_weights=model.weights,
                num_steps=int(init_num_steps),
            )
        return model

    @classmethod
    def from_periodogram(
        cls,
        periodogram: Periodogram,
        n_knots: int,
        degree: int,
        diffMatrixOrder: int = 3,
        parametric_model: jnp.ndarray | None = None,
        knot_kwargs: Optional[dict] = None,
    ):
        """
        Construct LogPSplines model from periodogram data.

        This factory method handles the complete setup of a log P-spline model,
        including knot placement, basis construction, penalty matrix setup, and
        initial weight estimation.

        Parameters
        ----------
        periodogram : Periodogram
            Input periodogram containing frequency grid and power measurements
        n_knots : int
            Number of interior knots to place. More knots provide greater
            flexibility but may lead to overfitting without sufficient penalty
        degree : int
            Polynomial degree of B-spline basis (0-5)
        diffMatrixOrder : int, default=2
            Order of difference penalty matrix for smoothness control
        parametric_model : jnp.ndarray, optional
            Known parametric component to include in the model
        knot_kwargs : dict, default={}
            Additional arguments passed to knot placement algorithm.
            May include placement strategy, boundary conditions, etc.

        Returns
        -------
        LogPSplines
            Fully initialized model ready for MCMC sampling

        Examples
        --------
        Basic model construction:

        >>> model = LogPSplines.from_periodogram(pdgrm, n_knots=10, degree=3)

        High-resolution model for detailed spectral features:

        >>> model = LogPSplines.from_periodogram(
        ...     pdgrm,
        ...     n_knots=25,
        ...     degree=3,
        ...     diffMatrixOrder=2,
        ...     knot_kwargs={'placement': 'adaptive'}
        ... )
        """
        if knot_kwargs is None:
            knot_kwargs = {}

        parametric_np = (
            None
            if parametric_model is None
            else np.asarray(parametric_model, dtype=np.float64)
        )
        knots = init_knots(
            n_knots,
            periodogram,
            parametric_np,
            **knot_kwargs,
        )
        # compute degree based on the number of knots
        # Evaluate basis at actual normalized frequencies to preserve geometry
        fmin, fmax = float(periodogram.freqs[0]), float(periodogram.freqs[-1])
        denom = (fmax - fmin) if fmax > fmin else 1.0
        grid = (np.asarray(periodogram.freqs) - fmin) / denom
        model = cls.from_knots(
            knots=knots,
            degree=degree,
            diffMatrixOrder=diffMatrixOrder,
            n=periodogram.n,
            grid_points=grid,
            parametric_model=parametric_model,
            log_target=jnp.log(periodogram.power),
            init_num_steps=5000,
        )
        return model

    @property
    def log_parametric_model(self) -> jnp.ndarray:
        """
        Logarithm of the parametric model component.

        Returns
        -------
        jnp.ndarray, shape (n,)
            Log of parametric model, defaulting to log(1) = 0 if None provided
        """
        if not hasattr(self, "_log_parametric_model"):
            if self.parametric_model is None:
                self.parametric_model = jnp.ones(self.n)
            self._log_parametric_model = jnp.log(self.parametric_model)
        return self._log_parametric_model

    @property
    def order(self) -> int:
        return self.degree + 1

    @property
    def n_knots(self) -> int:
        return len(self.knots)

    @property
    def n_basis(self) -> int:
        if self.basis is not None:
            return int(self.basis.shape[1])
        return self.n_knots + self.degree - 1

    def __call__(
        self, weights: jnp.ndarray | None = None, use_parametric_model=True
    ) -> jnp.ndarray:
        """
        Evaluate the log power spectral density.

        Computes the weighted sum of B-spline basis functions plus the
        logarithmic parametric component.

        Parameters
        ----------
        weights : jnp.ndarray, shape (n_basis,), optional
            Spline coefficients. Uses stored weights if None
        use_parametric_model : bool, default=True
            Whether to include the parametric component in evaluation

        Returns
        -------
        jnp.ndarray, shape (n,)
            Log power spectral density values at periodogram frequencies

        Examples
        --------
        >>> log_psd = model(weights)
        >>> psd = jnp.exp(log_psd)  # Convert to linear scale

        >>> # Evaluate spline component only (no parametric model)
        >>> log_spline_only = model(weights, use_parametric_model=False)
        """
        if weights is None:
            weights = self.weights
        if weights is None:
            raise ValueError("weights must be provided or initialized.")
        ln_para = self.log_parametric_model
        if not use_parametric_model:
            ln_para = jnp.zeros_like(ln_para)
        return build_spline(self.basis, weights, ln_para)

    def plot_basis(self, outdir: str | None = None):
        """
        Visualize B-spline basis functions and penalty matrix structure.

        Creates a three-panel plot showing:
        1. Individual B-spline basis functions
        2. Basis function overview
        3. Penalty matrix structure (sparsity pattern)

        Parameters
        ----------
        outdir : str, optional
            Directory to save the plot. If None, displays interactively

        Examples
        --------
        >>> model.plot_basis()  # Display plot
        >>> model.plot_basis(outdir="./diagnostics")  # Save to file
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        plot_basis(np.asarray(self.basis), axes=axes[:2])
        plot_penalty(np.asarray(self.penalty_matrix), ax=axes[2])
        plt.tight_layout()
        if outdir is not None:
            fig.savefig(f"{outdir}/basis_plot.png", bbox_inches="tight")


@jax.jit
def build_spline(
    ln_spline_basis: jnp.ndarray,
    weights: jnp.ndarray,
    log_parametric: jnp.ndarray,
) -> jnp.ndarray:
    return jnp.einsum("ij,j->i", ln_spline_basis, weights) + log_parametric
