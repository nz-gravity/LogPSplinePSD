"""Proper complex coefficient sums on grids or paired time-frequency sites."""

from dataclasses import dataclass, replace

import numpy as np

from log_psplines.data.spectral_utils import Y_to_U


def _grid_axis(values: np.ndarray, name: str) -> np.ndarray:
    axis = np.asarray(values, dtype=float)
    if (
        axis.ndim != 1
        or not axis.size
        or not np.isfinite(axis).all()
        or np.any(np.diff(axis) <= 0)
    ):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return axis.copy()


@dataclass
class WishartGridData:
    """Normalized proper complex statistics Y=U U^H and observation counts.

    u_re/u_im: (T,F,C,R) on a grid or (Q,C,R) at paired time/frequency
    sites; counts: scalar or (T,F)/(Q,). Each input coefficient
    has covariance S in the stated PSD units. Y is a SUM, never an average;
    R is factor width, independent of counts. Widths above C are compressed
    once. Rank-one and zero-count cells need no artificial diagonal power.
    reference_time/frequency retain the grids before masking/pooling. Time
    centring is defined on reference_time and must survive receiving pooled
    data. Counts assert the adopted independent proper complex observation
    model; this container cannot establish independence of transform tiles.
    """

    u_re: np.ndarray
    u_im: np.ndarray
    time: np.ndarray
    frequency: np.ndarray
    counts: int | float | np.ndarray = 1
    reference_time: np.ndarray | None = None
    reference_frequency: np.ndarray | None = None
    units: str = "data_unit^2/Hz"
    normalization: str = "proper_complex_covariance_psd"

    def __post_init__(self) -> None:
        real, imag = np.asarray(self.u_re), np.asarray(self.u_im)
        if np.iscomplexobj(real) or np.iscomplexobj(imag):
            raise ValueError("u_re and u_im must be real arrays")
        if (
            real.shape != imag.shape
            or real.ndim not in (3, 4)
            or min(real.shape) < 1
        ):
            raise ValueError("factors must have shape (T,F,C,R) or (Q,C,R)")
        for name in ("time", "frequency"):
            axis = np.asarray(getattr(self, name), dtype=float)
            if real.ndim == 4:
                axis = _grid_axis(axis, name)
            elif (
                axis.ndim != 1
                or axis.size != real.shape[0]
                or not np.isfinite(axis).all()
            ):
                raise ValueError(f"{name} must be finite with shape (Q,)")
            setattr(self, name, axis.copy())
        if real.ndim == 4 and real.shape[:2] != (
            len(self.time),
            len(self.frequency),
        ):
            raise ValueError("factors must have shape (T,F,C,R) with C,R >= 1")
        observation_shape = real.shape[:-2]
        counts = np.asarray(self.counts, dtype=float)
        if counts.ndim != 0 and counts.shape != observation_shape:
            raise ValueError(
                "counts must be scalar or match the observation shape"
            )
        if (
            not np.isfinite(counts).all()
            or np.any(counts < 0)
            or np.any(counts != np.floor(counts))
        ):
            raise ValueError("counts must be finite nonnegative integers")
        self.counts = np.broadcast_to(counts, observation_shape).copy()
        active = self.counts > 0
        if (
            not np.isfinite(real[active]).all()
            or not np.isfinite(imag[active]).all()
        ):
            raise ValueError("active factors must be finite")
        self.u_re = np.where(active[..., None, None], real, 0.0).astype(float)
        self.u_im = np.where(active[..., None, None], imag, 0.0).astype(float)
        if real.shape[-1] > real.shape[-2]:
            factors = Y_to_U(self.Y.reshape(-1, self.p, self.p)).reshape(
                *observation_shape, self.p, self.p
            )
            self.u_re, self.u_im = factors.real, factors.imag
        for name in ("time", "frequency"):
            reference = getattr(self, f"reference_{name}")
            reference = _grid_axis(
                np.unique(getattr(self, name))
                if reference is None
                else reference,
                f"reference_{name}",
            )
            axis = getattr(self, name)
            if axis.min() < reference[0] or axis.max() > reference[-1]:
                raise ValueError(f"{name} must lie inside its reference grid")
            setattr(self, f"reference_{name}", reference)

    @classmethod
    def from_coefficients(
        cls,
        coefficients: np.ndarray,
        time: np.ndarray,
        frequency: np.ndarray,
        *,
        units: str = "data_unit^2/Hz",
        normalization: str = "proper_complex_covariance_psd",
    ) -> "WishartGridData":
        """Prepare normalized complex vectors (T,F,C) or (T,F,C,replicate).

        Real WDM coefficients require a different likelihood/quadrature;
        casting them to complex does not make them proper complex data.
        """
        values = np.asarray(coefficients)
        if not np.iscomplexobj(values):
            raise ValueError(
                "proper complex coefficients required; real WDM coefficients are unsupported"
            )
        if values.ndim == 3:
            values = values[..., None]
        if values.ndim != 4:
            raise ValueError(
                "coefficients must have shape (T,F,C[,replicate])"
            )
        return cls(
            values.real,
            values.imag,
            time,
            frequency,
            counts=values.shape[-1],
            units=units,
            normalization=normalization,
        )

    @classmethod
    def from_scattered_coefficients(
        cls,
        coefficients: np.ndarray,
        time: np.ndarray,
        frequency: np.ndarray,
        *,
        reference_time: np.ndarray | None = None,
        reference_frequency: np.ndarray | None = None,
        units: str = "data_unit^2/Hz",
        normalization: str = "proper_complex_covariance_psd",
    ) -> "WishartGridData":
        """Prepare complex (Q,C[,replicate]) vectors at paired finite sites."""
        values = np.asarray(coefficients)
        if not np.iscomplexobj(values):
            raise ValueError(
                "proper complex coefficients required; real WDM coefficients are unsupported"
            )
        if values.ndim == 2:
            values = values[..., None]
        if values.ndim != 3:
            raise ValueError("coefficients must have shape (Q,C[,replicate])")
        return cls(
            values.real,
            values.imag,
            time,
            frequency,
            counts=values.shape[-1],
            reference_time=reference_time,
            reference_frequency=reference_frequency,
            units=units,
            normalization=normalization,
        )

    @property
    def is_grid(self) -> bool:
        """Whether observations occupy a rectangular time-frequency grid."""
        return self.u_re.ndim == 4

    @property
    def is_scattered(self) -> bool:
        """Whether observations occupy paired time-frequency sites."""
        return self.u_re.ndim == 3

    @property
    def p(self) -> int:
        return self.u_re.shape[-2]

    @property
    def U(self) -> np.ndarray:
        """Compact factors (T,F,C,R) or (Q,C,R); R is not a count."""
        return self.u_re + 1j * self.u_im

    @property
    def Y(self) -> np.ndarray:
        """Summed outer products (T,F,C,C) or (Q,C,C)."""
        factors = self.U
        return factors @ factors.conj().swapaxes(-1, -2)

    @property
    def empirical_spectrum(self) -> np.ndarray:
        """Y/count in PSD units; omitted cells are NaN for display only."""
        return np.divide(
            self.Y,
            self.counts[..., None, None],
            out=np.full((*self.counts.shape, self.p, self.p), np.nan + 0j),
            where=self.counts[..., None, None] > 0,
        )

    def mask(self, observed: np.ndarray) -> "WishartGridData":
        """Keep observed (T,F) cells or (Q,) sites, retaining references."""
        observed = np.asarray(observed)
        if observed.dtype != np.bool_ or observed.shape != self.counts.shape:
            raise ValueError("observed must be boolean and match counts shape")
        return replace(self, counts=np.where(observed, self.counts, 0))
