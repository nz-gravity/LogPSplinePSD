"""
Abstract base classes and factory pattern for models and data.
"""

from typing import Union, Optional, Any, Dict
from abc import ABC, abstractmethod

import arviz as az
import jax.numpy as jnp

from ..datatypes import Periodogram
from ..datatypes.multivar import MultivarFFT
from ..psplines import LogPSplines, MultivariateLogPSplines


# Abstract data types
class BaseData(ABC):
    """Abstract base for all data types."""

    @property
    @abstractmethod
    def n_freq(self) -> int:
        """Number of frequency bins."""
        pass

    @property
    @abstractmethod
    def freq(self) -> jnp.ndarray:
        """Frequency array."""
        pass


class BaseModel(ABC):
    """Abstract base for all P-spline models."""

    @property
    @abstractmethod
    def n_knots(self) -> int:
        """Number of knots in the B-spline basis."""
        pass

    @property
    @abstractmethod
    def degree(self) -> int:
        """Degree of the B-spline basis functions."""
        pass


class Model(ABC):
    """Abstract base class for PSD models."""
    pass


class ModelFactory:
    """Factory for creating appropriate models based on data type."""

    @staticmethod
    def create_model(
        data: Union[Periodogram, MultivarFFT],
        n_knots: int = 10,
        degree: int = 3,
        diffMatrixOrder: int = 2,
        knot_kwargs: dict = {},
        parametric_model: Optional[jnp.ndarray] = None,
    ) -> Union[LogPSplines, MultivariateLogPSplines]:
        """
        Create an appropriate model instance based on input data type.

        Parameters
        ----------
        data : Periodogram or MultivarFFT
            Input data
        n_knots : int, default 10
            Number of knots for B-splines
        degree : int, default 3
            Degree of B-spline basis
        diffMatrixOrder : int, default 2
            Order of difference matrix for smoothness penalty
        knot_kwargs : dict, optional
            Additional keyword arguments for knot allocation
        parametric_model : jnp.ndarray, optional
            Known parametric component (univariate only)

        Returns
        -------
        LogPSplines or MultivariateLogPSplines
            Initialized model instance
        """

        if isinstance(data, Periodogram):
            # Univariate case
            return LogPSplines.from_periodogram(
                data,
                n_knots=n_knots,
                degree=degree,
                diffMatrixOrder=diffMatrixOrder,
                parametric_model=parametric_model,
                knot_kwargs=knot_kwargs,
            )
        elif isinstance(data, MultivarFFT):
            # Multivariate case
            if parametric_model is not None:
                raise ValueError("parametric_model is not supported for multivariate data. "
                               "Parametric models are only available for univariate Periodogram data.")
            return MultivariateLogPSplines.from_multivar_fft(
                data,
                n_knots=n_knots,
                degree=degree,
                diffMatrixOrder=diffMatrixOrder,
                knot_kwargs=knot_kwargs,
            )
        else:
            raise ValueError(f"Unsupported data type: {type(data)}. "
                           "Expected Periodogram or MultivarFFT.")
