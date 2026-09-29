"""Synthetic stationary and locally stationary benchmark datasets."""

from log_psplines.example_datasets.ls2_data import LS2Data
from log_psplines.example_datasets.tvvar_data import TVData, TVVARData
from log_psplines.example_datasets.varma_data import VARMAData

__all__ = ["VARMAData", "LS2Data", "TVData", "TVVARData"]
