import os

import numpy as np
import pytest
import xarray as xr

from log_psplines.arviz_utils.to_arviz import _prepare_attributes_and_dims
from log_psplines.datatypes.multivar import EmpiricalPSD, _get_coherence
from log_psplines.samplers.base_sampler import SamplerConfig


def test_prepare_attributes_filters_nonserialisable(tmp_path):
    psd = np.ones((3, 2, 2), dtype=np.complex128)
    emp = EmpiricalPSD(
        freq=np.array([0.1, 0.2, 0.3]),
        psd=psd,
        coherence=_get_coherence(psd),
        channels=np.array([0, 1]),
    )
    cfg = SamplerConfig(
        outdir=None,
        extra_empirical_psd=[emp],
        extra_empirical_labels=["Welch"],
        extra_empirical_styles=[{"zorder": -10}],
    )

    attributes = {}
    coords = {}
    dims = {}
    samples = {"x": np.zeros((1, 1))}
    sample_stats = {}

    _prepare_attributes_and_dims(
        cfg,
        attributes,
        samples,
        coords,
        dims,
        sample_stats,
        data=None,
        model=None,
    )

    assert "extra_empirical_psd" not in attributes
    assert "extra_empirical_styles" not in attributes

    ds = xr.Dataset(attrs=attributes)
    out = tmp_path / "attrs.nc"
    ds.to_netcdf(out)
    assert os.path.exists(out)
