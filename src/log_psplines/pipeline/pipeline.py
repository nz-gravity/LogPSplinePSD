"""InferencePipeline and PipelineResult."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr

from .stages import NUTSStage, StageResult, VIStage


def _vi_result_to_idata(result: StageResult) -> xr.DataTree:
    """Wrap VI posterior means into a minimal xr.DataTree."""
    if not result.init_values:
        return xr.DataTree()
    ds = xr.Dataset(
        {
            k: xr.DataArray(np.asarray(v)[None, None, ...])
            for k, v in result.init_values.items()
        }
    )
    return xr.DataTree(children={"posterior": xr.DataTree(dataset=ds)})


@dataclass
class PipelineResult:
    """Outputs from InferencePipeline.run()."""

    vi_coarse: Optional[StageResult]
    vi: Optional[StageResult]
    idata: xr.DataTree

    def save(self, outdir: str) -> None:
        os.makedirs(outdir, exist_ok=True)
        self.idata.to_netcdf(os.path.join(outdir, "inference_data.nc"))
        if self.vi is not None and self.vi.losses is not None:
            np.save(
                os.path.join(outdir, "vi_losses.npy"),
                np.asarray(self.vi.losses),
            )
        if self.vi_coarse is not None and self.vi_coarse.losses is not None:
            np.save(
                os.path.join(outdir, "vi_coarse_losses.npy"),
                np.asarray(self.vi_coarse.losses),
            )


class InferencePipeline:
    """Sequential vi_coarse → vi → nuts inference pipeline.

    Each stage receives ``init_values`` from the previous stage so that
    downstream optimisation / sampling starts near the posterior mode.
    """

    def __init__(
        self,
        model_fn: Callable,
        full_model_kwargs: dict,
        coarse_model_kwargs: Optional[dict],
        vi_stage: VIStage,
        nuts_stage: NUTSStage,
        *,
        rng_key: Union[int, jax.Array] = 42,
        verbose: bool = False,
        only_vi: bool = False,
    ) -> None:
        self.model_fn = model_fn
        self.full_model_kwargs = full_model_kwargs
        self.coarse_model_kwargs = coarse_model_kwargs
        self.vi_stage = vi_stage
        self.nuts_stage = nuts_stage
        self.rng_key = rng_key
        self.verbose = verbose
        self.only_vi = only_vi

    def run(self) -> PipelineResult:
        """Execute the pipeline and return a PipelineResult."""
        rng = (
            jax.random.PRNGKey(self.rng_key)
            if isinstance(self.rng_key, int)
            else self.rng_key
        )

        vi_coarse: Optional[StageResult] = None
        init_values: Optional[Dict[str, jnp.ndarray]] = None

        if self.coarse_model_kwargs is not None:
            rng, key = jax.random.split(rng)
            vi_coarse = self.vi_stage.run(
                self.model_fn,
                self.coarse_model_kwargs,
                init_values=None,
                rng_key=key,
                verbose=self.verbose,
            )
            init_values = vi_coarse.init_values

        rng, key = jax.random.split(rng)
        vi = self.vi_stage.run(
            self.model_fn,
            self.full_model_kwargs,
            init_values=init_values,
            rng_key=key,
            verbose=self.verbose,
        )
        init_values = vi.init_values

        if self.only_vi:
            idata = _vi_result_to_idata(vi)
            return PipelineResult(vi_coarse=vi_coarse, vi=vi, idata=idata)

        rng, key = jax.random.split(rng)
        idata = self.nuts_stage.run(
            self.model_fn,
            self.full_model_kwargs,
            init_values=init_values,
            rng_key=key,
            verbose=self.verbose,
        )
        return PipelineResult(vi_coarse=vi_coarse, vi=vi, idata=idata)
