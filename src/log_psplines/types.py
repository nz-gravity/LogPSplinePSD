import os
from typing import Optional, TypedDict


class SamplerKwargs(TypedDict, total=False):
    alpha_phi: float
    beta_phi: float
    alpha_delta: float
    beta_delta: float
    rng_key: int
    verbose: bool
    outdir: Optional[str]


class NUTSKwargs(SamplerKwargs, total=False):
    target_accept_prob: float
    max_tree_depth: int
    dense_mass: bool


class MHKwargs(SamplerKwargs, total=False):
    target_accept_rate: float
    adaptation_window: int
    adaptation_start: int
    step_size_factor: float
    min_step_size: float
    max_step_size: float


class SplineKwargs(TypedDict, total=False):
    n_knots: int
    degree: int
    diffMatrixOrder: int
    knot_kwargs: dict
