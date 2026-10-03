# Opt-in diagnostics

Existing fits collect no guide-density diagnostics unless requested. Their training/posterior RNG streams and legacy early-stopping rule are preserved. Scalar VI dispatch, initialization and timings reuse the user's existing local implementation.

```python
from log_psplines import PowerConfig, fit

config = PowerConfig(
    method="vi",
    vi_guide="lowrank:10",
    vi_steps=15000,
    vi_posterior_draws=4096,
    vi_early_stopping=False,
    vi_diagnostics={
        "seeds": (8101, 8102, 8103),
        "num_particles": 4096,
        "chunk_size": 128,
        "evaluation_seeds": (8201, 8202, 8203, 8204),
        "evaluation_particles": 32,
        "checkpoint_steps": (1000, 5000, 15000),
    },
)
result = fit(data, config, model=spline)
state = result.vi.diagnostics
print(state.metadata["weights"])
state.save("guide-checkpoint")
result.to_netcdf("result.nc")
```

`StationaryConfig` exposes the same `vi_diagnostics` and `vi_early_stopping` options. Stationary Cholesky factors retain separate guide checkpoints. Independent factor draws use distinct diagnostic seeds; aligned joint log ratios are summed before fitting joint k. Shared latent sites are rejected rather than treated as independent factors.

For a prepared NumPyro model, call the existing `fit_vi` with `diagnostics=VIDiagnosticConfig(...)`. Supply a fingerprint of all closed-over data, basis, prior/configuration, normalization and source identity. Without one, the state records an unavailable fingerprint and cannot pass a same-target comparison. `model_args` iterators are materialized once.

Recover the trained density only against the identical prepared model:

```python
from log_psplines.diagnostics.variational import (
    VIDiagnosticState, rebuild_guide,
)

state = VIDiagnosticState.load("guide-checkpoint")
guide = rebuild_guide(
    state, prepared_model,
    target_fingerprint=matching_fingerprint,
)
q_unconstrained = guide.get_posterior(state.params)
```

The versioned reconstruction recipe supports built-in diagonal, low-rank and full Gaussian guides. It checks the NumPyro version, fingerprint, latent site order, shapes, dtypes and support transform schema. It does not reconstruct an executable model from numerical data or estimate the original guide with a KDE. Flow/custom reconstruction and optimizer-state resumption are explicitly unsupported.

`packed_log_densities` evaluates the NumPyro potential and matching guide density in unconstrained coordinates, including prior correction factors and one support Jacobian. Deterministic coefficients/spectral pixels are not extra latent density dimensions. Native PSIS is preferred; missing APIs retain an explicit unsupported status. ArviZ Stats supplies smoothed weights through a tested input-convention adapter, followed by explicit log-weight normalization.

Constant weights have undefined tail shape and near-full weight ESS. Invalid/nonfinite densities and insufficient tails receive separate statuses. Pareto k thresholds and the sample-size-dependent reference concern importance proposals; low k does not establish raw VI posterior accuracy. No diagnostic produces a universal all-clear score.

A separately requested `"stopping_rule": "noise_aware"` uses paired fixed-seed objective changes, Monte Carlo noise and packed-location stability. This is an optimization heuristic, not posterior validation. Diagnostics alone retain the legacy stopping rule. The benchmark disables stopping and compares increasing actual step counts.

The lower-level `fit_vi` accepts `optimization_particles=1` (the unchanged
default) independently of all diagnostic particle counts. `optimizer_lr` may
be an Optax schedule; supply numerical `optimizer_lr_metadata` to describe the
schedule in persisted optimization records. This records the schedule's
configuration and does not serialize or reconstruct callable code. The
fixed-target experiment in `examples/vi_nuts_validation/archive/optimize.py` uses these
options through the existing fitting path.
