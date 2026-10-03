# VI development and validation

Work on branch `hacking-vi-nuts-validation`. These are bounded research studies;
there is no validated production VI preset or established speed advantage.

| Active file | Role |
|---|---|
| `stationary_next.py` | Frozen AR(4) continuation and offline reporting; original stages 0–2 only |
| `stationary.py` | AR(4) target, reference and feature machinery |
| `extensions.py` | Bounded two-channel stationary and scalar time-varying comparison controller |
| `extension_report.py`, `verify_extensions.py` | Offline figures/report, raw-latent reference audit and checkpoint/guide verification |
| `extension_targets.py` | Native targets, frozen numerical inputs and all-draw physical functionals |
| `study_common.py` | Exact inherited schedule, joint draws, packing, MMD and persistence helpers |
| `comparison.py` | Inherited MC-aware posterior and paired checkpoint screens |

Historical drivers and their sweep specifications live in [archive/](archive/README.md).
Active studies never import them. The helper extraction preserves the original
numerical formulas; new feature roles extend the screens to cross spectra,
coherence, multiple roughness scales and temporal contrasts.

- [Stationary continuation result](../../docs/development/vi_nuts_validation/stationary_next_report.md)
- [Multivariate and time-varying report](../../docs/development/vi_nuts_validation/extensions_report.md)
- [Essential code and archived work](../../docs/development/vi_nuts_validation/branch_inventory.md)
- [Opt-in fitting diagnostics](../../docs/development/vi_nuts_validation/usage.md)

Numerical archives under `runs/` are ignored by Git. Preserve them separately;
checking out this branch does not restore observations, references or guides.
The extensions require the existing exact-power control and stationary optimizer
protocol. They freeze one VAR record and every numerical target before fitting.
A guide checkpoint contains no optimizer state.

```bash
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/extensions.py prepare
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/extensions.py execute
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/verify_extensions.py
PYTHONPATH=src JAX_ENABLE_X64=true .venv/bin/python examples/vi_nuts_validation/extension_report.py
```

`execute` retains completed and failed attempts. Each target gets four NUTS chains
and at most one tuning repair. Full covariance uses seeds 7101–7103, the inherited
40k cosine horizon, eight training particles and a constant 1e-4 tail through 60k.
Hierarchy is locked until at least one fixed fit passes both final primary
agreement and both late stability intervals. Repeatability requires all three
fits and all three seed pairs. One saved-guide draw refinement is allowed per
seed; it cannot change optimization. No diagonal reruns, parameterization sweeps,
flows, raw moving-periodogram changes, benchmarks or untouched-seed confirmation
are included. A failed prerequisite stops that case.
