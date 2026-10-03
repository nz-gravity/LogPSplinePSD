# Optimization on the fixed exact time-varying target

The optimization controls materially improve this VI fit, but no tested
setting yet meets all declared stability and repeatability screens. The
diagnostic study has identified optimization effects; it has not established
posterior validation or isolated irreducible guide-family error.

We ran 18 primary fits (three learning-rate schedules × one/eight full-data
optimization particles × three matched seeds), then three exploratory
extensions of the best observed setting. All use the same diagonal guide,
saved observations, prior, 8×6 basis, likelihood, noncentered parameterization
and float64 precision. The accepted four-chain NUTS reference was reused.
No observations were generated and no new NUTS jobs were run.

The frozen target fingerprint and numerical basis/penalty arrays match the
earlier control. The current model reproduces 128 archived log-joint values
exactly. Constant-rate/one-particle runs reproduce all three original
15,000-step guide checkpoints exactly. Each extension also reproduces its
selected 40,000-step prefix exactly; it reruns from the same initialization
rather than loading an optimizer state.

## What changed at the same update budget

At 40,000 updates, changing the original constant 0.01/one-particle setting
to warmup/cosine decay with eight particles reduces the worst selected
spectral mean discrepancy, across the three seeds, from **1.96 to 0.087
NUTS SD**. The maximum late feature-mean drift falls from 2.50 to 0.366
NUTS SD, and maximum final feature-mean spread across seeds falls from 1.14
to 0.142 NUTS SD. Particle work differs despite equal update counts.

The particle effect depends on the schedule: at constant 0.01, eight particles
reduce the worst spectral mean discrepancy from 1.96 to 0.527 SD; at constant
0.003, the change is 0.731 to 0.701 SD. These are results on this fixed
development target, not a universal optimization prescription.

Eight-particle cosine runs have fixed-evaluation negative ELBOs between
-847.983 and -847.973, with evaluation MCSE about 0.03. Similar objective
values alone would miss the remaining checkpoint and seed differences.
Objective evaluation uses 256 particles on each of eight fixed keys, plus
three independent keys; this is separate from training and PSIS particles.

## Does the low-rate tail stabilize the guides?

We selected the eight-particle cosine setting after inspecting the primary
experiment. Three exploratory fits follow the same schedule through step
40,000, then hold the rate at 0.0001 through step 80,000. All runs and failed
screen flags are retained.

| Quantity | Eight-particle cosine, 40k | Same setting plus low-rate tail, 80k |
|---|---:|---:|
| Maximum late feature-mean drift / NUTS SD | 0.366 | 0.118 |
| Maximum late latent-mean drift / NUTS SD | 0.468 | 0.204 |
| Maximum late relative feature SD drift | 4.92% | 0.91% |
| Maximum final seed feature-mean spread / NUTS SD | 0.142 | 0.104 |
| Late checkpoint pairs within all screens | 0/6 | 1/6 |
| Final seed pairs within all screens | 2/3 | 2/3 |

The late intervals are 30k→35k/35k→40k in the primary experiment and
60k→70k/70k→80k in the extension. A setting must satisfy both intervals
for every seed and all final seed pairs. The screens require every declared
feature and latent mean to change by at most 0.1 reference SD, relative SD
changes at most 5%, event-probability changes at most 0.01, and paired
objective changes within 0.2 plus twice their MCSE. These are stated
development tolerances, not convergence proofs.

The tail makes the SDs and objectives substantially steadier. Its largest
feature-mean drift occurs in band power at t=0.25; the latent mean checks
still fail in several coefficient directions. Even this substantially
improved fit does not satisfy the full answer to “repeatable, stable fitted
guides” under the declared screen. The numerical failure is now much smaller
than in the original setting, and should be interpreted at that scale.

## What remains against NUTS

At 80,000 steps, the worst selected spectral mean discrepancy is 0.1005
reference SD across seeds. Median selected spectral SD ratios remain
**1.183–1.189**. Functional MMD² is 0.00630–0.00649, compared with a
descriptive NUTS-versus-NUTS baseline of 0.000559. Reference autocorrelation
precludes interpreting this as an ordinary IID two-sample test.

Repeated native Pareto k spans 0.485–0.942, and the smallest smoothed weight
ESS fraction is 0.0156 (about 64 of 4096 particles). The improved means do
not certify a useful joint importance proposal. No posterior was reweighted.

Optimization therefore explains a substantial part of the original mean
discrepancy. The residual spectral spread and joint discrepancy remain
measured facts, but their attribution solely to diagonal-guide shape would
be premature. This experiment supplies no evidence for changing the
observation model, abandoning VI or introducing flows. Predictive adequacy,
calibration, untouched-seed confirmation and new parameterizations remain
unexecuted.

## Evidence and reproducibility

- [Primary 18-fit protocol and results](optimization_report.md)
- [Three-fit low-rate extension](optimization_tail_report.md)
- [Runner](../../../examples/vi_nuts_validation/archive/optimize.py) and
  [saved-draw analysis](../../../examples/vi_nuts_validation/archive/optimization_report.py)
- Primary artifacts: `runs/vi-optimization-exact-tv`; extension artifacts:
  `runs/vi-optimization-exact-tv-tail`. Each contains frozen input/reference
  hashes, complete joint draws, numerical guide checkpoints, paired feature
  and latent changes, objective evaluations, timing and PSIS records.

All 134 repository tests pass with float64 and NumPyro 0.22.0, including the
five slow scientific regression tests and the new scheduled/multi-particle
Gaussian contract. Ruff and whitespace checks pass. All 21 inference jobs
completed; the initial preflight batching error was repaired before fitting
and its log is retained. Passing implementation checks does not validate a
variational posterior.
