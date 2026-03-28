# Run X Duration Comparison

Figure: `fig1b_runx_xyz_duration_overlay_freq_units.pdf`

Data overlay: Hidden

PSD unit: `Hz$^2$/Hz`

Notes:
- `n` is the block-trimmed sample count actually used in inference with `7`-day blocks.
- `ESS` and `runtime` are read from `compact_run_summary.json`.
- `RIAE`, `L2`, `coverage`, and `CI width` are computed from the plotted CI curves on the saved frequency grid.
- `CI width` is the mean diagonal 90% interval width.

| Run | Nominal duration (days) | n | ESS median | Runtime (s) | RIAE | L2 | Coverage | CI width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 month | 30 | 483,840 | 7,138 | 277.6 | 0.0457 | 0.0551 | 61.6% | 3.218e-10 |
| 3 months | 90 | 1,451,520 | 6,695 | 497.2 | 0.0413 | 0.0531 | 56.8% | 1.947e-10 |
| 1 year | 365 | 6,289,920 | 3,359 | 1638.9 | 0.0407 | 0.0544 | 27.5% | 6.612e-11 |

## Run Paths

- `1 month`: `/Users/avi/Documents/projects/LogPSplinePSD/docs/studies/lisa/runs/run_x_30d_d2_k48_uniform_no_excision/k48_d2_kmuniform_wwtukey0p1_ewhann_nc8192_bd7d_ta0.8_td10_viOff_tauOff/seed_0`
- `3 months`: `/Users/avi/Documents/projects/LogPSplinePSD/docs/studies/lisa/runs/run_x_90d_d2_k48_uniform_no_excision/k48_d2_kmuniform_wwtukey0p1_ewhann_nc8192_bd7d_ta0.8_td10_viOff_tauOff/seed_0`
- `1 year`: `/Users/avi/Documents/projects/LogPSplinePSD/docs/studies/lisa/runs/run_x_d2_k48_uniform_no_excision/k48_d2_kmuniform_wwtukey0p1_ewhann_nc8192_bd7d_ta0.8_td10_viOff_tauOff/seed_0`
