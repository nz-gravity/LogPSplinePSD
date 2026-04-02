# Run X Duration Comparison

Figure: `fig1b_runx_xyz_duration_overlay_freq_units.pdf`

Data overlay: Raw periodogram

PSD unit: `Hz$^2$/Hz`

Notes:
- `n` is the block-trimmed sample count actually used in inference with `7`-day blocks.
- `ESS` and `runtime` are read from `compact_run_summary.json`.
- `RIAE`, `L2`, `coverage`, and `CI width` are computed from the plotted CI curves on the saved frequency grid.
- `CI width` is the mean diagonal 90% interval width.

| Run | Nominal duration (days) | n | ESS median | Runtime (s) | RIAE | L2 | Coverage | CI width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 month | 30 | 483,840 | 4,870 | 905.5 | 0.0213 | 0.0212 | 84.0% | 4.836e-10 |
| 3 months | 90 | 1,451,520 | 3,840 | 1186.6 | 0.0137 | 0.0151 | 85.9% | 2.838e-10 |
| 1 year | 365 | 6,289,920 | 201 | 1547.3 | 0.0089 | 0.0091 | 49.7% | 9.911e-11 |

## Run Paths

- `1 month`: `runs/run_paper_30d_k100_d2_kmuniform_nc8192_null_excision/k100_d2_kmuniform_wwtukey0p1_ewhann_nc8192_bd7d_ta0.8_td10_viOff_tauOff/seed_0`
- `3 months`: `runs/run_paper_90d_k100_d2_kmuniform_nc8192_null_excision/k100_d2_kmuniform_wwtukey0p1_ewhann_nc8192_bd7d_ta0.8_td10_viOff_tauOff/seed_0`
- `1 year`: `runs/run_paper_365d_k100_d2_kmuniform_nc8192_null_excision/k100_d2_kmuniform_wwtukey0p1_ewhann_nc8192_bd7d_ta0.8_td10_viOff_tauOff/seed_0`
