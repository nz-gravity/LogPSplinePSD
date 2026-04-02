# Knot Scoring Analysis

## Score Concentration

| Method | Diag top10 mass (mean) | Diag effective fraction (mean) | Offdiag top10 mass | Offdiag effective fraction |
| --- | ---: | ---: | ---: | ---: |
| spectral | 0.2724 | 0.7851 | 0.3027 | 0.7437 |
| cholesky | 0.1165 | 0.9947 | 0.1879 | 0.9272 |

## Posterior Metrics

| Method | Coverage | RIAE matrix | L2 matrix | Runtime (s) |
| --- | ---: | ---: | ---: | ---: |
| spectral | 0.7684 | 0.1161 | 0.1335 | 23.51 |
| cholesky | 0.7850 | 0.1140 | 0.1311 | 14.72 |

## Deltas (spectral - cholesky)

- diag_top10_mass_mean: 0.1559
- diag_effective_fraction_mean: -0.2096
- coverage: -0.0166
- riae_matrix: 0.0021
- l2_matrix: 0.0024
- runtime (s): 8.79
