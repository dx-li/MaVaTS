# Development checkpoint — 2026-09-05

The original goal remains active: comprehensive major published matrix-valued
time-series methods, with scientific benchmarks, examples, stable numerics and
an extensible Python library. This checkpoint is substantial implementation
progress, not a declaration that the literature is fully covered.

## Implemented and integrated

MAR projection, ALS, separable MLE, multi-lag and reduced-rank models, joint EBIC
rank selection; alpha-PCA, projected PCA, lagged matrix factors; constrained,
matrix Kendall, matrixwise Huber and refined CP factors; TOPUP/TIPUP and their
iterative tensor variants. Simulators, VAR/naive baselines, subspace/error
metrics and chronological evaluation support the estimators. Legacy numerical
bugs are corrected while compatibility entry points remain.

## Verification at this checkpoint

- 252 tests pass on Python 3.13 with NumPy 2.5.2 / SciPy 1.18.1.
- The same 252 tests pass on Python 3.10 with NumPy 1.26.4 / SciPy 1.13.1.
- Local line coverage is 92%; independent equation tests and statistical
  recovery checks, not this percentage alone, support the method claims.
- Both executable example modules run successfully.
- Quick and standard benchmark artifacts contain 120 and 300 runs respectively,
  with zero errors and zero unconverged fits for their retained seeds. Source
  hashes match the code used to generate them. These are integration experiments,
  not complete replications of the papers or broad performance guarantees.
- Source distribution, wheel and generated API documentation build successfully.
- Formatting and whitespace checks pass.
- CI is configured for Linux, macOS and Windows on Python 3.10–3.13, plus a
  minimum-dependency job. Remote CI results must be checked separately.

## Still required toward the full objective

Use [methods.md](methods.md) and [roadmap.md](roadmap.md) as the coverage ledger.
Major open branches include sparse/Bayesian MAR, threshold and regime factors,
two-way dynamic models, simultaneous decorrelation, matrix volatility models,
structural break methods, ARMA/cointegration, tensor autoregression, partial
constraints, paper-specific rank criteria and comprehensive inferential APIs.
CP is the unthresholded refined estimator, Huber includes a documented descent
safeguard, and reduced-rank MAR covers RR.LS but not RR.CC.

Benchmark work remains for weak/no signals, cancellation, near instability,
nonseparable noise, real datasets, multi-step forecasts, inference coverage,
dimension grids and warm-up/memory-aware speed comparisons. Rust is not yet
justified by a measured bottleneck; the current accelerated contractions use
NumPy/SciPy with independent numerical references.
