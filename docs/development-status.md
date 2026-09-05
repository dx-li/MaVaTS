# Development checkpoint — 2026-09-05

The original goal remains active: comprehensive major published matrix-valued
time-series methods, with scientific benchmarks, examples, stable numerics and
an extensible Python library. This checkpoint is substantial implementation
progress, not a declaration that the literature is fully covered.

## Third checkpoint: additive dynamics, volatility and online monitoring

Added additive two-way dynamic factors with automatic spatial rank selection,
conditional latent-factor scoring and forecasting; Matrix GARCH filtering,
simulation, full/diagonal conditional Gaussian QMLE and covariance forecasting;
and sequential matrix-factor monitoring with all four boundary families.
Finite-horizon Gaussian-reference calibration is a documented extension, not
an exact finite-sample guarantee for matrix-data false alarms. Each family has
independent equation tests, method notes and an executable example.

The combined suite passes 582 tests in both dependency environments listed
below, with 94% line coverage. All nine examples run locally and against an
independently installed wheel. Source/wheel
builds, generated API docs and formatting checks pass. Extreme-scale GARCH
tests cover unrepresentable covariance rejection, inactive-energy overflow,
and invalid multistart fallback without losing earlier valid fits. NumPy complex
scalar options are rejected instead of silently discarding imaginary parts.

The new retained studies add 80 paired dynamic/volatility experiments and 4,000
monitoring paths, with zero execution failures; all fitted advanced-model
optimizers and automatic-rank iterations converged for these retained seeds.
They report failures, optimization and rank convergence,
active constraints, early alarms, right-censored detections and Monte Carlo
uncertainty explicitly. The full GARCH fit does not outperform the diagonal
restriction in this small design; this is retained, not explained away as an
implementation success criterion. See the
[results and limitations](../benchmarks/results/advanced-summary.md).
All benchmark drivers now fingerprint numerical source before execution and
flag edits during a run. Timings are not warmed-up performance rankings.

These implementations remain scoped branches. Dynamic factors do not yet offer
automatic lag selection or general latent-state filtering; Matrix GARCH lacks
inferential APIs and multistep covariance forecasts; monitoring does not cover
disappearing factors, automatic baseline-rank selection or multiplicity control.
Major open families include matrix ARMA/cointegration, tensor autoregression,
transition-rank methods, iterative Huber factors and partial constraints.
Real-data studies and wider stress/performance grids remain necessary. The
comprehensive-library goal remains active; this is not a release-completeness claim.

## Second checkpoint: matrix extensions and inference (retained history)

Added two-regime threshold factors (including unequal automatic ranks),
continuous spike-and-slab sparse MAR EMVS, simultaneous matrix decorrelation
with fixed-threshold and adjacent-ratio grouping, and MAR(1) asymptotic
inference/specification tests. Each has independent equation tests, a dedicated
method note and an executable example. Sparse prior-aware scale updates and
decorrelation supplement boundary conventions are documented modifications.

The combined numerical and benchmark-protocol suite passes 402 tests in both
dependency environments listed below, with 93% line coverage. All six executable
examples, source/wheel builds, API docs and formatting checks pass locally.
CI includes the new examples and benchmark smoke drivers. The retained matrix-extension
artifact contains 200 runs with zero errors/unconverged fits. The inference
study contains 4,400 experiments with zero failures, including finite-sample
undercoverage and violated-separability cases rather than hiding those results.
Core quick/standard artifacts have been regenerated against the expanded source.
See [benchmark interpretation](../benchmarks/results/extensions-summary.md).

These additions cover specific branches, not their entire families. Sparse
posterior MCMC/credible intervals, threshold-variable selection and multiple
thresholds, decorrelation prewhitening/recursive partitions, and broad factor
inference remain open. At that checkpoint, absent matrix families included two-way
dynamic factors, matrix volatility, online structural breaks, ARMA and
cointegration. Tensor autoregression and transition-rank methods also remain
open. The roadmap and full objective are unchanged.

## First checkpoint (retained history)

### Implemented and integrated

MAR projection, ALS, separable MLE, multi-lag and reduced-rank models, joint EBIC
rank selection; alpha-PCA, projected PCA, lagged matrix factors; constrained,
matrix Kendall, matrixwise Huber and refined CP factors; TOPUP/TIPUP and their
iterative tensor variants. Simulators, VAR/naive baselines, subspace/error
metrics and chronological evaluation support the estimators. Legacy numerical
bugs are corrected while compatibility entry points remain.

### Verification at the first checkpoint

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

### Open work recorded at the first checkpoint

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
