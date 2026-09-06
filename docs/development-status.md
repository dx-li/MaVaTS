# Development checkpoint — 2026-09-05

The original goal remains active: comprehensive major published matrix-valued
time-series methods, with scientific benchmarks, examples, stable numerics and
an extensible Python library. This checkpoint is substantial implementation
progress, not a declaration that the literature is fully covered.

## Fourth checkpoint: cointegration, multi-term autoregression and entrywise robustness

Added fixed-rank cointegrated matrix autoregression with LS and profiled
separable Gaussian MLE, lagged differences, an unrestricted intercept, level
forecasts and complete I(1) coefficient diagnostics. Added multilinear tensor
autoregression with multiple terms/lags, projection/LS/MLE, matrix multi-term
canonicalization, local higher-order CP projection, and full-sum companion
stability. Added entrywise iterative Huber regressions for both loadings and
factor scores, robust held-out transforms and the preprint's rank criteria.

All three methods explicitly identify their primary manuscript versions.
In particular IHR targets the 2023 preprint, not an unverified equivalent of
the renamed accepted work. Optional numerical choices and missing inference
are documented rather than inherited from an estimator's name.

The suite passes 758 tests in both dependency environments below, with 94%
line coverage. Independent reviewers checked reduced-rank likelihood blocks,
whitened intercepts, full cointegration complements, tensor multi-term updates,
rank-normalization conventions and robust loss differences. Regressions cover
translation-invariant convergence, near-limit covariances, compensating tensor
coefficient/covariance scales and amplitude-invariant cancellation diagnostics.
All 12 examples pass locally and from an independently installed wheel.
Source/wheel builds, API docs and formatting checks pass.

New benchmark drivers compare common-origin one- and five-step forecasts,
including a vector VECM with the same total cointegration rank, and distinguish
entrywise from whole-matrix contamination. Known ranks and term counts supply
extra information and are labeled. Projection initialization, oversized rank
pilots, final optimizers and held-out robust-score solves have separate
convergence diagnostics. The timing recorder now preserves fitting-only time
even when later scoring fails and identifies the failure phase.

The retained structured-autoregression study has 540 records and no execution
errors. Four final records remain unconverged: two higher-order CP projections
and two tensor LS/MLE fits at the 200-sweep limit. Ten selected records have an
unfinished CP initialization/projection, separately from final convergence.
Weak adjustment challenges cointegrating-space recovery; underfit tensor models
remain competitive; nonseparable noise exposes large separable covariance errors.
All 9,100 older benchmark records were refreshed without changes to non-timing
results. See the [new results and limits](../benchmarks/results/structured-summary.md).

The entrywise robustness study adds 180 records with no execution errors and
all final fits converged. Five oversized rank pilots hit their iteration limit;
two paired IHR records share an unfinished held-out score solve. Those flags
remain explicit. IHR improves scattered-entry denoising in this design, but
does not outperform matrixwise Huber under whole-matrix contamination.

This remains an incomplete development library. Matrix ARMA, partially
constrained and multi-term factor models, transition-rank tensor AR, broader
rank/inference procedures, accepted-IHR reconciliation, real-data studies and
comprehensive performance/stress grids remain open. Published estimation,
automatic model selection and inferential theory are separate deliverables.

## Third checkpoint: additive dynamics, volatility and online monitoring (retained history)

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
