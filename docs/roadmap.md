# Scientific implementation roadmap

The goal is a dependable, broad scientific Python library for matrix-valued
time-series methodology, then tensor extensions. A passing small test suite or
a collection of estimator names does not establish that goal. The
[methods inventory](methods.md) tracks distinct estimators and publication status.

## Release acceptance contract

Each claimed paper implementation needs all of the following evidence:

1. A primary citation, exact estimator/algorithm identifier, stated assumptions,
   normalization, initialization, objective, stopping rule, and deviations from
   the paper. Separate point estimation from rank selection and inference.
2. A usable documented API with validated shape/rank/lag parameters, consistent
   fitted results, recoverable forecasts or common components, and convergence
   diagnostics for iterative estimators.
3. A small independent numerical oracle: explicit matrix/tensor algebra or the
   authors' reference software. Reusing the same optimized helper in the test
   and implementation is not independent evidence.
4. Identification-aware statistical checks, including known-signal recovery,
   singular/near-singular cases, scale changes, and a plausible violated-assumption
   setting. Confidence procedures additionally need coverage/size experiments.
5. A runnable example with interpretation and an actual benchmark artifact that
   records seed, data dimensions, method options, failures, elapsed time, software
   versions, and machine/BLAS/thread context.

Research implementations can be made available before all criteria pass, but
must say which evidence is missing. Method names must not hide substitutions:
ridge is not Lasso, HOOI is not iTIPUP, covariance PCA is not lagged factor
analysis, and static factor extraction plus a VAR does not by itself reproduce
a published dynamic factor estimator.

## Work in dependency order

| Priority | Deliverable | Dependencies and exit evidence |
| --- | --- | --- |
| 1 | Numerical core and data conventions | Time-first shapes; explicit vectorization convention; QR/SVD least squares; symmetric eigensolvers; normalization and subspace metrics; local RNG; no hidden global state |
| 1 | MAR(1) projection/ALS/MLE; MAR(p) | Independent dense coefficient oracle, objective descent, known dynamics, identified separable covariance, forecasts and companion stability |
| 1 | Alpha-PCA, lagged matrix factors, projected estimators | Exact nonzero-mean alpha statistic, all lag covariance pairs, rotation-invariant reconstruction, projection update oracle |
| 1 | Benchmarks and executable examples for every available method | Repeated statistical experiments and runtime measurements; examples execute in CI and explain estimator assumptions |
| 2 | Robust Kendall and Huber matrix factors | Correct pair kernels/loss; heavy tails and contamination; duplicate/constant data; publish exact versus approximate pair mode |
| 2 | Reduced-rank MAR and sparse MAR | Correct weighted reduced-rank updates or specified penalized/posterior objective; rank/support recovery; chronological tuning |
| 2 | CP matrix factors and constrained factors | Generalized eigenproblem with identification checks; arbitrary full-rank constraint bases and partial constraints |
| 2 | Rank criteria, standard errors, confidence intervals, residual diagnostics | Paper-specific null distributions or valid dependent resampling; Monte Carlo size/power/coverage evidence |
| 3 | Two-way dynamic factors and simultaneous decorrelation | Dynamic identification, block structure, forecasting comparisons against matrix and vector baselines |
| 3 | Threshold/regime methods and online breaks | Threshold search bounds, regime transition handling, false-alarm/delay and no-leakage tests |
| 3 | Matrix GARCH, ARMA, cointegration, envelope/spatial/network models | Positive-definite/stationarity/invertibility restrictions; likelihood references; model-specific simulations and diagnostics |
| 4 | TOPUP/TIPUP and iterative variants for general tensors | Matrix special-case equivalence; distinct lag contraction oracles; cancellation experiment; memory growth measurements |
| 4 | Multilinear TenAR, Tucker transition AR, CP dynamic tensors | Input/output mode convention; matrix reductions; higher-order independent oracle; rank/model selection |
| 4 | Robust, missing-data, time-varying and semiparametric tensor methods | Match dedicated papers; missingness/temporal boundary assumptions and observed-only likelihood checks |
| Continuous | Distribution, documentation, maintainability, performance | Reproducible builds, license/citations, compatibility policy, supported-version CI, reviewable performance evidence |

Priority 4 does not prevent early tensor foundations when they share the correct
matrix implementation. It prevents tensor breadth from displacing missing major
matrix methods.

## Benchmark protocols

### Statistical accuracy

Use paired random seeds and the same generated observations for all eligible
estimators. Include at least zero/no signal, strong and weak factors, correlated
row/column innovations, nearly unstable dynamics, nonzero means, and heavy-tailed
or contaminated noise. Clearly label which designs satisfy each estimator's
assumptions. A method can perform badly under misspecification without being
incorrect.

For MAR, vary `T`, both spatial dimensions, persistence, coefficient rank/sparsity,
and separable versus nonseparable noise. Report relative error of the identified
operator `B ⊗ A`, held-out one-step and multi-step errors, convergence failures,
and covariance error when covariance is estimated. Include vectorized VAR/ridge,
entrywise AR, and a last-value or mean forecast as baselines. Avoid materializing
the full operator solely to score a large problem when contraction identities
give the same metric.

For factors, report principal angles or projector distance for each mode,
relative common-component error, rank recovery frequency, and held-out
reconstruction or forecasting error. Raw loading differences are invalid without
alignment. Compare alpha-PCA variants on nonzero means, projected PCA on weak
signal, lagged estimators on temporally white versus correlated noise, and
TOPUP/TIPUP on a deliberate cancellation design. Factor forecasting must fit its
dynamics using training scores only.

For robust methods, include elliptical Student-t innovations, matrix-level
outliers and entrywise outliers separately. For sparse models report precision,
recall and support error. For threshold/break procedures report location error,
false-positive frequency and detection delay. For interval/test procedures use
enough independent repetitions to attach Monte Carlo uncertainty to nominal
coverage or rejection rates. A 95% interval passing once is not coverage evidence.

Use chronological holdouts or rolling origins. All centering, scaling, rank,
lag, penalty, alpha, and robust-threshold tuning occurs on the training portion
or nested chronological validation. Report failures rather than averaging only
successful fits. Smoke benchmarks are for execution validation; publication
comparisons need repetition counts, uncertainty, and retained raw results.

### Numerical correctness and speed

Measure wall time after warm-up with fixed BLAS thread count, multiple repeats,
and paired dimensions; report median and dispersion, not one favorable run.
Include peak memory or an explicitly labeled allocation estimate for outer
products, Kronecker operators, and lag blocks. Test tall, wide, singleton, and
rank-deficient inputs. Verify finite outputs and residual/objective tolerances
after scaling data up and down by many orders of magnitude.

Benchmark NumPy/SciPy contractions and factorized solves first. Add Rust or other
compiled kernels only after profiling identifies a Python or memory bottleneck,
and keep a readable numerical reference. Compiled code is not itself evidence
of speed or stability. Changed contraction order requires accuracy as well as
runtime comparison on ill-conditioned inputs.

## Completion audit

Before claiming broad coverage, reconcile every inventory row against current
source, runnable examples, tests, and retained benchmark results. Record missing
paper features explicitly, particularly inference and automatic model selection.
Independently compare at least one nontrivial case per implemented estimator
against an author implementation or an independently derived dense oracle.
Recheck publication metadata and add substantive new matrix families as the
literature grows. Remaining rows stay open; completion is not defined by the
subset that happens to be implemented first.
