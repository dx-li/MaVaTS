# MaVaTS

Every public method has paper attribution in its API docstring and the
[method-to-paper citation index](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/citations.md). [BibTeX references](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/references.bib)
and [method notes](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/methods.md) distinguish published algorithms from
documented extensions, simulations and conventional baselines.

Scientific Python methods for matrix- and tensor-valued time series: structured
autoregression, cointegration, factor estimation, volatility, sequential monitoring,
decorrelation, inference, simulation, and reproducible comparisons.

This is the **0.2.0a1 alpha prerelease** of the development rebuild. The aim is broad, dependable coverage
of published methodology. It is not yet a complete replacement for specialist
research implementations. The [method inventory](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/methods.md) states exactly
which estimators are implemented and which paper features remain open.

## Install the alpha prerelease

```bash
python -m pip install "mavats==0.2.0a1"
```

An explicit version selects this alpha; it is not a stable-release upgrade.
The API may change before 0.2.0 final. See the
[release notes](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/CHANGELOG.md) for compatibility and coverage limits.

## Install the development checkout

Requires Python 3.10 or newer, NumPy and SciPy. From this repository:

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m examples.quickstart
```

PyPI's older stable 0.1.5 package does not contain this rebuilt API. Examples,
method guides and benchmark drivers are in the source distribution and GitHub;
the wheel contains the importable library. The live API website tracks `main`.

## Forecast matrix observations

### Real-world walkthroughs and figures (development docs)

Start with the [air-quality data protocol](https://github.com/dx-li/MaVaTS/blob/main/docs/real-world-examples.md)
and [method-by-method visual gallery](https://github.com/dx-li/MaVaTS/blob/main/docs/gallery/index.md).
These new examples use openly licensed observed data, training-only preprocessing,
explicit missingness, chronological evaluation and retained numerical diagnostics.
Forecasts, same-day reconstruction, and assumption-sensitive inference are labelled
separately. They are teaching applications, not a claim that every model fits this dataset.

The gallery is newer than the `0.2.0a1` release. To reproduce it, use a current
repository checkout (the released wheel does not include examples):

```bash
python -m pip install -e '.[examples]'
python -m examples.air_quality --methods mar-als projected-pca --output /tmp/mavats-gallery
```

Omit `--methods` to rebuild every walkthrough; expensive local optimizers take
longer. Existing synthetic examples remain useful where known truth is needed.

### Minimal synthetic forecasting example

```python
import numpy as np
from mavats import fit_mar
from mavats.simulation import simulate_mar

X = simulate_mar(
    300, np.diag([0.6, 0.8]), np.diag([0.7, 0.5, 0.6]), random_state=42
)
model = fit_mar(X, method="als")
future = model.forecast(5)       # (5, 2, 3)
print(model.converged, model.spectral_radius)
```

MAR supports projection, alternating least squares, separable Gaussian MLE,
multiple lags, intercepts, reduced-rank coefficients and EBIC rank selection.
Iterative fits report their objective history and convergence. Stability is
diagnosed rather than imposed; a converged fit need not be stationary.

`fit_envelope_mar(X, envelope_dims=(2, 2))` links shared row/column response
spaces with reducing blocks of the innovation covariance. It fits the EMAR
likelihood, not reduced-rank ALS. Dimensions and lag order are fixed; local
optimization, warmup and failure diagnostics are explicit. See the
[envelope example](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/envelope_mar.py) and
[paper equations, numerical choices and limitations](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/envelope-notes.md).

`fit_sparse_mar` adds continuous spike-and-slab EM variable selection for
zero-mean MAR(1). Its inclusion probabilities are conditional on a fitted
posterior mode; they are not posterior averages or credible intervals.
Forecasts use all fitted coefficients, including those outside selected support.
See the [sparse MAR example](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/sparse_mar.py) and
[prior and algorithm conventions](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/sparse-notes.md).

For stationary, unconstrained, unpenalized zero-intercept MAR(1),
`mar_inference` supplies plug-in standard errors and marginal Wald intervals;
`mar_specification_test` tests whether a VAR(1) operator has one Kronecker term.
These require the published large-sample assumptions, including iid innovations;
MLE inference additionally requires a separable innovation covariance. They do
not provide post-selection inference for sparse or reduced-rank fits, and a
large specification-test p-value does not establish the model.

`fit_marma(X, ar_order=1, ma_order=1, method="mle")` adds bilinear moving-average
terms, using the paper's minus-MA sign convention. Conditional LS and separable
Gaussian likelihood use local multistart optimization; full-polynomial stability
and invertibility are enforced by default. These checks do not establish minimal
orders or structural identification. Forecasts refilter supplied history without
refitting, and covariance forecasts condition on fitted parameters. See the
[matrix ARMA example](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/matrix_arma.py) and
[conditioning and optimization notes](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/marma-notes.md).

## Recover factor spaces

```python
from mavats import fit_projected_pca, fit_tensor_factor
from mavats.simulation import simulate_factor

data = simulate_factor(200, shape=(12, 15), ranks=(2, 3), random_state=7)
fit = fit_projected_pca(data.observations, ranks=(2, 3))
signal = fit.signal
scores = fit.transform(data.observations)
reconstruction = fit.inverse_transform(scores)

tensor = simulate_factor(200, (6, 8, 5), (2, 2, 2), random_state=7)
tensor_fit = fit_tensor_factor(
    tensor.observations, ranks=(2, 2, 2), method="tipup", iterative=True
)
```

Matrix factor methods include alpha-PCA, projected PCA, all-pair lag covariance
estimation, robust matrix Kendall and Huber estimation, known loading constraints,
and refined CP generalized eigenanalysis with nonorthogonal components.
TOPUP, TIPUP, iTOPUP and iTIPUP accept matrices and higher-order tensors.
Modern Tucker results use orthonormal loadings and expose scores, signal,
residuals and transformations. Compare loading **spaces**, since individual
factor coordinates are not identified.

`fit_partial_constrained_factor` estimates shared loading groups inside and
outside supplied constraint spaces, including their cross-factor blocks.
`fit_multiterm_constrained_factor` separates identifiable overlapping constrained
components and solves their scores jointly. Transforming held-out observations
is contemporaneous denoising, not forecasting. See the
[partial-constraint example](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/partial_constraints.py) and
[paper-version, rank and identification limits](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/partial-constraints-notes.md).

`fit_two_way_dynamic` fits the additive model `X[t] = F[t] L.T + Lambda G[t].T`.
It combines covariance quasi-likelihood, conditional scores and pooled scalar
autoregressions, with separate row and column effects rather than a Tucker core.
Ranks can be supplied or selected through iterative residual projections.
See the [two-way dynamic example](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/two_way_dynamic.py) and
[normalization and forecasting conventions](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/dynamic-notes.md).

`fit_threshold_factors` estimates two regimes controlled by an observed variable
aligned with each matrix observation. It supports known or estimated thresholds,
unequal row/column ranks across regimes, and an inspectable trimmed search
profile. It uses the published regime-specific lagged cross moments. The
[threshold example](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/threshold_factors.py) includes held-out projection;
[method notes](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/threshold-notes.md) explain identification and time alignment.

`fit_matrix_decorrelation` instead keeps all coordinates and estimates an
invertible transformation into rectangular component series. Grouping uses a
chosen correlation threshold or `grouping="ratio"` for the published adjacent
correlation-ratio selector. Ratio grouping cannot select all-singleton partitions
and requires at least three coordinates in each nontrivial mode. `blocks()` and
`inverse_blocks()` connect separate component models with reconstruction; the
estimator does not fit those forecasting models itself. See the
[decorrelation notes](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/decorrelation-notes.md) for equations and limitations.

## Cointegration and multi-term autoregression

`fit_cmar` fits a matrix error-correction model with bilinear cointegrating
spaces, lagged differences and an optional unrestricted intercept. Least-squares
and separable Gaussian likelihood fits are distinct options. Its forecasts are
matrix **levels**, not differences. `i1_diagnostics()` checks the full companion
and cointegration rank condition; it is not a statistical test or an automatic
stationarity constraint. Ranks must be specified. See the
[cointegration example](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/cointegrated_mar.py) and
[assumptions and conventions](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/cointegration-notes.md).

`fit_tensor_ar` supports multiple Kronecker terms at each specified lag for
matrices and higher-order tensors. `terms=(2,1)` means two terms at lag one and
one term at lag two. Projection, least squares and separable likelihood have
different objectives. Higher-order projection uses local CP approximation and
retains its own convergence diagnostics; it is not a globally best CP guarantee.
See the [tensor autoregression example](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/tensor_autoregression.py) and
[identification and allocation limits](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/tensor-autoregression-notes.md).

`fit_ihr_factor` separately fits entrywise Huber regressions for loadings **and**
factor scores, with robust out-of-sample transforms. `select_ihr_ranks` exposes
the preprint's oversized-pilot rank rules. This implementation explicitly targets
the 2023 IHR preprint; equivalence with the renamed accepted work remains
unverified. It supplies no inferential standard errors. See the
[entrywise contamination example](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/entrywise_huber.py) and
[version boundary and threshold policy](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/ihr-notes.md).

## Model conditional covariance

`fit_matrix_garch` estimates the trace-identified first-order matrix GARCH model
with full dynamic matrices or `dynamics="diagonal"`. Both retain full triangular
covariance intercepts. It fits zero-conditional-mean observations or residuals;
no mean model is fitted implicitly. `filter_matrix_garch` applies fixed parameters
chronologically, and `forecast_one()` returns the exact next conditional
covariance. Multistep covariance forecasts, standard errors and an
estimation-adjusted portmanteau test remain unimplemented.

Optimization reports all starts, convergence, feasibility and active bounds.
The default spectral constraints do not by themselves prove stationarity.
See the [matrix GARCH example](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/matrix_garch.py) and
[model and constraint notes](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/volatility-notes.md).

## Monitor matrix factor changes

`MatrixFactorMonitor` processes one new matrix at a time with fixed training
length, ranks and monitoring horizon. It re-estimates the opposite-mode PCA
projection on every trailing window and implements the journal-version power
randomization. Maximum and weighted partial-sum procedures are available.
Monitoring stops at its first alarm or horizon exhaustion; reported indices
are one-based, and an alarm time is not an estimated break location.
`state_dict()` and `from_state()` preserve the window, diagnostics and local
randomization state for continuation.

`calibrate_monitor` separately supplies finite-horizon, zero-drift iid Gaussian
reference thresholds: exact for maxima and simulated for partial sums. These
are an explicit extension, not finite-sample false-alarm guarantees for matrix
data, whose randomized scores retain a data-dependent drift. See the
[online monitoring example](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/online_monitoring.py),
[monitor assumptions](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/monitoring-notes.md), and
[calibration scope](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/calibration-notes.md).

## Methods, examples and comparisons

`select_tensor_rank` implements Han, Chen and Zhang's (2022) IC/ER criteria
for TOPUP, TIPUP and their iterative rank-reselecting variants. All five
penalties per criterion are available; strength and physical-unit penalty
constants are explicit. `tensor_rank_stability` evaluates supplied nested
subsamples and penalty grids, retaining failed or unconverged fits and
reporting when no admissible stability interval exists. See the
[rank-selection example](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/factor_rank_selection.py),
[stability example](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/rank_stability.py), and
[paper-version and numerical scope](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/rank-selection-notes.md).

- [Scientific coverage and primary citations](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/methods.md)
- [Numerical conventions and compatibility](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/numerics.md)
- [Executable examples](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/quickstart.py)
- [Core method gallery](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/method_gallery.py)
- [MAR inference example](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/mar_inference.py) and [assumptions](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/inference-notes.md)
- [Additive dynamic factors](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/two_way_dynamic.py), [matrix GARCH](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/matrix_garch.py), and [online monitoring](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/examples/online_monitoring.py)
- [Benchmark protocol and retained results](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/benchmarks/README.md)
- [Remaining work and acceptance criteria](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/roadmap.md)
- [Contributing](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/CONTRIBUTING.md)

```bash
python -m benchmarks.run --quick --repeats 2 --output benchmark-smoke.json
python -m benchmarks.run --repeats 20 --output benchmark-results.json
```

Benchmarks retain per-replicate errors, failures, convergence, seeds, wall times,
and environment details. Forecasting uses chronological test observations;
factor studies compare estimated spaces and reconstructed signals to known truth.
Oracle constraints and ranks are labeled explicitly. These synthetic stress
tests are not full reproductions of the source papers' simulation studies.
Dedicated studies also examine sparse support recovery, threshold estimation,
decorrelation partitions, and MAR interval coverage and specification-test
size/power. Coverage and rejection rates are empirical results with Monte Carlo
uncertainty, not guarantees for other data-generating processes.

## Citation and license

Please cite the methodological papers associated with the estimators you use;
[references.bib](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/references.bib) provides citations. Code is licensed under
the [MIT license](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/LICENSE). Implementations are derived from published equations,
with paper-specific limitations and extensions documented beside each API.
