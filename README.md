# MaVaTS

Scientific Python methods for matrix- and tensor-valued time series: structured
autoregression, factor estimation, volatility, sequential monitoring,
decorrelation, inference, simulation, and reproducible comparisons.

This is the **0.2 development rebuild**. The aim is broad, dependable coverage
of published methodology. It is not yet a complete replacement for specialist
research implementations. The [method inventory](docs/methods.md) states exactly
which estimators are implemented and which paper features remain open.

## Install the development checkout

Requires Python 3.10 or newer, NumPy and SciPy. From this repository:

```bash
python -m pip install -e '.[dev]'
python -m pytest
python -m examples.quickstart
```

The older package on PyPI does not contain this development API.

## Forecast matrix observations

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

`fit_sparse_mar` adds continuous spike-and-slab EM variable selection for
zero-mean MAR(1). Its inclusion probabilities are conditional on a fitted
posterior mode; they are not posterior averages or credible intervals.
Forecasts use all fitted coefficients, including those outside selected support.
See the [sparse MAR example](examples/sparse_mar.py) and
[prior and algorithm conventions](docs/sparse-notes.md).

For stationary, unconstrained, unpenalized zero-intercept MAR(1),
`mar_inference` supplies plug-in standard errors and marginal Wald intervals;
`mar_specification_test` tests whether a VAR(1) operator has one Kronecker term.
These require the published large-sample assumptions, including iid innovations;
MLE inference additionally requires a separable innovation covariance. They do
not provide post-selection inference for sparse or reduced-rank fits, and a
large specification-test p-value does not establish the model.

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

`fit_two_way_dynamic` fits the additive model `X[t] = F[t] L.T + Lambda G[t].T`.
It combines covariance quasi-likelihood, conditional scores and pooled scalar
autoregressions, with separate row and column effects rather than a Tucker core.
Ranks can be supplied or selected through iterative residual projections.
See the [two-way dynamic example](examples/two_way_dynamic.py) and
[normalization and forecasting conventions](docs/dynamic-notes.md).

`fit_threshold_factors` estimates two regimes controlled by an observed variable
aligned with each matrix observation. It supports known or estimated thresholds,
unequal row/column ranks across regimes, and an inspectable trimmed search
profile. It uses the published regime-specific lagged cross moments. The
[threshold example](examples/threshold_factors.py) includes held-out projection;
[method notes](docs/threshold-notes.md) explain identification and time alignment.

`fit_matrix_decorrelation` instead keeps all coordinates and estimates an
invertible transformation into rectangular component series. Grouping uses a
chosen correlation threshold or `grouping="ratio"` for the published adjacent
correlation-ratio selector. Ratio grouping cannot select all-singleton partitions
and requires at least three coordinates in each nontrivial mode. `blocks()` and
`inverse_blocks()` connect separate component models with reconstruction; the
estimator does not fit those forecasting models itself. See the
[decorrelation notes](docs/decorrelation-notes.md) for equations and limitations.

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
See the [matrix GARCH example](examples/matrix_garch.py) and
[model and constraint notes](docs/volatility-notes.md).

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
[online monitoring example](examples/online_monitoring.py),
[monitor assumptions](docs/monitoring-notes.md), and
[calibration scope](docs/calibration-notes.md).

## Methods, examples and comparisons

- [Scientific coverage and primary citations](docs/methods.md)
- [Numerical conventions and compatibility](docs/numerics.md)
- [Executable examples](examples/quickstart.py)
- [Core method gallery](examples/method_gallery.py)
- [MAR inference example](examples/mar_inference.py) and [assumptions](docs/inference-notes.md)
- [Additive dynamic factors](examples/two_way_dynamic.py), [matrix GARCH](examples/matrix_garch.py), and [online monitoring](examples/online_monitoring.py)
- [Benchmark protocol and retained results](benchmarks/README.md)
- [Remaining work and acceptance criteria](docs/roadmap.md)
- [Contributing](CONTRIBUTING.md)

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
[references.bib](docs/references.bib) provides citations. Code is licensed under
the [MIT license](LICENSE). Implementations are derived from published equations,
with paper-specific limitations and extensions documented beside each API.
