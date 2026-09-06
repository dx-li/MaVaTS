# Cointegration, tensor autoregression and entrywise robustness

These synthetic integration studies use independent series as replications, not
as complete paper replications or broad estimator rankings. All methods within a
scenario receive the same training data and evaluation observations. Raw records
retain errors, separate convergence diagnostics, options, seeds and source
fingerprints. Timings measure complete fits without repeated warm-up.

## Cointegrated matrix forecasts

[structured-ar.json](structured-ar.json) retains 180 cointegration experiments:
ten replications, three regimes and six methods. Matrices are 3×4, with 600
training observations, one lagged difference, an unrestricted intercept and
true mode ranks (1,2). The vector VECM receives the same total rank of two;
the unrestricted VAR uses two level lags. Known ranks are extra information.

Both forecast horizons use the same 46 origins from a 50-observation holdout.
Fits are training-only; forecasts use observations strictly before their origin,
and five-step predictions recurse without intermediate future observations.
Overlapping origins are not treated as independent replications. Values below
are mean MSE ± standard error across ten series.

| Estimator | Isotropic, five steps | Separable, five steps | Weak adjustment, five steps |
| --- | --- | --- | --- |
| CMAR LS, known ranks | 5.1197 ± .1676 | 4.8004 ± .2415 | 5.9088 ± .2758 |
| CMAR separable MLE, known ranks | 5.1191 ± .1679 | 4.7590 ± .2404 | 5.4579 ± .2641 |
| Vector Gaussian VECM, known total rank | 5.2294 ± .1615 | 4.9102 ± .2464 | 6.7192 ± .4546 |
| Unrestricted VAR(2) | 6.8448 ± .5040 | 6.6409 ± .3528 | 7.5152 ± .3922 |
| Random walk | 5.3554 ± .1681 | 4.9343 ± .2189 | 5.3624 ± .2699 |
| Known-parameter oracle | 5.0883 ± .1586 | 4.7088 ± .2114 | 5.2660 ± .2521 |

Paired MLE-minus-random-walk differences are −.2363 ± .0357, −.1753 ± .0688
and +.0955 ± .1027 respectively. The weak-adjustment design is an important
negative result: the random walk remains competitive, while mean estimated
cointegrating-space errors are .7674 (LS) and .4812 (MLE), versus .0285/.0279
under isotropic innovations. Numerical I(1) compatibility does not establish
accurate space recovery: all retained CMAR fits pass that coefficient check.
All final CMAR fits converged within their three-start, 200-sweep budget.

## Multi-term tensor autoregression

The same artifact retains 360 autoregression experiments: ten replications,
two shapes (3×4 matrices and 2×3×2 tensors), three innovation regimes and six
methods. Training length is 500, holdout length 50, and the same 46 origins are
used for one- and five-step forecasts. True term counts are (2,1) at lags (1,2).
LS and MLE use two starts and at most 200 sweeps. Projection uses an unrestricted
VAR followed by Kronecker/CP approximation; higher-order CP is a local procedure.

| Estimator, 2×3×2 tensors | Isotropic, one step | Separable, one step | Nonseparable, one step |
| --- | --- | --- | --- |
| Projection, known terms | 1.0156 ± .0146 | 1.0352 ± .0329 | 1.0442 ± .0277 |
| LS, known terms | 1.0149 ± .0148 | 1.0236 ± .0313 | 1.0314 ± .0268 |
| Separable MLE, known terms | 1.0142 ± .0147 | 1.0232 ± .0294 | 1.0242 ± .0266 |
| LS, underfit terms (1,1) | 1.0119 ± .0137 | 1.0217 ± .0279 | 1.0244 ± .0246 |
| Unrestricted VAR(2) | 1.0583 ± .0189 | 1.0647 ± .0349 | 1.0625 ± .0279 |
| Known-parameter oracle | 1.0034 ± .0149 | 1.0066 ± .0264 | 1.0104 ± .0243 |

The deliberately underfit model often has lower forecast error despite missing
a true term. These weak secondary terms do not make known true term counts an
automatic finite-sample advantage. Matrix results and five-step scores remain
in the raw artifact. For matrices, MLE one-step MSE is 1.0325/1.0117/1.0189
across the three regimes.

Under nonseparable innovations, mean MLE relative covariance error is .6374
for tensors and .6220 for matrices, compared with .0475/.0727 under separable
innovations. Competitive point forecasts do not validate a misspecified
innovation covariance. No retained MLE uses covariance flooring, and all fitted
tensor transition sums have companion spectral radius below one.

### Convergence limits retained

Across all 540 structured experiments there are zero execution errors and four
unconverged final records, all for 2×3×2 tensors:

- Isotropic projection, seed 1194: one CP projection reaches its iteration limit.
- Separable LS and MLE, seed 1194: selected fits reach 200 sweeps; neither
  start converges.
- Nonseparable projection, seed 1197: one CP projection reaches its limit.

Ten selected records have at least one unconverged CP initialization/projection,
which is distinct from final fitting convergence. The LS/MLE limit records have
finite scores and modest block condition/cancellation diagnostics;
they are not relabeled converged. All finite results, including these records,
are included in the tables. No seed-specific rerun or post-hoc budget increase
is used to remove limits.

## Entrywise versus whole-matrix contamination

[entrywise-robust.json](entrywise-robust.json) retains 180 experiments: ten
replications, three regimes and six methods. Each series has 250 training and
30 held-out 8×10 matrices, true ranks (2,2), temporally dependent Gaussian
factors and Gaussian entry noise of standard deviation .5. Contamination adds
noise of standard deviation 8 to either 5% of individual entries or 5% of whole
matrices. The underlying signal and clean noise are paired across regimes.

The table reports held-out relative common-signal reconstruction error, mean ±
replication SE. This is contemporaneous denoising using the held-out observations,
not forecasting. Contaminated observations remain in the score.

| Estimator | Gaussian | Entry outliers | Matrix outliers |
| --- | --- | --- | --- |
| IHR, known ranks | .0687 ± .0034 | .0762 ± .0032 | .2363 ± .0392 |
| IHR, automatic ratio ranks | .0687 ± .0034 | .0762 ± .0032 | .2363 ± .0392 |
| Matrixwise Huber, known ranks | .0672 ± .0033 | .2470 ± .0084 | .2108 ± .0356 |
| Matrix Kendall, known ranks | .0673 ± .0033 | .2514 ± .0082 | .2111 ± .0356 |
| Alpha=0 PCA, known ranks | .0672 ± .0033 | .2535 ± .0082 | .2443 ± .0306 |
| Projected PCA, known ranks | .0672 ± .0033 | .2478 ± .0085 | .2240 ± .0330 |

IHR estimates robust factor scores as well as robust loading spaces. Its
entry-contamination advantage here does not extend to all regimes: paired
IHR-minus-matrixwise-Huber errors are +.0015 ± .0005 for Gaussian data,
−.1708 ± .0068 for entry outliers, and +.0255 ± .0226 for matrix outliers.
Entrywise and matrixwise thresholds act on different residual units; both are
fixed from their respective training pilots, not equated numerically.

There are zero execution errors and all 180 final fits are marked converged.
Nevertheless, five of the 30 oversized IHR rank pilots reach 100 iterations:
Gaussian seeds 1693/1698/1699 and matrix-outlier seeds 1696/1699. All select
(2,2), so their final refits match the known-rank estimates; selection recovery
does not erase unfinished pilot optimization. Auto-rank timings include both
the oversized pilot and the final refit.

For matrix-outlier seed 1698, both IHR variants also have an unfinished held-out
score solve: the maximum score norm is 2.052×10⁻⁹ after 200 iterations, above
the requested 10⁻⁹ tolerance. These two records represent the same fitted model
and observations, not independent failures. All other IHR record-level held-out
solve flags are true. The table retains these scores without loosening tolerance
or replacing their convergence diagnostics.

## Remaining validation

Larger dimensions, weak/no-factor regimes, near-instability, order/rank selection,
broader covariance structures, real datasets and warmed-up memory-aware timing
remain necessary. These experiments do not establish global optimization,
inferential calibration or comprehensive literature coverage.
