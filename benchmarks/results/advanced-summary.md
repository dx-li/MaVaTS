# Dynamic, volatility and monitoring integration results

These are specified synthetic integration studies, not full paper replications.
All methods in a scenario share data and chronological holdouts. Raw artifacts
retain tuning, seeds, convergence, constraints, software, BLAS/thread context and
source fingerprints. No warmed-up speed or broad superiority claim is made.

## Additive two-way dynamic forecasts

The [advanced artifact](advanced-matrix.json) contains five replications of each
design. Additive data have shape 8×10, row/column ranks (1,2), 500 training
observations and 50 held-out one-step origins. Each forecast uses only its
origin's preceding observed matrix after training-only estimation.

| Estimator | Diagonal factor dynamics MSE | Coupled factor dynamics MSE |
| --- | --- | --- |
| Two-way, known true ranks | 2.2407 ± .0273 | 2.2866 ± .0296 |
| Two-way, automatically selected ranks | 2.2407 ± .0273 | 2.2866 ± .0296 |
| MAR ALS | 2.3441 ± .0197 | 2.3671 ± .0231 |
| Ridge VAR (fixed ridge 1) | 2.7026 ± .0245 | 2.7028 ± .0266 |
| Zero | 4.4926 ± .1539 | 4.8274 ± .1804 |
| Inaccessible latent conditional-mean oracle | 2.2047 ± .0277 | 2.2047 ± .0277 |

Values are mean ± replication standard error, not independent-method uncertainty.
Known-rank estimates receive extra information. All auto-rank fits select (1,2)
in these strong-factor examples; optimization and rank stabilization are checked
separately. The coupled design violates the estimator's diagonal factor-dynamics
assumption and can induce cross-factor marginal dependence. Its performance
does not validate the fitted assumptions. The oracle knows latent past states,
which ordinary forecasts do not observe.

## Matrix GARCH covariance forecasting

The full nonsymmetric 2×2 GARCH design uses 600 training observations and 50
held-out covariance forecasts. All filters continue the training state and
predict covariance before seeing its scored observation. The true parameters
satisfy the documented sufficient stationarity bound. Both fits use two starts,
300 maximum iterations and the spectral constraint option.

| Estimator | Negative Gaussian log score | Relative covariance error |
| --- | --- | --- |
| Matrix GARCH, diagonal dynamics | 4.6965 ± .1926 | .1034 ± .0185 |
| Matrix GARCH, full dynamics | 4.7351 ± .2179 | .1899 ± .0233 |
| Training constant covariance | 4.7454 ± .2189 | .2261 ± .0246 |
| Oracle conditional covariance | 4.6862 ± .1796 | 0 |

Lower is better for both scores. Values are mean ± replication SE across five
series. Full dynamics have more free parameters and do not outperform the
diagonal restriction here despite being the generating model. Weak off-diagonal
dynamics, local optimization and estimation variance matter; this small study
does not isolate their separate contributions. Inspect selected-start convergence
and active bounds in the raw records before using a fit. A successful optimizer
flag does not establish stationarity or a global minimum.

## Sequential monitoring

[monitoring.json](monitoring.json) contains group summaries and calibration
metadata; [monitoring.jsonl](monitoring.jsonl) retains 4,000 runs: 200 matrix
series per method/regime, five methods and four regimes. No execution failures
occurred. The standard matrix dimensions are 20×15, training length 50 and
monitoring horizon 100. The planned event begins at monitoring step 34.

| Boundary | Stationary-null alarm rate | Wilson 95% interval | Pre-change alarm rate |
| --- | --- | --- | --- |
| Asymptotic maximum | .040 | [.0204,.0769] | .005 |
| Finite-horizon Gaussian maximum | .070 | [.0422,.1141] | .010 |
| Gaussian partial sum, eta=.25 | .035 | [.0171,.0705] | .000 |
| Gaussian Darling–Erdős, eta=.5 | .040 | [.0204,.0769] | .025 |
| Gaussian Rényi, eta=.75 | .050 | [.0274,.0896] | .050 |

Nominal alpha is .05. These estimates are compatible with considerable
Monte Carlo variation, not exact calibration guarantees. Gaussian-reference
calibration uses an independent seed and a fixed 20,000-path sample for the
partial-sum families; Gaussian maximum calibration is analytic. Reusing one
calibration across the study makes the reported empirical rates conditional
on that realized calibration. The matrix-data drift still remains.

All planted space switches and factor increases were detected among paths
without an early alarm; early alarms are **not** counted as successful detections.
The at-risk counts are 199/198/200/195/190 for the table's respective methods.
Mean delays, conditional on detection, are respectively 3.84/3.84/4.45/4.42/4.56
steps for a space switch and 4.40/4.37/5.18/5.12/5.35 for factor increases.
Zero misses against these strong alternatives do not establish power against
small changes, weak factors or disappearing factors.

Doubling factor amplitude without changing the loading space produced the same
realized alarm counts as the null in these paired replications. This is a useful
negative control, but it violates stationary second moments and is not an
additional stationary-null size experiment. Other regimes and methods reuse
the seeds, so their counts must not be pooled as independent trials.

Remaining benchmark work includes larger spatial dimensions, weak/no factors,
misspecified projection ranks, heavy tails and temporal noise dependence,
near-boundary volatility, multistep forecasts, real datasets, and calibrated
inference beyond the currently supported procedures.
