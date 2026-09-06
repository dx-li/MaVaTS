# Reproducible benchmarks

Run from the repository root after installing `.[dev]`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m benchmarks.run --repeats 20 --output benchmark-results.json
```

`--quick` uses small dimensions for CI execution checks. Default runs use
500 training observations, 8×10 MAR observations, 20×25 matrix factors, and
10×12×8 tensor factors. `--seed` fixes the first replication seed, incremented
for each replicate. Every competing method within a scenario sees the same data.

The JSON stores raw runs, errors, convergence flags, warnings, iterations, timing,
dimensions, seeds, Python/NumPy/SciPy versions, platform, BLAS and thread settings.
Failures remain in the result file and make the runner exit unsuccessfully.
Unconverged fits remain visible and are counted separately. Estimator defaults
are those in the source version used for the run; explicitly set ridge is 1,
factor ranks are (2,2), tensor ranks are (2,2,2), and factor lags are 2.

## Tasks and interpretation

| Task | Scenarios | Scores |
| --- | --- | --- |
| MAR forecasts | Isotropic, separably correlated, reduced-rank innovations/dynamics | Held-out one-step MSE, identified Kronecker operator error, fitted spectral radius |
| Matrix factors | Gaussian, radial Student-t(3), 5% matrix outliers | Relative signal error, mean normalized loading projector error |
| Tensor factors | Gaussian AR factors with white observation noise | Signal and projector error for all four lag estimators |
| Matrix CP | Three components with distinct AR persistence and observation noise | Signal error, permutation/sign-invariant individual component error, eigenproblem conditioning |

Forecast coefficients are fitted once using the training window. Each subsequent
one-step forecast receives observations through its own origin, never future
values. Zero, training mean, last value, unstructured VAR/ridge and oracle MAR
baselines use the same origins. Ridge strength is fixed here, not selected with
test data. Use `mavats.model_selection.rolling_forecast` for refitted expanding or
fixed-window evaluation; any tuning belongs inside its training-only factory.

Factor reconstruction is **in-sample** recovery against known signal. It is not
forecast accuracy. The elliptical stress case applies one radial multiplier per
time point to the Gaussian factor signal and noise. Matrix-outlier scores include
corrupted times; a robust loading estimator can still have poor factor scores
on those observations. The known constraints and reduced coefficient ranks
receive extra information and are labeled `oracle`; do not rank them as if they
had the same information as ordinary fits.

Huber uses the median initial PCA residual norm as its fixed threshold. CP uses
rank three, lags 1–3, the estimator's observed-data PCA proxies, and unit-norm
nonorthogonal loadings. CP identification failures are recorded as failures;
complex eigenpairs are never silently converted into a real fitted model.

Timing excludes data generation and scoring. The currently retained runs measure
complete estimator calls without per-estimator repeated warm-up. They support
basic cost inspection, not claims of precise steady-state performance. For
publication-grade speed comparisons, add warm-up, fixed threads, sufficiently
many runs, peak memory, and paired uncertainty summaries. Five replications are
an integration sample, not strong statistical evidence.

## Retained artifacts

- `results/quick.json`: small execution/accuracy run.
- `results/standard.json`: default-size paired repetitions with raw measurements.
- [Summary table](results/summary.md): mean errors, replicate SEs and median times.
- `results/matrix-extensions.json`: 20 paired repetitions of sparse MAR,
  unequal-rank threshold factors and decorrelation/block forecasting.
- `results/inference.json` and `results/inference.jsonl`: MAR inference
  metadata/group summaries and individual replications respectively.
- [Extension results and calibration](results/extensions-summary.md): accuracy,
  paired forecast differences, coverage uncertainty and specification size/power.

Run the additional drivers with the same thread environment as above:

```bash
python -m benchmarks.matrix_extensions --repeats 20 --output benchmarks/results/matrix-extensions.json
python -m benchmarks.inference --repeats 200 --output benchmarks/results/inference.json
```

The extension driver offers `--quick` for small CI problems. Decorrelation
timing includes transformation, block fits and predictions, explicitly labeled
in each record; sparse/threshold fitting times exclude scoring. Threshold
held-out reconstruction uses observed held-out matrices and is not forecasting.
The adjacent-ratio grouping rule cannot produce all singleton groups, so it
must not be scored as a failed implementation for that outcome.

The inference driver uses sample sizes 200 and 800 by default. It retains all
errors and reports successful/total counts; unlike an execution smoke runner,
it does not discard a whole statistical study or exit with an error merely
because some fitted intervals were unavailable. Rates explicitly condition
on successful fits. Unit tests separately inject failures to verify that they
remain visible. MLE under nonseparable innovations is labeled misspecified.
Coverage standard errors use independent series, and specification rejection
rates include Wilson bounds. This is asymptotic calibration evidence, not an
assertion that every interval attains its nominal level.

Regenerate the summary using
`python -m benchmarks.summarize benchmarks/results/standard.json --output benchmarks/results/summary.md`.
The raw JSON includes SHA-256 hashes of source modules and the benchmark driver.
Runners snapshot those files before starting their experiments and record
`source_changed_during_run` afterward. A nonempty list makes the runner exit
unsuccessfully, preventing a long run from being silently attributed to source
edits made while it was executing. The source snapshot describes the checkout
at the beginning of the run; do not edit numerical source during experiments.

## Additive dynamics, covariance forecasts and sequential monitoring

```bash
python -m benchmarks.advanced_matrix --repeats 5 --output benchmarks/results/advanced-matrix.json
python -m benchmarks.monitoring --repeats 200 --output benchmarks/results/monitoring.json
```

Use the thread environment shown above for retained performance measurements.
Both drivers support `--quick` for smaller smoke problems. The advanced driver
compares additive two-way dynamics with MAR, ridge VAR, zero and inaccessible
latent-state oracle forecasts. Known-rank fits receive the simulation's true
ranks; auto-rank fits must select them from training data. Optimization and rank
selection convergence are separate diagnostics. Coupled factor dynamics are
explicitly a misspecified comparison, not a validation of diagonal-AR assumptions.

Its volatility design fits both full and diagonal-dynamics Matrix GARCH to
nonsymmetric full-model data, then filters held-out observations from the
training state without refitting. Each covariance precedes its scored matrix.
The `negative_gaussian_log_score` includes the Gaussian normalizing constant;
**lower is better**. Dense covariance relative errors use the simulation truth.
Constant covariance is estimated from training only; the oracle covariance
receives extra information. Local failures, active constraints and unconverged
selected starts remain visible. Five replications provide integration evidence,
not a comprehensive volatility estimation study or a warmed-up speed ranking.

The monitoring driver runs five arms spanning all boundary families: asymptotic
maximum, finite-horizon Gaussian maximum, partial sum (eta=.25), Darling–Erdős
(eta=.5), and delayed Rényi (eta=.75). Gaussian calibrations are fixed before
the data study, using separate calibration seeds and 20,000 reference paths
where needed. Each method/regime shares the replication's matrix-data seed
and independent Gaussian-randomization seed. Comparisons are therefore paired.

The standard design has 50 training matrices of size 20×15, a monitoring horizon
of 100, a rank-one strong Gaussian factor, projection rank two and power eight.
Regimes are stationary null, loading-space switch, factor increase and unchanged
space with doubled factor amplitude. The last is a nonstationary negative
control, not a stationary-null size experiment. The common one-based event or
comparison step is 34; it remains a reference split in the null scenario.

Records stop at the first alarm or horizon and retain the consumed count and
indices. `max_drift` refers only to consumed observations. Summaries distinguish
pre-event alarms from detections among paths still at risk at the change.
Missed detections remain right-censored rather than being assigned zero delay.
Reported mean delay conditions on detection. Wilson intervals use independent
matrix-series replications, not time points or pooled methods. Summary JSON
points to raw JSONL rows, including failed experiments. A Gaussian reference
boundary does not imply exact finite-sample matrix-data false-alarm control.

See [advanced matrix results](results/advanced-summary.md) for retained outcomes
and limitations. Strong alternatives here do not establish local power,
disappearing-factor detection, or robustness to weak factors/heavy tails.

## Cointegration, multi-term autoregression and entrywise robustness

```bash
python -m benchmarks.structured_ar --repeats 10 --output benchmarks/results/structured-ar.json
python -m benchmarks.entrywise_robust --repeats 10 --output benchmarks/results/entrywise-robust.json
```

The structured driver uses independent dense simulation transitions, checked
against indexed matrix/tensor recurrences. It fits once on the training period
and scores one- and five-step forecasts at the same 46 origins in a 50-observation
holdout. Five-step scores concern the terminal forecast, not an average of
one-step forecasts supplied with future observations. Overlapping origins are
not treated as independent replications for uncertainty summaries.

CMAR designs have 600 training matrices of size 3×4, cointegrating ranks (1,2),
one lagged difference and an unrestricted intercept. Common-trend directions
have unit roots, while cointegrating coordinates have stable AR(2) roots.
Isotropic, separable and weak-adjustment regimes are paired. Compared methods
are CMAR LS/MLE given the true mode ranks, a vector Gaussian VECM given the
same total rank, unrestricted VAR(2), random walk and known-parameter oracle.
The vector VECM baseline partials out short-run terms and profiles residual
covariance; its full-rank limit is tested against unrestricted VAR. CMAR's fitted
I(1) compatibility diagnostic is distinct from optimizer convergence.

Multi-term autoregression designs have 500 training observations, matrix shape
3×4 or tensor shape 2×3×2, and term counts (2,1) at lags one and two.
Nonsymmetric coefficient matrices obey a conservative sum-of-norms stationarity
bound. Isotropic, separable and nonseparable innovation regimes are explicit;
the last violates the likelihood fit's covariance assumption. Projection/LS/MLE
fits with true term counts are compared with under-specified (1,1) LS, unrestricted VAR
and known-transition oracle. Projection initialization, final optimization,
covariance regularization and term-cancellation diagnostics remain separate.
These are not tests of automatic term selection or global CP approximation.

The entrywise study has 250 training and 30 held-out 8×10 matrices, true ranks
(2,2), Gaussian AR factors and Gaussian entry noise of standard deviation .5.
It compares no contamination, independently corrupted 5% of entries, and
whole-matrix corruption at 5% of times; additive corruption has standard
deviation 8. All regimes share the underlying signal and clean noise.

IHR with known ranks and with oversized-pilot ratio rank selection is compared
with known-rank matrixwise Huber, matrix Kendall, alpha=0 PCA and projected PCA.
Default thresholds are fixed using training pilots; entrywise thresholds and
matrix Frobenius thresholds have different units and are not equated. Auto-rank
timings include oversized fitting and refitting the selected ranks. Pilot
convergence, final convergence and held-out robust-score convergence are all
reported. Reconstruction errors include contaminated observations and target
the clean common signal. Held-out reconstruction is contemporaneous denoising,
not forecasting. Rank pilots may exhaust their budget even when the selected
final fit converges; that outcome is not silently relabeled successful.

Both drivers offer `--quick` for execution checks. Known ranks/term counts and
oracle coefficients supply extra information and are labeled accordingly.
Retain failures and Monte Carlo uncertainty; these fixed synthetic designs
do not establish broad dominance or reproduce every paper experiment.
See [structured and entrywise results](results/structured-summary.md) for
retained accuracy, paired differences, covariance misspecification and explicit
projection/optimizer/rank-pilot limits.

These artifacts were generated locally during the rebuild. Rerun after changing
algorithms or dependencies. Further benchmark coverage remains tracked in
[the roadmap](../docs/roadmap.md): weak/no factors, deliberate TIPUP cancellation,
near instability, broader nonseparable innovations, rank selection, broader inference,
real datasets, and profiling across dimension grids.
