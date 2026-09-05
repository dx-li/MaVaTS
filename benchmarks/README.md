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

These artifacts were generated locally during the rebuild. Rerun after changing
algorithms or dependencies. Further benchmark coverage remains tracked in
[the roadmap](../docs/roadmap.md): weak/no factors, deliberate TIPUP cancellation,
near instability, broader nonseparable innovations, rank selection, broader inference,
real datasets, and profiling across dimension grids.
