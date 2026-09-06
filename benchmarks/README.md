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

The eight original retained studies were generated at implementation checkpoint
`cf8948774929a75e52b1e9491154db3dacdb4d1c`. Subsequent citation-only source edits
change byte-level hashes; removing docstrings leaves identical executable ASTs.
Their original source fingerprints remain unmodified and must not be read as
hashes of the later citation-edited files. New smoke runs fingerprint their own
checkout normally. Method attribution, including the benchmark VECM and Wilson
intervals, is in [the citation index](../docs/citations.md).

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

## Matrix ARMA and partial/overlapping constraints

The [retained results and limitations](results/marma-constraints-summary.md)
report paired series-level errors and explicit optimizer/identification limits.

```bash
python -m benchmarks.marma --repeats 5 --output benchmarks/results/marma.json
python -m benchmarks.constrained_extensions --repeats 10 --output benchmarks/results/constrained-extensions.json
```

These new reports fingerprint their own source tree, separately from the eight
historical reports above. `--quick` reduces the designs for execution checks;
it is not the retained accuracy study. The generators build dense vector
recurrences or Kronecker signal designs independently of the fitted APIs.

### Conditional MARMA forecasts

Five paired replications use 500 training and 50 held-out 2×3 matrices from a
stable/invertible MARMA(1,1), with 300 burn-in observations. AR and MA operators
do not commute. The four regimes are isotropic, separable correlated innovations,
near AR/MA cancellation, and nonseparable covariance. All parameters and the
covariance construction are explicit in `marma_scenarios.py`; no test-set tuning
is performed. The sign convention is `X[t]=C+Phi X[t-1]+E[t]-Theta E[t-1]` in
column-major vectorization.

Known-order conditional MARMA LS and separable Gaussian MLE are compared with
MAR ALS/MLE, unrestricted VAR(1)/VAR(5), zero forecasts, and true-parameter
forecasts with extra latent past innovations. That last comparator has more
information than a fitted zero-prefix conditional filter, not just better
parameter estimates. Every method is trained once. At the same 46 held-out
origins, horizons one and five use only the observed prefix; MARMA refilters
that prefix but does not refit. Horizon-five forecasts do not observe the four
intervening outcomes. Forecast covariance and Gaussian scores omit parameter
uncertainty. A near-cancelling model can forecast well while identifying its
separate AR and MA operators poorly.

MARMA uses two local starts, 300 final iterations, tolerance 1e-8, and explicit
full-companion stability/invertibility bounds `rho <= 1-1e-6`. Each MLE start
has a separate LS warm-up of at most 100 iterations. Timings include all starts
and initialization but exclude forecast scoring. Final convergence, raw gradient
norm, feasibility, constraint residuals, and initialization stopping flags are
separate fields. SLSQP success is neither a global-optimum nor a structural
identification certificate. All finite unconverged outcomes remain in accuracy
summaries and are separately counted. The protocol targets Tsay (2024), not an
exact replication of its simulation table or an exact stationary likelihood.

### Constrained-factor denoising

Ten paired replications use 400 training and 30 held-out 8×10 matrices with
Gaussian white measurement noise of standard deviation .8. The partial model
has supplied/complement row ranks (1,2), column ranks (2,1), AR(.55) cores and
200 burn-in observations. Four regimes retain all cross-factor interactions,
remove them, weaken complement factors, or rotate the supplied constraints
away from the true spans. The last regime uses exactly the same observations
as the full-interaction regime: only the prior constraints are wrong. Its
nominal supplied/complement ranks are therefore also misspecified. Metadata
distinguishes generating ranks (1,2)/(2,1) from the actual ranks relative to
the wrong supplied spans, (1,3)/(2,2). Fixed-group-rank labels refer to the
generating groups, not knowledge of the rotated projection ranks.

Known-group-rank, ratio-selected, and diagonal-only partial fits are compared
with a fully constrained fit, unconstrained lagged factors, projected PCA,
and known-loading projection. Known ranks and constraints are extra information.
The diagonal-only fit imposes an additional scientific restriction that is
correct only in the matching regime. The known-loading comparator knows the
full Tucker loading spaces but **not** the diagonal-only zero cross blocks;
it is not a universal error lower bound. Automatic group-rank search is capped
by the available block dimensions. In the smaller quick design its column
constraint search cannot reach the generating rank two; quick is smoke-only.

Three additional regimes have two rank-(1,1) terms with two-dimensional
constraint spans: orthogonal, overlapping (cosine .7), and near-overlapping
(cosine .98). Competing-span estimation with joint scores is compared with
unadjusted independent single-term fits summed together, lagged factors,
projected PCA, and known-loading joint projection. The independent sum is a
deliberately unadjusted comparator, **not** the paper's multi-term estimator.
The generators align the actual loading directions, not merely unused parts
of the constraint spaces. Component recovery, total signal recovery, surviving
loading ranks and joint score conditioning are distinct diagnostics.

All held-out factor errors measure **contemporaneous denoising**, using the
current noisy matrix with frozen training loadings; they are not forecasts.
Errors target the clean common signal. Direct linear-algebra procedures report
`converged=True, n_iter=0`; this says nothing about statistical recovery under
incorrect constraints or weak signal. The target is Chen, Tsay and Chen's
accessible manuscript v3, with paper-version and algebraic-extension boundaries
in [the constraint notes](../docs/partial-constraints-notes.md).

## Published factor-rank criteria and penalty stability

```bash
python -m benchmarks.factor_ranks --repeats 10 --output benchmarks/results/factor-ranks.json
python -m benchmarks.rank_stability --repeats 10 --output benchmarks/results/rank-stability.json
```

The rank study compares all 40 Han–Chen–Zhang (2022) combinations: TOPUP/TIPUP,
noniterative/iterative rank reselection, IC/ER, and penalties 1–5. Training has
300 matrices of size 10×12, with 30 held-out observations and true non-null
ranks (2,2). Search caps are (4,4), centering is training-only, h0=1, c=1,
c0=.1, nu=0, and iterations stop at 100 sweeps or rank/projector tolerance
1e-8. Comparators are four matched-cap adjacent-ratio initializations with
fixed ranks thereafter, known-loading projection (not known mean), and the
training mean. Timings include rank selection, pilots and refits, but exclude
simulation and scoring. All methods use identical data within each replicate.

`rank_scenarios.py` independently generates orthogonal mode loadings scaled by
sqrt(d_k), independent stationary unit-variance AR(.65) cores and white unit
variance Gaussian noise. Weak designs divide each mode's second loading column
by d_k**.25 (weakest strength exponent .5), keeping cores/noise paired with
the strong design. Four extra IC2 fits receive this true nu=.5: these are
extra-information comparisons, not estimated strength. Cancellation uses core
AR coefficients .65*[[1,-1],[-1,1]], making both population lag-one TIPUP
moments exactly zero while TOPUP remains rank two. A separate h0=2 comparison
restores TIPUP's population information. White-factor cores have AR zero;
all positive-lag signal moments vanish, violating dynamic identification.
No-factor data have zero clean signal and true ranks (0,0). Serial-noise
data use stationary entrywise AR(.5) measurement error, violating the white
error assumption. Stationary starts plus 200 burn-in steps are used. Independent
population-moment tests verify cancellation without reusing estimator algebra.

Retained diagnostics include rank histories, canonical zero-signal ranks,
projector changes, stopping reasons, normalized terminal spectra and physical
log penalties. Rank accuracy compares raw mode selections to generating ranks;
a mixed tuple containing zero still represents an identically zero centered
signal. Held-out errors measure contemporaneous denoising, not forecasting.
Absolute MSE remains defined in the no-factor design; relative error is omitted
there. Statistical failure, convergence failure and execution failure are
different outcomes, and none are silently discarded. These are integration
stress designs, not replications of the paper's simulation tables.

The stability study evaluates TIPUP and iTIPUP IC2 under strong, weak,
lag-one cancellation and no-factor designs. Each path uses 25 geometrically
spaced c values from .01 to 100 and three nested spatial prefixes of sizes
(6,8), (8,10), (10,12), paired with time prefixes 100,200,300. Caps remain
(4,4). The variance uses divisor three. Plateau selection requires zero
empirical variance over at least two adjacent gridpoints and convergence of
every subsample fit, rejects maximum-rank and zero-rank plateaus, and picks
the lower-middle point of the first admissible interval per mode. This is an
explicit finite-grid convention, not a guarantee or automatic grid search.
When both modes yield choices, a joint full-training refit is evaluated; its
ranks may differ from modewise path choices. Missing choices are retained
without fallback. Each path also retains a separately timed fixed-c=1 fit;
path time and joint-refit time are separate, and tuning cost must not be omitted
from an end-to-end comparison. A no-factor run is a negative control for the positive-rank
plateau convention, not an appropriate use of that convention as a null test.
Full cell results require memory proportional to grid size times retained
subsample arrays. Quick modes reduce dimensions, samples and grids and are
execution checks only. Seeds start at 2573 for rank comparisons; each driver's
report records its exact seed and configuration.

## Envelope MAR

[Retained results and initialization audit](results/envelope-summary.md) report
both the initial and updated 720-fit studies, with source fingerprints and paired
Monte Carlo uncertainty. The same 80 series are reused, not independent reruns.

```bash
python -m benchmarks.envelope --quick --repeats 1 --output benchmark-envelope-smoke.json
python -m benchmarks.envelope --repeats 10 --output benchmarks/results/envelope.json
```

The full design crosses order 1/2 with high-immaterial variance,
high-material variance, isotropic covariance and material/complement covariance
coupling. Each of 80 generated series (10 seeds per cell) has shape `(385,4,5)`:
350 training observations and 35 held out. Nine methods share each series,
giving 720 fit records. The quick mode uses `(155,3,4)`, two regimes, and is
an execution check, not statistical evidence.

Shared nominal response spaces have dimensions `(2,2)`. Nonsymmetric,
lag-specific coefficients include immaterial-past predictors, and are rescaled
to `sum(||A_l||_2*||B_l||_2)=0.7` for a sufficient stability condition. Both
covariance factors have trace equal to their dimension, holding the total
innovation variance fixed. The coupled regime adds nonzero material/complement
cross covariance while preserving positive definiteness; the nominal response
spaces are then **not** reducing spaces. Space errors still target the nominal
output spaces there, not the expanded true envelopes. These are controlled
package designs, not reproductions of the paper's simulations.

Comparators are EMAR with nominal, under-, over- and full dimensions; MAR MLE;
MAR ALS; rank-bounded MAR ALS; unrestricted VAR; and a known-parameter oracle.
All receive the true lag order; nominal envelope dimensions and low-rank bounds
are also extra supplied information, not selected on the data. Multi-lag
rank-bounded ALS is a documented extension of the first-order RR.LS estimator.
The oracle gets true parameters but no future innovations. All fits use only
the training prefix, with no test-based tuning. EMAR uses three fresh Grassmann
starts plus the preceding envelope, inner gradient tolerance 1e-6, 200 inner
iterations and 100 outer iterations (40 in quick mode), likelihood tolerance
1e-8 and a 100-iteration unrestricted MLE warmup. The warmup's coefficient spaces
supply an additional first envelope start. MAR baselines use 200 sweeps,
tol=1e-8, and no covariance floor. Timings include warmup and all optimization.

Scores use the same 31 held-out origins at horizons 1, 3 and 5, updating observed
history without refitting. Raw entrywise MSE and error against the true
conditional mean are both retained: raw MSE can obscure estimation differences
when innovation noise dominates. The independent dense impulse recursion gives
the exact Gaussian forecast noise floor. Operator error pools lag-specific
Kronecker operators, not unidentified factor pairs. Covariance error compares
full Kronecker covariance, not arbitrarily scaled row/column factors. Every
failure, convergence flag, warmup flag and selected terminal gradient remains
visible. Retained results and their paired Monte Carlo summaries are reported
separately; convergence is not a claim of global likelihood optimality.

These artifacts were generated locally during the rebuild. Rerun after changing
algorithms or dependencies. Further benchmark coverage remains tracked in
[the roadmap](../docs/roadmap.md): broader weak/no-factor and cancellation grids,
near instability, nonseparable innovations, rank selection and inference,
real datasets, and profiling across dimension grids.
