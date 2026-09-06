# Real-world examples: read this before interpreting the figures

The [method-by-method gallery](gallery/index.md) pairs the implemented estimator
families and major algorithm variants with executable examples and computed
figures on observed data. Every page links its paper attribution, exact fitting
configuration, diagnostics and numerical arrays. The API reference remains
[online](https://dx-li.github.io/MaVaTS/mavats.html); the detailed
[scientific inventory](methods.md) explains implementation boundaries.

**This is a teaching gallery, not an empirical validation study.** Air quality
is a useful naturally matrix-valued example, but it cannot establish the
assumptions of cointegration, conditional covariance, robust estimation,
high-dimensional inference or sequential monitoring simultaneously. Pages
explicitly flag those mismatches. Existing [synthetic examples](../examples/)
provide complementary illustrations with known signals or planted changes.
Do not present this gallery as a leaderboard, paper replication, pollutant
source-attribution study, medical assessment or regulatory assessment.

## Data and preprocessing

We use [Chen's Beijing Multi-Site Air Quality dataset](https://doi.org/10.24432/C5RK5G)
from UCI, licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The [source and extract provenance](../examples/air_quality/data/README.md)
describe attribution, modifications, license and exact download hashes.

The bundled extract uses January–December 2014, Aotizhongxin, Changping, Dingling
and Dongsi stations, and PM2.5, PM10 and NO2. Concentration units are micrograms
per cubic metre. Station selection is alphabetical and for teaching convenience,
not geographic representativeness. No simulated values are labelled observations.

1. Preserve every recorded calendar day. Do not remove missing days and then
   pretend remaining days are equally spaced.
2. Average available readings over each recorded half-day (00–11, 12–23 hours),
   retaining hourly counts. This yields `(365, 4, 3, 2)` tensors: **day × station
   × pollutant × half-day**, not a reshaped one-dimensional list. The matrix
   examples aggregate half-day means with their hourly counts to `(365, 4, 3)`.
3. Require at least 9 hourly readings for a half-day cell or 18 for a daily
   cell. Mark others missing. These are declared teaching thresholds, not
   externally certified quality-control rules.
4. Split before estimating preprocessing: January–September (273 days) for
   training; October–December (92 days) held out. Fill missing cells using
   **training-only cell medians**, without interpolation or access to future
   test readings. Retain a mask and report imputed training/test cell counts.
5. Apply `log(1 + concentration)`, subtract each cell's training mean, and divide
   all entries by one training-derived scalar standard deviation. Fixed training
   centering makes the three alpha-PCA variants coincide here; the existing
   nonzero-mean simulation illustrates their actual distinction.
6. Never score imputed targets. They can still affect predictors, fitted spaces,
   covariance diagnostics and nominal inference; imputation uncertainty is not
   modelled. These effects are limitations, not solved by the observed-only mask.

The log transform is nonlinear and changes the model being fitted. It reduces
scale disparities but does not establish stationarity, separability, Gaussian
innovations or independence. Figures use **standardized log units**, not original
concentration units; log-scale squared errors do not imply original-scale accuracy.
No weather/seasonal adjustment is included. A single contiguous held-out quarter
does not give repeated-sample uncertainty or assess all seasons and locations.

## Reproduce a walkthrough

The gallery was added after `0.2.0a1`: use the development checkout, not just the
published wheel. Core numerical functions are unchanged by this documentation work.

```bash
git clone https://github.com/dx-li/MaVaTS.git
cd MaVaTS
python -m pip install -e '.[examples]'
python -m examples.air_quality --methods mar-als projected-pca --output /tmp/mavats-gallery
```

Data are bundled; normal runs require no download, account or API key. Plots
use the headless Matplotlib backend. Omit `--methods` for the complete gallery.
For comparable runtimes, set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`,
`MKL_NUM_THREADS=1` and `VECLIB_MAXIMUM_THREADS=1` before launching Python.
Methods with local optimization can take minutes. Files in the chosen output
directory with matching method names are regenerated; use a fresh directory to
preserve a previous run. Keep subsets in separate directories so an index
cannot be mistaken for a full run.

The common input setup is:

```python
from examples.air_quality.data import load, preprocess
from mavats import fit_mar

dates, concentrations, counts = load()
x, observed, preprocessing = preprocess(concentrations, counts, train=273)
model = fit_mar(x[:273], method="als", fit_intercept=True, max_iter=500)
# On October 1, no October 1 observation enters the prediction:
prediction = model.forecast(1, history=x[:273])[0]
print(model.converged, prediction.shape)
```

Tensor pages use `preprocess(..., tensor=True)`. See the
[configuration catalog](../examples/air_quality/cases.py) and
[complete runner](../examples/air_quality/__main__.py) for constraints, lag
alignment, rank grids and stateful filtering. Source hashes and dependency
versions are retained in [gallery provenance](gallery/provenance.json).
Minor numerical differences across platforms are expected; no exact-pixel
or exact-floating-point reproducibility is promised.

## What the figures do and do not measure

| Output | What data enter it? | Interpretation |
| --- | --- | --- |
| Rolling one-day forecast | Parameters fixed after training; history ends before each target day | Observed-only transformed-space MSE against previous-day and fixed-training-mean baselines. No test-period refit or hyperparameter search. Tensor forecasts predict both next-day half-days before either is observed. |
| Factor reconstruction | Training loadings plus the target observation for its score | Same-day compression discrepancy, **not forecast error or measured clean-signal recovery**. Bigger ranks often lower this discrepancy by construction. |
| Rank plot | Training data and predeclared candidate grids only | Criterion output, spectra or subset stability; not known true ranks or a calibrated absence-of-factor test. |
| Intervals/specification test | Transformed training data under unverified assumptions | Nominal, assumption-sensitive diagnostics only. Serial dependence, imputation and model misspecification can invalidate coverage/p-values. |
| Conditional covariance | Training fit, then chronological filtering of log-changes | Predicted variation before each new change. Zero conditional mean remains unverified; this is not a forecast of concentration. |
| Monitoring | Fixed design and training period, one new day at a time until the first alarm | One randomized path; no known true break or measured false-alarm probability. Do not restart until a desired alarm appears. |
| Decorrelation | Training transform applied to held-out observations | Before/after correlation structure; not independence or predictive validation. |

The displayed first cell is fixed in advance, not selected for a good-looking
fit. MSE uses all eligible cells rather than that one trace. The previous-day
baseline may contain an imputed previous value; its target uses the same observed
mask as the model. Models can legitimately lose to a baseline. An optimizer's
`converged=True` is a numerical stopping result, not evidence of a correct model.

## Coverage and maintenance

The gallery covers public fitting/selection procedures and major variants,
including conventional baselines. Companion filters, result transforms and
forecast methods are exercised within these workflows. Simulation functions,
array operations, metrics, result containers and legacy wrappers do not receive
redundant observational fits: their references are in the complete
[method-to-paper index](citations.md), and synthetic workflows show their role.
Known-truth synthetic benchmarks remain separate from this observational gallery.

Checks enforce catalog coverage, data hashes/calendar, training-only
preprocessing, excluded imputed targets, gallery links/assets, stored metric
recomputation and documentation smoke execution. Every run keeps rejected and
unconverged cases. Do not suppress failure diagnostics or relax identification
guards to make the gallery appear complete.

Before a new scientific claim: use a suitable independent dataset, justify the
model assumptions, predeclare tuning/validation, inspect residuals and stability,
include relevant baselines, and account for uncertainty and multiple comparisons.
