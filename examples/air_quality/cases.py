"""Explicit, predeclared teaching configurations for the real-data gallery.

Scientific references and method boundaries are indexed in docs/citations.md.
These configurations are illustrative, not selected using the held-out quarter.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Case:
    id: str
    title: str
    api: str
    kind: str
    question: str
    caution: str
    options: dict = field(default_factory=dict)
    tensor: bool = False


CASES = []


def add(id, title, api, kind, question, caution, tensor=False, **options):
    CASES.append(Case(id, title, api, kind, question, caution, options, tensor))


FORECAST = "Can yesterday's station-by-pollutant pattern predict tomorrow's transformed observations?"
FACTOR = (
    "Can a small shared loading space summarize new station-by-pollutant observations?"
)
FCAUTION = "Scores use each held-out observation itself: this is reconstruction, not forecasting. The unobserved clean signal is unknown, so discrepancy is not denoising accuracy."

for method in ("projection", "als", "mle"):
    add(
        f"mar-{method}",
        f"MAR: {method.upper()}",
        "fit_mar",
        "forecast",
        FORECAST,
        "A bilinear conditional-mean model is an assumption, not evidence of pollutant transport or causation. Projection and ALS optimize different objectives.",
        method=method,
        fit_intercept=True,
        max_iter=500,
    )
add(
    "mar-multilag",
    "Two-lag MAR",
    "fit_mar",
    "forecast",
    FORECAST,
    "Two lags were fixed before evaluation. Inspect full companion stability, not each lag separately.",
    order=2,
    fit_intercept=True,
    max_iter=500,
)
add(
    "mar-reduced-rank",
    "Reduced-rank MAR",
    "fit_mar",
    "forecast",
    FORECAST,
    "Ranks (1, 1) are illustrative restrictions, not discovered physical laws. This is RR.LS, not RR.CC.",
    ranks=(1, 1),
    max_iter=500,
)
add(
    "mar-rank-selection",
    "MAR rank selection",
    "select_mar_rank",
    "forecast",
    FORECAST,
    "EBIC searches only the supplied training grid. Selection uncertainty is not included in subsequent forecasts.",
    max_ranks=(2, 2),
    max_iter=500,
)
add(
    "envelope-mar",
    "Envelope MAR",
    "fit_envelope_mar",
    "forecast",
    FORECAST,
    "Fixed envelope dimensions (1, 1) link mean and covariance spaces. Local convergence does not validate the reducing-space assumption.",
    envelope_dims=(1, 1),
    max_iter=100,
)
for method in ("ls", "mle"):
    add(
        f"marma-{method}",
        f"Matrix ARMA: {method.upper()}",
        "fit_marma",
        "forecast",
        FORECAST,
        "Orders (1, 1) are fixed; each forecast filters only available history. Feasibility is not structural identification or global optimality.",
        method=method,
        n_starts=1,
        max_iter=500,
    )
    add(
        f"cmar-{method}",
        f"Cointegrated MAR: {method.upper()}",
        "fit_cmar",
        "forecast",
        "What would fixed-rank error correction predict for pollutant log-levels?",
        "Deliberate applicability check: these pollutant series are not established I(1) processes. A fitted cointegration rank is not a unit-root or cointegration test; do not interpret it as a long-run equilibrium.",
        ranks=(1, 1),
        method=method,
        n_starts=1,
        max_iter=500,
    )
for method in ("projection", "ls", "mle"):
    add(
        f"tensor-ar-{method}",
        f"Tensor AR: {method.upper()}",
        "fit_tensor_ar",
        "forecast",
        "Can station × pollutant × half-day patterns predict the next day's two half-days?",
        "The third spatial mode is actual recorded half-day, not an arbitrary reshape. One term is fixed; no automatic term selection or inference is claimed.",
        tensor=True,
        method=method,
        terms=(1,),
        max_iter=200,
    )
add(
    "sparse-mar",
    "Sparse MAR",
    "fit_sparse_mar",
    "forecast",
    FORECAST,
    "Training centering does not prove a zero conditional mean. Conditional slab probabilities are not causal links or calibrated posterior uncertainty.",
    max_iter=500,
)
for method in ("projection", "als", "mle"):
    add(
        f"inference-{method}",
        f"MAR marginal intervals: {method.upper()}",
        "mar_inference",
        "inference",
        "How are transition-operator intervals inspected on transformed air-quality series?",
        "Exploratory diagnostic only: iid innovations and other population assumptions are unverified and missing-data imputation is ignored by the covariance formula. These are not validated real-data confidence guarantees or simultaneous intervals.",
        method=method,
    )
add(
    "specification",
    "Kronecker specification test",
    "mar_specification_test",
    "specification",
    "Is the empirical transition compatible with a single Kronecker term under the test's assumptions?",
    "Its p-value is nominal and asymptotic; serial dependence and imputation can invalidate calibration. It tests neither stationarity nor all MAR assumptions.",
)
for alpha in (-1, 0, 1):
    add(
        f"alpha-pca-{alpha + 1}",
        f"Alpha-PCA: alpha={alpha}",
        "fit_alpha_pca",
        "factor",
        FACTOR,
        FCAUTION
        + " Training centering makes the alpha variants coincide here; see the synthetic nonzero-mean example for their distinction.",
        ranks=(1, 1),
        alpha=alpha,
    )
for id, title, api in (
    ("projected-pca", "Projected PCA", "fit_projected_pca"),
    ("lagged-factor", "Lagged matrix factors", "fit_lagged_factor"),
    ("kendall", "Matrix Kendall factors", "fit_matrix_kendall"),
    ("huber", "Matrixwise Huber factors", "fit_huber_factor"),
    ("ihr", "Entrywise Huber factors", "fit_ihr_factor"),
):
    add(
        id,
        title,
        api,
        "factor",
        FACTOR,
        FCAUTION
        + " Pollution peaks are genuine possible events, not known sensor outliers; downweighting is not automatically desirable.",
        ranks=(1, 1),
    )
for method in ("topup", "tipup"):
    for iterative in (False, True):
        name = ("i" if iterative else "") + method
        add(
            name,
            name.upper() + " tensor factors",
            "fit_tensor_factor",
            "factor",
            "Can shared station, pollutant and half-day spaces summarize the daily tensor?",
            FCAUTION
            + " Lagged identification requires temporal factor signal; TIPUP can suffer signal cancellation.",
            tensor=True,
            ranks=(1, 1, 1),
            method=method,
            iterative=iterative,
        )
for criterion in ("ic", "er"):
    add(
        f"tensor-rank-{criterion}",
        f"Tensor rank selection: {criterion.upper()}",
        "select_tensor_rank",
        "rank",
        "How many station and pollutant factors does the training criterion select?",
        "The penalty and rank cap are predeclared; IC can select zero, ER searches positive ranks. Small dimensions make asymptotic interpretation especially tentative.",
        max_ranks=(2, 2),
        criterion=criterion,
        penalty=2,
    )
add(
    "rank-stability",
    "Rank stability path",
    "tensor_rank_stability",
    "stability",
    "Does training-only rank selection persist across subsets and penalty choices?",
    "A finite-grid plateau is not a certificate of true rank. The nested (3, 2) subset forces the fixed rank caps to (1, 1). No positive rank strictly below this cap exists, so the selector correctly cannot return an interior plateau. Use a larger panel before attempting a non-boundary stability choice; do not silently substitute one.",
)
for method in ("ratio", "threshold"):
    add(
        f"ihr-rank-{method}",
        f"IHR rank selection: {method}",
        "select_ihr_ranks",
        "rank",
        "What ranks does an oversized robust training pilot suggest?",
        "The pilot rank is fixed at (2, 2); unknown true ranks may violate the oversized-pilot assumption. Pilot convergence and rank validity are different questions.",
        max_ranks=(2, 2),
        method=method,
    )
add(
    "constrained",
    "Constrained factors",
    "fit_constrained_factor",
    "factor",
    FACTOR,
    FCAUTION
    + " Equal station loadings and a shared particulate loading are transparent illustrative restrictions, not externally established atmospheric science.",
    ranks=(1, 1),
)
add(
    "partial-constrained",
    "Partially constrained factors",
    "fit_partial_constrained_factor",
    "factor",
    FACTOR,
    FCAUTION
    + " The shared-station and particulate spans are illustrative; complementary spaces permit departures and cross-block interactions.",
    row_ranks=(1, 1),
    column_ranks=(1, 1),
)
add(
    "multiterm-constrained",
    "Multi-term constrained factors",
    "fit_multiterm_constrained_factor",
    "factor",
    FACTOR,
    FCAUTION
    + " Two disjoint pollutant spans separate particulate and NO2 components algebraically; this does not attribute emissions to sources.",
    ranks=[(1, 1), (1, 1)],
)
add(
    "cp",
    "CP matrix factors",
    "fit_cp_factor",
    "factor",
    FACTOR,
    FCAUTION
    + " Rank one is predeclared. CP identification guards can legitimately reject a dataset; do not weaken them to obtain a picture.",
    rank=1,
)
add(
    "two-way-dynamic",
    "Two-way dynamic factors",
    "fit_two_way_dynamic",
    "forecast",
    FORECAST,
    "This is an additive row/column model, not Tucker dynamics. Plug-in forecasts omit parameter and latent-state uncertainty.",
    ranks=(1, 1),
    max_iter=200,
)
add(
    "threshold",
    "Threshold factors",
    "fit_threshold_factors",
    "threshold",
    "Do shared loading spaces differ with the previous day's average PM2.5?",
    FCAUTION
    + " The lagged threshold variable is available before the target day. The estimated split is not a regulatory threshold, causal effect or health boundary.",
    ranks=((1, 1), (1, 1)),
)
add(
    "decorrelation",
    "Matrix decorrelation",
    "fit_matrix_decorrelation",
    "decorrelation",
    "Can bilinear transformations separate components for subsequent modelling?",
    "Shown correlations are contemporaneous diagnostics and include training-median imputations. Small correlations do not establish independence, temporal separation or forecast quality.",
)
add(
    "matrix-garch",
    "Matrix GARCH",
    "fit_matrix_garch",
    "volatility",
    "How does a conditional covariance model track variation in pollutant log-changes?",
    "Differencing and training centering do not prove zero conditional mean. Covariance trace is not a pollution-level forecast, health-risk estimate or calibrated prediction interval.",
    dynamics="diagonal",
    n_starts=1,
    max_iter=500,
)
add(
    "monitor",
    "Sequential factor monitoring",
    "MatrixFactorMonitor",
    "monitor",
    "When does a fixed-design monitoring procedure first flag a new loading pattern?",
    "There is no known true break date. This tiny spatial panel does not validate large-dimensional false-alarm theory; an alarm is not proof of a pollution event.",
)
add(
    "calibrated-monitor",
    "Gaussian-reference monitor calibration",
    "calibrate_monitor",
    "monitor",
    "How is a precomputed reference boundary used in an actual observation stream?",
    "The boundary calibrates zero-drift iid Gaussian reference draws, not the air-quality data. No empirical false-alarm rate is estimated from this single path.",
)
for strategy in ("last", "mean"):
    add(
        f"naive-{strategy}",
        f"Naive baseline: {strategy}",
        "fit_naive",
        "forecast",
        FORECAST,
        "This is a transparent benchmark, not a scientific pollutant model.",
        strategy=strategy,
    )
add(
    "var",
    "Unrestricted VAR baseline",
    "fit_var",
    "forecast",
    FORECAST,
    "Dense vectorization discards separable structure; this small panel keeps the parameter count manageable.",
)
