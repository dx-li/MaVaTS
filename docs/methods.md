# Methods and scientific coverage

This is a coverage inventory, not a claim that every cited paper is implemented.
The scope is time-indexed matrices and tensors, with matrix methods taking priority.
References were checked against primary papers and author manuscripts on 2026-09-05.
Machine-readable citations are in [references.bib](references.bib); implementation
acceptance criteria and remaining work are in [roadmap.md](roadmap.md).
The [method-to-paper index](citations.md) maps every public procedure and result
class, including legacy APIs and utilities, to its applicable references.

For runnable observed-data applications and method-specific figures, start with
the [real-world example protocol](real-world-examples.md) and
[visual gallery](gallery/index.md). The gallery distinguishes execution coverage
from statistical suitability: one dataset cannot validate every model below.

## Shared notation and interpretation

A matrix series has shape `(T, m, n)`, with observation `X[t]`. A tensor series has
shape `(T, d1, ..., dK)`. Time is never treated as an interchangeable spatial mode.
Column-major vectorization gives `vec(A X B.T) = (B ⊗ A) vec(X)`.
Factor loading coordinates depend on normalization and rotation: comparisons
should use loading spaces or reconstructed common components. Autoregressive
coefficient pairs also have scale ambiguity: `(c A, B/c)` gives the same operator.
These are mathematical ambiguities, not estimation failures.

## Core estimator inventory

The following estimators and inferential procedures are implemented in the
working tree. Their scope does not establish reproduction of every source
paper's simulations or full inferential theory. Public functions return fitted
result objects; the older `estimate_*` functions are compatibility interfaces.
Tests, examples, and retained benchmark reports provide separate implementation
and empirical evidence.

| Implemented API | Statistical object | Paper | Coverage boundary |
| --- | --- | --- | --- |
| `fit_mar(method="projection")` | Nearest Kronecker approximation to an unrestricted VAR coefficient | Chen, Xiao & Yang (2021), `chen2021mar` | Dense VAR allocation; projection is not bilinear least squares |
| `fit_mar(method="als")` | Bilinear conditional mean | Same paper | Conditional point estimation; scale-invariant ridge is an optional extension |
| `fit_mar(method="mle")` | Bilinear mean and separable innovation covariance | Same paper | Covariance flooring, if applied, produces a stabilized likelihood estimate; diagnostics report it |
| `fit_mar(order=p)` | Sum of lag-specific bilinear operators | MAR extension | One Kronecker term per lag; no stationarity constraint |
| `fit_envelope_mar` | Shared response reducing spaces linking bilinear means and separable innovation covariance | Samadi & De Alwis (2026), `samadi2026envelope` | Fixed positive dimensions/order, conditional Gaussian likelihood and local Grassmann optimization; no sparse SEMAR, selection, standard errors or distinct one-sided model; [source and numerical notes](envelope-notes.md) |
| `fit_marma`, `filter_marma`, `simulate_marma` | Bilinear AR and moving-average operators, conditional innovations and forecasts | Tsay (2024), `tsay2024marma` | One Kronecker term per lag, fixed orders; conditional LS and separable Gaussian likelihood with local optimization; full-polynomial stability/invertibility enforced by default, but minimality and structural identification not certified; no exact stationary likelihood or inference |
| `fit_cmar` | Bilinear cointegrating spaces and short-run matrix error correction | Li & Xiao (2024 manuscript), `li2024cointegration` | Fixed ranks; LS and separable Gaussian MLE, unrestricted difference intercept and level forecasts; no rank tests, coefficient intervals or imposed I(1) constraint |
| `fit_tensor_ar` | Multiple Kronecker terms per lag for matrices and higher-order tensors | Li & Xiao (2021 manuscript), `li2021tenar` | Projection/LS/separable MLE, explicit lag/term counts; local higher-order CP initialization, not a global approximation guarantee; no automatic term selection or inference |
| `fit_mar(ranks=(r,s))` | Reduced-rank bilinear mean via ALS | Xiao et al. manuscript, `xiao2022rrmar` | RR.LS for order one/ridge zero; the distinct RR.CC estimator is absent |
| `fit_sparse_mar` | Continuous spike-and-slab EM variable selection for MAR(1) | Celani, Pagnottoni & Jones (2024), `celani2024sparse` | Zero-mean, separable Gaussian innovations; posterior-mode estimation with documented prior-preserving scale updates; no MCMC or credible intervals |
| `mar_inference` | MAR(1) coefficient/operator plug-in covariance and marginal Wald intervals | Chen, Xiao & Yang (2021), Theorems 2–4 | Projection, ALS, separable MLE; stationary unconstrained fits without intercept/ridge or active covariance flooring; iid innovations; no HAC or post-selection inference |
| `mar_specification_test` | Wald test of one-Kronecker structure in a VAR(1) operator | Same paper, Section 4.2 | Dense unrestricted VAR and normal-space covariance; asymptotic chi-square calibration, not a test of every MAR assumption |
| `fit_alpha_pca` | Mean and contemporaneous second-moment loading spaces | Chen & Fan, `chen2021alphapca` | Legacy covariance routines are not a validated full inferential API |
| `fit_projected_pca` | Refined row/column loading spaces | Yu et al. (2022), `yu2022projection` | Default is one simultaneous projected update; repeated simultaneous updates are an extension; ranks fixed after initialization |
| `fit_lagged_factor` | Loading spaces from all column-pair lag covariances | Wang, Liu & Chen (2019), `wang2019matrixfactor` | Requires temporal signal; contemporaneous PCA is a different estimator |
| `fit_two_way_dynamic` | Additive row/column factors with covariance quasi-likelihood and pooled diagonal AR dynamics | Yuan et al. (2023), `yuan2023dynamic` | Algorithms A1/1/2, conditional scores, explicit/selected ranks and plug-in forecasts; not a Tucker factor model or full temporal likelihood; no inference or Kalman smoothing |
| `fit_threshold_factors` | Two regime-specific loading spaces and observed-variable threshold | Liu & Chen (2022), `liu2022threshold` | Known/estimated threshold, trimmed spectral-complement profile, unequal regime ranks; no multiple-threshold or threshold-variable-selection extension |
| `fit_matrix_decorrelation` | Invertible bilinear transformation and rectangular component partitions | Han et al. (2024), `han2024decorrelation` | Signed-lag moments and correlation-threshold or adjacent-ratio grouping; no optional VAR prewhitening or recursive irregular partitioning |
| `fit_tensor_factor(method="topup")`, `method="tipup"` | Tucker loading spaces from outer/inner lag products | Chen, Yang & Zhang (2022), `chen2022tensorfactor` | Inner-product signal cancellation is an actual model limitation |
| `fit_tensor_factor(iterative=True)` | Sequential loading-space updates after projection along other modes | Han et al. (2024), `han2024iterative` | Ranks fixed after initialization; rank reselection uses the separate API below; no loading inference |
| `select_tensor_rank` | TOPUP/TIPUP IC/ER with five penalties each, optionally updating ranks and spaces sequentially | Han, Chen & Zhang (2022), `han2022rank` | Equations (6)–(8) and Remark 5 in author manuscript v3; explicit strength exponent and physical-unit penalty multipliers; IC includes zero, ER positive ranks only; [notes](rank-selection-notes.md) |
| `tensor_rank_stability` | Subsample IC rank variance over explicit penalty grids | Same paper, Remark 8 and Section 5.4 | Explicit nested subsets and training prefixes; documented finite-grid plateau convention, failure/convergence diagnostics and no fallback when no interval qualifies; [notes](rank-stability-notes.md) |
| `fit_matrix_kendall` | Robust loading spaces and projected factor scores | He et al. (2025), `he2025kendall` | Exact pairs by default; optional pair sampling is an approximation; projected scores themselves are not robust |
| `fit_constrained_factor` | Loading spaces within supplied row/column spans | Chen, Tsay & Chen (2020), `chen2020constrained` | Single-term fully constrained model |
| `fit_partial_constrained_factor` | Shared loading groups in supplied and complementary spans, including cross blocks | Same paper, author manuscript v3, Section 3.4 | Separate block moments summed as equation (14); optional diagonal-block restriction; explicit or ratio-selected group ranks; no constraint tests or forecasting dynamics |
| `fit_multiterm_constrained_factor` | Separate terms with potentially overlapping constraint spans | Same paper, author manuscript v3, Section 3.3 Remark 3 | Competing-span annihilation with survival guards; joint SVD score reconstruction and more-than-two-term annihilation explicitly documented extensions; no unidentified score fallback |
| `fit_cp_factor` | Nonorthogonal rank-one matrix components via refined generalized eigenanalysis | Chang et al. (2023), `chang2023cp` | Specified rank, unthresholded moments; rejects singular, complex or unseparated eigenproblems |
| `fit_huber_factor` | Matrixwise Huber factor loss with weighted projection | He et al., `he2024huber` | Fixed threshold and explicit ranks; simultaneous paper update has a documented sequential descent safeguard; entrywise Huber inference is absent |
| `fit_ihr_factor`, `select_ihr_ranks` | Entrywise robust loading/core regressions and oversized-pilot rank rules | He et al. (2023 preprint v1), `he2023ihr` | Fixed pilot-calibrated threshold, robust transforms, two rank criteria; no adaptive block-MAD or inference; equivalence with renamed accepted work unverified |
| `fit_matrix_garch` | Trace-identified first-order conditional covariance QMLE | Yu, Li, Jiang & Zhu, `yu2024garch` | Full or diagonal dynamic matrices, multiple starts and constraint diagnostics; zero conditional mean; no QMLE standard errors, factor-GARCH or estimation-adjusted portmanteau test |
| `filter_matrix_garch`, `simulate_matrix_garch` | Fixed-parameter covariance recursion, Gaussian simulation and exact next-step covariance | Same paper, accepted equations (4)–(9) | Zero-state conditional initialization or supplied chronological state; no closed-form multistep covariance forecast |
| `MatrixFactorMonitor` | Sequential randomized test of increases in factor rank or joint loading span | He, Kong, Trapani & Yu (2024), `he2024breaks` | Journal power transformation and per-window PCA; fixed ranks/horizon, maximum/partial-sum procedures, first-alarm stopping and resumable state; no disappearing-factor branch or simultaneous-mode calibration |
| `calibrate_monitor` | Finite-horizon zero-drift iid Gaussian reference critical values | Explicit calibration extension | Exact Gaussian maxima; simulated whole-path partial-sum quantiles with Monte Carlo uncertainty; not finite-sample matrix-data false-alarm control |

### Bilinear autoregression

The first-order model is

\[
X_t=M+A(X_{t-1}-M)B^\top+E_t.
\]

Alternatively an implementation can use an intercept `C`; then
`C = M - A M B.T`. Fitting a sample-centered model and estimating a conditional
intercept jointly are different finite-sample procedures and must be documented.
For MAR(1), causality requires `rho(A) rho(B) < 1`. For multiple lags it requires
the full vectorized companion operator to have spectral radius below one;
checking each lag separately is insufficient. ALS minimizes summed squared
Frobenius residuals. Gaussian MLE uses covariance `Sigma_c ⊗ Sigma_r`, including
log determinants and covariance normalization. Projection solves a different
problem by rearranging the unrestricted VAR coefficient and taking its leading
singular triplet. [Paper and open manuscript](https://arxiv.org/abs/1812.08916).

`fit_sparse_mar` implements the continuous-normal-mixture EMVS branch of
[Celani, Pagnottoni & Jones](https://doi.org/10.1007/s11222-024-10402-y), with
Beta inclusion weights, inverse-Wishart covariance priors and a shared Gamma
scale. Conditional GLS coefficient updates and covariance modes optimize a
recorded observed posterior. Reciprocal coefficient/covariance scaling preserves
likelihood but changes the priors; the implementation therefore optimizes those
scale directions instead of applying the paper's arbitrary Frobenius balancing.
These algorithmic modifications and source-equation corrections are documented
in [sparse-notes.md](sparse-notes.md). Coefficient scales depend on the priors.
Inclusion probabilities are conditional plug-in quantities at a local mode,
and a probability above 0.5 defines support without forcing coefficients to
zero. Neither global convergence nor posterior uncertainty is established by
EM convergence. General tensor EMVS, lag-factor MAR*(P), and MCMC remain open.

### MAR inference and specification

`mar_inference` implements the paper's projection delta covariance and constrained
ALS/MLE covariance sandwiches using `N=T-1` sample moments. Parameters use
column-major `[vec(A), vec(B)]` throughout, including the permutation from the
paper's ALS/MLE `vec(B.T)` convention. Unit Frobenius norm identifies A's scale;
a fixed `sign_anchor` selects its sign and must correspond to a nonzero true
entry. Operator intervals avoid coefficient-pair sign ambiguity. All intervals
are marginal, and entries whose underlying A and B coefficients both vanish
have degenerate first-order variances outside ordinary Wald coverage theory.

The inference API accepts converged stationary MAR(1) fits without an intercept,
penalty, reduced-rank constraints, or active MLE covariance flooring. Theorems
2–4 assume iid innovations with finite second moments and nonsingular A, B and
innovation covariance; ALS/MLE additionally invoke the paper's absolute-continuity
Condition R, and MLE requires correct covariance separability. Sample numerical
guards cannot verify these population assumptions. This is not long-run/HAC
covariance estimation or inference after model selection.

`mar_specification_test` evaluates the paper's Section 4.2 statistic in an
orthonormal normal-space basis, avoiding an arbitrary pseudoinverse tolerance
for the structurally singular projected covariance. Under its null assumptions,
the reference distribution has `(m²-1)(n²-1)` chi-square degrees of freedom.
It tests one-Kronecker structure of the transition operator, and does not test
innovation whiteness, covariance separability, or all model assumptions. A
non-rejection is not evidence that these other conditions hold. Dense dimension
guards bound allocations; neither API claims high-dimensional asymptotic validity.

### Static and lagged matrix factors

The common-component model is `X_t = R F_t C.T + E_t`. Under the scaled convention
`R.T R = m I` and `C.T C = n I`, scores are `F_t = R.T X_t C / (m n)`.
Under orthonormal loadings, the denominator disappears; mixing these conventions
silently changes reconstructions.

Alpha-PCA takes the leading eigenspaces of

\[
M_R(\alpha)=\frac{1}{mn}\left\{\frac1T\sum_t X_tX_t^\top
  +\alpha\bar X\bar X^\top\right\},\quad \alpha\geq-1,
\]

and its transposed column analogue. Thus `alpha=-1` removes the sample mean
component, while `alpha=0` uses uncentered second moments. A nonzero mean test
case is necessary to distinguish the two. [Chen & Fan](https://arxiv.org/abs/2001.01890).

Projected estimation first estimates one loading space, projects each observation
onto it, and estimates the other space from the projected second moment. It uses
cross-sectional factor strength rather than requiring the lagged factor signal
to be nonzero. Alternating projections can have a least-squares interpretation.
This API repeats simultaneous updates when `max_iter>1` and does not claim
monotone descent for that extension.
[Yu et al.](https://arxiv.org/abs/2003.10285),
[He et al.](https://arxiv.org/abs/2112.04186).

The Wang–Liu–Chen estimator aggregates `Omega_ij(h) Omega_ij(h).T` over all
column pairs `(i,j)` and positive lags `h`, where `Omega_ij(h)` is the cross
covariance of column `i` at one time and column `j` at a lagged time. White noise
across time disappears in the population lag covariance; serially correlated
idiosyncratic noise can therefore invalidate the identifying argument. Taking
only `sum_t X[t] @ X[t+h].T` contracts column pairs and changes the estimator.
[Wang, Liu & Chen](https://arxiv.org/abs/1610.01889).

### Cointegrated matrix autoregression

`fit_cmar` fits `Delta X[t] = A1 X[t-1] A2.T +
sum(B1[i] Delta X[t-i] B2[i].T) + D + E[t]`, with supplied mode ranks
`rank(A1)=r1`, `rank(A2)=r2`. The full cointegration rank is r1*r2; the
cointegrating vectors are the **right** singular spaces beta1/beta2, giving
stationary combinations `beta1.T X[t] beta2` under the model assumptions.
An ordinary stationary reduced-rank MAR does not implement this model.

Mode-block LS residualizes nuisance regressors and uses a reduced-rank
least-squares projection. Gaussian MLE profiles the current-mode covariance
while whitening the opposite mode; merely applying the LS rank projection to
weighted data changes that likelihood update. SVD solves, covariance checks,
normalization, starts and rejected-update diagnostics are explicit. An optional
unrestricted intercept belongs in the difference equation. Forecasts recursively
return levels from the last difference_lags+1 observations.

The full vectorized companion and `alpha_perp.T Gamma beta_perp` condition are
checked by a bounded dense diagnostic. Full complements cannot be replaced by
Kronecker products of the mode-wise complements. Unit-root tolerance and fitted
rank conditioning are numerical diagnostics, not statistical evidence for an
I(1) population. Automatic cointegration ranks, restrictions on deterministic
terms, I(2) models and asymptotic coefficient inference remain absent.
[Primary manuscript](https://arxiv.org/html/2409.10860v1) and
[implementation notes](cointegration-notes.md).

### Entrywise iterative Huber regression

`fit_ihr_factor` alternates separate row, column and factor Huber regressions,
using the new loading blocks sequentially. Every new observation's robust core
is also obtained from a Huber regression. Matrixwise Huber and Kendall loading
estimators instead retain ordinary projected factor scores. These methods
therefore respond differently to scattered entry and whole-matrix outliers.

The default freezes a projected-pilot residual MAD threshold from preprint
Section 4.4. The repeatedly updated block-specific MAD variant is distinct and
not implemented. SVD normalization preserves the signal; public loadings are
orthonormal, while rank-criterion spectra retain the paper's dimension-scaled
loading convention. Rank-pilot convergence and final-fit convergence are
separate. Direct piecewise loss differences resolve inner IRLS roundoff without
weakening the requested score tolerance or clipping objective histories.

This explicitly targets [arXiv:2306.03317v1](https://arxiv.org/html/2306.03317v1).
The author lists a renamed accepted **Winsorized Mean Matrix Factor Model**;
the accepted text was unavailable for reconciliation. No accepted-equivalence,
standard-error, adaptive-threshold or post-selection inference claim is made.
See [version boundary, assumptions and diagnostics](ihr-notes.md).

### Additive two-way dynamic factors

`fit_two_way_dynamic` implements Yuan et al.'s additive model
`Y_t = F_t L.T + Lambda G_t.T + E_t`, with `Y_t` of shape `(n,m)`,
`F_t` of shape `(n,r)` and `G_t` of shape `(m,c)`. Public `ranks=(c,r)`
and `orders=(q,p)` follow row/column order; the paper lists the opposite order.
Loadings satisfy `L.T L=m I_r` and `Lambda.T Lambda=n I_c`. Factor marginal
covariances are diagonalized, sorted, and paired with a sign convention;
coincident factor variances still prevent individual-coordinate identification.

Supplement Algorithm A1 initializes through alternating residual PCA. Main
Algorithm 1 then optimizes an additive covariance quasi-likelihood through
heterogeneous quadratic loading updates and full conditional-moment covariance
EM. The noise update includes the conditional cross-effects between F and G;
dropping those terms changes the estimator. A spectral inverse action avoids
materializing the full vectorized covariance. Final conditional factor scores
are Gaussian linear predictors under the working covariance, not ordinary
projections or Kalman-smoothed states. Step two fits a scalar AR to each factor
column, pooling across units. The paper's innovation-variance denominators
are `n*T` and `m*T`; they differ from lag-adjusted effective sample sizes.

Algorithm 2 selects positive ranks using iterated opposite-mode projections
and residual-moment eigenvalue ratios. It reports stabilization, cycles and
iteration limits. Outer and inner optimization diagnostics remain separate
from rank-selection convergence. Mean centering and an optional reported
variance floor are explicit computational extensions. Forecasts recurse the
factor AR processes and predict the idiosyncratic component as zero; they need
not be full observation conditional means when residuals are serially dependent.
No automated noise-start search, lag selection, factor inference or Kalman
filtering is supplied. [Primary paper](https://doi.org/10.1093/jrsssb/qkad077),
[implementation notes](dynamic-notes.md), and
[example](../examples/two_way_dynamic.py).

### Threshold factors and simultaneous decorrelation

`fit_threshold_factors` uses the published 2022 Liu–Chen estimator: each lagged
cross moment masks the origin regime only, divides by T, and retains the original
temporal offsets. It then sums cross-moment products over all column pairs and
lags, repeating on transposes for column spaces. This differs from the authors'
earlier four-partition preprint. With an unknown threshold, extreme observations
estimate fixed loading-space complements; the sum of four projected spectral
norms scores every candidate in the trimmed interval. Missing ranks use separate
eigenvalue ratios and remain fixed through the search. Results retain unequal
core shapes, original time indices and the complete evaluated profile.
Informative temporal signal and a true threshold inside the trimming bounds are
assumptions. The supplied threshold variable must be aligned and available for
its intended use; projection of observed held-out matrices is not forecasting.
See [threshold-notes.md](threshold-notes.md) and the
[author manuscript](https://par.nsf.gov/servlets/purl/10351724).

`fit_matrix_decorrelation` implements marginal whitening, signed-lag all-coordinate
moments, and two full-dimensional rotations from Han et al. Its separate partially
transformed row/column series supply the grouping correlations. The default
correlation threshold is tuning, with no significance interpretation.
`grouping="ratio"` instead implements equation (11), searching adjacent ratios
up to half the number of unordered coordinate pairs, as in the authors'
supplementary code. `ratio_delta` controls smoothing and `ratio_max_edges` can override the
search bound. This selector always chooses an edge in a nontrivial mode, cannot
produce all singletons, and cannot use a dimension-two mode with only one pair.
The transform remains invertible; it does not reduce factor rank or establish
independence. Marginal nonsingularity and separation of the relevant eigenspaces
are needed. [Decorrelation notes](decorrelation-notes.md) distinguish the printed
equations, supplementary-code conventions, and remaining prewhitening and
recursive-partition extensions.

### Trace-identified matrix GARCH

`fit_matrix_garch` implements the accepted first-order model of Yu, Li, Jiang
and Zhu. Unnormalized row/column shape states follow BEKK-type recursions
driven by the preceding observation and preceding shape state. A separate
scalar recursion tracks `y_t=E(||X_t||_F² | past)`. Covariance factors are
`U_t=y_t*S1_t/tr(S1_t)` and `V_t=S2_t/tr(S2_t)`, so
`Cov(vec_F(X_t) | past)=V_t ⊗ U_t` and its trace is `y_t`.
The code distinguishes trace-one column factors V from column marginal
covariances `y_t*V_t`. The recursions never substitute trace-normalized factors
for the unnormalized states.

`dynamics="full"` estimates all entries of the four dynamic matrices;
`"diagonal"` restricts those matrices and still permits full triangular
intercepts. Intercept diagonals are positive and their first entries fixed
at one. Scalar modes omit unidentified shape dynamics. Fitting uses the
paper's zero-state conditional Gaussian quasi-likelihood, analytic recursive
objective gradients and multiple constrained starts. Numerical parameter
boxes, selected starts, invalid evaluations and active constraints are visible.
An optimizer success flag does not establish global optimality or identification;
dynamic-matrix signs are observationally ambiguous.

The default spectral constraints implement accepted Assumption 3.3(i), which
alone does not prove stationarity. A separate `constraint="sufficient"` uses
the stronger Frobenius bound in accepted Theorem 1; failure of that sufficient
bound does not prove nonstationarity. `constraint="none"` retains positivity
and numerical optimization bounds. These constraints operate in the original
data units. No implicit mean subtraction or mean-model estimation is performed.

Filtering supports exact chronological continuation through its returned state.
`forecast_one()` computes the next conditional covariance exactly. Multistep
expectations of the normalized shape ratios do not close under substitution
of expected states, and no such approximation is offered. Gaussian simulation
returns the covariance states used to generate observations. Standardized
residuals use Cholesky whitening; they do not implement the paper's symmetric-root
residual coordinates or estimation-adjusted portmanteau inference. QMLE
standard errors, the portmanteau test and the matrix factor-GARCH branch remain
open. [Accepted publication](https://doi.org/10.1080/01621459.2024.2415719),
[implementation notes](volatility-notes.md), and
[example](../examples/matrix_garch.py).

### Sequential factor monitoring and reference calibration

`MatrixFactorMonitor` implements the journal-version procedure of He, Kong,
Trapani and Yu. The monitored baseline rank and opposite-mode projection rank
are fixed using a stable training period. Each arriving matrix replaces the
oldest observation in a fixed-length trailing window. Opposite-mode PCA is
re-estimated on that window, and **all** its observations are projected using
the new loading estimate. The next non-spiked eigenvalue is normalized by
the average projected eigenvalue and the dimension-dependent rate. The
journal's power transformation, followed by independent Gaussian randomization,
differs from the exponential transformation in the early arXiv version.
The default power eight entails the paper's finite 32nd-moment assumptions;
smaller powers have their corresponding moment requirements.

Maximum and weighted partial-sum statistics are implemented, including
Darling–Erdos normalization at eta=1/2 and delayed Renyi monitoring at eta>1/2.
The horizon is fixed before observing the monitoring sequence. Default Gumbel
boundaries are available for maxima and eta=1/2; other partial-sum exponents
require a supplied critical value. The procedure stops at the first strict
crossing or horizon exhaustion. `update_many` consumes only that prefix;
it does not restart automatically or revise ranks after an alarm. All reported
time indices are one-based, with observation index equal to training length
plus monitoring index. An alarm index includes detection delay and is not a
retrospective change-point estimate. Saved state includes the chronological
window, accumulated statistics, diagnostics and local random-number state.

The implemented alternatives cover newly appearing factors and changes that
enlarge the combined pre/post loading span. Disappearing factors, zero baseline
ranks, weak-factor extensions, and calibration across simultaneous modes or
restarts remain absent. Population moment, factor-strength and training-stability
assumptions cannot be verified by numerical input checks.
[Published paper](https://doi.org/10.1214/24-AOS2410),
[monitoring notes](monitoring-notes.md), and
[example](../examples/online_monitoring.py).

`calibrate_monitor` is a separate extension targeting a **finite, discrete
horizon of iid N(0,1) draws with zero drift**. Maxima use an exact Gaussian
quantile expressed on the monitor's normalized scale. Partial sums simulate
entire independent paths and calibrate their maximal statistic, including
the chosen delay and exponent. The conservative order-statistic choice has
an unconditional reference exceedance bound averaging over calibration
randomness; it is not a conditional guarantee for a particular simulated
threshold. Binomial order-statistic intervals describe quantile Monte Carlo
uncertainty, not matrix-data false-alarm uncertainty. `monitor_kwargs` transfers
the calibrated horizon/statistic/exponent/delay without changing the design.

Actual matrix-monitor scores retain a nonnegative data-dependent drift, even
under a finite-sample no-break model. Thus neither Gaussian-reference calibration
nor the asymptotic boundary supplies exact finite-sample matrix-data false-alarm
control. Empirical null rejection and detection-delay studies remain distinct
evidence, and multiple monitors require separate multiplicity control.
[Calibration notes](calibration-notes.md).

### Tucker time-series factors

For tensors, `X_t = F_t ×1 A1 ... ×K AK + E_t`. TOPUP unfolds a lagged outer
product; TIPUP first contracts matching coordinates in other modes, then unfolds.
Both use nonzero time lags to isolate dynamic factors. TOPUP can require
substantially more memory; TIPUP can cancel nonzero signals when contracted
components oppose one another. Their tradeoff must appear in benchmarks.
[Chen, Yang & Zhang](https://arxiv.org/abs/1905.07530).

Iterative variants project the observations onto current loading spaces in the
other modes and recompute a TOPUP or TIPUP estimate. Stopping should compare
subspace projectors so sign flips and rotations do not masquerade as lack of
convergence. Static HOOI is a useful baseline, but does not reproduce these
lagged estimators. [Han et al.](https://arxiv.org/abs/2006.02611).

## Further matrix families and incomplete paper features

These families are required for broad scientific coverage. Implemented branches
are identified below; their remaining paper features and benchmark evidence are
separate deliverables. An implemented branch does not cover an entire family.

| Family | Defining method and assumptions | Implementation acceptance evidence |
| --- | --- | --- |
| Robust matrix Kendall factors | Pairwise normalized matrix differences under matrix-elliptical structure; [He et al.](https://arxiv.org/abs/2207.09633), `he2025kendall` | Exact small-sample pair sum; translation/scale invariance; duplicate observations; heavy-tailed loading-space recovery; explicit pair-subsampling semantics |
| Huber matrix factors | Huber loss on matrix residual norms with weighted projection; [He et al.](https://arxiv.org/abs/2112.04186), `he2024huber` | Actual Huber objective descent, threshold calibration, Gaussian and contamination comparisons; entrywise Huber regression must have a different label |
| Iterative Huber regression | Fixed-threshold entrywise regression and preprint rank criteria implemented; [He et al. v1](https://arxiv.org/abs/2306.03317v1), `he2023ihr` | Remaining: accepted-version reconciliation, adaptive block-MAD, robust centering and inferential APIs; not interchangeable with matrixwise residual reweighting |
| Reduced-rank MAR | Rank-constrained left/right autoregressive matrices; [Xiao, Han, Chen & Liu manuscript](https://yuefenghan.github.io/papers/Reduced_Rank_MAR.pdf), `xiao2022rrmar` | Whitened reduced-rank regression against a dense reference; rank preservation; rank-selection recovery; ordinary truncation of the fitted coefficient is not the weighted solution |
| Sparse and Bayesian MAR | Continuous-normal-mixture MAR(1) EMVS implemented; [Celani, Pagnottoni & Jones (2024)](https://doi.org/10.1007/s11222-024-10402-y), `celani2024sparse` | Remaining: spike-and-slab MCMC, posterior interval coverage, lag-factor MAR*(P), tensor variants, and broader prior/local-mode sensitivity comparisons |
| Envelope MAR | Fixed-order, fixed-positive-dimension Gaussian EMAR implemented; [Samadi & De Alwis (2026)](https://doi.org/10.1080/07350015.2025.2537404), `samadi2026envelope` | Dense likelihood/profile and tangent-gradient oracles, reducing covariance blocks, paired misspecification benchmarks; remaining: dimension/order selection, sparse SEMAR, separate one-sided models, inference, zero-dimensional boundaries and broader local-mode/performance assessment |
| CP matrix factors | `X_t = sum_j f_jt a_j b_j.T + E_t`; generalized eigenanalysis with reduced-space refinement; [Chang et al. (2023)](https://arxiv.org/abs/2112.15423), `chang2023cp` | Nonorthogonal identifiable components up to permutation/scale; generalized-eigen residuals; repeated-root diagnostics; compare published refined estimator |
| Constrained matrix factors | Fully/partially constrained and identifiable multi-term loading estimators implemented; [Chen, Tsay & Chen (2020)](https://arxiv.org/abs/1710.06075), `chen2020constrained` | Shared block-moment, full-projector and joint-score oracles; remaining: journal-v3 reconciliation, calibrated constraint/rank tests, inference, weak temporal signal and broader near-overlap evidence |
| Matrix ARMA | Rank-one-per-lag conditional LS and separable Gaussian MLE, filtering and forecasting implemented; [Tsay (2024)](https://doi.org/10.1111/insr.12558), `tsay2024marma` | Dense innovation and Gaussian likelihood oracles, analytic gradient and noncommuting impulse responses; remaining: minimality/order selection, exact stationary likelihood, multiple Kronecker terms per lag and inferential APIs |
| Two-way dynamic factors | Additive quasi-likelihood estimator, conditional scores, pooled diagonal AR and iterative rank selection implemented; [Yuan et al. (2023)](https://doi.org/10.1093/jrsssb/qkad077), `yuan2023dynamic` | Remaining: automated noise-start sensitivity, factor-loading/bias-corrected inference, lag-selection assessment and broader misspecification comparisons; Kalman filtering is a separate extension |
| Simultaneous decorrelation | Bilinear transform with threshold/ratio grouping implemented; [Han et al. (2024)](https://arxiv.org/abs/2103.09411), `han2024decorrelation` | Remaining: optional VAR prewhitening, recursive irregular partitions, alternative moment spectral functions, and broader partition/forecast comparisons |
| Threshold factors | Single-threshold, unequal-rank factor estimator implemented; [Liu & Chen (2022)](https://doi.org/10.1111/sjos.12576), `liu2022threshold` | Remaining: multiple thresholds, threshold-variable selection, and broader weak-regime/threshold-location experiments; no threshold confidence interval or existence test is supplied |
| Matrix GARCH | Full/diagonal first-order trace-identified QMLE, chronological filtering, simulation and exact next covariance implemented; [accepted Yu et al. paper](https://doi.org/10.1080/01621459.2024.2415719), `yu2024garch` | Remaining: QMLE covariance/standard errors, estimation-adjusted portmanteau size/power, factor-GARCH, and validated multistep forecasting; default spectral feasibility is not stationarity proof |
| Online structural breaks | Journal-version power randomization, per-window projected eigenvalues, maximum/partial sums and resumable state implemented; [He et al. (2024)](https://doi.org/10.1214/24-AOS2410), `he2024breaks` | Remaining: disappearing/weak factors, zero-rank baselines, restart/multiple-mode calibration and broader null-size/delay evidence; the separate finite-H Gaussian reference does not prove finite-sample matrix null control |
| Rank and statistical inference | MAR(1) Wald covariance and Kronecker specification test, all 40 IC/ER factor-rank variants and explicit IC stability paths implemented; [Han, Chen & Zhang (2022)](https://arxiv.org/abs/2011.07131v3), `han2022rank` | Remaining: factor-loading inference, strength estimation, calibrated ER rank-zero extension, weak identification, long-run covariance and post-selection inference; IC zero selection is not a calibrated no-factor test |

The reduced-rank MAR reference is an author manuscript. Secondary bibliographies
disagree on its publication status; no unverified journal DOI is asserted here.

## Tensor extensions and emerging branches

`fit_tensor_ar` implements multilinear tensor autoregression as a sum of lagged
mode products, with independently specified term counts per lag. Projection,
least-squares and tensor-normal likelihood are distinct targets. For two spatial
modes, multi-term transition rearrangements have an exact SVD projection and
orthogonal canonical representation. Higher-order projection instead uses local
CP-ALS; a best low-rank tensor approximation may not exist. Initial projection
convergence, final-fit convergence, term cancellation and identification
diagnostics are retained separately. Bounded dense accessors diagnose stability
of the **sum** of all lag terms; component-wise stability is insufficient.
Dense projection can be bypassed by explicit or random initialization.
Term/lag selection, inferential covariance, and global CP optimization are not
provided. [Li & Xiao (2021 manuscript)](https://arxiv.org/abs/2110.00928),
`li2021tenar`; [implementation notes](tensor-autoregression-notes.md).

Low-rank tensor autoregression instead constrains the **transition tensor**,
with separate input and output loading modes. Its nuclear-norm estimators and
nonconvex Tucker estimator are not equivalent to multilinear rank-one
Kronecker autoregression. [Wang, Zheng & Li (2024)](https://arxiv.org/abs/2101.04276),
`wang2024lrtar`.

CP factors for dynamic tensors use nonorthogonal loading vectors and scalar
dynamic factors, with a specialized high-order projection procedure. Generic
CP-ALS on the complete space-time array does not implement this method.
[Han et al. (2024)](https://arxiv.org/abs/2110.15517), `han2024cp`.

Cointegrated MAR now has the fixed-rank branch described above; its rank tests
and asymptotic inference remain open. Other tracked branches include
time-varying factors ([Chen et al. manuscript](https://arxiv.org/abs/2404.01546)),
EM/Kalman matrix factor estimation
([Barigozzi & Trapin manuscript](https://arxiv.org/abs/2502.04112)), and
two-way threshold MAR ([Yu et al. manuscript](https://arxiv.org/abs/2407.10272)).
These are tracked as manuscript methods unless a journal version is separately
verified. Missing-data estimation, semiparametric factors, tensor pre-averaging,
envelope MAR, matrix count series, and network/spatial restrictions
also require dedicated implementations; none follows automatically from
supporting arbitrary tensor shapes.

## Scope and evidence

This inventory is intentionally maintained as research evolves. A useful
discovery index is the [Rutgers authors' bibliography](https://statistics.rutgers.edu/resources-of-analysis-of-tensor-time-series),
but inclusion decisions and formulas should be checked against each original
paper. A statistical method is covered only when its actual estimating equations,
assumptions, identified output, numerical behavior, and empirical comparisons
are represented. Citations or API stubs alone do not establish coverage.
