# Conditional matrix ARMA

The implementation follows the rank-one-per-lag family in Ruey S. Tsay
(2024), *Matrix-Variate Time Series Analysis: A Brief Review and Some New
Developments*, International Statistical Review 92(2), 246–262,
[DOI:10.1111/insr.12558](https://doi.org/10.1111/insr.12558).
The article appeared online in November 2023; 2024 is the journal-volume year.
The [author's institutional full text](https://knowledge.uchicago.edu/records/prg0h-r9h91)
is the primary equation source. No author software was copied.

## Model and conditioning

For an m-by-n series, equation (4) is

`X[t] = C + sum_i A[i] X[t-i] B[i]' + E[t] - sum_j L[j] E[t-j] R[j]'`.

Each lag has one bilinear term, not an unrestricted vector coefficient and
not the paper's broader rank-r sum. The MA sign is **minus**. With column-major
vectorization, equation (5) gives `Phi[i] = kron(B[i], A[i])` and
`Theta[j] = kron(R[j], L[j])`. Omitting all MA lags gives MAR; omitting AR
lags gives matrix MA; orders (0,0) give a matrix white-noise model with a
possible constant.

Section 2.5 conditions on the first `t0 = max(p,q)` observed matrices, fixes
their innovations to zero, and scores only times `t0,...,T-1` in zero-based
Python indexing. The filtering recursion is
`E[t] = X[t] - C - AR[t] + sum_j L[j] E[t-j] R[j]'`.
This is a conditional likelihood, not an exact stationary or Kalman likelihood.
In particular, computing residuals throughout the conditioning prefix would
define a different objective. `filter_marma` retains full-length innovations
(zero prefix), fitted values (NaN prefix), and a conditioning mask. A fit's
`residuals` and `fitted_values` select only its scored suffix.

## Objectives and numerical algorithm

`fit_marma(..., method="ls")` minimizes the mean, per scored time, of the
squared Frobenius innovations. `method="mle"` jointly estimates the
coefficient pairs, unrestricted intercept when requested, and positive-definite
separable innovation covariance `Sigma = V kron U` by conditional Gaussian
maximum likelihood. With N scored times, its negative log likelihood is

`0.5 * [N*m*n*log(2*pi) + N*(n*logdet(U)+m*logdet(V))`
`       + sum_t trace(V^-1 E[t]' U^-1 E[t])]`.

The quadratic term in printed equation (30) lacks the necessary factor 1/2;
the implementation uses the Gaussian density above, including all constants.
The optimized objective is this expression divided by N, in numerically
centered/scaled training units. The returned `log_likelihood` is instead the
total conditional likelihood in original physical units. LS does not claim a
Gaussian likelihood: its `log_likelihood` is None. Its forecast covariance uses
the unrestricted, uncentered second moment of fitted innovations with divisor
N, without a degrees-of-freedom correction. MLE uses its fitted separable
covariance and no covariance floor or ridge regularization.

This is direct local numerical optimization of the paper's objectives, not a
claim to reproduce its suggested initialization exactly. The automatic start
fits an unrestricted VAR(p) and projects each lag onto its nearest rank-one
Kronecker product; it then supplies small, nonzero MA coefficients. Additional
starts perturb the factors using a local reproducible random generator. MLE
starts without supplied covariance first run conditional LS and initialize the
separable covariance with 100 flip-flop updates. This fixed flip-flop budget is
an initializer, not the final MLE convergence criterion. A user-provided MLE
start with both covariance factors bypasses LS warm-up. Failed attempts remain
in `runs`; finite unconverged candidates can be returned and are explicitly
marked unconverged. Selection uses the smallest retained objective, not the
optimizer's success flag.

The exact gradient differentiates through the entire innovation recursion.
For MLE set `Q[t] = U^-1 E[t] V^-1 / N`, then run the reverse recursion
`H[t] = Q[t] + sum_j L[j]' H[t+j] R[j]`, with out-of-suffix terms zero.
For example, `dC = -sum H[t]`,
`dA[i] = -sum H[t] B[i] X[t-i]'`, and
`dL[j] = sum H[t] R[j] E[t-j]'`. LS uses `Q[t]=2 E[t]/N`.
Covariance gradients are propagated through lower Cholesky factors with
log-diagonal coordinates. This avoids freezing old residuals while optimizing
the MA parameters, which would not minimize the recursive objective.

## Admissibility, gauges, and convergence

By default `enforce_admissibility=True` and `stability_margin=1e-6`.
SLSQP constrains the **complete** AR and MA companion spectral radii to
`rho <= 1 - stability_margin`. These are the paper's stationary/invertible
polynomial regions, inset by an explicit numerical margin; separate modewise
bounds would be unnecessarily restrictive. Both polynomials have the form
`I - sum_lag coefficient[lag] z**lag`, because of the minus-MA convention.
First-order radii are computed as products of mode radii; higher-order models
use full summed Kronecker companion matrices. Constraint derivatives use
numerical differences, while objective derivatives are analytic.

Automatic/perturbed starts outside the interior are contracted by multiplying
each lag-l right factor by `c**l`, which contracts every companion root by c.
Supplied infeasible starts are rejected, not silently modified. Margins too
small to make `1-margin < 1` in floating point are rejected. If a final solver
point is infeasible or nonfinite, the best finite feasible point encountered
is retained with convergence False. Feasibility, constraint residuals, raw
gradient norm, and the optimizer's convergence status are separate diagnostics.
SLSQP's `ftol=tol` checks several convergence conditions: its success flag is
not a raw-gradient or KKT certificate. The optional
`enforce_admissibility=False` is an explicitly unrestricted extension using
BFGS; it may return unstable or noninvertible local fits. A finite likelihood
alone is not evidence of admissibility.

Each lag is optimized in a fixed-pivot chart (largest initial left-factor
entry fixed to one), then returned with unit-Frobenius left factor and a
deterministic sign. Distinct starts may use distinct charts. These charts are
local numerical parameterizations; a solution whose preferred pivot vanishes
can require another start. The row-covariance Cholesky first diagonal is fixed
to one to remove the separable covariance scale ambiguity. Equivalent finite
huge/tiny coefficient gauges are balanced before contractions.

Neither these gauges nor stationary/invertible polynomials establish the
structural identification conditions in Section 2.1. Left coprimeness,
minimal orders, and the joint terminal-coefficient rank condition are **not
certified**. In particular, identical AR and MA polynomials cancel to white
noise. `structural_identification_certified` is always False. Weakly identified
or nearly cancelling models may have very different local coefficients with
similar forecasts and likelihoods. There is no claim of global optimization,
unique estimates, or finite-sample inference.

`max_iter` limits each final optimization, not the whole multistart call.
Each separate MLE LS warm-up has budget `min(max_iter,100)` and tolerance
`max(tol,1e-5)`. Its iterations, status, history, and message are recorded
separately. A failed or unconverged warm-up must not be counted as a successful
final optimization. All-starts failure raises an error with the failure reasons.

## Forecasting, scaling, and scope

`result.forecast(h)` uses the stored training innovations; passing
`history=complete_prefix` refilters that full prefix with fixed parameters
and the same zero-prefix convention. It never refits parameters or accesses
later observations. Filtering separate chunks does not preserve innovation
state. Forecast means recurse with future innovations set to zero.

For full column-vector forecast covariance, `Psi[0]=I` and
`Psi[s]=sum_i Phi[i] Psi[s-i] - Theta[s]` (missing terms zero).
The horizon-h covariance is `sum_{s=0}^{h-1} Psi[s] Sigma Psi[s]'`.
Multiplication order matters for noncommuting coefficients. Even if Sigma is
separable, the multi-step covariance generally is not. Returned arrays have
shape `(h,m*n,m*n)` and omit parameter-estimation and presample-state
uncertainty. Dense companion/covariance allocations are guarded by
`max_dense_dimension=256`; raising the guard is an explicit memory choice.

When an intercept is fitted, the training mean is removed only as a numerical
reparameterization. Data are scaled by their maximum centered deviation. The
physical constant is restored by `C = C_work*scale + mean - sum A mean B'`;
it need not be a stationary-mean parameter. Fixed-zero-intercept fits are not
centered. Objectives therefore avoid arbitrary unit/offset constants in their
stopping criteria. Filtering and forecasting balance scales and reject
nonfinite represented results. An otherwise valid factored covariance whose
dense variances underflow or overflow cannot be returned as a spurious zero
matrix. Positive-definite covariance is required; degenerate data can fail MLE
without regularization. Extremely ill-conditioned designs remain a numerical
limitation, not an automatically certified statistical identification result.

`simulate_marma` uses local independent matrix-normal draws, zero presample
states, and an explicit finite burn-in. It requires a stable AR polynomial;
noninvertible MA coefficients are allowed when generating data. Finite burn-in
is not an exact stationary draw. Automatic order/rank selection, general
rank-r-per-lag MARMA, seasonal terms, exact stationary likelihood, coprimeness
tests, standard errors, and hypothesis tests are not implemented.

Independent tests construct dense triangular conditional Gaussian systems,
differentiate their likelihood numerically, and construct dense joint future
shock maps. They also cover nonsquare and noncommuting matrices, the MA sign,
conditioning, scalar/multilag boundaries, cancellation, extreme equivalent
gauges, physical units, fit recovery, causality, and invalid starts.
