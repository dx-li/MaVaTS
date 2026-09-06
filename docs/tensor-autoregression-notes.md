# Multilinear tensor autoregression

`fit_tensor_ar` implements the multilinear TenAR model of Zebang Li and Han
Xiao, [*Multi-linear Tensor Autoregressive Models*](https://arxiv.org/abs/2110.00928),
arXiv:2110.00928v1 (October 3, 2021). The arXiv record and the authors'
[current Rutgers bibliography](https://statistics.rutgers.edu/resources-of-analysis-of-tensor-time-series)
still identify this as a manuscript; no newer journal version was verified.
Equation numbers below refer to [the primary manuscript](https://arxiv.org/html/2110.00928v1).
The authors' [tensorTS software manuscript, Section 2.1](https://yuefenghan.github.io/papers/R_software_paper_tensorTS.pdf)
provides an additional exposition, not a replacement source for the equations.

## Model, layout and identification

For observations `(T,d1,...,dK)`, with K at least two, equation (8) is

\[
X_t=\sum_{i=1}^{p}\sum_{r=1}^{R_i}
 X_{t-i}\times_1 A_1^{(ir)}\cdots\times_K A_K^{(ir)}+E_t.
\]

Every A is a square mode matrix; none is required to have low matrix rank.
`terms=(2,1)` supplies two terms at lag one and one at lag two.
`terms=(2,1), lags=(1,3)` explicitly omits lag two. Counts must be positive;
the lag sequence must be increasing and unique. Coefficients are returned as
`result.coefficients[lag_index][term][mode]`, where the first index addresses
`result.lags`, not necessarily every integer lag. Fitted values and residuals
correspond to the common response interval `X[max(lags):]`.

Spatial vectorization is column-major (Fortran order), with time kept separate:

\[
\Phi_i=\sum_r A_K^{(ir)}\otimes\cdots\otimes A_1^{(ir)}.
\]

The low-dimensional restriction is low CP rank of a **rearranged** transition
matrix, equation (4). It is not a low-Tucker-rank restriction on the order-2K
transition tensor. The latter is a distinct estimator family. It is also not
a dynamic factor model or a Tucker core followed by a VAR.

Within each term the first K-1 coefficient matrices have Frobenius norm one.
Their largest-magnitude entries are nonnegative; compensating signs and
scales are transferred to the final mode, preserving the complete term.
Scale transfer is delayed and exponent-balanced, so intermediate underflow
or overflow does not destroy compensating gauges across three or more modes.
Terms are sorted by decreasing product Frobenius norm. These conventions
remove scaling/sign/permutation ambiguity but do not establish identification.

For matrices with multiple terms, Proposition 1 additionally requires
Frobenius orthogonality across terms in both modes and distinct nonzero
singular values of the rearranged transition. A skinny-QR/small-SVD
canonicalization supplies the orthogonal representation without constructing
that dense transition. Repeated singular values are diagnosed, not artificially
separated. For K>=3, the generalized Kruskal condition is sufficient.
The implementation reports the easy sufficient case in which every matrix of
vectorized mode coefficients has full column rank. It does not attempt the
combinatorial Kruskal-rank calculation. Failure to certify that condition is
not proof of nonidentification. Zero components are explicitly flagged.

## Least squares and likelihood blocks

Equations (9)--(10) minimize the complete residual sum of squares. For a
coefficient block `(i,r,k)`, remove all other lag/term predictions from the
response to form Y and define

\[
W_t=X_{t-i,(k)}\Phi_{-k}^{(ir)\top}.
\]

The mode-k unfolding keeps remaining mode indices in Fortran order.
The exact conditional coefficient update is

\[
A_k^{(ir)}=
 \left(\sum_tY_tQ_kW_t^\top\right)
 \left(\sum_tW_tQ_kW_t^\top\right)^{-1},
\]

where Q is identity for LS, or the inverse Kronecker product of the other
mode covariances for MLE. The left mode covariance cancels from this normal
equation. Code does not form these normal equations: mode products and
separable whitening build a rectangular regression design, and SVD least
squares solves it without squaring its condition number. Rank-deficient
coefficient designs fail with a diagnostic instead of silently selecting an
unidentified coefficient vector. Updates are sequential across lags, terms
and modes; full residuals are recomputed after canonicalization.

MLE additionally assumes iid Gaussian innovations with separable covariance
`Sigma_K kron ... kron Sigma_1`, equation (7). Let `N=T-max(lags)` and
`D=product(shape)`. The conditional negative log likelihood, including its
Gaussian constant, is

\[
\frac12\left\{ND\log(2\pi)+N\sum_k(D/d_k)\log|\Sigma_k|
 +\sum_t e_t^\top(\Sigma_K\otimes\cdots\otimes\Sigma_1)^{-1}e_t\right\}.
\]

The mode covariance update following equation (11) is

\[
\Sigma_k=\frac{\sum_tE_{t,(k)}\Sigma_{-k}^{-1}E_{t,(k)}^\top}{N(D/d_k)}.
\]

The covariance sweep runs from the last mode to the first, using updated
factors. All but the final covariance factor are normalized to unit Frobenius
norm, transferring scale into the final factor. This is covariance-factor
identification, not a change to the innovation covariance.

There are indexing/constant slips in the manuscript's general-p display:
some residual/LS expressions retain `t-1` instead of `t-i`, the likelihood
retains `T-1`, and the Gaussian constant lacks D. The implementation follows
the model in equation (8), the correctly indexed coefficient update in
Section 3.2, and the full conditional Gaussian density over N usable responses.
For p=1 the covariance denominator is exactly `(T-1)*(D/d_k)`.

## Initialization and local optimization

`init="projection"` first estimates the unrestricted VAR jointly over the
specified lags. Each lag coefficient is rearranged and approximated by a
rank-R tensor, equations (12)--(13). K=2 uses exact truncated SVD. For K>=3
the implementation uses complete-mode CP-ALS, an explicitly local numerical
solver for the paper's projection objective. It does not claim the global
best approximation, the incoherence-based guarantees of other tensor
algorithms discussed in Section 3.3, or existence of a best rank-R tensor.

Each start retains per-lag `TensorARProjection` records containing the solver
type, objective history, iteration count, convergence flag, count of deficient
CP regression blocks, and component-cancellation ratio. A limited CP
initializer can still lead to a converged LS/MLE fit; the two convergence
flags remain distinct. Standalone `method="projection"` chooses starts by
the sum of projection losses, not by the prediction SSE. Its top-level
objective is nevertheless the resulting prediction SSE; projection losses
are in the separate records.

Dense unrestricted VAR projection requires a full-column-rank design and
`D*number_of_lags <= max_dense_dimension` (default 256). It does not silently
change method when that requirement fails. Use `init="random"` or supply
`initial` for larger or shorter designs. Subsequent LS/MLE contractions never
allocate the D-by-D transition or full covariance.

For projection-initialized MLE, covariance initialization follows Section 3.3:
form unrestricted VAR residual second moments and recursively apply nearest
Kronecker SVDs, orienting factors positively. A positive semidefinite
initializer need not be positive definite. Singular initializations/updates
are rejected unless an explicit relative covariance floor is requested.
Random/user coefficient starts without `covariance_initial` instead use
isotropic residual variance, a documented computational alternative.

`n_starts` keeps every run. Random starts and perturbations are seeded; no
true coefficient is used unless the caller explicitly supplies it. LS/MLE
select the smallest finite retained fitting objective, even if the selected
run has not converged. A failed block or an increasing objective rolls back
the entire current sweep, preserving the last completed iterate and marking
that run failed. Inspect `status`, `message`, `converged`, histories, condition
numbers, and selected-start diagnostics. If no start supplies a finite
initial objective, fitting raises with all failure messages.

LS stops on relative SSE change; MLE stops when the change in NLL per scalar
response is below `tol`. The latter is invariant to the additive likelihood
constant induced by data units. These tests establish numerical objective
stabilization, not global optimality or small parameter uncertainty.

The CP projection and fitting routines reject component cancellation ratios
above 1e8: the ratio is the sum of individual transition-term Frobenius norms
divided by the norm of their sum, computed from small Gram matrices. This is
a numerical degeneracy safeguard, not a statistical rank test or proof that
all noncompact degeneracies have been detected. An optional covariance floor
is likewise a stabilization extension; its use is exposed and must not be
called unregularized MLE. An increasing floored sweep is still rolled back.

## Units, forecasts and omissions

Observed data are divided by their largest absolute entry. Coefficients are
unchanged by a scalar unit change. Objective histories use working units:
LS SSE multiplies by `data_scale**2` for physical SSE; physical NLL adds
`residuals.size*log(data_scale)`. `log_likelihood` performs the latter
conversion. Returned covariance factors use physical units, with all scale
in the final factor. Nonrepresentable covariance factors are rejected.
The guarded dense covariance accessor additionally checks positive definiteness
after Kronecker materialization. No hidden covariance ridge is applied.

Default `center=False` follows the zero-mean model. `center=True` estimates a
fixed sample mean once and uses `mean + sum(term(history-mean))` for forecasts;
it does not profile a free intercept or account for mean-estimation uncertainty.
`forecast(steps, history=...)` accepts chronological time-first history and
recursively substitutes predicted tensors at future lags. This is a plug-in
linear predictor, and a conditional mean when future errors are martingale
differences (in particular under the iid Gaussian MLE model). Serial
uncorrelatedness alone does not establish that stronger interpretation.

Stability uses the **sum** of all term transitions and the full VAR companion
including omitted-lag zero blocks. Individual term radii do not establish
multiterm/multilag stability. The exact one-term, lag-one Kronecker shortcut
uses logarithmic products to avoid intermediate underflow. Other dense
diagnostics have allocation guards; exceeding the companion guard raises
rather than reporting an approximate radius as a certificate. Stability is
not enforced during fitting. Forecast uncertainty and intervals are absent.

Independent tests cover nonsquare K=3 entrywise contractions; complete
multiterm/gapped-lag LS and GLS sweeps reconstructed with dense regression
designs; mode covariance updates and full Gaussian likelihood; MAR projection,
LS and MLE special cases; matrix multiterm orthogonality; forecasts and companion
stability; known-transition recovery; failures, centering, and extreme units.
These tests are implementation evidence, not reproductions of all manuscript
experiments or its asymptotic theory. See the executable
[tensor autoregression example](../examples/tensor_autoregression.py).

Remaining: information-criterion order/term selection (equations 14--18),
coefficient inference, near-boundary identification analysis, global CP
optimization guarantees, nonseparable likelihood fitting, penalized/high-order
transition alternatives, and forecast uncertainty. These are separate
deliverables; implementing TenAR does not cover the low-rank transition-tensor
family.
