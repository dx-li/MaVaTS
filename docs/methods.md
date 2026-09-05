# Methods and scientific coverage

This is a coverage inventory, not a claim that every cited paper is implemented.
The scope is time-indexed matrices and tensors, with matrix methods taking priority.
References were checked against primary papers and author manuscripts on 2026-09-05.
Machine-readable citations are in [references.bib](references.bib); implementation
acceptance criteria and remaining work are in [roadmap.md](roadmap.md).

## Shared notation and interpretation

A matrix series has shape `(T, m, n)`, with observation `X[t]`. A tensor series has
shape `(T, d1, ..., dK)`. Time is never treated as an interchangeable spatial mode.
Column-major vectorization gives `vec(A X B.T) = (B ⊗ A) vec(X)`.
Factor loading coordinates depend on normalization and rotation: comparisons
should use loading spaces or reconstructed common components. Autoregressive
coefficient pairs also have scale ambiguity: `(c A, B/c)` gives the same operator.
These are mathematical ambiguities, not estimation failures.

## Core estimator inventory

The following point estimators are implemented in the working tree. The focused
autoregression, factor, tensor, robust, and constrained tests passed on
2026-09-05. This does not establish reproduction of every source paper's
simulations or full inferential theory. Public functions return fitted result
objects; the older `estimate_*` functions are compatibility interfaces.

| Implemented API | Statistical object | Paper | Coverage boundary |
| --- | --- | --- | --- |
| `fit_mar(method="projection")` | Nearest Kronecker approximation to an unrestricted VAR coefficient | Chen, Xiao & Yang (2021), `chen2021mar` | Dense VAR allocation; projection is not bilinear least squares |
| `fit_mar(method="als")` | Bilinear conditional mean | Same paper | Conditional point estimation; scale-invariant ridge is an optional extension |
| `fit_mar(method="mle")` | Bilinear mean and separable innovation covariance | Same paper | Covariance flooring, if applied, produces a stabilized likelihood estimate; diagnostics report it |
| `fit_mar(order=p)` | Sum of lag-specific bilinear operators | MAR extension | One Kronecker term per lag; no stationarity constraint |
| `fit_mar(ranks=(r,s))` | Reduced-rank bilinear mean via ALS | Xiao et al. manuscript, `xiao2022rrmar` | RR.LS for order one/ridge zero; the distinct RR.CC estimator is absent |
| `fit_alpha_pca` | Mean and contemporaneous second-moment loading spaces | Chen & Fan, `chen2021alphapca` | Legacy covariance routines are not a validated full inferential API |
| `fit_projected_pca` | Refined row/column loading spaces | Yu et al. (2022), `yu2022projection` | Default is one simultaneous projected update; repeated simultaneous updates are an extension; ranks fixed after initialization |
| `fit_lagged_factor` | Loading spaces from all column-pair lag covariances | Wang, Liu & Chen (2019), `wang2019matrixfactor` | Requires temporal signal; contemporaneous PCA is a different estimator |
| `fit_tensor_factor(method="topup")`, `method="tipup"` | Tucker loading spaces from outer/inner lag products | Chen, Yang & Zhang (2022), `chen2022tensorfactor` | Inner-product signal cancellation is an actual model limitation |
| `fit_tensor_factor(iterative=True)` | Sequential loading-space updates after projection along other modes | Han et al. (2024), `han2024iterative` | Ranks fixed after initialization; no full rank-selection/inference implementation |
| `fit_matrix_kendall` | Robust loading spaces and projected factor scores | He et al. (2025), `he2025kendall` | Exact pairs by default; optional pair sampling is an approximation; projected scores themselves are not robust |
| `fit_constrained_factor` | Loading spaces within supplied row/column spans | Chen, Tsay & Chen (2020), `chen2020constrained` | Single-term fully constrained model; partial and multi-term estimators are absent |
| `fit_cp_factor` | Nonorthogonal rank-one matrix components via refined generalized eigenanalysis | Chang et al. (2023), `chang2023cp` | Specified rank, unthresholded moments; rejects singular, complex or unseparated eigenproblems |
| `fit_huber_factor` | Matrixwise Huber factor loss with weighted projection | He et al., `he2024huber` | Fixed threshold and explicit ranks; simultaneous paper update has a documented sequential descent safeguard; entrywise Huber inference is absent |

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

These families are required for broad scientific coverage. Where a point
estimator appears above, the remaining paper features and benchmark evidence
are still separate deliverables. All other entries below are unimplemented.

| Family | Defining method and assumptions | Implementation acceptance evidence |
| --- | --- | --- |
| Robust matrix Kendall factors | Pairwise normalized matrix differences under matrix-elliptical structure; [He et al.](https://arxiv.org/abs/2207.09633), `he2025kendall` | Exact small-sample pair sum; translation/scale invariance; duplicate observations; heavy-tailed loading-space recovery; explicit pair-subsampling semantics |
| Huber matrix factors | Huber loss on matrix residual norms with weighted projection; [He et al.](https://arxiv.org/abs/2112.04186), `he2024huber` | Actual Huber objective descent, threshold calibration, Gaussian and contamination comparisons; entrywise Huber regression must have a different label |
| Iterative Huber regression | Entrywise robust regression for factor/loadings inference; [He et al.](https://arxiv.org/abs/2306.03317), `he2023ihr` | Validate each block update and associated inference; not interchangeable with matrixwise residual reweighting |
| Reduced-rank MAR | Rank-constrained left/right autoregressive matrices; [Xiao, Han, Chen & Liu manuscript](https://yuefenghan.github.io/papers/Reduced_Rank_MAR.pdf), `xiao2022rrmar` | Whitened reduced-rank regression against a dense reference; rank preservation; rank-selection recovery; ordinary truncation of the fitted coefficient is not the weighted solution |
| Sparse and Bayesian MAR | Spike-and-slab MCMC and EM variable selection; [Celani, Pagnottoni & Jones (2024)](https://doi.org/10.1007/s11222-024-10402-y), `celani2024sparse` | Recover supports, compare posterior/EM targets, convergence diagnostics, interval coverage; a ridge option is not sparse estimation |
| CP matrix factors | `X_t = sum_j f_jt a_j b_j.T + E_t`; generalized eigenanalysis with reduced-space refinement; [Chang et al. (2023)](https://arxiv.org/abs/2112.15423), `chang2023cp` | Nonorthogonal identifiable components up to permutation/scale; generalized-eigen residuals; repeated-root diagnostics; compare published refined estimator |
| Constrained matrix factors | Known linear loading constraints, including partial and multi-term constraints; [Chen, Tsay & Chen (2020)](https://arxiv.org/abs/1710.06075), `chen2020constrained` | Constraint residuals, equivalent orthonormal bases, partial-constraint cases, misspecified-constraint experiment |
| Two-way dynamic factors | Two-way dynamic dimension reduction; [Yuan et al. (2023)](https://doi.org/10.1093/jrsssb/qkad077), `yuan2023dynamic` | Reproduce the specified dynamic estimation procedure and forecasting protocol; static PCA followed by any VAR is not sufficient |
| Simultaneous decorrelation | Bilinear transformation to mutually uncorrelated submatrix series; [Han et al. (2024)](https://arxiv.org/abs/2103.09411), `han2024decorrelation` | Recover block partitions and cross-lag decorrelation; retain invertible reconstruction and forecast comparison |
| Threshold factors | Regime-dependent loadings selected by an observed threshold variable; [Liu & Chen (2022)](https://doi.org/10.1111/sjos.12576), `liu2022threshold` | Regime-specific cross-lag moment sums, threshold-location consistency, trimmed search, unequal regime ranks |
| Matrix GARCH | Conditional row/column covariance dynamics with an identified trace process; [Yu et al.](https://arxiv.org/abs/2306.05169), `yu2024garch` | Positive definite covariance at every step; likelihood reference; covariance forecast scoring and portmanteau size/power |
| Online structural breaks | Monitor non-spiked eigenvalues with sequential randomization; [He et al. (2024)](https://arxiv.org/abs/2112.13479), `he2024breaks` | Null false-alarm control and detection delay across repetitions; genuinely sequential state with no future observations |
| Rank and statistical inference | Rank criteria from [Han, Chen & Zhang (2022)](https://arxiv.org/abs/2011.07131), `han2022rank`; loading inference in the alpha-PCA/projection papers | Null/no-factor cases, weak factors, finite-sample coverage and size, long-run covariance checks; an eigenvalue ratio alone does not implement every rank criterion |

The reduced-rank MAR reference is an author manuscript. Secondary bibliographies
disagree on its publication status; no unverified journal DOI is asserted here.

## Tensor extensions and emerging branches

Multilinear tensor autoregression uses a sum of lagged mode products. The
projection, least-squares, and matrix-normal/tensor-normal likelihood estimators
are distinct targets. [Li & Xiao (2021 manuscript)](https://arxiv.org/abs/2110.00928),
`li2021tenar`.

Low-rank tensor autoregression instead constrains the **transition tensor**,
with separate input and output loading modes. Its nuclear-norm estimators and
nonconvex Tucker estimator are not equivalent to multilinear rank-one
Kronecker autoregression. [Wang, Zheng & Li (2024)](https://arxiv.org/abs/2101.04276),
`wang2024lrtar`.

CP factors for dynamic tensors use nonorthogonal loading vectors and scalar
dynamic factors, with a specialized high-order projection procedure. Generic
CP-ALS on the complete space-time array does not implement this method.
[Han et al. (2024)](https://arxiv.org/abs/2110.15517), `han2024cp`.

Other tracked branches include cointegrated MAR with bilinear cointegrating
spaces ([Li & Xiao manuscript](https://arxiv.org/abs/2409.10860), `li2024cointegration`),
time-varying factors ([Chen et al. manuscript](https://arxiv.org/abs/2404.01546)),
EM/Kalman matrix factor estimation
([Barigozzi & Trapin manuscript](https://arxiv.org/abs/2502.04112)), and
two-way threshold MAR ([Yu et al. manuscript](https://arxiv.org/abs/2407.10272)).
These are tracked as manuscript methods unless a journal version is separately
verified. Missing-data estimation, semiparametric factors, tensor pre-averaging,
matrix ARMA, envelope MAR, matrix count series, and network/spatial restrictions
also require dedicated implementations; none follows automatically from
supporting arbitrary tensor shapes.

## Scope and evidence

This inventory is intentionally maintained as research evolves. A useful
discovery index is the [Rutgers authors' bibliography](https://statistics.rutgers.edu/resources-of-analysis-of-tensor-time-series),
but inclusion decisions and formulas should be checked against each original
paper. A statistical method is covered only when its actual estimating equations,
assumptions, identified output, numerical behavior, and empirical comparisons
are represented. Citations or API stubs alone do not establish coverage.
