# Envelope matrix autoregression

Reference: S. Yaser Samadi and Tharindu P. De Alwis (2026),
*Envelope Matrix Autoregressive Models*, JBES 44(2), 397–412,
[DOI and final article](https://doi.org/10.1080/07350015.2025.2537404).
The final article and the
[author-deposited accepted manuscript](https://ircommons.uwf.edu/esploro/outputs/journalArticle/Envelope-Matrix-Autoregressive-Models/99381474285106600)
were checked against equations (13), (15)–(16) and Algorithm 1.
This implementation is an independently tested transcription of the estimator,
not a comparison against unavailable author software or a full paper replication.

## Model and scope

For observations `(T,m,n)`, order `p`, positive dimensions `(u,v)`,

\[
X_t=\mu+\sum_{l=1}^p A_l X_{t-l} B_l^\top+E_t,\quad
A_l=R\eta_l,\quad B_l=C\xi_l.
\]

The shared orthonormal output bases are `R:(m,u)` and `C:(n,v)`.
Inputs are not sandwiched between envelope projectors: immaterial past
coordinates may predict material responses. Both covariance factors are reduced
by the corresponding response space:

\[
\Sigma_r=R\Omega_rR^\top+R_0\Omega_{r0}R_0^\top,
\qquad
\Sigma_c=C\Omega_cC^\top+C_0\Omega_{c0}C_0^\top.
\]

Covariance blocks are positive definite. This mean–covariance linkage is what
distinguishes EMAR from ordinary low-rank regression. Full `(m,n)` dimensions
recover the unrestricted separable MAR likelihood, although a different local
optimization path can yield a different stationary point.

`fit_envelope_mar` implements fixed-order, fixed-positive-dimension Gaussian
conditional point estimation. It does **not** implement zero-dimensional null
models, the separate one-sided identity-coefficient models of Section 3.5,
SEMAR, order/dimension selection, asymptotic standard errors, sandwich inference,
confidence intervals, missing-data methods, or a stationarity constraint.
In particular, a full column envelope with a free B is not the paper's model
with B fixed to the identity. Model identification requires more than numerical
convergence; sample fitting cannot certify the population assumptions.

## Conditional sample and block likelihood

Exactly `N=T-p` observations are scored. With an intercept, center responses
and **each lag separately over these same N rows**. Write centered responses
as Y and the stacked design as
`F_t = concatenate([Z_lt @ B_l.T], axis=rows)`. This is algebraically
`V_t B.T` for the paper's block-diagonal V, without constructing V.

For fixed column coefficients and covariance, compute joint-lag weighted LS
`A_tilde` using whitened SVD least squares. The moment matrices are

\[
S_0=\frac1{Nn}\sum_tY_t\Sigma_c^{-1}Y_t^\top,\qquad
M=\frac1{Nn}\sum_t(Y_t-\widetilde A F_t)
\Sigma_c^{-1}(Y_t-\widetilde A F_t)^\top.
\]

Equation (15) minimizes the actual Grassmann profile

\[
f(U)=\log|U^\top M U|+\log|U^\top S_0^{-1}U|.
\]

Then `A=P_U A_tilde` and `Sigma_r=P_U M P_U+Q_U S0 Q_U`.
Transpose everything for the column update, with divisor **N*m**, not N*n.
The complete Gaussian negative conditional log likelihood is

\[
\tfrac12\left[Nmn\log(2\pi)+Nn\log|\Sigma_r|
+Nm\log|\Sigma_c|+
\sum_t\operatorname{tr}(\Sigma_r^{-1}E_t\Sigma_c^{-1}E_t^\top)\right].
\]

No degrees-of-freedom correction is applied to likelihood covariance updates.
The physical intercept is restored from the response mean minus the fitted
operators applied to the separate lag means.

## Numerical choices and diagnostics

- Divide data by its maximum absolute entry before fitting. Covariance factors
  split this scale on return, so their Kronecker product is the physical
  covariance. Physical log likelihood uses an additive log-scale correction
  without forming the potentially unrepresentable covariance product.
- Use Cholesky triangular solves for whitening and Cholesky solves for the
  profile gradient; never invert weighted normal equations. Joint-lag SVD
  regression differs numerically from cycling over lag pairs.
- Optimize the Grassmann logdet objective by QR retraction, transported
  conjugate-gradient directions and Armijo line search. Analytic horizontal
  gradient is the projection of
  `2 M U (U.T M U)^-1 + 2 S0^-1 U (U.T S0^-1 U)^-1`.
  Independent moment rescaling changes only an additive profile constant.
- Each block uses fresh smallest-residual-eigenvector, smallest-response-
  eigenvector and random starts (as allowed by `inner_starts`). The previous
  envelope is always an additional start. Eigenvectors are initializers,
  **not** the envelope estimator. Fixed `random_state` controls only a local RNG.
- Default initialization is an unrestricted MAR MLE as in Algorithm 1;
  its convergence is reported separately. User coefficient/covariance starts
  and initial envelope bases are supported. Large automatic projection
  warmups are replaced by identity coefficient initialization.
  When initial envelopes are not supplied, the leading left singular vectors
  of the warmup's concatenated lag coefficients provide an additional first
  start for each space (using the warmup's documented per-lag gauges).
  Covariance eigenvectors alone can get trapped in immaterial directions when
  material variance is high. The warmup spaces remain merely initializers;
  the logdet optimization and competing fresh starts are still performed.
- Rank-deficient weighted designs and numerically singular residual moments
  raise. The latter uses a `10*eps*||S0||_2` eigenvalue threshold to reject
  saturated regressions with roundoff-sized residuals. Very ill-conditioned
  but mathematically finite problems may therefore also be rejected.
  No covariance floor, ridge, pseudodeterminant or hidden regularization is used.
- Objective history starts at the first complete EMAR sweep, not at the
  unrestricted warmup, whose larger parameter space has a different optimum.
  A subsequent likelihood increase above a relative `1e-10` allowance is
  rejected and diagnosed. Roundoff-sized increases may be retained.
- `converged` requires both the outer relative likelihood criterion and
  selected row/column inner gradient criteria. Inspect `optimization_history`
  for all start outcomes and budgets. Inner gradient convergence, outer
  likelihood convergence, global optimality and time-series stability are
  distinct claims. Only the first two are stopping conditions.
- Each nonzero lag pair uses unit-Frobenius left factor and a positive largest
  absolute entry; the opposite factor receives the reciprocal scaling. Zero
  left factors are not replaced by vectors outside the fitted envelope.
  A resulting unidentified zero design is rejected on a later update.
- Row covariance has unit Frobenius norm in scaled-data units. Result dense
  coefficients, residual covariance and higher-order companion diagnostics
  have a configurable allocation guard. Matrix-form forecasts do not allocate
  the vectorized transition or covariance.

## Source discrepancies and explicit interpretation

The implementation uses the internally consistent conditional Gaussian model,
not literal evaluation of dimensionally inconsistent displays:

1. Equation (14)'s displayed determinant multiplicities interchange m and n;
   equation (7) and the Gaussian density give the multiplicities above.
   The displayed inverse of an ambient rank-deficient complement block is
   interpreted as the restricted precision `R0 Omega_r0^-1 R0.T`.
2. Algorithm 1's row-complement covariance subscript differs across versions:
   the accepted manuscript has a column-conditioned subscript, while the
   final PDF's extracted algorithm and equation (15) use the correct
   row-conditioned moment. The implemented update follows equation (15).
3. Scaling by the sign of a designated entry fails when that entry is zero.
   We choose the largest-magnitude entry for sign identification. For p>1,
   every nonzero lag pair has its own reciprocal scale ambiguity; normalizing
   only the concatenated coefficients does not remove these p gauges.
4. The algorithm's displayed covariance rescaling preserves the Kronecker
   product but does not enforce the stated row-unit-norm convention.
   We use `Sigma_r/d, d*Sigma_c` with `d=||Sigma_r||_F`.
5. Rectangular envelope coordinates do not have ordinary spectral radii.
   Stability uses actual square A and B for p=1, and the full vectorized
   companion for p>1. It is not inferred from individual lag stability.

The paper's displayed generic parameter count subtracts two gauges; for
arbitrary independent nonzero lag pairs there are p coefficient gauges plus
one covariance gauge. The regular-stratum count is
`p*(m*u+n*v) + m*(m+1)/2+n*(n+1)/2-p-1`, plus `m*n` for a free intercept.
Zero lag terms and other nonregular boundaries need separate treatment.
No selection criterion is exposed here, so this count is not used to report
unvalidated AIC/BIC dimension choices or consistency guarantees.

## Evidence and examples

`tests/test_envelope.py` independently checks dense Kronecker Gaussian
likelihoods, weighted vectorized coefficient fits, profiled likelihood identity,
finite-difference tangent gradients, a fine two-dimensional angle oracle,
full-envelope MAR equivalence, reducing covariance blocks, multiple lags,
intercepts, causal forecasts, scale equivariance, singularity guards and
honest nonconvergence. See [the example](../examples/envelope_mar.py) and
[benchmark protocol](../benchmarks/README.md#envelope-mar).
