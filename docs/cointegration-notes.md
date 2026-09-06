# Cointegrated matrix autoregression

`fit_cmar` implements the fixed-rank error-correction model of Li and Xiao,
[Cointegrated Matrix Autoregression Models](https://arxiv.org/abs/2409.10860).
The primary version was rechecked for this implementation: arXiv lists only
v1, September 17, 2024, without a journal reference. The
[Rutgers tensor-time-series reference page](https://statistics.rutgers.edu/resources-of-analysis-of-tensor-time-series),
dated June 14, 2026, likewise lists the preprint. Do not confuse this with later
papers having similar titles or silently import their different model structures.

## Model and supported interface

For time-first observations `X[t]` of shape `p` by `q`, equations (3)–(6) are

```
Delta X[t] = A1 @ X[t-1] @ A2.T
             + sum_i B_i1 @ Delta X[t-i] @ B_i2.T + D + E[t]
A1 = alpha1 @ beta1.T                  rank(A1) = r1
A2 = alpha2 @ beta2.T                  rank(A2) = r2
Pi = kron(A2, A1)                      rank(Pi) = r1*r2
Gamma_i = kron(B_i2, B_i1)
beta = kron(beta2, beta1)
```

`difference_lags=k` counts lagged differences, not level lags. The equivalent
level VAR has order `k+1`; the first fitted response is observation `k+2` under
one-based indexing. `fitted_differences` and `residuals` have `T-k-1` entries.
The stationary combinations, under the model's I(1) assumptions, are
`beta1.T @ X[t] @ beta2`. Either one-sided contraction alone need not be stationary.

Both mode ranks are supplied, positive, no larger than their dimensions, with
`r1*r2 < p*q`. One full mode is allowed, including a singleton mode that recovers
the ordinary fixed-rank vector VECM. Zero cointegration rank and the full-rank
stationary boundary are not handled by this fitting interface. The diagnostic
function can inspect those coefficient boundaries explicitly.

`intercept=True` includes an **unrestricted matrix intercept in differences**;
it can induce a deterministic trend along common-trend directions. Setting it
false fixes `D=0`. No demeaning, detrending, restricted cointegration intercept,
seasonal regressor, deterministic time regressor or exogenous regressor is
silently inserted. All tuning and rank choices must precede held-out evaluation.

The short-run matrices are not rank constrained. The level coefficients are
`I+Pi+Gamma_1`, `Gamma_2-Gamma_1`, ..., `-Gamma_k` (or `I+Pi` at `k=0`). These
are generally **sums** of Kronecker products. Consequently this is not obtained
by calling a stationary reduced-rank MAR estimator on levels or differences.

## Independent derivation and manuscript corrections

The implementation follows the residual objectives of Sections 3.1–3.2. Several
displayed v1 formulas cannot be executed literally:

- Section 2.3 says beta contains left singular vectors of `A=alpha beta.T`.
  Algebra requires its **right** singular space. The implementation returns
  orthonormal right singular vectors and `alpha=A@beta`.
- Equation (7) has inconsistent level indexing; its residual must use `X[t-1]`
  as defined in (3)/(5), not the contemporaneous response level.
- The introductory columnwise derivation sometimes uses `Sigma1` for right
  whitening. A `p` by `q` observation must be right-whitened using `Sigma2`.
- The general RRR updates misprint cross-moment inverse subscripts and omit
  nuisance residualization in the coefficient expression. Equation (10) also
  drops the difference symbol on short-run regressors. The code retains the
  lagged differences and fully residualizes the level and response designs.
- The printed covariance updates omit the opposite-mode dimension divisor.
  Differentiating (7) gives `Sigma1=sum(R Sigma2^-1 R.T)/(N*q)` and
  `Sigma2=sum(R.T Sigma1^-1 R)/(N*p)`, where `N=T-k-1` is the usable sample size.

For a row update, stack all whitened columns over usable times. Write the
regression `y=A1*x+Psi*z`, where `x=X[t-1] A2.T`, and `z` stacks the opposite-side
short-run transformed differences plus the column-specific intercept design.
For MLE, right-whiten **all** these blocks, including the intercept design,
with the current opposite covariance. Let `y.z,x.z` denote residualization on z.
Define `Sxx=(x.z)(x.z).T` and `Syx=(y.z)(x.z).T`.

For LS, the corrected block solution is

```
U = r1 leading eigenvectors of Syx @ solve(Sxx, Syx.T)
A1 = U @ U.T @ Syx @ inverse(Sxx)
Psi = (Syz - A1 @ Sxz) @ inverse(Szz)
```

The code does not form those inverses or squared-condition normal equations.
It equilibrates design columns, uses SVD for nuisance residualization and
full-rank regression, and truncates the fitted response's singular space.

For Gaussian MLE, first obtain the unrestricted regression residual scatter
`Omega`. Its positive-definite root supplies an additional response whitening.
Do RRR in those whitened response coordinates, then transform the coefficient
back. Equivalently, with U the leading eigenvectors of
`Omega^-1/2 Syx Sxx^-1 Syx.T Omega^-1/2`,

```
A1 = Omega^1/2 @ U @ U.T @ Omega^-1/2 @ Syx @ inverse(Sxx)
```

Update the nuisance coefficients and row covariance from the new residuals.
Transpose everything for the column update, then repeat full sweeps. This
profiles the updated mode's covariance jointly with its reduced-rank coefficient;
freezing that response covariance would instead define another block algorithm.
The original-unit Gaussian likelihood uses `Sigma2 kron Sigma1` with **column-major**
vectorization, includes Gaussian constants, and is independently checked against
a dense multivariate Gaussian oracle.

## Identification, initialization and convergence

Each paired coefficient has reciprocal scale/sign ambiguity. The code fixes
`||A1||_F=1` and `||B_i1||_F=1`, absorbing scale into the paired right matrix.
It makes the largest-absolute left entry positive by a simultaneous sign flip.
Rotations or repeated singular values still prevent interpreting individual
beta columns as unique; use `cointegration_projectors` for invariant comparisons.
Separable covariance scale is identified by `trace(row_covariance)=p`, with the
scale absorbed into `column_covariance`.

Default `init='ols'` computes an unrestricted dense VECM regression **only as an
initializer**, followed by nearest-Kronecker approximation and mode-rank
truncation. The final estimate always performs the actual structured alternating
updates. `init='random'` starts from random rank-projector directions and small
identity short-run matrices without a vector fit. `initial` can supply a mapping
containing `A1`, `A2`, `short_run_left`, `short_run_right`, and optional physical-unit
`intercept`; it overrides the initializer. Arrays and a supplied Generator are
copied. There is no global RNG mutation.

The default three starts use the base initializer and independent local random
perturbations. Their size `0.3` is explicit implementation tuning, not a
paper-prescribed optimal default. `max_iter=200` and `tol=1e-8` govern objective
stabilization: LS uses relative loss increments without an additive unit
constant; MLE uses absolute negative-log-likelihood increments per scalar entry.
The latter is invariant to additive log-unit constants. Descent allowances use
the same respective scales. This prevents a large level offset (absorbed by the
unrestricted intercept) from changing convergence merely through numerical
data scaling. Every start retains its objective history, completed
iterations, failure text and minimum equilibrated design singular ratio. A
failed start with no complete sweep is excluded; a later failure retains its
last complete sweep. The smallest completed finite objective wins, even when
that start has not converged. Always inspect `.converged` and `.runs`.
No claim of global optimization or population identification follows from a
small objective increment. Adding starts can improve a local fit.

`rcond=1e-12` defines a relative numerical rank threshold. Singular block
designs, a lower-than-requested coefficient rank, and singular MLE scatter
matrices are rejected. No pseudoinverse estimator, ridge or covariance floor
is silently substituted. The unrestricted dense **initializer** alone permits
an underdetermined least-squares solution; subsequent block fits enforce their
identifiability checks. A conditional Gaussian MLE need not exist on every sample.

## Scale handling and allocation limits

All fitting operates on `X/max(abs(X))`. Coefficients and subspaces are therefore
unchanged by global nonzero rescaling, up to floating-point roundoff. Intercepts,
residuals, forecasts and scores return in original units. `objective_history`
records scaled mean Frobenius loss for LS or scaled mean Gaussian negative log
likelihood for MLE. `log_likelihood` restores the Gaussian scaling Jacobian and
is `None` for LS. Physical MLE covariances must be representable; overflow or
singularity after restoring units raises an informative error.

Forecasts and cointegrating-score contractions are scaled before multiplication
and reject unrepresentable final results. Forecasting keeps only the required
lag window in each calculation, recursively predicts **levels**, and never
refits or inspects future observations. The stored training history is a copy.

Dense initializers and diagnostics are not scalable matrix-sized operations.
`max_dense_dimension=256` guards `p*q*(k+1)` for the initializer/companion, and
also guards materialized dense result operators and cointegrating vectors.
Explicitly increase it only after considering memory and cubic-time costs.
Use a random or supplied start to fit without a dense initializer; matrix
forecasting and mode-wise projectors do not need dense vector operators.
`i1_diagnostics(max_dense_dimension=...)` allows a separately explicit override.

## I(1) diagnostics and statistical scope

`cmar_i1_diagnostics` checks the complete vectorized level companion. Its
eigenvalues are reciprocals of the characteristic-polynomial roots. It expects
exactly `p*q-r1*r2` roots near `+1` and all remaining roots strictly inside the
unit circle. It also tests the numerical nonsingularity of

```
alpha_perp.T @ (I - sum Gamma_i) @ beta_perp
```

using full orthogonal complements of the **dense** long-run operator's left and
right spaces, not products of mode-wise complements. This distinguishes
semisimple I(1) roots from, for example, an I(2) Jordan block. When numerically
nonsingular, `long_run_impact` returns the usual common-trend impact matrix.
Root tolerance defaults to `1e-6`; near-boundary or nonnormal systems require
care. No mode-wise spectral-radius shortcut suffices for the summed level model.

`compatible=True` means only that these **fitted coefficient** checks pass at
the supplied tolerances. The rank restriction itself imposes unit roots: passing
the checks is not evidence that the observed process is actually integrated,
that ranks are correct, or that a population model satisfies the assumptions.
Fitting does not constrain the remaining roots to stability.

The paper's Assumption 2/Theorem 1 require stable nonunit roots and the full I(1)
rank condition, not merely positive low ranks. Its asymptotic theory is
fixed-dimensional and depends on innovation assumptions. Least squares does
not require separable innovation covariance, while the stated Gaussian MLE
does. Normality is needed for the stated LS cointegrating-space limits; do not
promote moment-only statements for other coefficients to those limits. There
are different convergence rates along stochastic and deterministic trend
directions (`T` and `T**1.5`), so stationary-MAR standard errors are inappropriate.
The preprint states finite second moments in some theorems while a proof invokes
fourth moments; no inferential feature here relies on resolving that discrepancy.

The paper uses known ranks in its simulations and supplies no new matrix rank
test. This implementation therefore supplies **no automatic rank selection,
Johansen critical values, p-values, standard errors, bootstrap intervals,
cointegration-restriction tests or trading guidance**. Vectorized Johansen
analysis could be an honestly labeled comparison, not a substitute for the
structured estimator or matrix-specific rank inference. Zero-rank/full-rank
boundary fitting, restricted deterministic terms, higher-order integration and
weak-identification inference remain separate work.

## Validation

Tests use independently assembled dense moment RRR equations with symmetric
whitening, dense multivariate Gaussian likelihoods, and full level-VAR forecast
recurrences. They cover LS/MLE, intercept choices, zero/one/two difference lags,
the vector limit, planted bilinear spaces, I(1)/I(2)/explosive/non-+1 unit roots,
scale equivariance at `+/-1e150` and `+/-1e-150`, input ownership, multistart
failures, allocation guards and extreme-score cancellation. These equation and
integration checks are not an independent claim of asymptotic calibration.
The installed-package example generates data inline: `python -m examples.cointegrated_mar`.
