# Additive two-way dynamic factors

`fit_two_way_dynamic` implements the two-step quasi-likelihood estimator in
[Yuan, Gao, He, Huang and Guo (2023), JRSS B 85, 1517–1537](https://doi.org/10.1093/jrsssb/qkad077),
main Algorithms 1 and 2, with initialization from supplementary Algorithm A1.
The [primary article](https://academic.oup.com/jrsssb/article/85/5/1517/7249210)
links its supplementary material; Sections A.2–A.6 provide the numerical
derivations used here. This model has additive row and column effects:

```text
Y[t] = F[t] L.T + Lambda G[t].T + E[t]
F[t] = sum(F[t-lag] diag(phi[lag])) + epsilon[t]
G[t] = sum(G[t-lag] diag(gamma[lag])) + eta[t].
```

It is not a bilinear `R H[t] C.T` model. The paper's covariance fit is static
in its first stage, but it optimizes a specific additive covariance model;
generic PCA followed by a VAR does not implement that stage. Stage two fits
shared scalar autoregressions to conditional factor scores.

## Shapes and identification

| Public quantity | Shape | Paper notation |
| --- | --- | --- |
| observations | `(T,n,m)` | `Y[t]` has n rows, m columns |
| `row_loadings` | `(n,c)` | Lambda |
| `column_loadings` | `(m,r)` | L |
| `F` | `(T,n,r)` | row-specific column-factor scores |
| `G` | `(T,m,c)` | column-specific row-factor scores |
| `ranks` | `(c,r)` | public spatial order; paper uses `(r,c)` |
| `orders` | `(q,p)` | public spatial order; paper uses `(p,q)` |
| `row_ar` | `(q,c)` | gamma |
| `column_ar` | `(p,r)` | phi |

The loading normalizations are `L.T L=m I_r` and
`Lambda.T Lambda=n I_c`, unlike the unit-norm convention in the package's
bilinear factor models. Marginal factor covariances Psi_F and Psi_G are
diagonal with decreasing entries. Each loading column has a nonnegative
first entry (paper I3); an exactly zero first entry uses the largest-magnitude
entry as a deterministic sign fallback. If factor variances coincide,
individual loading columns and their diagonal dynamics need not be
identifiable; deterministic eigenvectors do not resolve statistical
nonidentification. Compare subspaces or the additive signal in that case.

Both ranks must be positive and strictly below their spatial dimensions.
The implementation does not test for no factors. The paper assumes
uncorrelated factor processes across units, diagonal marginal factor and
innovation covariances, stationary scalar AR dynamics, and idiosyncratic
errors uncorrelated across matrix entries. Errors may have weak serial
dependence and heterogeneous variances. The fitted working covariance
uses a scalar average error variance. `is_stable` diagnoses fitted AR roots;
the routine does not impose stationarity or fit residual serial dynamics.

## Estimation details

Supplement Algorithm A1 obtains initial L/F from column PCA, then alternates
PCA of `Y-F L.T` for Lambda/G and PCA of `Y-Lambda G.T` for L/F. Crucially,
these scores are computed from the current residuals. This initialization
is followed by the quasi-likelihood fit, not returned as the final estimator.
Variance initialization follows A.30. The default uses a single initial
noise variance; users can explore different loading initializations. The
paper recommends multiple noise initializations, which are not automated.

For the paper's row-major vectorization, the working covariance is

```text
K = I_n kron (L Psi_F L.T)
    + (Lambda Psi_G Lambda.T) kron I_m + noise I_nm
ell = -logdet(K) - mean_t(vec_C(Y[t]).T inv(K) vec_C(Y[t])).
```

This is equation (6), without the irrelevant Gaussian multiplier and
constants; it is not the likelihood of the full temporal covariance.
The implementation never constructs the nm-by-nm covariance. Diagonalizing
the small factor covariances and completing the normalized loading bases
gives bases V/U and eigenvalues a/b of the column/row signal covariance.
The matrix inverse action is then

```text
W[t] = U @ ((U.T @ Y[t] @ V) / (b[:,None]+a[None,:]+noise)) @ V.T.
```

The conditional scores are `F=W L Psi_F` and `G=W.T Lambda Psi_G`, equivalent
to the Gaussian linear predictors in A.13–A.18. They are contemporaneous
scores, not Kalman-smoothed estimates. The additive overlap is apportioned
using covariance information, so the two effects are not interchangeable
with projections `Y L/m` and `Y.T Lambda/n`.

Algorithm 1 alternates three blocks. L and Lambda each maximize a sum of
heterogeneous quadratic forms under their orthogonality constraint
(equations 10–12). Rank one uses a leading eigenvector; higher ranks use
the shifted positive-definite polar/SVD iteration, with the paper's .01
shift. In computing the quadratic forms, a direct positive spectral formula
replaces the subtraction of nearly equal inverse terms in A.9. If a
row-signal covariance eigenvalue is b and the column-factor variance is f,
the weight is `f/((noise+b)*(noise+b+m*f))`. This is algebraically the same
weight as equation (10), with improved numerical behavior.

The third block performs full conditional-moment EM on Psi_F, Psi_G and
noise (equation 13, supplementary A.18–A.21). Off-diagonal updated factor
covariances are retained through the inner EM loop. The noise update
includes conditional dependence between F and G, implemented using
`Var(signal|Y)=noise I-noise**2 inv(K)`. Adding separate F/G conditional
variances without their cross-covariance overestimates noise. At the end
of that block, each factor covariance is diagonalized and ordered and its
loading is rotated identically. The square-root transformation in paper
Algorithm 1 simplifies to this rotation; all fitted observation covariances
are preserved.

Step two solves equation (7) with QR/SVD-based least squares on each
factor column, pooling across units. The innovation estimates use the
paper's equation (8) denominators **n*T and m*T**, not the effective count
after dropping lags. Marginal factor covariance and AR innovation variance
are distinct outputs. Since they are estimated in separate stages, they
need not satisfy an exact finite-sample stationary covariance equation.

## Rank selection and numerical conventions

Algorithm 2 starts from user-supplied rank bounds, defaulting to half each
spatial dimension. At each iteration it uses the leading c row eigenvectors
of the original second moment to remove a row projection, chooses r from
the residual column-moment eigenvalue ratios, then reverses the modes using
that r. The residual moments divide by T only. Each denominator receives
`ratio_ridge*max(n**(-.5),m**(-.5),T**(-.5))` in the original data units.
The square roots in this expression can be lost in text-only rendering
of the paper; the implementation follows Algorithm 2's mathematical markup.
Rank iterations stop on stabilization, a repeated cycle, or the iteration
limit. `rank_history`, `rank_converged`, `rank_cycle` and the last residual
spectra make those outcomes explicit. Rank convergence is separate from
quasi-likelihood convergence.

For arithmetic, all data are divided by `data_scale=max(abs(X))`. This
rescaling does not alter the target covariance likelihood; rank ridge terms
are adjusted to retain their original units. Rank spectra and objective
histories use the scaled units. Physical quasi-likelihood is obtained by
subtracting `2*n*m*log(data_scale)` from the stored objective. Scores,
noise/marginal/innovation variances, reconstructed signals and forecasts
are returned in physical units. Unrepresentable physical variances raise
an error; this estimator cannot represent every covariance generated by
an arbitrarily large finite input.

The optional variance floor bounds noise and covariance eigenvalues by
`variance_floor*mean(centered_scaled_X**2)` in working units. It is a
restricted parameter space, not part of the unrestricted paper estimator;
`variance_regularized` reports whether an update activates it. Setting zero
removes this bound apart from the smallest normal machine number.
`center=True` is a separate extension: a training mean is removed and
restored in predictions. The default retains the zero-mean specification.

Outer, loading, covariance EM and rank loops are all bounded. Objective
ascent is checked to roundoff; failures raise instead of claiming a
successful fit. `converged` requires a small outer improvement and convergence
of both loading blocks and the final covariance EM block. It certifies no
global optimum or parameter-error bound. Per-iteration inner counts/flags
use `(row_loading,column_loading,covariance_EM)` order; covariance EM
histories include their initial objective.

## Forecasting and verification

`forecast` recurses the pooled diagonal AR models from the latest conditional
scores and reconstructs `mean + F L.T + Lambda G.T`. This is a plug-in
common-component prediction with a zero residual prediction. When residuals
are serially dependent it need not be the full observation conditional mean.
The paper's forecasting experiments (equations 22, 24–25) use held-out
observations; the example likewise keeps loadings and dynamics trained only
on the training window and scores each predictor contemporaneously.

The numerical tests independently construct a full joint latent Gaussian
model for nonsquare observations with r=3 and c=2, verify its observation
covariance, inverse action, factor means, full covariance EM and noise
cross-effects, and compare loading quadratic forms to the paper expansion.
Additional checks cover polar tangent stationarity, rotation invariance,
the exact projected-residual rank iteration, pooled AR(2), simulated
subspace/dynamic/noise recovery, centering/scaling and recursive forecasts.
Automatic lag selection, full VAR factor transitions, Kalman filtering,
heterogeneity bias-corrected inference, and uncertainty intervals remain
outside this implementation.
