# Entrywise iterative Huber regression

## Primary version boundary

The implemented target is He, Kong, Liu and Zhao's
[2023 preprint, arXiv:2306.03317v1](https://arxiv.org/html/2306.03317v1),
not an asserted reproduction of its accepted successor. On 2026-09-05,
[Yong He's publication page](https://heyongstat.github.io/) lists the same
arXiv identifier under **Winsorized Mean Matrix Factor Model**, Journal of
Business and Economic Statistics, in press (2026+). ArXiv still lists only v1.
The author's [Paper repository](https://github.com/heyongstat/Paper) has no
IHR/renamed manuscript; searches did not locate accepted full text or a publisher
DOI. Algorithm and assumption equivalence therefore remains unresolved.

## Objective and implementation

`fit_ihr_factor` minimizes the average **entrywise** Huber loss of
`X[t,i,j] - row[i] @ core[t] @ column[j]`, with quadratic branch `e²/2` and
linear branch `tau*abs(e)-tau²/2`. This differs from `fit_huber_factor`, which
assigns a single matrix weight and uses ordinary projected factors.

One sweep updates all row regressions, then all column regressions using the
new rows, then all factor regressions using both new loading matrices.
Regressions have no additional intercept or ridge. Each fixed-threshold convex
subproblem uses IRLS with weights `min(1,tau/abs(e))` and SVD-based weighted least
squares, without explicitly forming normal-equation inverses. Column scaling
protects conditioning; rank-deficient designs are rejected. The preceding
coefficients warm-start each block. Singular updated loading matrices retain
the last accepted sweep with a failure reason.

Loadings are normalized by SVD and rotated using two core-moment eigensystems;
these transformations preserve the fitted signal. Public loadings are
orthonormal, so public factors equal `sqrt(p1*p2)` times factors under the
paper's `R.T@R=p1*I`, `C.T@C=p2*I` convention. `eigenvalues` uses **paper core
units after dividing X by data_scale**. Axis signs are deterministic; ties still
leave axes unidentified. Compare reconstructed signals or loading projectors.

The fixed default threshold is `1.345*1.483*median(abs(pilot residuals))`, using
one projected-PCA sweep and then freezing the value, following Section 4.4's
calibration. The paper's Section 2 also describes adaptive, regression-specific
MAD updates; that different variant is not implemented. A zero pilot MAD requires
an explicit positive threshold. An explicit threshold is in original entry units
and must scale with the data. `center=True` removes an ordinary nonrobust mean.

Every accepted outer objective is evaluated and recorded in normalized data
units; multiply by data_scale twice to restore loss units when representable.
Rejected increases are not concealed by clipping history values. Convergence
requires relative loss and signal stability plus successful inner scores.
`numerical_tolerance` permits stopping at the previous iterate when a proposed
increase is only roundoff, signal change is small, and inner solves succeeded.
`block_history` stores every attempted sweep's three groups of inner diagnostics,
including any final rejected sweep. Strict inner tolerances can legitimately
produce `descent_stalled` rather than claiming convergence beyond numerical
resolution. Neither convergence flag establishes a global optimum.
Inner descent checks evaluate the piecewise Huber loss **difference** along the
coefficient step directly, avoiding cancellation from subtracting total losses.
Inner histories retain the independently evaluated totals, which can differ
upward by roundoff even for a verified decreasing step; they are never clipped.

## Transform and rank APIs

`result.transform(X)` estimates new factors by entrywise Huber regression;
it does not use ordinary projection. Each observation is scaled and solved
separately using the training mean, loadings and threshold, so another observation
cannot influence even its numerical stopping criterion. Request
`return_diagnostics=True` to receive factors and inner diagnostics; otherwise
unfinished solves produce a warning. `inverse_transform` reconstructs those
factors. These are reconstructions using observed matrices, **not forecasts**.
Nonfinite original-unit outputs are rejected, not clipped.

`select_ihr_ranks(X,max_ranks,method=...)` first fits oversized dimensions.
The ratio method searches below each fitted dimension and returns log-ratio
diagnostics; its absolute ridge is in original variance units, so scaling data
by `a` requires scaling that ridge by `a²` for invariance. Log-domain evaluation
avoids squaring extreme data scales. The threshold method uses the paper's
leading-eigenvalue-relative empirical cutoffs and is scale invariant. It may
return zero ranks without silently forcing a factor; zero cannot be passed to
the positive-rank fitting API. Inspect the overfit pilot's convergence. These
criteria do not implement a calibrated no-factor test or post-selection inference.

## Tests and inference limits

Tests independently solve scalar clipped-score equations and dense convex
regressions, explicitly construct Kronecker designs, check all three blocks of
one sweep, verify signal-preserving normalization and paper-unit spectra, and
exercise the quadratic limit, cell contamination, threshold calibration,
out-of-sample causality and extreme scale changes.

No standard errors, confidence intervals or tests are implemented. The preprint's
theory assumes conditional error independence across time/rows/columns,
symmetric regular conditional densities, bounded conditional second moments,
compact strong-factor parameters and separated factor spectra. Loading CLTs
add nonsingular information and different row/column dimension-growth regimes;
these are not generic guarantees for dependent heavy-tailed errors or every
finite optimization path. Adaptive thresholds, weak factors, missing entries,
robust centering, inference and accepted-version reconciliation remain missing.
