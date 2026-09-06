# Simultaneous matrix decorrelation

`fit_matrix_decorrelation` implements the bilinear transform in Han, Chen,
Zhang and Yao, [Simultaneous Decorrelation of Matrix Time Series](https://arxiv.org/abs/2103.09411).
Equation numbers below refer to arXiv v2, Sections 2.2–2.3.

The method preserves every matrix coordinate. Its target is a partition into
rectangular component series with no cross-block linear dependence at any lag.
It estimates two invertible transformations rather than factor loadings with a
reduced rank. Sampling error and misspecification can prevent exact separation.

## Implemented equations

For centered observations, define the row and column marginal covariances

\[
 S_r=(Tq)^{-1}\sum_tX_tX_t^\top,\qquad
 S_c=(Tp)^{-1}\sum_tX_t^\top X_t.
\]

All cross-coordinate lagged covariance matrices in (6), including negative
and zero lags, contribute to the two moment matrices in (7). The orthogonal
eigenvectors give rotations `row_rotation` and `column_rotation`. In (8),

\[
 U_t=B_*^\top S_r^{-1/2}(X_t-\bar X)S_c^{-1/2}A_*.
\]

The implementation groups the separate, partially transformed series in (9)
using the maximum absolute cross-correlations in (10). It then permutes both
rotations so connected groups occupy contiguous coordinates. The inverse is

\[
 X_t=S_r^{1/2} B_* U_t A_*^\top S_c^{1/2}+\bar X.
\]

## Tuning and limitations

The graph connects a pair when its correlation exceeds
`correlation_threshold`, a scalar or a `(row, column)` pair. The default 0.2
is a convenience, with no significance interpretation. Inspect the returned
correlation matrices and compare several thresholds. `lags` controls the
moment construction; `correlation_lags` controls grouping. Both include the
corresponding negative lags and zero, and must be smaller than the sample size.

Alternatively, `grouping="ratio"` implements the adjacent-ratio selector (11).
The authors' [journal supplement](https://doi.org/10.6084/m9.figshare.21641763.v2),
file [real data analysis.R](https://ndownloader.figshare.com/files/38371890),
resolves the printed formula's unspecified final ordered statistic: it searches
only the first half of all unordered coordinate pairs, rounding down, and uses
zero smoothing. Our independent implementation preserves that search bound and
first-maximum rule. `ratio_delta>0` supplies the ridge in (11).
`ratio_max_edges` allows explicit alternative bounds below the total pair count.
The source file's MD5 is `a0abf2a1ec176462e3e5775779b12ff1`.

The ratio selector always selects at least one edge. It cannot return all
singletons in a nontrivial mode. A mode of dimension two has only one pair, so
no adjacent ratio exists: use threshold grouping instead. A scalar mode needs
no grouping. At zero smoothing, the implementation treats `0/0` as its
continuous positive-ridge limit, one, and positive/zero as infinity. All-zero
scores therefore select one edge too; tied edges follow lexicographic order.
These boundary conventions are explicit extensions because the reference
script only demonstrates larger modes with nonzero correlations.

Optional VAR prewhitening, recursive irregular partitions, and alternative
moment spectral functions remain unimplemented. The reference script also uses
only nonnegative moment lags and trimmed variance estimates. Both grouping
modes here retain the printed signed-lag and full-variance equations (7) and
(10), so this implementation does not reproduce every reference-script choice.

The estimator assumes stationary second moments and invertible marginals.
Repeated moment eigenvalues across different latent groups can prevent
identification. Estimated finite-lag separation is not an independence test.
Correlations use whole-series variances and `T-abs(lag)` covariance divisors;
they can exceed one in finite samples. Identically zero scalar coordinates
contribute zero rather than undefined ratios.

## Numerical and output conventions

SVD of each marginal data matricization supplies its symmetric square root
and inverse square root. The default relative singular-value cutoff is
`sqrt(machine epsilon)`; a deficient marginal raises an error without silently
introducing a ridge. Scale-normalized storage avoids squaring large input
units. Under `X -> c*X`, `U -> U/c`; this is a consequence of the two marginal
normalizations, not a loss of invertibility.

Moment eigenvalues are reordered to match each grouped rotation. Mixing
matrices and transforms use original units. `blocks()` extracts training
components, `blocks(new_data)` applies the fitted transformation, and
`inverse_blocks(predictions)` reconstructs forecasts after separate modeling.
No forecasting model is imposed by the decorrelation estimator.

The tests compare every moment, correlation, and transformation against
direct equation implementations. Two planted simulations check recovery of
independent scalar AR components and dependent rectangular VAR blocks.
