# Threshold matrix factors

`fit_threshold_factors` implements the two-regime estimator in Liu and Chen,
“Identification and estimation of threshold matrix-variate factor models,”
Scandinavian Journal of Statistics 49 (2022), 1383–1417,
[DOI](https://doi.org/10.1111/sjos.12576). Equations below refer to the
[author manuscript deposited with NSF](https://par.nsf.gov/servlets/purl/10351724).

The model permits different row and column loading spaces and different numbers
of factors in each regime:

\[
X_t=R_i F_{t,i} C_i^\top+E_t,\qquad
i=1\text{ if }z_t<r_0,\quad i=2\text{ otherwise}.
\]

The Python result uses zero-based regime labels. The observed variable `z[t]`
must correspond to `X[t]`. For a threshold variable derived from a lagged matrix,
construct it from the past and truncate both arrays to the same time indices.
The implementation never compresses time before calculating temporal lags.

For each lag `h`, source column `u`, destination column `v`, and regime `i`,
equation (9) is

\[
\widehat\Omega_{x,i,uv}(h,r)=\frac1T\sum_{t=1}^{T-h}
  x_{t,u}x_{t+h,v}^\top I_{t,i}(r).
\]

Equation (10) sums `Omega @ Omega.T` over all column pairs and lags. Its
leading eigenvectors estimate the row loading space. Repeating the procedure
on transposed matrices estimates the column loading space. Cross moments are
uncentered, and their denominator is **T**, not the number of selected origins
or T minus the lag. Crucially, the published 2022 method partitions only the
**source observation**. It is different from the authors' 2019 arXiv preprint,
which additionally partitions the future observation and sums four partition
cross-moment products. It is also different from PCA on separate regime samples.

With an unknown threshold, lower and upper trimming cutoffs define two pure
extreme subsets, assumed to bracket the true threshold. Their loading-space
complements are estimated once. Candidate thresholds are scored with the sum
of four **spectral** norms in equation (11):

\[
\widehat G(r)=\sum_{s=1}^2\sum_{i=1}^2
 \|\widehat B_{s,i}(\eta_i)^\top\widehat M_{s,i}(r)
                   \widehat B_{s,i}(\eta_i)\|_2.
\]

By default, every unique observed threshold-variable value strictly inside the
trimming bounds is evaluated. Supplying `candidates` explicitly uses a restricted
grid. The complete evaluated profile is returned; exact score ties select the
smallest threshold. An estimated threshold then partitions all training origins
for a final loading-space fit. With `threshold` supplied, the extreme subsets and
search are bypassed.

Missing row/column ranks use equation (12)'s eigenvalue-ratio heuristic on the
extreme subsets for a searched threshold, or full regime subsets for a known
threshold. The upper rank-search bound is half the smaller of T and the relevant
spatial dimension (at least one). A relative machine-precision floor prevents
division by zero at exact rank deficiency. Ranks selected from the extreme
subsets remain fixed during threshold search and the final fit. This heuristic
does not establish absence of factors or prove rank consistency.

The implementation normalizes observations by a single global scale before
forming cross moments. Subspaces and the minimizing threshold are invariant
to this scaling. Returned spectra and profile scores refer to normalized data;
their original-unit counterparts require multiplication by `data_scale**4`,
which can overflow and is therefore left to callers. Returned factors and
reconstructions are in original units. The moment calculation holds one source
column's cross moments at a time, avoiding a full `(p*q, p*q)` covariance array.
Search is exhaustive, so large samples and spatial dimensions can be expensive.

`loadings[i]` contains orthonormal row and column bases. `factors[i]` contains
cores only for observations assigned to that regime, in temporal order;
`indices[i]` maps them to the original time axis. This supports unequal core
shapes without padding. `transform(X_new, z_new)` and
`inverse_transform(cores, z_new)` use training loadings and the fitted threshold.
`reconstruct` is projection/denoising, not a forecast, and does not refit anything.

The method assumes informative factor cross moments at the selected lags and
temporally uncorrelated idiosyncratic noise. Arbitrary future-dependent choices
of z can invalidate those conditions. The caller must establish z's provenance
and its availability for any forecasting exercise. Weak dynamics, insufficient
extreme observations, a true threshold outside the trim interval, or nested
regime spaces can impair identification. The profile is not a confidence interval
or a test for a threshold's existence. Numerically zero moments and ranks above
their numerical rank are rejected instead of returning arbitrary eigenspaces.

This implementation covers **one threshold, two regimes, and unequal regime
ranks**. The paper's multi-threshold and threshold-variable-selection extensions
are not implemented. `examples/threshold_factors.py` demonstrates training and
held-out projection. `tests/test_threshold.py` independently reconstructs every
cross moment with explicit loops and checks the spectral criterion, temporal
alignment, unequal-rank recovery, scale invariance, and degenerate inputs.
