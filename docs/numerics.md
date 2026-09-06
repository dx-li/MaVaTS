# Numerical conventions and compatibility

## Data and identification

All observations are finite real arrays with time on axis zero. Matrices have
shape `(T, rows, columns)`; tensors have `(T, d1, ..., dK)`. NaN, infinity,
complex observations, empty modes, and invalid ranks/lags are rejected. Missing
data require dedicated methods; replacing missing values with zero changes the
statistical problem. Inputs are not mutated.

Matrix vectorization stacks columns: `vec(A @ X @ B.T) = kron(B, A) @ vec(X)`.
Autoregression fits use coefficients multiplying previous observations on the
left and right. `MARResult.left` and `.right` always retain a leading lag axis.
`.A` and `.B` are conveniences for first-order models. The `.coefficients`
property materializes dense Kronecker transition operators, so avoid it for
large spatial dimensions. MAR(1) stability can be checked without that allocation.

Modern Tucker loading matrices have orthonormal columns. Factor scores compensate
for this normalization. `result.signal` equals
`result.inverse_transform(result.factors)`. Centered models restore the fitted
training mean during reconstruction. New observations use that same training
mean. Signs and rotations of loadings are not uniquely identified; compare
projectors, principal angles, or signals using `mavats.metrics`.

## Solvers and stopping

Regression blocks use SVD least squares, including rank-deficient designs.
Ridge regression uses augmented designs. This avoids explicit inverse matrices
and avoids squaring the design condition number through normal equations.
Reduced-rank MAR truncates the fitted response in the design-induced metric;
truncating the unconstrained coefficient's ordinary SVD is a different estimator.

Factor moment calculations scale observations before accumulating quadratic or
fourth-order products. `FactorResult.eigenvalues` therefore describe the scaled
moments, as documented in the result class. TOPUP chooses between exact blocked
outer contractions and a dual time-Gram identity according to dimensions.
Matrix Kendall calculations normalize differences before forming their kernels.

Iterative estimators expose convergence and iteration counts. MAR has an objective
history; a local optimum is not certified as globally optimal. Recursive
simultaneous projected PCA and iterative lag estimators have subspace stopping
criteria and must not be interpreted as universally decreasing a reconstruction
loss. Separable MLE exposes whether covariance flooring was active. Such a fit
is a stabilized likelihood estimate; it is not an unmodified singular Gaussian
MLE. Exact zero residual variance makes the nonsingular likelihood undefined.

`MARResult.data_scale` records the common scale removed before centering and
fitting. ALS objective histories use normalized-data units (including the
appropriately transformed ridge penalty); MLE histories are normalized-data
negative log-likelihoods. `.log_likelihood` restores physical units. MLE
covariances split the data scale between row and column factors to avoid
overflow from explicitly squaring the scale. Their Kronecker product is the
physical innovation covariance when representable. The row covariance's
Frobenius norm is `data_scale`; legacy wrappers retain their old unit-norm
convention whenever the companion covariance can be represented.

Huber histories likewise use normalized-data squared units. Its threshold is
reported in original Frobenius-norm units. CP loadings are unit-length but
generally nonorthogonal, so its scores use joint least squares rather than
the separate orthogonal projections used by Tucker results.

Rank estimation by eigenvalue ratios is a heuristic unless the paper's assumptions
and tuning conditions hold. The default cannot detect a zero-factor model.
Use fixed ranks when comparing methods at equal information, and separate oracle
rank results from data-selected ranks.

Threshold factors retain the original time indices when masking lag origins.
Their moment eigenvalues and threshold profile scores use normalized fourth-order
units; reconstructed signals use original units. Decorrelation is invertible,
not rank reduction: its two marginal whitenings imply latent coordinates scale
as `1/c` when the observations scale as `c`. See the dedicated
[threshold](threshold-notes.md) and [decorrelation](decorrelation-notes.md) notes.

Sparse EMVS has priors in specified data units. Arbitrarily rescaling its
coefficient/covariance factors preserves the likelihood but changes those priors.
Its additional scale updates maximize the conditional posterior, not an arbitrary
unit-norm convention. Inclusion scores are plug-in probabilities, not marginal
posterior draws; support masks do not truncate forecast coefficients. See
[sparse notes](sparse-notes.md).

MAR inference uses a fixed nonzero coefficient sign anchor for local
identification and normalized-data moments for unit-invariant standard errors.
Conditioning guards cannot certify population identification or finite-sample
coverage. Supported assumptions and allocation limits are in
[inference notes](inference-notes.md).

Additive two-way dynamic factors have a different identification from Tucker
factors: the row/column loadings have Gram matrices equal to their spatial
dimension times identity, and separate F/G effects overlap. Conditional
Gaussian scores apportion that overlap. The covariance EM noise update includes
cross-effects; omitting them changes the estimator. Variance flooring and mean
centering are explicit extensions, with activation diagnostics. See
[dynamic notes](dynamic-notes.md).

Matrix GARCH filters preserve unnormalized row/column shape recursions while
normalizing the output covariance traces. Positivity comes from triangular
intercepts, positive scalar trace and Cholesky checks, not an undocumented
floor. Numerical scaling transforms observation-shock parameters consistently;
constraint diagnostics refer to original units. Gaussian conditional likelihood
gradients differentiate the complete state recursion. Optimization remains
local, and an earlier feasible iterate can be retained after failed termination
without marking that run converged. See [volatility notes](volatility-notes.md).

Sequential monitoring recomputes the opposite-mode projection on every trailing
window. Drift powers, randomized partial sums and crossings use signed-log
arithmetic, so display saturation does not determine the alarm. State snapshots
preserve the local random stream and stop policy. One-based alarm indices are
explicit and are not delay-corrected change-point estimates. Finite-horizon
Gaussian reference [calibration](calibration-notes.md) is separate from the
paper's asymptotic matrix-data validity conditions.

Cointegrated matrix autoregression fits differences but forecasts levels. Full
cointegration complements and summed level operators are required for I(1)
diagnostics; mode-wise roots do not suffice. Dense accessors and initialization
have explicit allocation guards. Equivalent reciprocal coefficient gauges are
normalized with exponent-balanced scale transfer, not squared norms. Near-limit
physical covariances are scaled before symmetrization. LS stopping is relative
to residual loss, while MLE uses per-entry likelihood increments, so an
unrestricted level shift does not turn the stopping rule into an absolute
small-loss cutoff. See [cointegration notes](cointegration-notes.md).

Multi-term tensor autoregression contracts each lag/term separately without
materializing the vector transition during fitting. Matrix multi-term
canonicalization uses a small QR/SVD representation; higher-order CP projection
is explicitly local. Initial projection, final optimization, identification and
stability are separate diagnostics. Coefficient and covariance scale transfer
must consider every mode together to preserve compensating extreme gauges.
See [tensor autoregression notes](tensor-autoregression-notes.md).

Entrywise IHR solves convex loading and factor regressions, including held-out
cores. Its inner descent check evaluates the piecewise loss difference directly
instead of subtracting nearly equal total losses. The requested score tolerance
is retained. Per-observation transform scaling prevents a later extreme matrix
from changing an earlier observation's stopping criterion. Paper-normalized
rank spectra, public orthonormal loadings and physical threshold units are
distinguished in [IHR notes](ihr-notes.md).

## ARMA and overlapping constraints

Matrix ARMA differentiates the full conditional innovation recursion with an
analytic reverse pass. Centering and scaling are numerical reparameterizations
when the intercept is fitted; returned parameters and likelihoods use original
units. Default local optimization constrains both full companion radii, not
individual mode radii. Optimizer termination, feasibility, LS initialization of
MLE, and structural identification are separate diagnostics. Dense allocation
limits apply to full-polynomial checks and forecast covariance. Conditional
prefix innovations are zero; covariance forecasts omit parameter uncertainty.
See [MARMA notes](marma-notes.md) for sign and conditioning conventions.

Partial constrained factors use a single observation normalization before
computing and summing separate block moments. Multi-term constrained factors
annihilate competing spaces, check surviving loading ranks, and solve scores
with a full-column-rank joint SVD. Independent projections are not the joint
score solution for overlapping components. Allocation and identification
guards are explicit; see [constraint notes](partial-constraints-notes.md).

## Migration from 0.1

Legacy imports from `mavats.MAR`, `mavats.alphaPCA` and `mavats.factormodel` remain.
Legacy alpha-PCA retains `sqrt(rows)` and `sqrt(columns)` loading normalization
and tuple return values. Modern entry points return named result objects.

The rebuild intentionally changes numerically incorrect behavior in 0.1:

- MAR projection now regresses future observations on past observations.
- Conditional coefficient updates solve on the correct side.
- Alpha-PCA reconstruction no longer divides by the spatial size twice.
- Invalid/degenerate inputs produce informative errors rather than accidental
  divisions by zero or low-level shape errors.

Legacy covariance helpers have algebraic HAC checks and validation, but that is
not a finite-sample coverage study. A comprehensive public inference API remains
open. Library code does not seed or consume NumPy's global RNG.

## Scope and scaling limits

MAR projection and unrestricted VAR store dense product-dimension coefficients.
Multi-lag exact stability diagnostics store a dense companion. Exact matrix
Kendall is quadratic in the observation count; `max_pairs` explicitly opts into
sampling with replacement. TOPUP remains more computationally demanding than
TIPUP despite blocking and the dual formulation. No claim of superiority over
specialized compiled software is made without a matching benchmark.
