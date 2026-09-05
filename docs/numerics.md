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
