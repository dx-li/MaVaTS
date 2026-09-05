# Trace-identified Matrix GARCH

`mavats.volatility` implements the full first-order matrix GARCH model of
Yu, Li, Jiang and Zhu, *Matrix GARCH Model: Inference and Application*,
JASA **120(551), 1747–1762 (2025)**, published online November 22, 2024.
[Publisher](https://doi.org/10.1080/01621459.2024.2415719).
Equation and theorem numbers below follow the
[accepted manuscript linked by the author](https://www.researchgate.net/publication/371399004_Matrix_GARCH_model_Inference_and_application).
The older [arXiv v1](https://arxiv.org/abs/2306.05169) does not contain the
accepted version's strict-stationarity theorem and has different later numbering.

## Model and identification

For zero-conditional-mean observations of shape `(T,m,n)`, equations (4)–(9) are

\[
S_{1,t}=A_0A_0^\top+A_1X_{t-1}X_{t-1}^\top A_1^\top+A_2S_{1,t-1}A_2^\top,
\]
\[
S_{2,t}=B_0B_0^\top+B_1X_{t-1}^\top X_{t-1}B_1^\top+B_2S_{2,t-1}B_2^\top,
\]
\[
y_t=w+\alpha\|X_{t-1}\|_F^2+\beta y_{t-1},\qquad
U_t=y_t S_{1,t}/\operatorname{tr}(S_{1,t}),\quad
V_t=S_{2,t}/\operatorname{tr}(S_{2,t}).
\]

The conditional covariance of column-vectorized observations is `V[t] ⊗ U[t]`.
The row covariance `E(X[t] X[t].T | past)` equals U, while the column covariance
equals **y V**, not V. Both covariances have trace y, and V has trace one.
The state matrices S1/S2 must remain unnormalized in their recursions; replacing
them with U/V changes the model.

`MatrixGARCHParameters` stores A0, A1, A2, B0, B1, B2, w, alpha and beta. A0/B0
are lower triangular, with first diagonal fixed to one to remove scale ambiguity.
The paper permits nonnegative diagonal entries subject to positive definite
states. This implementation uses strictly positive intercept diagonals so the
zero-initialized likelihood is defined without any additional covariance ridge.
The dynamic matrices may have arbitrary signed entries. Each of A1, A2, B1 and
B2 can independently be negated without changing any covariance path. Reported
coefficients retain the selected start's signs; covariance paths and likelihoods
are invariant comparison targets. Normalization alone does not establish complete
parameter identification in degenerate cases.

When a mode has dimension one, its normalized shape is identically one, so its
shape dynamics cannot be identified. The parameter class requires those dynamic
matrices to be zero, and fitting omits them. With both modes scalar, the model
is the usual recursion `y[t]=w+alpha*x[t-1]**2+beta*y[t-1]` and
`x[t]=sqrt(y[t])*z[t]`. The manuscript's scalar explanatory sentence drops the
square root; the implementation follows equation (1) and its covariance model.

## Filtering, likelihood and simulation

`filter_matrix_garch` uses the paper's feasible conditional likelihood, equations
(12)–(13), initializing X0, S1, S2 and y to zero. It updates the covariance
**before** evaluating the first observation. Consequently first-row shape is
A0 A0.T, first-column shape is B0 B0.T, and first trace is w. These are conditional
initial values, not an unconditional stationary initialization. Filtering takes
fixed parameters and never estimates or removes a mean. A separate fitted mean
model would introduce additional estimation uncertainty not covered here.

The implementation evaluates Gaussian negative log-likelihood contributions

\[
\ell_t=\tfrac12\{mn\log(2\pi)+n\log|U_t|+m\log|V_t|
  +\|L_{U,t}^{-1}X_tL_{V,t}^{-\top}\|_F^2\},
\]

where the L matrices are lower Cholesky factors. Triangular solves replace
explicit inverses of the full Kronecker covariance. No dense `(mn,mn)` covariance
is allocated unless requested through `covariance(index)` or a forecast's
`covariance()` accessor. Any loss of positive definiteness or nonfinite recursive
state raises an error rather than modifying the model with an undocumented floor.

The result stores row covariances, trace-one column factors, traces, per-time
negative log-likelihoods including the Gaussian constant, and Cholesky-standardized
residuals. The latter have Gaussian identity covariance under the model but are
not the paper's symmetric-square-root residual coordinates. They do not supply
the paper's portmanteau test automatically.

`simulate_matrix_garch` draws Gaussian matrices using the current Cholesky
covariance factors. Finite burn-in from the zero state approximates a stationary
draw only under suitable parameters. The simulation returns the actual covariance
paths and the state immediately before retained observations. Passing this
`initial_state` to filtering reproduces those paths exactly. No non-Gaussian
innovation family, matrix-t likelihood, or inference is claimed.

## Conditional QMLE

`fit_matrix_garch(dynamics="full")` estimates every entry in the four dynamic
matrices. Its free parameter count for m,n greater than one is
`3 + (2*m*m + m*(m+1)/2 - 1) + (2*n*n + n*(n+1)/2 - 1)`.
`dynamics="diagonal"` restricts only A1/A2/B1/B2; A0/B0 retain full lower
triangular structure. Both fit the same trace-identified recursion. Diagonal
fitting is an explicit restriction, not a substitute for the full model.

The optimizer is SLSQP with analytic first derivatives through every state update
and trace normalization. Nonlinear-constraint derivatives use numerical
differences. Intercept diagonals and w use logarithmic coordinates. The data are
divided by their largest absolute entry for likelihood conditioning. If
`X_work=X/scale`, then `A1_work=scale*A1`, `B1_work=scale*B1`, and
`w_work=w/scale**2`; other parameters and unnormalized shape states are unchanged.
Returning to original units reverses these transformations. Likelihoods gain
`m*n*log(scale)` per observation; all returned covariances use original units.
ARCH coefficients therefore have a data-unit dependence that must not be ignored
when checking parameter constraints after standardization.

Optimization uses an explicitly bounded working-coordinate domain:

- log w lies within `parameter_bound` of log mean squared Frobenius norm;
- free log intercept diagonals lie in `[-parameter_bound, parameter_bound]`;
- remaining matrix coefficients lie in that same numeric interval;
- alpha and beta lie between zero and `parameter_bound`.

These are numerical domain choices, not limits prescribed universally by the
paper. The API reports coordinates near their bounds. Changing the bounds or
starting values can change a local optimum. Default starts use sample marginal
Cholesky shapes and small nonzero diagonal dynamics; further starts perturb the
first feasible start reproducibly. An explicit original-unit `initial` is allowed.
No true parameter is used by the default initialization.

The result retains all starts' objectives, solver statuses/messages, iteration
counts, feasibility, active bounds and objective histories. Histories need not
decrease monotonically for constrained SLSQP. `gradient_norm` is an unconstrained
score diagnostic, not a KKT residual at active constraints; `n_evaluations` is
the optimizer-reported evaluation count, excluding additional callback checks.
An exact-last-point cache reuses likelihood/gradient results across solver,
callback and terminal checks; it never merges approximately equal points.
Original-unit covariance factors must remain positive definite with positive
represented entry variances: finite but underflowed covariances are rejected,
without adding a numerical floor. Invalid physical-unit parameters and spectral
trial values receive finite optimizer penalties. An unrepresentable randomized
start is contracted toward the known feasible initializer; if 40 contractions
are insufficient, `start_fallback=True` records reuse of that initializer.
These failed trial points do not discard earlier valid optimization runs.
When a run terminates with invalid or infeasible parameters, or a failed run
ends worse than an earlier feasible iterate, that earlier point is retained
with `retained_earlier_iterate=True` and `converged=False`. The terminal constraint
violation remains visible. The lowest-objective finite feasible retained fit is
selected across starts, even if it has not converged. Users must inspect that flag.
Convergence is not a guarantee of a global optimum or of identified coefficients.

## Positivity and stationarity are different

The parameter class enforces w positive, alpha/beta nonnegative, and positive
definite intercept products. These ensure positive covariance recursions, not
stationarity. Optional fitting constraints, evaluated in **original data units**,
have distinct meanings:

- `constraint="spectral"` applies Assumption 3.3(i)'s bounds on
  `rho(A1⊗A1+A2⊗A2)`, `rho(B1⊗B1+B2⊗B2)` and `alpha+beta`.
  It does not certify strict stationarity or all of Assumption 3.3.
- `constraint="sufficient"` instead imposes the accepted manuscript's Theorem 1
  bound, `||A1||_F**2 + ||B1||_F**2 + alpha +
  max(||A2||_F**2, ||B2||_F**2, beta) < 1`, with a chosen margin. Under Gaussian
  innovations this is sufficient for an ergodic stationary solution with finite
  second moment. It is not necessary.
- `constraint="none"` applies only positivity and the compact numerical domain.

The paper's main 3×3 simulation has A1=B1=.3 I, A2=B2=.6 I, alpha=.3 and beta=.6.
Its sufficient bound is **1.92**, while its separate spectral bounds are
**(.45,.45,.9)**. Rejecting that simulation merely because the sufficient bound
exceeds one would misrepresent the paper. Both diagnostics are exposed separately.

## Forecasts, tests and remaining work

`forecast_one()` gives the exact covariance of the next observation from the
terminal state and last observed matrix. Filtering subsequent observations with
that state produces chronological one-step covariances without refitting. It
cannot use the observation currently being predicted. The full-model example
shows held-out filtering and the equality of its first covariance with the
training-origin forecast.

Multiple-step covariance expectations do not close under the trace normalizations.
In particular, the expectation of a normalized shape is not the normalization
of its expectation, and an average of Kronecker covariances is generally not a
single Kronecker covariance. No such plug-in approximation is labeled an exact
forecast. A future multistep API should simulate paths, average full conditional
covariances, and report Monte Carlo uncertainty; it is not implemented here.

Independent tests explicitly construct the full 2×3 Kronecker covariance at each
time and compare every likelihood contribution and recursive state. They check
the full analytic gradient against finite differences of that dense oracle,
scalar GARCH reduction and parameter recovery, simulation/filter agreement,
independent coefficient-sign changes, scaling, causality and optimization failure
diagnostics. These tests are implementation and small-simulation evidence, not
a reproduction of every experiment or inferential theorem in the paper.

Missing branches remain: QMLE sandwich standard errors and boundary inference,
the estimation-adjusted portmanteau test, matrix factor GARCH, higher orders,
non-Gaussian simulation/likelihood families, and multistep simulation forecasts.
The paper's QMLE theory requires more than a positive-definite filter: its
population identification and stationarity assumptions, moment conditions,
interior parameters, and nonsingular information must hold. No automatic
population-identification test is supplied by this implementation.
