# MAR inference and specification testing

`mar_inference` implements plug-in versions of Theorems 2–4 in
[Chen, Xiao and Yang (2021)](https://arxiv.org/abs/1812.08916) for projection,
least squares and separable maximum likelihood. `mar_specification_test`
implements Section 4.2's test of a single Kronecker VAR(1) coefficient.
These are asymptotic procedures, not finite-sample guarantees.

## Identification and assumptions

The process must be a stationary, zero-mean MAR(1) with iid innovations,
finite second moments, nonsingular coefficient matrices and nonsingular
innovation covariance. ALS/MLE also use the paper's absolute-continuity
Condition R; MLE inference requires correctly separable covariance. The APIs
do not cover estimated intercepts, multiple lags, regularization, selected
ranks, active covariance flooring, HAC dependence or post-selection inference.
Numerical checks reject unsuitable sample fits but cannot establish these
population assumptions from the observations.

The left coefficient has unit Frobenius norm. A fixed `sign_anchor=(i,j)`
makes that entry positive by jointly flipping A and B in a copied result.
Its population value must be nonzero. This avoids relying on a largest-entry
sign rule, which is discontinuous at opposite-signed ties. The supplied fit
is not mutated. Operator inference for `B ⊗ A` is unaffected by joint signs.
At an operator entry whose two coefficient factors both vanish, the
first-order delta-method variance degenerates; ordinary Wald coverage is
not justified there.

## Covariance construction

All vectorization is column-major. The parameter order is consistently
`[vec(A), vec(B)]`; the permutation from the paper's `vec(B.T)` convention
is applied in the ALS/MLE Jacobian. Returned covariance is that of the
estimator, including division by `N=T-1`, not its square-root-N limit.

For projection, the unrestricted VAR covariance `Gamma^-1 ⊗ Sigma` is
rearranged and propagated through the rank-one projection derivative. For
ALS, the covariance is the constrained sandwich `H^-1 M H^-1 / N`, with
`H = mean(J.T J) + gamma gamma.T`, `gamma=[vec(A),0]`, and
`M = mean(J.T Sigma J)`. MLE replaces the Gram matrix with
`mean(J.T Sigma^-1 J)` and uses that weighted Gram as the sandwich middle.
Sample expectations are accumulated in bounded time batches. A common
input scale is removed before moments are calculated; coefficient standard
errors are invariant to data units.

`confidence_interval(level=.95)` returns marginal operator intervals;
`target="left"` and `target="right"` address identified coefficient entries.
They are not simultaneous confidence bands. `operator_covariance()` explicitly
materializes a dense covariance with `(m*n)^4` entries. Dense inference is
guarded by `max_dimension`; increasing the limit is a deliberate memory choice.

The specification statistic uses an orthonormal basis of the rank-one
matrix's normal space. A positive-definite solve there is equivalent to the
paper's singular projected covariance pseudoinverse. Its asymptotic reference
is chi-square with `(m²-1)(n²-1)` degrees of freedom. Both spatial dimensions
must exceed one. A small p-value rejects the coefficient restriction; a
large p-value does not prove it, and a true null can occasionally be rejected.

## Evidence and boundaries

Independent tests compare the projection derivative with finite differences,
ALS/MLE covariances with explicit per-observation Jacobian formulas, and the
specification statistic with the dense pseudoinverse expression. Tests also
cover sign charts, rejected fits and input scaling by `1e±150`.

The retained [inference study](../benchmarks/results/inference.json) contains
summaries and points to raw JSONL replications. It compares sample sizes 200
and 800 under isotropic, separable and nonseparable innovations. MLE under
nonseparable innovations is explicitly a violated-assumption experiment.
Coverage uncertainty uses independent series, not correlated entries of one
operator. Failures remain recorded; reported rates state their conditioning
on successful fits. Specification rejection rates include Wilson binomial
bounds even when all or none of the replications reject. See the
[results interpretation](../benchmarks/results/extensions-summary.md) for
finite-sample undercoverage rather than assuming nominal calibration.
