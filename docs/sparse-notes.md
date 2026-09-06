# Sparse MAR posterior and implementation scope

`fit_sparse_mar` implements the continuous-normal-mixture EMVS branch for
zero-mean MAR(1), following [Celani, Pagnottoni and Jones (2024),
Sections 3–4 and Appendix C](https://doi.org/10.1007/s11222-024-10402-y).
The paper is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
the implementation and derivations below include the corrections and
algorithmic modifications described here.

The observation model is `X[t] = A X[t-1] B.T + E[t]`, with
`vec_F(E[t]) ~ N(0, C ⊗ R)`. Every coefficient independently has variance
`tau0` or `tau1`, both strictly positive, with mode-specific mixing weight
`theta[k]`. Priors are `theta[k] ~ Beta(alpha,beta)`,
`Sigma[k] | xi ~ IW(xi Omega[k], nu[k])`, and
`xi ~ Gamma(shape=eta1, rate=eta2)`. Defaults use `tau0=.01`, `tau1=4`,
`Omega[k]=I/d[k]`, `nu[k]=d[k]+2`, and uniform Beta weights. The uniform
Beta default differs from the paper's dimension-dependent simulation
settings. Covariance priors use the original data units; centering or
standardizing should be an explicit modeling choice.

Up to fixed prior normalization constants, the observed log posterior is

```text
ell = log p(Y | Z,A,B,R,C)
    + sum[k,i] log((1-theta[k]) N(phi[k,i];0,tau0)
                  + theta[k] N(phi[k,i];0,tau1))
    + sum[k] ((alpha-1) log(theta[k]) + (beta-1) log(1-theta[k]))
    + sum[k] (nu[k]*d[k]/2 log(xi)
              - (nu[k]+d[k]+1)/2 logdet(Sigma[k])
              - xi/2 tr(Omega[k] inv(Sigma[k])))
    + (eta1-1) log(xi) - eta2*xi.
```

The E-step computes slab probabilities using log-sum-exp, and expected
coefficient precisions `(1-p)/tau0 + p/tau1`. Each coefficient block is an
exact penalized GLS conditional mode. An SVD compresses the time design;
an augmented least-squares solve avoids normal equations and never builds
the observation-space Kronecker covariance. Row and column factors need
dense solves with `m²` and `n²` columns respectively, so this is not a
matrix-free estimator for extremely large modes.

For `N=T-1`, the row covariance mode is
`(sum E[t] inv(C) E[t].T + xi Omega_row)/(N*n + nu_row + m + 1)`;
the column mode follows by transposition. The coefficient prior is
independent of covariance, so it contributes no additional covariance
degrees of freedom. The mixing weight uses the Beta conditional mode,
with expected inclusion counts. The shared scale uses Appendix C's
Gamma mode formula.

The published equations contain notation inconsistencies: the Gamma prose
calls eta2 a scale, whereas its log posterior and C13 use a rate; B6 has a
Beta parameter typo; some design cross-products are transposed inconsistently.
The code follows the stated hierarchical density and dimensionally valid
conditional derivatives, checked against independent multivariate-normal,
inverse-Wishart, Beta and Gamma density evaluations.

The paper's equation (8) arbitrarily balances Frobenius norms after updates.
Although reciprocal scaling preserves the likelihood, it changes the
normal-mixture and inverse-Wishart priors. This implementation instead adds
two exact conditional scale maximizations. For the coefficients it minimizes
`c² sum(D_A*A²) + c⁻² sum(D_B*B²)` using the current E-step precisions; for
the covariance factors it maximizes their IW prior along `(c R,C/c)`.
These operations preserve the current EM surrogate or improve it, and
accelerate convergence in the otherwise weakly identified scale directions.
Only a joint coefficient sign convention is applied to the output. Fitted
factor magnitudes are prior-dependent; they are not forced to unit norms.

`log_posterior_history` contains the same observed objective each iteration.
An objective decrease beyond roundoff raises a numerical error. Convergence
means a small relative posterior increment, not a certified global maximum
or a bound on parameter error. Different initial conditions or priors can
produce different local modes; stability is diagnosed, not enforced.
Temporally constant data are rejected. Degenerate exact-fit data can drive
the shared covariance scale to its boundary, so a finite covariance mode
need not exist for every dataset; the routine raises on nonfinite or
numerically decreasing objectives instead of adding an undocumented ridge.

`row_inclusion` and `column_inclusion` are *conditional plug-in* probabilities
at the fitted mode, not posterior averages over parameter uncertainty.
Support masks use a probability strictly greater than .5. Because both
mixture components are continuous, fitted coefficients are not forced to
zero, and forecasts retain all fitted coefficients. The example reports
sparse support, operator recovery and genuinely held-out one-step errors.
There are no posterior draws, credible intervals, MCMC convergence checks,
MAR*(P) lag-factor estimation, or general tensor EMVS in this implementation.
