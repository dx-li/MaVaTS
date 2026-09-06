# Sequential matrix-factor monitoring

`MatrixFactorMonitor` implements the randomized, fixed-horizon procedure of He,
Kong, Trapani and Yu (2024), *Annals of Statistics* 52(4), 1646–1670,
[DOI 10.1214/24-AOS2410](https://doi.org/10.1214/24-AOS2410). The authoritative
equations used here are in the authors'
[accepted manuscript](https://raw.githubusercontent.com/heyongstat/Paper/main/Online%20Change-point%20Detection%20for%20Matrix-valued%20Time%20Series%20with%20Latent%20Two-way%20Factor%20Structure/manuscript.pdf)
and [supplement](https://raw.githubusercontent.com/heyongstat/Paper/main/Online%20Change-point%20Detection%20for%20Matrix-valued%20Time%20Series%20with%20Latent%20Two-way%20Factor%20Structure/supplement.pdf).
The [arXiv v1](https://arxiv.org/html/2112.13479v1) uses a different transformation
and must not be substituted for the accepted power transformation.

The [authors' OLCPM reference implementation](https://raw.githubusercontent.com/cran/OLCPM/master/R/function.R),
`gen.psi.tau.proj`, clarifies rolling projection estimation. It recomputes the
opposite-mode principal components in each current window; it does not fix them
at training or estimate them using future monitoring data. This implementation
was independently derived from the equations; no GPL source was copied.

## Window statistic

Let `m` be the training length, `p` the monitored dimension, `q` the opposite
dimension, `k=rank`, and `l=projection_rank`. Input arrays are time-first. At
monitoring step `tau`, the window is observations `tau+1,...,m+tau`, with indices
starting at one. The initial training window is not itself tested. Equations
(2.1)–(2.2) and the rolling definition preceding (3.9) give

```
G_tau = sum_window(X_t.T @ X_t) / (m*p)
Q_tau = l leading orthonormal eigenvectors of G_tau
M_tau = sum_window(X_t @ Q_tau @ Q_tau.T @ X_t.T) / (m*q)
beta = log(p) / log(q*m)
delta = epsilon + max(0, 1 - 1/(2*beta))
u_tau = p**(1-delta) * eigenvalue_(k+1)(M_tau) / trace(M_tau)
psi_tau = u_tau**power
y_tau = psi_tau + z_tau,             z_tau independently N(0,1)
S_tau = sum(y_1,...,y_tau)
```

Eigenvalues are ordered decreasingly. The last four definitions correspond to
(3.9)–(3.13); the sum starts at one despite the manuscript's inconsistent zero
index in (3.13). Column monitoring transposes each matrix and applies the same
equations. All matrices in a window are reprojected with its new `Q_tau`.
The paper assumes zero means: this implementation does not silently center.

Both ranks are positive and fixed by the caller before monitoring. They may be
known from the design or selected on stable training data. The projection rank
should cover the true opposite-mode rank and must be below its dimension.
The monitored `(k+1)` eigenvalue must be attainable within a window, so
`k+1 <= m*l`. No rank is reselected after the stream starts.

Default `power=8` and `epsilon=0.05` reproduce the accepted Gaussian simulation
tuning; they are not universal data-driven defaults. Any finite `power>=2` is
accepted, with the corresponding population moment assumptions. The resulting
`delta` must lie below one. There is no default projection rank or horizon.

## Boundaries and calibration

All comparisons are strict `statistic > critical_value`. A supplied
`critical_value` always uses the normalized scale below, not a raw `y` or
partial-sum boundary. `H` is the prespecified horizon; `d` is the Renyi delay.

| Procedure | Recorded statistic | Reference critical value |
| --- | --- | --- |
| `maximum` | `(y_tau-B_H)/A_H` | Standard Gumbel |
| `partial_sum`, `0<=eta<0.5` | `H**(eta-0.5)*abs(S_tau)/tau**eta` | `sup(0<u<=1) abs(W(u))/u**eta` |
| `partial_sum`, `eta=0.5` | `a_H*abs(S_tau)/sqrt(tau)-b_H` | Standard Gumbel |
| `partial_sum`, `0.5<eta<1` | `d**(eta-0.5)*abs(S_tau)/tau**eta`, eligible `tau>=d` | Brownian supremum with exponent `1-eta` |

For the maximum procedure, (3.28)–(3.32) give
`B_H=sqrt(2*log(H))-(log(log(H))+log(4*pi))/(2*sqrt(2*log(H)))` and
`A_H=B_H/(1+B_H**2)`. For `eta=0.5`, (3.15), (3.19) give
`a_H=sqrt(2*log(log(H)))` and
`b_H=2*log(log(H))+0.5*log(log(log(H)))-0.5*log(pi)`.
The other boundaries are (3.14), (3.16), (3.18), (3.20).

Maximum requires `H>=2`; `eta=0.5` requires `H>=3`. These merely make the formulas
defined and do not make an asymptotic approximation accurate. Renyi defaults to
`max(1,floor(log(H)))`, the reference-software choice with an explicit finite-H
lower bound. Its asymptotic theory needs `d` to diverge and `d/H` to vanish;
choosing any legal small-H delay does not ensure those approximations. Pre-delay
observations and Gaussian draws contribute to `S_tau` but cannot trigger an alarm.

If no threshold is supplied, maximum and `eta=0.5` use
`-log(-log(1-alpha))`, where default `alpha=0.05` is a horizon-wide asymptotic
false-alarm level. Other partial-sum schemes require a caller-supplied positive
critical value; no unverified closed Brownian quantile is invented. A separately
calibrated finite-horizon zero-drift Gaussian threshold can be supplied to any
scheme. Such calibration controls a Gaussian *reference*, not the full
finite-sample matrix-data null: conditional on the observations, the actual
`y_tau` are independent `N(psi_tau,1)`, with nonzero finite-sample drift.

## Sequential and numerical contract

- `update(matrix, noise=None)` consumes one observation and returns a frozen
  `MonitorStep`. Invalid data, an invalid draw, or an undefined zero projected
  trace leaves the current window, history, and RNG unchanged.
- `update_many(batch)` is identical to repeated calls and returns only the
  consumed prefix. Validation is per observation: a bad later observation does
  not roll back earlier updates, and data after a stop are never inspected.
  Calling it on a stopped monitor returns an empty tuple. A stopped `update`
  raises `RuntimeError`.
- Stop occurs at the first crossing or after exactly `H` consumed matrices.
  `alarm_step` is a one-based post-training index; `alarm_observation` adds `m`.
  It is a detection time, not a retrospectively corrected change-point estimate.
  There is no lookahead, horizon extension, automatic restart, or retrospective
  rank refit. Monitoring both modes requires a separate multiplicity policy.
- A local Gaussian generator is independent of NumPy's global RNG. A supplied
  `Generator` is cloned. Supplying `noise` skips its internal draw and supports
  equation-oracle tests or external randomization; deterministic or non-Gaussian
  values invalidate Gaussian calibration.
- `state_dict()` produces detached strict-JSON-compatible state, including
  configuration, the current window, diagnostics, and RNG. `from_state()` resumes
  without drawing or reprocessing history. It validates basic consistency, not
  provenance of manually edited scientific records. No pickle is needed.
- Each covariance calculation first divides the window by its maximum absolute
  observation. Eigenvalue ratios are unchanged, even when direct covariance
  squaring would overflow or underflow. `data_scale`, `eigenvalue_scaled`, and
  `trace_scaled` expose this scaling. A zero-trace window is rejected; no ridge
  is silently substituted. Eigenvalues are symmetrized and negative numerical
  roundoff is clipped at zero.
- Power drift, signed cumulative sums, and crossings use logarithmic magnitude
  arithmetic. Unrepresentable display values saturate at a finite float or zero
  with `saturated=True`; crossing decisions use the unsaturated log values.
  `log_drift` may be infinite and is encoded explicitly in checkpoints. Results
  remain floating-point approximations near exact numerical ties.

The buffer takes `O(m*p*q)` storage and the retained diagnostic history `O(H)`.
Each update recomputes both covariances and eigendecompositions. This prioritizes
the changing-projection equations and a clear online contract over cached
low-rank updates.

## Assumptions, supported changes, and omissions

Main-paper Assumptions 1–3 and supplement A.1–A.4 require a clean stable training
period, strong factors of fixed positive ranks, nonsingular factor second
moments, suitable weak spatial/temporal dependence and maximal inequalities,
controlled factor-error correlation, and finite `4*power` moments of factors
and errors. Thus `power=8` needs finite 32nd moments, not just finite variance.
Dimensions and training size must grow in the paper's joint asymptotic regime;
the null drift must vanish sufficiently quickly over the entire growing horizon.
In particular the horizon restriction accompanying the limit theorems is
`H * (p**(-delta) * l_pqm)**power -> 0`, as well as `H=Omega(m)`, with
`l_pqm=(1/q+1/m+p/sqrt(m*q))*(log(p)**2*log(q)*log(m))**(1+small_positive)`.
These are population/asymptotic assumptions, not runtime-verifiable guarantees.

The target alternatives are newly appearing factors and changes in loading
spaces that enlarge the joint pre/post span. A changed space can produce a
temporary eigenvalue spike while the rolling window straddles the break; a new
factor can produce a persistent spike. Rotations within an unchanged loading
span and pure factor-variance changes are not promised detectable alternatives.

Disappearing factors are **not implemented**. Supplement B.3 instead monitors
a reciprocal transformation of the baseline-rank eigenvalue and needs extra
rate assumptions, including `m / p**(power*(1-delta)) -> 0`; reversing a sign
would not implement that method. Also absent are zero-rank/weak-factor variants,
multiple-randomization retrospective voting wrappers, automatic tuning, and
simultaneous row/column inference. The supported eta range `[0,1)` is explicit.

## Validation

`tests/test_monitoring.py` uses direct, unscaled loop equations independent of
the implementation's covariance and signed-log helpers. It tests both modes,
all four boundary families, hand-computable strict crossings, causal prefix
consumption, five local bit generators, strict JSON continuation, transactional
input failures, global scaling up to `1e150` and down to `1e-150`, and overflowing
power drift before a Renyi delay. A paired seeded planted-factor example checks
that a structural alternative produces an alarm without turning this one draw
into a false-alarm or power claim. Run `python -m examples.online_monitoring`.
