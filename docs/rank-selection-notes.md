# Published dynamic tensor rank selection

`mavats.rank_selection.select_tensor_rank` implements Han, Chen and Zhang,
*Rank Determination in Tensor Factor Model*, Electronic Journal of Statistics
16(1), 1726–1803 (2022), [journal DOI](https://doi.org/10.1214/22-EJS1991),
[primary manuscript v3](https://arxiv.org/html/2011.07131v3).
It is separate from the older unpenalized adjacent-ratio heuristics: existing
fixed-rank TOPUP/TIPUP estimators are not changed.

## Moments, criteria, and penalties

For each mode, unfold each observation as a `d_k × d_-k` matrix. TOPUP preserves
all pairs of its lagged fibers, forms past-first lag outer-product averages,
and sums their Gram matrices. TIPUP first sums matching fibers into the
past-first matrix cross-moment and then squares. Each lag uses denominator
`T-h`; lags 1 through `h0` are accumulated without an extra `h0` average.
The resulting eigenvalues have observation-to-the-fourth-power units.

Equations (1)–(2) give

- IC: minimize `sum(lambda[j], j > m) + m*G` over `0 <= m <= m*`.
- ER: minimize `(lambda[m+1] + H)/(lambda[m] + H)` over `1 <= m <= m*`.

The `penalty` integer selects equations (7)–(8) exactly. Put `d=prod(d_k)`,
`D=h0*d**(2-2*nu)`, `L=log(d*T/(d+T))`, `M=log(min(d,T))`, and
`M_k=log(min(d_k,T))`:

| Number | IC penalty G | ER penalty H |
| --- | --- | --- |
| 1 | `D*L/T` | `c0*h0` |
| 2 | `D*(1/T+1/d)*L` | `h0*d**2/T**2` |
| 3 | `D*M/T` | `h0*d**2/(T**2*d_k**2)` |
| 4 | `D*(1/T+1/d)*M` | `h0*d**2/(T**2*d_k**2) + h0*d_k**2/T**2` |
| 5 | `D*(1/T+1/d)*M_k` | `h0*d**2/(T**2*d_k) + h0*d*d_k/T**2` |

Defaults are `criterion='er'`, `penalty=1`, `c0=.1`, and `nu=0`.
The default IC strength setting is not an estimate: ideally `nu=delta1`, the
weakest-factor strength exponent. Assumption IV allows
`0 <= delta0 <= delta1 <= 1`; satisfying the additional rate conditions is
essential. A true `delta1` supplied in a simulation is oracle information and
must not be described as an untuned rank procedure. `nu` affects IC, not ER.

## Sequential iteration and dimensions

All 40 combinations (two moment types, two criteria, five penalty families,
with/without iteration) are implemented. Equation (6) updates modes in order,
using newly updated lower-mode spaces and previous higher-mode spaces.
Every update reselects the rank. Initial noniterative choices are inflated to
`min(2*r, r+3, d_k)` for projection, unless `initial_ranks` is explicit.
The dimension cap is an implementation boundary convention. Search bounds
remain separate and default to `floor(d_k/2)`; they must be less than `d_k`.

Penalties always retain the ORIGINAL `d` and `d_k`, including in iterative
updates. Only moments use the smaller projected tensors. Besides the displayed
paper notation, this is corroborated by the author's tensorTS implementation:
[R/tenFM.R](https://github.com/cran/tensorTS/blob/master/R/tenFM.R), where the
original `dim(x)` supplies `ddd` for `tensor.bic`/`tensor.ratio` after projection.
The paper, not package-specific variant formulas, governs the ten penalties.
There is no numbered Algorithm 1 in this rank paper's v3 HTML: the implemented
algorithm is equation (6) and Remark 5's initialization/stopping instructions.

Stopping requires both unchanged ranks and maximum Frobenius loading-projector
change below `tol`. A maximum-iteration exit is explicitly unconverged. History
contains the initial unprojected criteria followed by complete sequential
sweeps; final spectra retain their actual update context, not a silently
recomputed simultaneous moment. `rank_history` starts with the projection
initialization; `initial_ranks` separately retains raw unprojected choices.

## Scaling and numerical limits

Internally observations are divided once by their maximum magnitude (with a
small representability floor). This is numerical protection only. The fixed
physical penalty is divided by that scale to the fourth power in log space;
it is never replaced by a multiple of the largest sample eigenvalue.
`penalty_multiplier` can be scalar or one positive value per mode. Its
`log_penalty_multiplier` alternative supports constants whose exponentials
cannot be represented and cannot be combined with a nondefault ordinary
multiplier. To preserve a criterion under `X -> a*X`, add `4*log(abs(a))` to
each log multiplier. Keeping its physical multiplier fixed can change ranks.

Each `RankSelectionStep` exposes normalized `eigenvalues`, physical
`log_eigenvalue_scale`, physical `log_penalty`, candidate ranks, and scores.
IC scores share the factor `exp(log_score_scale)`; ER scores are dimensionless
ratios. Physical/log criterion values are also recorded. IC uses the exact
identity `IC(m)-IC(m-1)=G-lambda[m]` for selection, preventing a large common
tail from obscuring a weak factor. ER compares the log improvement from one,
so displayed ratios rounding to one do not force a spurious tie. Exact ties
choose the smallest candidate rank. No relative eigenvalue floor is added.
Roundoff still limits distinguishable eigenvalues; scaling safeguards do not
make arbitrarily ill-conditioned spectra identifiable. Unrepresentable
physical factors/signals/residuals raise explicit errors rather than returning
infinite arrays.

## Zero ranks, held-out use, and boundaries

IC supports zero, including empty cores and their inverse transformation.
Raw modewise criteria may select mixed zero/positive ranks; any zero mode
implies the entire centered Tucker signal is zero. `ranks` preserves the raw
choices, while `signal_ranks` reports all zero in this case. Empty projected
modes yield exact zero moments and can make subsequent iterative IC choices
zero. This is a transparent degenerate execution convention, not an additional
positive-rank consistency theorem. ER searches strictly positive ranks: an
all-zero spectrum selects one by tie convention, not evidence for a factor.
The paper's suggested mock eigenvalue for ER rank zero is not specified enough
to implement uniquely and is not invented here.

At least one spatial mode is supported, including the vector special case,
but each dimension must be at least two. Full-mode-rank search would require
the modifications mentioned in Remark 7 and is not provided. The vector
boundary `d=T=2` makes g1/g2 zero, violating the positive-penalty premise, and
is rejected for those families. The other positive families remain usable.

Centering defaults to true, following Remark 6 when removing the deterministic
mean preserves factor dimensions. A constant factor can violate that premise.
`transform` uses fixed training spaces and training mean; `inverse_transform`
adds that mean. Their numerical scales are per observation, so an extreme
later observation cannot underflow an earlier transformed score. Training
moments still use one common scale, as changing training units observation by
observation would change the estimator. Current held-out observations are required: this is denoising,
not forecasting. Numerical convergence does not verify serially white errors,
informative factor lags, strength assumptions, or correct ranks. In particular,
TIPUP can cancel lag signal before squaring, while TOPUP retains cross-fiber
information; lagged criteria cannot distinguish all serial-noise alternatives.

This selector does not tune the multiplier automatically. Remark 8/Section 5.4
subsample stability paths are a separate procedure requiring explicit grids
and plateau conventions. No rank confidence interval, post-selection
inference, or universal finite-sample guarantee is implemented.

## Verification

Independent tests construct dense lag moments with nested tensor indices,
evaluate all ten penalties and both criteria directly, and reproduce a
sequential projection sweep with original penalty dimensions. Other checks
cover strong-factor recovery for all 40 variants, exact cancellation,
weak-tail IC decisions, ER rounding against high-precision arithmetic,
positive/negative extreme rescalings, empty cores, and training-only held-out
projection. Run `python examples/factor_rank_selection.py` for a reproducible
strong-factor example with explicitly labeled held-out denoising scores.
