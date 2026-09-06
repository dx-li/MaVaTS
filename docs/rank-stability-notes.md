# IC penalty-constant subsample stability

Reference: Yuefeng Han, Rong Chen and Cun-Hui Zhang (2022), *Rank
Determination in Tensor Factor Model*, Electronic Journal of Statistics
16(1), 1726–1803, [DOI:10.1214/22-EJS1991](https://doi.org/10.1214/22-EJS1991).
The implemented source is [arXiv v3, Remark 8 and Section 5.4](https://arxiv.org/html/2011.07131v3).

## Published path

For every supplied positive penalty multiplier c and nested subsample j,
`tensor_rank_stability` runs the paper's IC selector with penalty
`c * g_k(d_j,T_j)`. It retains `ranks[k,c,j]` and computes

`S[k,c] = sum_j (ranks[k,c,j] - mean_j ranks[k,c,j])**2 / J`.

This is variance with **ddof=0**, including the final full-data subsample.
The conceptual j=0 empty subsample in Remark 8 is not fitted or included in
the variance. For fixed lag moments, increasing c decreases the IC rank; the
result reports observed monotonicity rather than forcing it. Iterative rank
updates couple modes, so fixed-moment monotonicity need not hold for the
fully iterated path.

Inputs are explicit and deterministic:

- `c_grid`: finite, positive, strictly increasing, at least two points.
- `spatial_subsets[j][k]`: the actual integer indices retained in spatial mode
  k. Each sequence of sets grows strictly and ends with all indices. Order
  within each vector is preserved and retained, not sorted or shuffled.
- `time_prefixes[j]`: exclusive stopping positions in the original training
  data. Prefix lengths are nondecreasing, can repeat, and end at `len(X)`.
  Each must exceed the maximum lag.
- `max_ranks`: fixed positive caps, one per mode or an explicitly broadcast
  scalar. Each cap must be strictly smaller than every subsample's mode size;
  there is no hidden clipping or subsample-dependent search cap.

At least two paired subsamples are required. The API intentionally does not
generate random subsets, choose a grid from held-out data, or supply an
automatic grid. These decisions affect the scientific result and must be
reported. Spatial growth is required in every mode, as specified in Remark 8;
time-only resampling is not this interface's procedure.

`method`, `penalty`, `lags`, `iterative`, `nu`, `center`, `max_iter`, and `tol`
are passed to `select_tensor_rank` with `criterion="ic"`. Dimensions and T in
the penalty are those of the current subsample, not those of full data.
Within an iterative cell they remain that cell's original dimensions, as in
the underlying selector. Each cell independently centers its own prefix and
initializes its own iteration; no full-data mean, loadings, or ranks leak into
smaller samples. The path defaults to noniterative TIPUP/IC2; iTIPUP/iTOPUP
require `iterative=True`. None of these defaults estimates factor strength:
`nu=0` is the paper's strong-factor tuning choice.

The penalty is in physical fourth-power data units, not relative to the
largest eigenvalue. Scaling X by a without changing c generally changes the
path. To preserve the criterion, scale the c grid by `abs(a)**4`, provided the
resulting grid is representable. The underlying selector supports logarithmic
penalties, but this explicit finite positive c-grid interface does not expose
an unrepresentable log-c grid.

## Finite-grid plateau choice

Remark 8's global variance-minimum language alone is insufficient: small c
can yield a zero-variance maximum-rank plateau, while large c can yield a
zero-rank tail. Section 5.4 discusses the **second stability interval** after
excluding the first, maximum-rank interval, under a positive-rank assumption.
The paper does not supply numerical definitions of approximately zero
variance, minimum interval width, or a representative finite-grid point.

`path.choose_plateaus(variance_tolerance=..., minimum_grid_points=...)`
therefore implements an explicitly labeled discretization:

1. Exclude failed cells. Unless `allow_unconverged=True`, exclude every c with
   any unconverged subsample fit. These points break intervals.
2. Find maximal runs of consecutive supplied gridpoints whose variance is at
   most the supplied tolerance and whose **full-sample rank is constant**.
3. Reject maximum-cap and zero-rank runs. Also reject runs with fewer than
   the required number of gridpoints (at least two).
4. Select the first remaining run in ascending c order, separately per mode.
   Return its lower-middle gridpoint, without interpolation or optimization
   between gridpoints.

The required width is a count of gridpoints, not a width in c or log(c).
Irregular grid spacing therefore matters. Interval endpoints, rejection
reasons, and whether the interval touches a grid boundary are reported.
Per-point reasons explain failure, unavailable variance, excess variance, or
unconverged fits. A grid beginning inside a plateau cannot establish that it
is literally the population's second interval; the boundary flag makes this
limitation visible. No region returns None for that mode's index, multiplier,
and rank, with `complete=False`; no endpoint fallback is supplied.

This rule does not assert that a chosen plateau identifies the true rank.
In particular, a too-small maximum cap excludes the truth, coarse grids can
miss intervals, different subsets can suppress factors, and weak factors or
TIPUP signal cancellation can give stable but incorrect ranks. Excluding zero
means the convenience choice is unsuitable for certifying no factors, even
though the complete IC path itself correctly includes rank zero.

The result's modewise selected multipliers may differ. A path evaluates a
common scalar c across modes at each point. For iterative IC, combining
modewise choices into a new multiplier vector changes the coupled iteration;
`selected_ranks` are **path ranks**, not promised ranks from that new fit.
Refit `select_tensor_rank` jointly with the chosen vector and report its own
convergence and ranks. This interface does not silently perform that refit.

## Diagnostics, cost, and validation

`cells[c][j]` retains the complete selector result, including loading spaces,
eigenvalue criteria, initialization, rank history, convergence, and iteration
count. Numerical exceptions are retained as cell error strings. Failed ranks
use -1, distinct from scientifically meaningful rank zero; their corresponding
variance is NaN. No failed or unconverged cases are silently removed from the
variance denominator. Unconverged finite ranks remain available for inspection.

There are exactly `len(c_grid) * len(spatial_subsets)` independent selector
calls. One subsample is materialized at a time, but all fit results are retained.
Large TOPUP or iterative grids can be expensive; pilot the explicit grid first.
No claims of asymptotic validity are added beyond the paper's fixed-rank,
white-noise, dependence, factor-strength, informative-lag and rate assumptions.
The finite-grid plateau convention itself has no separately established
consistency theorem here.

Independent tests reconstruct small dense physical-unit TOPUP/TIPUP spectra
and IC objectives, check the population-variance divisor, hand-construct
multiple plateaus, and verify failures, convergence exclusions, index identity,
prefix causality, centering, compensated unit changes, and planted rank recovery.
