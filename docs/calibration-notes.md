# Monitoring reference calibration

`calibrate_monitor` is an explicit extension to the asymptotic monitoring
boundaries in [He et al. (2024)](https://doi.org/10.1214/24-AOS2410).
It calibrates the finite, discrete horizon for **iid standard Gaussian,
zero-drift reference draws**. It does not make the observed matrix procedure
an exact finite-sample test. See [monitoring notes](monitoring-notes.md) for
the data-dependent drift and population assumptions.

```python
from mavats import MatrixFactorMonitor, calibrate_monitor

calibration = calibrate_monitor(
    100, statistic="partial_sum", eta=.25, random_state=91
)
monitor = MatrixFactorMonitor(
    training, rank=1, projection_rank=2, **calibration.monitor_kwargs
)
step = monitor.update(next_matrix)
```

The horizon, statistic, exponent and delayed-start options must match the
calibration. Hold them fixed before monitoring. Reusing one calibration in
repeated experiments is appropriate; selecting it after observing alarm
outcomes is not. Multiple modes, procedures or restarts need additional
multiplicity control.

For maximum monitoring, the exact Gaussian reference threshold is
`norm.isf(-expm1(log1p(-alpha)/horizon))`. This computes the small upper-tail
probability without subtracting nearly equal numbers. It is then converted
to the manuscript's Gumbel-normalized statistic scale. The reported Monte
Carlo interval has zero width because no Monte Carlo is used.

Partial-sum, Darling–Erdős (`eta=.5`) and Rényi (`eta>.5`) reference boundaries
simulate entire iid Gaussian paths and apply the same finite-horizon statistic
as the monitor. This is **not** a discretization claiming to equal the
continuous-time Brownian supremum quantile. It directly targets the finite
observation grid. Gaussian draws are processed in bounded batches; changing
batch size preserves seeded results.

With R simulated path maxima, the critical value is the order statistic at
`ceil((R+1)*(1-alpha))`, using one-based indexing. For a fresh independent
reference path and continuous statistics, exchangeability gives exceedance
probability at most alpha, averaged over the calibration sample too. This
does not guarantee conditional exact control for the realized threshold.
If the requested order exceeds R, the routine requires more replications
instead of returning an unjustified finite threshold.

`quantile_interval` bounds the true `(1-alpha)` Gaussian-reference quantile
using binomial order-statistic confidence limits. Infinite endpoints disclose
inadequate tail information. The interval is not a confidence interval for
the matrix-data false-alarm rate. Simulations of actual matrix series are
still required, with failures and early alarms retained.

Independent tests reconstruct the entire statistic path with scalar loops,
check the exact maximum CDF, verify seeded batching and global RNG isolation,
compare monitor diagnostics against the reference formulas, and check
empirical reference coverage on fresh Gaussian paths. Current SciPy
distribution documentation was used to verify inverse-survival and discrete
quantile conventions.
