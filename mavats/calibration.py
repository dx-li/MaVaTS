"""Finite-horizon iid Gaussian reference calibration for sequential monitors.

These are explicit extensions to the asymptotic boundaries in He et al. (2024),
not finite-sample null guarantees for the matrix-data monitoring procedure.
The Gaussian reference sets every data-dependent monitoring drift to zero.

References
----------
He, Y., Kong, X., Trapani, L. and Yu, L. (2024).
Online Change-Point Detection for Matrix-Valued Time Series with
Latent Two-Way Factor Structure. Annals of Statistics, 52, 1646-1670.
https://doi.org/10.1214/24-AOS2410
This paper supplies the monitoring statistic, NOT the finite-horizon
calibration implemented here. Exact Gaussian maxima and simulated
zero-drift path quantiles are explicit package extensions; they do
not establish finite-sample matrix-data false-alarm control.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import binom, norm

from ._validation import finite_scalar, positive_int, random_generator


@dataclass(frozen=True)
class MonitorCalibration:
    """A critical value on the monitor's transformed-statistic scale.

    ``quantile_interval`` is a Monte Carlo confidence interval for the true
    Gaussian-reference quantile, not a confidence interval for matrix-data
    false-alarm probability. Infinite endpoints indicate insufficient tail
    replications. No continuous-time Brownian supremum is approximated here:
    the discrete observation times and finite horizon are the target.

    References
    ----------
    He, Y., Kong, X., Trapani, L. and Yu, L. (2024).
    Online Change-Point Detection for Matrix-Valued Time Series with
    Latent Two-Way Factor Structure. Annals of Statistics, 52, 1646-1670.
    https://doi.org/10.1214/24-AOS2410
    This paper supplies the monitoring statistic, NOT the finite-horizon
    calibration implemented here. Exact Gaussian maxima and simulated
    zero-drift path quantiles are explicit package extensions; they do
    not establish finite-sample matrix-data false-alarm control.
    """

    critical_value: float
    alpha: float
    horizon: int
    statistic: str
    eta: float
    delay: int | None
    method: str
    replications: int
    confidence: float
    quantile_interval: tuple
    order_index: int | None

    @property
    def monitor_kwargs(self):
        """Constructor options for a monitor using exactly this calibration."""
        return dict(
            horizon=self.horizon,
            statistic=self.statistic,
            eta=self.eta,
            delay=self.delay,
            alpha=self.alpha,
            critical_value=self.critical_value,
        )


def _maximum_normalizers(horizon):
    root = np.sqrt(2 * np.log(horizon))
    b = root - (np.log(np.log(horizon)) + np.log(4 * np.pi)) / (2 * root)
    return b / (1 + b * b), b


def _reference_supremum(draws, eta, delay):
    """Whole-horizon partial-sum statistic, derived independently of monitor."""
    horizon = draws.shape[1]
    times = np.arange(1, horizon + 1, dtype=float)
    paths = np.abs(np.cumsum(draws, axis=1))
    if eta == 0.5:
        llh = np.log(np.log(horizon))
        a = np.sqrt(2 * llh)
        b = 2 * llh + 0.5 * np.log(llh) - 0.5 * np.log(np.pi)
        return np.max(a * paths / np.sqrt(times) - b, axis=1)
    start = delay - 1 if eta > 0.5 else 0
    scale = (delay if eta > 0.5 else horizon) ** (eta - 0.5)
    return np.max(scale * paths[:, start:] / times[start:] ** eta, axis=1)


def calibrate_monitor(
    horizon,
    *,
    statistic="maximum",
    eta=0.25,
    delay=None,
    alpha=0.05,
    replications=20000,
    confidence=0.95,
    random_state=None,
    batch_size=256,
):
    """Calibrate a finite-horizon zero-drift iid N(0,1) reference monitor.

    Maximum monitoring uses the exact Gaussian maximum quantile, expressed
    on the paper's Gumbel-normalized statistic scale. Partial-sum monitoring
    simulates whole Gaussian paths and records each path's maximum statistic.
    It covers eta<1/2, eta=1/2 and delayed Renyi eta>1/2. Pass the result's
    ``monitor_kwargs`` to ``MatrixFactorMonitor``.

    The simulated critical value is order statistic
    ceil((replications+1)*(1-alpha)). For a fresh independent Gaussian path,
    this gives an unconditional exceedance probability at most alpha, averaging
    over the calibration simulation too. It is not conditional exact control
    for the realized critical value. Binomial order-statistic bounds quantify
    its Monte Carlo uncertainty. Replications too few to give a finite
    conservative critical value are rejected. Reuse one calibration for a
    fixed design, never tune it to observed monitoring paths or alarm outcomes.

    Matrix monitoring adds a data-dependent nonnegative drift. Its actual
    false-alarm probability can differ even under an intended no-break model.
    Multiple monitors, modes or restarts require separate multiplicity control.
    Memory is bounded by at most one million Gaussian draws per batch.
    Seed reproducibility is invariant to ``batch_size``.

    References
    ----------
    He, Y., Kong, X., Trapani, L. and Yu, L. (2024).
    Online Change-Point Detection for Matrix-Valued Time Series with
    Latent Two-Way Factor Structure. Annals of Statistics, 52, 1646-1670.
    https://doi.org/10.1214/24-AOS2410
    This paper supplies the monitoring statistic, NOT the finite-horizon
    calibration implemented here. Exact Gaussian maxima and simulated
    zero-drift path quantiles are explicit package extensions; they do
    not establish finite-sample matrix-data false-alarm control.
    """
    horizon = positive_int(horizon, "horizon")
    eta = finite_scalar(eta, "eta", minimum=0)
    if eta >= 1:
        raise ValueError("eta must lie in [0,1)")
    alpha = finite_scalar(alpha, "alpha")
    confidence = finite_scalar(confidence, "confidence")
    if not 0 < alpha < 1 or not 0 < confidence < 1:
        raise ValueError("alpha and confidence must lie strictly between zero and one")
    if statistic not in ("maximum", "partial_sum"):
        raise ValueError("statistic must be 'maximum' or 'partial_sum'")
    if statistic == "maximum" and horizon < 2:
        raise ValueError("maximum monitoring requires horizon >= 2")
    if horizon > 1000000:
        raise ValueError(
            "horizon exceeds the one-million-step calibration allocation limit"
        )
    if statistic == "partial_sum" and eta == 0.5 and horizon < 3:
        raise ValueError("eta=.5 requires horizon >= 3")
    renyi = statistic == "partial_sum" and eta > 0.5
    if delay is not None and not renyi:
        raise ValueError("delay applies only to partial sums with eta > .5")
    delay = (
        (
            max(1, int(np.floor(np.log(horizon))))
            if delay is None
            else positive_int(delay, "delay")
        )
        if renyi
        else None
    )
    if delay is not None and delay > horizon:
        raise ValueError("delay cannot exceed horizon")
    replications = positive_int(replications, "replications", minimum=2)
    batch_size = positive_int(batch_size, "batch_size")
    if statistic == "maximum":
        # Avoid forming 1-(1-alpha)**(1/H), which loses tiny tail probabilities.
        tail = -np.expm1(np.log1p(-alpha) / horizon)
        a, b = _maximum_normalizers(horizon)
        critical = float((norm.isf(tail) - b) / a)
        if not np.isfinite(critical):
            raise ValueError(
                "alpha is too extreme for a finite floating-point boundary"
            )
        return MonitorCalibration(
            critical,
            alpha,
            horizon,
            statistic,
            eta,
            delay,
            "exact-gaussian-maximum",
            0,
            confidence,
            (critical, critical),
            None,
        )
    order = int(np.ceil((replications + 1) * (1 - alpha)))
    if order > replications:
        raise ValueError(
            "replications are too few for a finite conservative alpha quantile"
        )
    rng = random_generator(random_state)
    values = np.empty(replications)
    batch_size = min(batch_size, max(1, 1000000 // horizon))
    for start in range(0, replications, batch_size):
        stop = min(start + batch_size, replications)
        draws = rng.normal(size=(stop - start, horizon))
        values[start:stop] = _reference_supremum(draws, eta, delay)
    values.sort()
    tail = (1 - confidence) / 2
    lower = int(binom.ppf(tail, replications, 1 - alpha))
    upper = int(binom.ppf(1 - tail, replications, 1 - alpha)) + 1
    interval = (
        float(values[lower - 1]) if lower else -np.inf,
        float(values[upper - 1]) if upper <= replications else np.inf,
    )
    return MonitorCalibration(
        float(values[order - 1]),
        alpha,
        horizon,
        statistic,
        eta,
        delay,
        "monte-carlo-gaussian-paths",
        replications,
        confidence,
        interval,
        order,
    )
