"""Sequential randomized monitoring of matrix factor spaces.

Implements He, Kong, Trapani and Yu (2024), accepted AOS manuscript equations
(2.1)--(2.2), (3.9)--(3.20), and (3.28)--(3.32). The accepted manuscript uses
a power transformation, unlike arXiv v1. The authors' OLCPM implementation
clarifies that projection is re-estimated on each trailing window. This is an
independent implementation of those equations, not a translation of GPL code.

Primary manuscript and supplement:
https://github.com/heyongstat/Paper/tree/main/Online%20Change-point%20Detection%20for%20Matrix-valued%20Time%20Series%20with%20Latent%20Two-way%20Factor%20Structure
Publication: https://doi.org/10.1214/24-AOS2410

References
----------
He, Y., Kong, X., Trapani, L. and Yu, L. (2024).
Online Change-Point Detection for Matrix-Valued Time Series with
Latent Two-Way Factor Structure. Annals of Statistics, 52, 1646-1670.
https://doi.org/10.1214/24-AOS2410
Uses the accepted-manuscript power transformation and equations
(3.9)-(3.20), (3.28)-(3.32), not the earlier arXiv v1 transformation.
State management is a software interface, not an additional estimator.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np

from ._validation import as_series, finite_scalar, positive_int

_LOG_MAX = float(np.log(np.finfo(float).max))
_LOG_MIN = float(np.log(np.nextafter(0.0, 1.0)))
_MAX = np.finfo(float).max


def _signed_log(value):
    return (int(np.sign(value)), float(np.log(abs(value)))) if value else (0, -np.inf)


def _add_logs(left, right):
    """Add signed numbers without exponentiating their absolute magnitudes."""
    ls, lv = left
    rs, rv = right
    if ls == 0:
        return right
    if rs == 0:
        return left
    if ls == rs:
        return ls, float(np.logaddexp(lv, rv))
    if lv == rv:
        return 0, -np.inf
    if lv < rv:
        ls, lv, rv = rs, rv, lv
    # expm1 preserves cancellation when the log magnitudes are very close.
    return ls, float(lv + np.log(-np.expm1(rv - lv)))


def _scale_log(value, multiplier):
    sign, logabs = value
    return (sign, logabs + np.log(multiplier)) if sign else value


def _greater(left, right):
    ls, lv = left
    rs, rv = right
    if ls != rs:
        return ls > rs
    return (lv > rv if ls > 0 else lv < rv) if ls else False


def _display(value):
    """Finite diagnostic approximation, with an explicit saturation flag."""
    sign, logabs = value
    if not sign:
        return 0.0, False
    if logabs > _LOG_MAX:
        return float(sign * _MAX), True
    if logabs < _LOG_MIN:
        return float(sign * 0.0), True
    # exp(log(max_float)) can round to a slightly different representable
    # number; bound the diagnostic only, never the internal crossing test.
    return float(sign * min(np.exp(logabs), _MAX)), False


def _encode(value):
    """JSON-compatible state, including exact-zero log magnitudes."""
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_encode(item) for item in value]
    if isinstance(value, (float, np.floating)):
        if np.isneginf(value):
            return "-Infinity"
        if np.isposinf(value):
            return "Infinity"
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


@dataclass(frozen=True)
class MonitorStep:
    """One consumed observation and its complete monitoring diagnostics.

    Indices are one-based: ``observation_index = training_length + monitoring_index``.
    ``eigenvalue_scaled`` and ``trace_scaled`` are computed after dividing the
    current window by ``data_scale``; multiplying by its square restores data
    units when representable. All other statistics are invariant to global
    nonzero scaling of observations.

    ``statistic`` is compared strictly with ``critical_value``, on the scale
    specified by ``MatrixFactorMonitor``. Before a Renyi delay, it is recorded
    but not eligible to trigger an alarm. ``log_drift`` can be -inf for exactly
    zero drift or +inf for an unrepresentably large logarithm. Other numeric
    diagnostics are finite. If ``saturated`` is true, an under/overflowed
    diagnostic has been clipped; the crossing decision still uses signed-log
    arithmetic, not the clipped value.

    References
    ----------
    He, Y., Kong, X., Trapani, L. and Yu, L. (2024).
    Online Change-Point Detection for Matrix-Valued Time Series with
    Latent Two-Way Factor Structure. Annals of Statistics, 52, 1646-1670.
    https://doi.org/10.1214/24-AOS2410
    Uses the accepted-manuscript power transformation and equations
    (3.9)-(3.20), (3.28)-(3.32), not the earlier arXiv v1 transformation.
    State management is a software interface, not an additional estimator.
    """

    monitoring_index: int
    observation_index: int
    window_start_index: int
    data_scale: float
    eigenvalue_scaled: float
    trace_scaled: float
    normalized_eigenvalue: float
    log_drift: float
    drift: float
    noise: float
    randomized_value: float
    partial_sum: float
    statistic: float
    critical_value: float
    eligible: bool
    alarm: bool
    saturated: bool


class MatrixFactorMonitor:
    """Monitor a fixed matrix factor rank using a genuinely sequential state.

    Parameters
    ----------
    training : array_like, shape (m, rows, columns)
        Finite real observations from a stable training period, with m >= 2.
        The paper assumes zero means. No centering, standardization, rank
        selection, or retrospective change-point search is performed.
    rank : int
        Baseline rank of the monitored mode, fixed before monitoring. Must be
        positive and below its dimension. The (rank+1)-st eigenvalue is monitored.
    projection_rank : int
        Number of opposite-mode PCA directions used in every rolling window.
        Must be positive and below that mode's dimension. It should cover the
        true opposite-mode factor rank. Both ranks are supplied using training
        information only and remain fixed throughout monitoring.
    horizon : int
        Number of future observations allowed, fixed before monitoring. The
        maximum procedure requires at least 2; eta=0.5 requires at least 3.
        Alarm probability refers to this whole horizon, not one time point.
    mode : {"row", "column"}, default "row"
        The column procedure applies the same equations to transposed matrices.
    power : float, default 8
        Power r >= 2 in accepted-manuscript (3.10), not the exponential
        transformation in arXiv v1. The value 8 reproduces the Gaussian
        simulation choice; its theory requires finite 32nd moments of both
        factors and errors. Power 2 still requires finite eighth moments.
    epsilon : float, default 0.05
        Positive smoothing in delta=epsilon+max(0,1-1/(2*beta)), with
        beta=log(monitored_dimension)/log(opposite_dimension*m). The resulting
        delta must be below one. A fixed epsilon does not guarantee finite
        sample calibration.
    statistic : {"maximum", "partial_sum"}, default "maximum"
        Maximum uses the manuscript's one-sided Gumbel-normalized y statistic.
        Partial sums use absolute cumulative sums and the eta-specific scaling.
    eta : float, default 0.25
        Partial-sum exponent in [0,1). Eta<0.5 uses horizon scaling; eta=0.5
        uses Darling--Erdos normalization; eta>0.5 uses the Renyi procedure.
        Ignored by the maximum procedure.
    delay : int, optional
        First eligible monitoring step for eta>0.5. Defaults to
        max(1,floor(log(horizon))), following the reference software where
        positive. It must not exceed the horizon. Partial sums include draws
        before this step. Not accepted for other procedures.
    alpha : float, default 0.05
        Requested horizon-wide asymptotic false-alarm level. Used to generate
        the Gumbel critical value if critical_value is not supplied.
    critical_value : float, optional
        On the normalized statistic scale: Gumbel for maximum and eta=0.5;
        weighted Brownian-supremum for eta<0.5; Brownian exponent 1-eta for
        Renyi. Required for partial sums with eta != 0.5. A supplied value can
        also be a separately calibrated finite-horizon Gaussian reference
        threshold; this does not establish exact finite-sample matrix-data
        false-alarm control. The caller owns its calibration.
    random_state : int, numpy.random.Generator, or None
        Independent local randomization. A supplied Generator is copied, so
        its state is not advanced. Each accepted observation consumes one
        standard normal draw unless noise is explicitly supplied to update.

    Notes
    -----
    At step tau, only observations tau+1,...,m+tau enter the window. Re-estimate
    opposite-mode PCA on that window, then recompute the projected second
    moment for ALL window observations using the new projection. Normalize
    its (rank+1)-st eigenvalue by its trace/monitored_dimension, multiply by
    monitored_dimension**(-delta), raise to power, and add independent N(0,1).
    The resulting draws are conditionally independent N(drift_tau,1), not
    exactly iid standard normal under a finite-sample null.

    Monitoring ends at its first strict boundary crossing, or on consuming the
    horizon. update then raises RuntimeError; update_many returns the consumed
    prefix and leaves later observations untouched. No automatic restart or
    re-estimation of the baseline rank occurs. The reported detection time is
    an alarm time, not an estimate corrected for detection delay.

    Supports changes that enlarge the joint pre/post loading span and newly
    appearing factors. Disappearing factors, zero baseline ranks, weak-factor
    extensions, and simultaneous row/column multiplicity calibration are not
    implemented. See docs/monitoring-notes.md for population assumptions.

    References
    ----------
    He, Y., Kong, X., Trapani, L. and Yu, L. (2024).
    Online Change-Point Detection for Matrix-Valued Time Series with
    Latent Two-Way Factor Structure. Annals of Statistics, 52, 1646-1670.
    https://doi.org/10.1214/24-AOS2410
    Uses the accepted-manuscript power transformation and equations
    (3.9)-(3.20), (3.28)-(3.32), not the earlier arXiv v1 transformation.
    State management is a software interface, not an additional estimator.
    """

    def __init__(
        self,
        training,
        rank,
        projection_rank,
        horizon,
        *,
        mode="row",
        power=8,
        epsilon=0.05,
        statistic="maximum",
        eta=0.25,
        delay=None,
        alpha=0.05,
        critical_value=None,
        random_state=None,
    ):
        training = as_series(training)
        if not isinstance(mode, str) or mode not in ("row", "column"):
            raise ValueError("mode must be 'row' or 'column'")
        if not isinstance(statistic, str) or statistic not in (
            "maximum",
            "partial_sum",
        ):
            raise ValueError("statistic must be 'maximum' or 'partial_sum'")
        self.mode = mode
        self.statistic = statistic
        self.training_length = len(training)
        self.shape = training.shape[1:]
        p, q = self.shape if mode == "row" else self.shape[::-1]
        self.rank = positive_int(rank, "rank")
        self.projection_rank = positive_int(projection_rank, "projection_rank")
        if self.rank >= p or self.projection_rank >= q:
            raise ValueError(
                "rank and projection_rank must be below their mode dimensions"
            )
        if self.rank + 1 > self.training_length * self.projection_rank:
            raise ValueError("window cannot support the monitored eigenvalue")
        self.horizon = positive_int(horizon, "horizon")
        self.power = finite_scalar(power, "power", minimum=2)
        self.epsilon = finite_scalar(epsilon, "epsilon", minimum=0)
        if self.epsilon == 0:
            raise ValueError("epsilon must be strictly positive")
        beta = np.log(p) / np.log(q * self.training_length)
        self.delta = float(self.epsilon + max(0, 1 - 1 / (2 * beta)))
        if self.delta >= 1:
            raise ValueError("epsilon and dimensions must give delta < 1")
        self.eta = finite_scalar(eta, "eta", minimum=0)
        if self.eta >= 1:
            raise ValueError("eta must lie in [0,1)")
        self.alpha = finite_scalar(alpha, "alpha", minimum=0)
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie strictly between zero and one")
        if statistic == "maximum" and self.horizon < 2:
            raise ValueError("maximum normalization requires horizon >= 2")
        if statistic == "partial_sum" and self.eta == 0.5 and self.horizon < 3:
            raise ValueError("eta=0.5 normalization requires horizon >= 3")
        renyi = statistic == "partial_sum" and self.eta > 0.5
        if delay is not None and not renyi:
            raise ValueError("delay applies only to partial sums with eta > 0.5")
        self.delay = (
            (
                max(1, int(np.log(self.horizon)))
                if delay is None
                else positive_int(delay, "delay")
            )
            if renyi
            else 1
        )
        if self.delay > self.horizon:
            raise ValueError("delay cannot exceed the monitoring horizon")
        if critical_value is None:
            if statistic == "partial_sum" and self.eta != 0.5:
                raise ValueError(
                    "critical_value is required for eta != 0.5; supply a calibrated "
                    "Brownian or finite-horizon Gaussian reference quantile"
                )
            self.critical_value = float(-np.log(-np.log1p(-self.alpha)))
            self.calibration = "asymptotic-gumbel"
        else:
            self.critical_value = finite_scalar(critical_value, "critical_value")
            if (
                statistic == "partial_sum"
                and self.eta != 0.5
                and self.critical_value <= 0
            ):
                raise ValueError(
                    "Brownian-scale critical_value must be strictly positive"
                )
            self.calibration = "caller-supplied"
        self._configure_normalizers()
        self._window = training.copy()
        self._rng = (
            deepcopy(random_state)
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        self._steps = []
        self._sum = (0, -np.inf)
        # Reject a zero-trace training sample before an unusable monitor is
        # handed back. No random draw or monitoring observation is consumed.
        self._window_eigenvalue(self._window)

    def _configure_normalizers(self):
        if self.statistic == "maximum":
            root = np.sqrt(2 * np.log(self.horizon))
            self._b = float(
                root - (np.log(np.log(self.horizon)) + np.log(4 * np.pi)) / (2 * root)
            )
            self._a = float(self._b / (1 + self._b**2))
        elif self.eta == 0.5:
            nested = np.log(np.log(self.horizon))
            self._a = float(np.sqrt(2 * nested))
            self._b = float(2 * nested + 0.5 * np.log(nested) - 0.5 * np.log(np.pi))
        else:
            self._a, self._b = 1.0, 0.0

    @property
    def n_processed(self):
        """Number of consumed post-training observations."""
        return len(self._steps)

    @property
    def steps(self):
        """Immutable sequence of accepted monitoring diagnostics."""
        return tuple(self._steps)

    @property
    def alarm_step(self):
        """First alarm's one-based monitoring index, or None."""
        return (
            self._steps[-1].monitoring_index
            if self._steps and self._steps[-1].alarm
            else None
        )

    @property
    def alarm_observation(self):
        """First alarm's one-based index in training + monitoring, or None."""
        return (
            self.training_length + self.alarm_step
            if self.alarm_step is not None
            else None
        )

    @property
    def stop_reason(self):
        """'alarm', 'horizon', or None while monitoring remains active."""
        if self.alarm_step is not None:
            return "alarm"
        return "horizon" if self.n_processed == self.horizon else None

    @property
    def is_stopped(self):
        return self.stop_reason is not None

    @property
    def window(self):
        """Detached copy of the current chronological trailing window."""
        return self._window.copy()

    def _window_eigenvalue(self, window):
        scale = float(np.max(np.abs(window)))
        if scale == 0:
            raise ValueError("projected window has zero trace; monitoring is undefined")
        work = window / scale
        if self.mode == "column":
            work = work.transpose(0, 2, 1)
        _, p, q = work.shape
        gram = np.einsum("tij,tik->jk", work, work) / (len(work) * p)
        _, vectors = np.linalg.eigh((gram + gram.T) * 0.5)
        projection = vectors[:, -self.projection_rank :]
        projected = work @ projection
        moment = np.einsum("tij,tkj->ik", projected, projected) / (len(work) * q)
        spectrum = np.maximum(np.linalg.eigvalsh((moment + moment.T) * 0.5), 0)
        trace = float(np.trace(moment))
        if not np.isfinite(trace) or trace <= 0:
            raise ValueError("projected window has zero trace; monitoring is undefined")
        eigenvalue = float(spectrum[-self.rank - 1])
        normalized = float(p ** (1 - self.delta) * (eigenvalue / trace))
        if normalized == 0:
            log_drift = -np.inf
        else:
            with np.errstate(over="ignore"):
                log_drift = float(self.power * np.log(normalized))
        return scale, eigenvalue, trace, normalized, log_drift

    def _score(self, randomized, partial_sum, step):
        if self.statistic == "maximum":
            return _scale_log(_add_logs(randomized, _signed_log(-self._b)), 1 / self._a)
        absolute_sum = (abs(partial_sum[0]), partial_sum[1])
        if self.eta == 0.5:
            return _add_logs(
                _scale_log(absolute_sum, self._a / np.sqrt(step)),
                _signed_log(-self._b),
            )
        reference = self.horizon if self.eta < 0.5 else self.delay
        sign, logabs = absolute_sum
        return sign, float(
            logabs + (self.eta - 0.5) * np.log(reference) - self.eta * np.log(step)
        )

    def update(self, matrix, *, noise=None):
        """Consume one matrix; rejected inputs do not advance the state or RNG.

        ``noise`` optionally supplies the standardized Gaussian draw. It is
        intended for reproducible external randomization and equation oracles;
        supplying deterministic/non-Gaussian values forfeits Gaussian reference
        calibration. An explicit value does not consume the monitor's RNG.

        References
        ----------
        He, Y., Kong, X., Trapani, L. and Yu, L. (2024).
        Online Change-Point Detection for Matrix-Valued Time Series with
        Latent Two-Way Factor Structure. Annals of Statistics, 52, 1646-1670.
        https://doi.org/10.1214/24-AOS2410
        Uses the accepted-manuscript power transformation and equations
        (3.9)-(3.20), (3.28)-(3.32), not the earlier arXiv v1 transformation.
        State management is a software interface, not an additional estimator.
        """
        if self.is_stopped:
            raise RuntimeError(f"monitoring already stopped: {self.stop_reason}")
        raw = np.asarray(matrix)
        if raw.ndim != 2 or raw.shape != self.shape:
            raise ValueError("matrix dimensions must match the training observations")
        observation = as_series(raw[None], min_samples=1)[0]
        supplied = None if noise is None else finite_scalar(noise, "noise")
        candidate = np.concatenate((self._window[1:], observation[None]))
        scale, eigenvalue, trace, normalized, log_drift = self._window_eigenvalue(
            candidate
        )
        # Nothing below can fail for validated real scalars. Draw only after
        # the complete data-dependent calculation has succeeded.
        gaussian = float(self._rng.standard_normal()) if supplied is None else supplied
        drift_log = (0, -np.inf) if log_drift == -np.inf else (1, log_drift)
        randomized = _add_logs(drift_log, _signed_log(gaussian))
        partial_sum = _add_logs(self._sum, randomized)
        index = self.n_processed + 1
        score = self._score(randomized, partial_sum, index)
        eligible = index >= self.delay
        alarm = bool(eligible and _greater(score, _signed_log(self.critical_value)))
        displayed = [
            _display(value) for value in (drift_log, randomized, partial_sum, score)
        ]
        result = MonitorStep(
            monitoring_index=index,
            observation_index=self.training_length + index,
            window_start_index=index + 1,
            data_scale=scale,
            eigenvalue_scaled=eigenvalue,
            trace_scaled=trace,
            normalized_eigenvalue=normalized,
            log_drift=log_drift,
            drift=displayed[0][0],
            noise=gaussian,
            randomized_value=displayed[1][0],
            partial_sum=displayed[2][0],
            statistic=displayed[3][0],
            critical_value=self.critical_value,
            eligible=eligible,
            alarm=alarm,
            saturated=any(item[1] for item in displayed),
        )
        self._window = candidate
        self._sum = partial_sum
        self._steps.append(result)
        return result

    def update_many(self, observations, *, noise=None):
        """Consume an array's prefix until the first alarm or horizon exhaustion.

        The return tuple contains only consumed observations. Validation is
        per observation, matching successive update calls: a bad later value
        does not roll back earlier updates, and values after stopping are not
        inspected. An empty batch or an already stopped monitor returns ().

        References
        ----------
        He, Y., Kong, X., Trapani, L. and Yu, L. (2024).
        Online Change-Point Detection for Matrix-Valued Time Series with
        Latent Two-Way Factor Structure. Annals of Statistics, 52, 1646-1670.
        https://doi.org/10.1214/24-AOS2410
        Uses the accepted-manuscript power transformation and equations
        (3.9)-(3.20), (3.28)-(3.32), not the earlier arXiv v1 transformation.
        State management is a software interface, not an additional estimator.
        """
        if self.is_stopped:
            return ()
        batch = np.asarray(observations)
        if batch.ndim != 3 or batch.shape[1:] != self.shape:
            raise ValueError(
                "observations must be a time-first array matching training"
            )
        if noise is not None:
            draws = np.asarray(noise)
            if draws.ndim != 1 or len(draws) != len(batch):
                raise ValueError("noise must have one entry per supplied observation")
        results = []
        for index, matrix in enumerate(batch):
            if self.is_stopped:
                break
            results.append(
                self.update(matrix, noise=None if noise is None else draws[index])
            )
        return tuple(results)

    def state_dict(self):
        """Return detached, JSON-compatible state for exact continuation.

        Includes the trailing window, all diagnostics, configuration and local
        bit-generator state. No executable objects or pickle are needed.
        Infinite log drift is encoded as a string so strict JSON is supported.
        """
        configuration = dict(
            rank=self.rank,
            projection_rank=self.projection_rank,
            horizon=self.horizon,
            mode=self.mode,
            power=self.power,
            epsilon=self.epsilon,
            statistic=self.statistic,
            eta=self.eta,
            delay=(
                self.delay
                if self.statistic == "partial_sum" and self.eta > 0.5
                else None
            ),
            alpha=self.alpha,
            critical_value=self.critical_value,
        )
        return _encode(
            dict(
                schema_version=1,
                configuration=configuration,
                calibration=self.calibration,
                window=self._window,
                steps=[asdict(step) for step in self._steps],
                rng_state=deepcopy(self._rng.bit_generator.state),
            )
        )

    @classmethod
    def from_state(cls, state):
        """Restore a state_dict snapshot without consuming observations or draws.

        Snapshots are checked for format and basic consistency. They should
        come from this implementation; this is not a validator of scientific
        provenance for manually constructed or modified histories.
        """
        try:
            state = deepcopy(state)
            if not isinstance(state, dict) or state["schema_version"] != 1:
                raise ValueError("unsupported monitoring state schema")
            rng_state = state["rng_state"]
            kind = rng_state["bit_generator"]
            if kind not in ("PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64"):
                raise ValueError("unsupported random bit generator in state")
            bit_generator = getattr(np.random, kind)()
            bit_generator.state = rng_state
            result = cls(
                state["window"],
                **state["configuration"],
                random_state=np.random.Generator(bit_generator),
            )
            if state["calibration"] not in ("asymptotic-gumbel", "caller-supplied"):
                raise ValueError("unknown critical-value calibration in state")
            result.calibration = state["calibration"]
            steps = state["steps"]
            if not isinstance(steps, list) or len(steps) > result.horizon:
                raise ValueError("inconsistent number of monitoring steps")
            for index, entry in enumerate(steps, start=1):
                if result.is_stopped:
                    raise ValueError("state has observations after monitoring stopped")
                if entry["log_drift"] == "-Infinity":
                    entry["log_drift"] = -np.inf
                elif entry["log_drift"] == "Infinity":
                    entry["log_drift"] = np.inf
                step = MonitorStep(**entry)
                if (
                    step.monitoring_index != index
                    or step.observation_index != result.training_length + index
                    or step.window_start_index != index + 1
                ):
                    raise ValueError("inconsistent monitoring indices in state")
                if np.isnan(step.log_drift) or not np.isfinite(step.noise):
                    raise ValueError("invalid randomization history in state")
                numeric_fields = (
                    step.data_scale,
                    step.eigenvalue_scaled,
                    step.trace_scaled,
                    step.normalized_eigenvalue,
                    step.drift,
                    step.randomized_value,
                    step.partial_sum,
                    step.statistic,
                    step.critical_value,
                )
                if (
                    not all(np.isfinite(value) for value in numeric_fields)
                    or step.data_scale <= 0
                    or step.trace_scaled <= 0
                    or step.eigenvalue_scaled < 0
                    or step.normalized_eigenvalue < 0
                    or step.drift < 0
                    or step.critical_value != result.critical_value
                    or not all(
                        isinstance(value, bool)
                        for value in (step.eligible, step.alarm, step.saturated)
                    )
                ):
                    raise ValueError("invalid monitoring diagnostics in state")
                drift = (
                    (0, -np.inf) if step.log_drift == -np.inf else (1, step.log_drift)
                )
                randomized = _add_logs(drift, _signed_log(step.noise))
                result._sum = _add_logs(result._sum, randomized)
                expected_alarm = index >= result.delay and _greater(
                    result._score(randomized, result._sum, index),
                    _signed_log(result.critical_value),
                )
                if (
                    step.eligible != (index >= result.delay)
                    or step.alarm != expected_alarm
                ):
                    raise ValueError("inconsistent stopping decision in state")
                result._steps.append(step)
            return result
        except (KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError("invalid monitoring state") from exc
