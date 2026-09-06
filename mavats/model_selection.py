"""Chronological out-of-sample evaluation, including all preprocessing.

References
----------
Tashman (2000), *Out-of-Sample Tests of Forecasting Accuracy: An Analysis and
Review*, https://doi.org/10.1016/S0169-2070(00)00065-0, discusses rolling origins,
recalibration and fixed/rolling windows. This module implements that evaluation
design; it does not claim a new matrix-specific estimator or an iid error test.
"""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ._validation import as_series, positive_int
from .metrics import mean_squared_error


@dataclass
class ForecastEvaluation:
    """One entry per forecast origin; errors have shape (origin, horizon)."""

    origins: np.ndarray
    errors: np.ndarray
    fit_seconds: np.ndarray
    predictions: np.ndarray

    @property
    def mse(self):
        """Aggregate over forecast origins and horizons."""
        scale = np.max(self.errors)
        if scale == 0 or np.isinf(scale):
            return float(scale)
        return float(np.mean(self.errors / scale) * scale)


def rolling_forecast(X, fit, *, initial_train_size, horizon=1, step=1, window=None):
    """Evaluate a model factory on expanding or fixed chronological windows.

    ``fit(training_array)`` must return an object with ``forecast(horizon)``.
    Preprocessing and tuning belong inside the factory and can see only the
    supplied training data. Overlapping horizons are allowed and have dependent
    errors; this function does not attach iid standard errors to them.

    References
    ----------
    Tashman (2000), *Out-of-Sample Tests of Forecasting Accuracy: An Analysis
    and Review*, https://doi.org/10.1016/S0169-2070(00)00065-0. The callable
    factory and arbitrary matrix/tensor output shapes are software extensions
    of the rolling-origin evaluation design.
    """
    X = as_series(X, ndim=None)
    initial = positive_int(initial_train_size, "initial_train_size")
    horizon = positive_int(horizon, "horizon")
    step = positive_int(step, "step")
    if initial + horizon > len(X):
        raise ValueError(
            "series must contain an initial training window and full test horizon"
        )
    if window is not None:
        window = positive_int(window, "window")
        if window > initial:
            raise ValueError("window cannot exceed initial_train_size")
    origins = np.arange(initial, len(X) - horizon + 1, step)
    predictions, errors, timings = [], [], []
    for origin in origins:
        start = 0 if window is None else origin - window
        training = X[start:origin].copy()
        begin = perf_counter()
        model = fit(training)
        timings.append(perf_counter() - begin)
        raw_prediction = np.asarray(model.forecast(horizon))
        if np.iscomplexobj(raw_prediction) or raw_prediction.dtype.kind not in "biuf":
            raise ValueError("forecast must contain real numeric values")
        prediction = np.asarray(raw_prediction, dtype=float)
        actual = X[origin : origin + horizon]
        if prediction.shape != actual.shape or not np.isfinite(prediction).all():
            raise ValueError(
                "forecast must be finite and match (horizon, *spatial_shape)"
            )
        predictions.append(prediction)
        errors.append([mean_squared_error(y, yp) for y, yp in zip(actual, prediction)])
    return ForecastEvaluation(
        origins, np.asarray(errors), np.asarray(timings), np.asarray(predictions)
    )
