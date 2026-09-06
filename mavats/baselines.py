"""Unstructured and naive forecasting baselines for honest comparisons.

References
----------
Chen, Xiao and Yang (2021), *Autoregressive Models for Matrix-Valued Time
Series*, https://doi.org/10.1016/j.jeconom.2020.07.015, motivates the vector
autoregression comparison. Hoerl and Kennard (1970), *Ridge Regression: Biased
Estimation for Nonorthogonal Problems*,
https://doi.org/10.1080/00401706.1970.10488634, supplies the optional penalty.
Hyndman and Koehler (2006), *Another Look at Measures of Forecast Accuracy*,
https://doi.org/10.1016/j.ijforecast.2006.03.001, discusses naive/mean benchmarks.
These are conventional baselines, not additional matrix-specific estimators.
"""

from dataclasses import dataclass

import numpy as np

from ._validation import as_series, finite_scalar, positive_int


@dataclass
class VARResult:
    """Vector autoregression fitted to column-vectorized matrices.

    ``coefficients[lag]`` multiplies a column vector. Dense storage and least
    squares scale with the product of the spatial dimensions.
    """

    coefficients: np.ndarray
    intercept: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    history: np.ndarray
    rank: int
    singular_values: np.ndarray
    design_scale: float = 1.0

    @property
    def order(self):
        return len(self.coefficients)

    def forecast(self, steps, history=None):
        """Recursively forecast without using future observations."""
        steps = positive_int(steps, "steps")
        history = (
            self.history
            if history is None
            else as_series(history, min_samples=self.order)
        )
        if history.shape[1:] != self.history.shape[1:]:
            raise ValueError("history spatial dimensions differ from the fitted series")
        m, n = history.shape[1:]
        state = [x.ravel(order="F").copy() for x in history[-self.order :]]
        result = []
        for _ in range(steps):
            scale = (
                max(
                    np.max(np.abs(self.intercept)),
                    max(np.max(np.abs(x)) for x in state[-self.order :]),
                )
                or 1.0
            )
            with np.errstate(over="ignore", invalid="ignore"):
                value = self.intercept / scale
                for lag, coefficient in enumerate(self.coefficients, 1):
                    value += coefficient @ (state[-lag] / scale)
                value *= scale
            if not np.isfinite(value).all():
                raise FloatingPointError("VAR forecast exceeds floating-point range")
            state.append(value)
            result.append(value.reshape(m, n, order="F"))
        return np.asarray(result)

    @property
    def spectral_radius(self):
        p, d, _ = self.coefficients.shape
        companion = np.zeros((p * d, p * d))
        companion[:d] = np.concatenate(self.coefficients, axis=1)
        companion[d:, :-d] = np.eye((p - 1) * d)
        return float(max(abs(np.linalg.eigvals(companion))))


def fit_var(X, *, order=1, intercept=True, ridge=0.0):
    """Fit unrestricted VAR(p) using SVD least squares.

    Ridge minimizes ``sum(error**2) + ridge * sum(coefficients**2)``;
    the intercept is not penalized. It is implemented by augmented least
    squares, avoiding squared condition numbers from normal equations.
    A common scale is removed before centering and fitting. Returned
    ``singular_values`` describe that scaled augmented design; multiply by
    ``design_scale`` to obtain the original units when representable.

    References
    ----------
    Chen, Xiao and Yang (2021), *Autoregressive Models for Matrix-Valued Time
    Series*, https://doi.org/10.1016/j.jeconom.2020.07.015 (vector baseline).
    Optional ridge uses the quadratic penalty of Hoerl and Kennard (1970),
    *Ridge Regression: Biased Estimation for Nonorthogonal Problems*,
    https://doi.org/10.1080/00401706.1970.10488634. The matrix flattening,
    unpenalized intercept and scale-safe SVD are implementation choices.
    """
    order = positive_int(order, "order")
    X = as_series(X, min_samples=order + 1)
    ridge = finite_scalar(ridge, "ridge", minimum=0)
    if not isinstance(intercept, (bool, np.bool_)):
        raise ValueError("intercept must be boolean")
    t, m, n = X.shape
    scale = float(max(np.max(np.abs(X)), np.sqrt(ridge))) or 1.0
    flat = (X / scale).transpose(0, 2, 1).reshape(t, m * n)
    design = np.concatenate(
        [flat[order - lag : t - lag] for lag in range(1, order + 1)], axis=1
    )
    target = flat[order:]
    dx = design.mean(axis=0) if intercept else np.zeros(design.shape[1])
    dy = target.mean(axis=0) if intercept else np.zeros(target.shape[1])
    a, b = design - dx, target - dy
    if ridge:
        a = np.concatenate((a, (np.sqrt(ridge) / scale) * np.eye(a.shape[1])))
        b = np.concatenate((b, np.zeros((design.shape[1], b.shape[1]))))
    beta, _, rank, singular = np.linalg.lstsq(a, b, rcond=None)
    offset = dy - dx @ beta
    with np.errstate(over="ignore", invalid="ignore"):
        fitted = ((design @ beta + offset) * scale).reshape(-1, n, m).transpose(0, 2, 1)
        offset = offset * scale
        residuals = X[order:] - fitted
    if not all(np.isfinite(value).all() for value in (beta, fitted, offset, residuals)):
        raise FloatingPointError(
            "VAR fitted parameters or residuals exceed floating-point range"
        )
    coefficients = beta.reshape(order, m * n, m * n).transpose(0, 2, 1).copy()
    return VARResult(
        coefficients,
        offset,
        fitted,
        residuals,
        X[-order:].copy(),
        int(rank),
        singular,
        scale,
    )


@dataclass
class NaiveResult:
    """Constant forecasts using the last matrix, training mean, or zero."""

    value: np.ndarray

    def forecast(self, steps, history=None):
        if history is not None:
            raise ValueError("refit this baseline to update its forecast origin")
        return np.repeat(self.value[None], positive_int(steps, "steps"), axis=0)


def fit_naive(X, *, strategy="last"):
    """Fit ``last`` (random walk), ``mean``, or ``zero`` baseline.

    References
    ----------
    Hyndman and Koehler (2006), *Another Look at Measures of Forecast Accuracy*,
    https://doi.org/10.1016/j.ijforecast.2006.03.001, discusses random-walk and
    historical-mean benchmarks. Applying them entrywise to matrices/tensors
    and the fixed zero benchmark are elementary comparison conventions, not
    distinct published matrix estimators.
    """
    X = as_series(X, min_samples=1, ndim=None)
    if strategy == "last":
        value = X[-1].copy()
    elif strategy == "mean":
        scale = np.max(np.abs(X)) or 1.0
        value = (X / scale).mean(axis=0) * scale
    elif strategy == "zero":
        value = np.zeros(X.shape[1:])
    else:
        raise ValueError("strategy must be 'last', 'mean', or 'zero'")
    return NaiveResult(value)
