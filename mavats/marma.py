"""Rank-one matrix autoregressive moving-average models, with MINUS MA terms.

References
----------
Tsay, R. S. (2024), "Matrix-Variate Time Series Analysis: A Brief Review and
Some New Developments", International Statistical Review 92, 246--262,
https://doi.org/10.1111/insr.12558, equations (4)--(5), Sections 2.1, 2.4--2.5.
This module fits the recursive conditional objective, not a projected vector
ARMA estimate. Equation (30)'s missing quadratic factor 1/2 is corrected from
the Gaussian density. Optimization, initializers and numerical tolerances are
package choices. General rank-r and seasonal models and inference are absent.
"""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from ._validation import as_series, finite_scalar, positive_int
from .autoregression import nearest_kronecker_product


def _finite(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf" or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real numeric")
    result = np.asarray(raw, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return result.copy()


def _represented(value, name):
    if not np.isfinite(value).all():
        raise FloatingPointError(f"{name} exceeds floating-point range; rescale data")
    return value


def _multiply(value, *scales):
    """Delay scalar exponents until the represented product is formed."""
    mantissa, exponent = np.frexp(value)
    for scale in scales:
        part, power = np.frexp(scale)
        mantissa *= part
        exponent = exponent + power
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.ldexp(mantissa, exponent)


def _pair(left, right):
    scale = float(np.max(np.abs(left)))
    if not scale or not np.any(right):
        left = np.zeros_like(left)
        left[0, 0] = 1
        return left, np.zeros_like(right)
    unit = left / scale
    norm = float(np.linalg.norm(unit))
    left = unit / norm
    right = _represented(_multiply(right, scale, norm), "coefficient gauge")
    if left.flat[np.argmax(np.abs(left))] < 0:
        left, right = -left, -right
    return left, right


def _chol(covariance, name):
    scale = float(np.max(np.abs(covariance)))
    if not scale:
        raise ValueError(f"{name} must be positive definite")
    work = covariance / scale
    if not np.allclose(work, work.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    try:
        root = np.linalg.cholesky((work + work.T) * 0.5)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name} must be positive definite") from exc
    return root * np.sqrt(scale)


@dataclass(frozen=True)
class MARMAParameters:
    """Parameters of ``X=C+sum A Xpast B.T+E-sum L Epast R.T``.

    Each coefficient array has a leading lag axis; None or an empty sequence
    denotes zero lags. ``intercept`` is a required finite matrix, even when
    zero, and determines the spatial dimensions. Row/column covariances are
    either both omitted or both positive definite. Arrays are copied. No
    stability, invertibility or structural identification is presumed.

    References
    ----------
    Tsay (2024), "Matrix-Variate Time Series Analysis: A Brief Review and Some
    New Developments", equations (4)--(5), https://doi.org/10.1111/insr.12558.
    Parameter/result methods inherit this model reference.
    """

    ar_left: object
    ar_right: object
    ma_left: object
    ma_right: object
    intercept: np.ndarray
    row_covariance: np.ndarray | None = None
    column_covariance: np.ndarray | None = None

    def __post_init__(self):
        raw = np.asarray(self.intercept)
        if raw.ndim != 2 or not all(raw.shape):
            raise ValueError("intercept must be a nonempty matrix")
        m, n = raw.shape
        object.__setattr__(self, "intercept", _finite(raw, (m, n), "intercept"))
        for left_name, right_name in (("ar_left", "ar_right"), ("ma_left", "ma_right")):
            for name, dim in ((left_name, m), (right_name, n)):
                value = getattr(self, name)
                raw = np.asarray([] if value is None else value)
                if not raw.size:
                    value = np.empty((0, dim, dim))
                else:
                    if raw.ndim == 2:
                        raw = raw[None]
                    if raw.ndim != 3:
                        raise ValueError(f"{name} needs a leading lag axis")
                    value = _finite(raw, (len(raw), dim, dim), name)
                object.__setattr__(self, name, value)
            if len(getattr(self, left_name)) != len(getattr(self, right_name)):
                raise ValueError("left and right coefficients need matching lag counts")
        if (self.row_covariance is None) != (self.column_covariance is None):
            raise ValueError("supply both row and column covariances")
        if self.row_covariance is not None:
            for name, dim in (("row_covariance", m), ("column_covariance", n)):
                value = _finite(getattr(self, name), (dim, dim), name)
                _chol(value, name)
                object.__setattr__(self, name, value)

    @property
    def ar_order(self):
        return len(self.ar_left)

    @property
    def ma_order(self):
        return len(self.ma_left)

    @property
    def conditioning_length(self):
        return max(self.ar_order, self.ma_order)


def _canonical(parameters):
    if not isinstance(parameters, MARMAParameters):
        raise TypeError("parameters must be MARMAParameters")
    arrays = []
    for left, right in (
        (parameters.ar_left, parameters.ar_right),
        (parameters.ma_left, parameters.ma_right),
    ):
        pairs = [_pair(a, b) for a, b in zip(left, right)]
        arrays.extend(
            (
                np.array([a for a, _ in pairs]).reshape(left.shape),
                np.array([b for _, b in pairs]).reshape(right.shape),
            )
        )
    return MARMAParameters(
        *arrays,
        parameters.intercept,
        parameters.row_covariance,
        parameters.column_covariance,
    )


def _raw(parameters):
    return (
        parameters.ar_left,
        parameters.ar_right,
        parameters.ma_left,
        parameters.ma_right,
        parameters.intercept,
    )


def _innovations(X, arrays, return_fitted=False):
    A, B, L, R, constant = arrays
    t0 = max(len(A), len(L))
    errors = np.zeros_like(X)
    fitted = np.full_like(X, np.nan) if return_fitted else None
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(t0, len(X)):
            prediction = constant.copy()
            for lag, (a, b) in enumerate(zip(A, B), 1):
                prediction += a @ X[t - lag] @ b.T
            for lag, (a, b) in enumerate(zip(L, R), 1):
                prediction -= a @ errors[t - lag] @ b.T
            errors[t] = X[t] - prediction
            if return_fitted:
                fitted[t] = prediction
    _represented(errors, "recursive innovations")
    return (errors, fitted) if return_fitted else errors


def _gaussian_loss(errors, row_root, col_root):
    white = np.linalg.solve(row_root, errors)
    white = np.linalg.solve(col_root, white.transpose(0, 2, 1)).transpose(0, 2, 1)
    m, n = errors.shape[1:]
    loss = 0.5 * (
        m * n * np.log(2 * np.pi)
        + 2 * n * np.log(np.diag(row_root)).sum()
        + 2 * m * np.log(np.diag(col_root)).sum()
        + np.mean(np.sum(white * white, axis=(1, 2)))
    )
    return float(loss), white


@dataclass(frozen=True)
class MARMADiagnostics:
    """Numerical full-polynomial root checks, not structural identification.

    ``is_stable`` and ``is_invertible`` require companion radii below
    ``1-tolerance``. Roots near the boundary are not certified. Gauge choices
    do not establish left coprimeness or minimal ARMA orders (Tsay §2.1).
    """

    ar_eigenvalues: np.ndarray
    ma_eigenvalues: np.ndarray
    ar_spectral_radius: float
    ma_spectral_radius: float
    is_stable: bool
    is_invertible: bool
    tolerance: float
    structural_identification_certified: bool = False


def _operators(parameters, maximum, order=None):
    m, n = parameters.intercept.shape
    d = m * n
    if d * max(1, parameters.conditioning_length if order is None else order) > maximum:
        raise ValueError("dense operator/companion exceeds max_dense_dimension")
    result = []
    for left, right in (
        (parameters.ar_left, parameters.ar_right),
        (parameters.ma_left, parameters.ma_right),
    ):
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.array([np.kron(b, a) for a, b in zip(left, right)]).reshape(
                -1, d, d
            )
        result.append(_represented(values, "dense coefficients"))
    return result


def _roots(operators):
    order, d, _ = operators.shape
    if not order:
        return np.empty(0, dtype=complex)
    companion = np.zeros((d * order, d * order))
    companion[:d] = np.concatenate(operators, axis=1)
    if order > 1:
        companion[d:, :-d] = np.eye(d * (order - 1))
    return np.linalg.eigvals(companion)


def _radii(arrays):
    radii = []
    for left, right in ((arrays[0], arrays[1]), (arrays[2], arrays[3])):
        if not len(left):
            radii.append(0.0)
        elif len(left) == 1:
            # Exact Kronecker eigenvalue products, not separate mode bounds.
            radii.append(
                float(
                    max(abs(np.linalg.eigvals(left[0])))
                    * max(abs(np.linalg.eigvals(right[0])))
                )
            )
        else:
            operators = np.array([np.kron(b, a) for a, b in zip(left, right)])
            radii.append(float(max(abs(_roots(operators)))))
    return np.array(radii)


def _admissible_start(parameters, margin):
    arrays = [x.copy() for x in _raw(parameters)[:4]]
    for mode, radius in enumerate(_radii(arrays)):
        factor = min(1.0, 0.9 * (1 - margin) / radius) if radius else 1.0
        for lag in range(len(arrays[2 * mode + 1])):
            arrays[2 * mode + 1][lag] *= factor ** (lag + 1)
    return MARMAParameters(
        *arrays,
        parameters.intercept,
        parameters.row_covariance,
        parameters.column_covariance,
    )


def marma_diagnostics(parameters, *, tolerance=1e-8, max_dense_dimension=256):
    """Check both complete AR and minus-MA companion polynomials.

    References
    ----------
    Tsay (2024), "Matrix-Variate Time Series Analysis: A Brief Review and Some
    New Developments", Sections 2.1 and 2.4,
    https://doi.org/10.1111/insr.12558. Companion eigenvalues are reciprocals
    of polynomial roots. This tolerance-based numerical diagnostic does not
    certify minimality, left coprimeness or identify cancelling ARMA factors.
    """
    parameters = _canonical(parameters)
    maximum = positive_int(max_dense_dimension, "max_dense_dimension")
    tolerance = finite_scalar(tolerance, "tolerance", minimum=0)
    if tolerance >= 1:
        raise ValueError("tolerance must be below one")
    ar, ma = (_roots(x) for x in _operators(parameters, maximum))
    ar_radius = float(max(abs(ar), default=0))
    ma_radius = float(max(abs(ma), default=0))
    return MARMADiagnostics(
        ar,
        ma,
        ar_radius,
        ma_radius,
        ar_radius < 1 - tolerance,
        ma_radius < 1 - tolerance,
        tolerance,
    )


def _forecast_covariance(parameters, steps, maximum, covariance=None):
    steps = positive_int(steps, "steps")
    maximum = positive_int(maximum, "max_dense_dimension")
    ar, ma = _operators(parameters, maximum)
    d = parameters.intercept.size
    if covariance is None:
        if parameters.row_covariance is None:
            raise ValueError("innovation covariance is unavailable")
        u, v = parameters.row_covariance, parameters.column_covariance
        us, vs = np.max(abs(u)), np.max(abs(v))
        covariance = _represented(
            _multiply(np.kron(v / vs, u / us), us, vs), "innovation covariance"
        )
        if np.any(np.diag(covariance) <= 0):
            raise FloatingPointError(
                "positive innovation variances underflow in dense covariance"
            )
    impulses = [np.eye(d)]
    for h in range(1, steps):
        value = -ma[h - 1].copy() if h <= len(ma) else np.zeros((d, d))
        for lag in range(1, min(h, len(ar)) + 1):
            value += ar[lag - 1] @ impulses[h - lag]
        impulses.append(_represented(value, "forecast impulse response"))
    result, total = [], np.zeros((d, d))
    for impulse in impulses:
        with np.errstate(over="ignore", invalid="ignore"):
            total = total + impulse @ covariance @ impulse.T
        result.append(_represented(total.copy(), "forecast covariance"))
    return np.array(result)


@dataclass
class MARMAFilterResult:
    """Conditional innovations and plug-in forecasts for the module's model.

    Residuals have the full input length: the conditioning prefix is zero,
    not estimated noise. Fitted values there are NaN. ``log_likelihood`` scores
    only the suffix, with all Gaussian constants, if covariance is supplied.
    Forecast covariances condition on the fixed initialization and parameters;
    they omit parameter and presample-state uncertainty.
    """

    parameters: MARMAParameters
    residuals: np.ndarray
    fitted_values: np.ndarray
    conditioning_mask: np.ndarray
    log_likelihood: float | None
    history: np.ndarray = field(repr=False)

    def forecast(self, steps):
        """Recurse using past innovations and zero future innovations (Tsay §2.4)."""
        steps = positive_int(steps, "steps")
        parameters = self.parameters
        p, q = parameters.ar_order, parameters.ma_order
        if not p and not q:
            return np.repeat(parameters.intercept[None], steps, axis=0)
        values = [x.copy() for x in self.history[-max(p, 1) :]]
        noises = [x.copy() for x in self.residuals[-max(q, 1) :]]
        predictions = []
        for _ in range(steps):
            scales = [np.max(abs(parameters.intercept))]
            if p:
                scales.extend(np.max(abs(x)) for x in values[-p:])
            if q:
                scales.extend(np.max(abs(x)) for x in noises[-q:])
            scale = max(scales) or 1.0
            value = parameters.intercept / scale
            with np.errstate(over="ignore", invalid="ignore"):
                for lag, (a, b) in enumerate(
                    zip(parameters.ar_left, parameters.ar_right), 1
                ):
                    value = value + a @ (values[-lag] / scale) @ b.T
                for lag, (a, b) in enumerate(
                    zip(parameters.ma_left, parameters.ma_right), 1
                ):
                    value = value - a @ (noises[-lag] / scale) @ b.T
                value = value * scale
            value = _represented(value, "MARMA forecast")
            predictions.append(value)
            values.append(value)
            noises.append(np.zeros_like(value))
        return np.array(predictions)

    def forecast_covariance(self, steps, *, max_dense_dimension=256):
        """Dense horizon covariances from full impulse responses (Tsay §2.4)."""
        return _forecast_covariance(self.parameters, steps, max_dense_dimension)


def filter_marma(X, parameters):
    """Filter the complete supplied history using the paper's zero-prefix rule.

    For t0=max(p,q), condition on X[:t0], set E[:t0]=0 and evaluate only
    X[t0:]. This is a conditional filter, not exact stationary/Kalman filtering.
    A new call restarts the conditioning convention; do not filter chunks
    separately to continue the same innovation state. Future observations
    never affect an earlier innovation for fixed parameters.

    References
    ----------
    Tsay (2024), "Matrix-Variate Time Series Analysis: A Brief Review and Some
    New Developments", equations (4), (30), Section 2.5,
    https://doi.org/10.1111/insr.12558. The Gaussian quadratic term uses the
    necessary factor 1/2 omitted in printed equation (30).
    """
    parameters = _canonical(parameters)
    X = as_series(X, min_samples=parameters.conditioning_length + 1)
    if X.shape[1:] != parameters.intercept.shape:
        raise ValueError("X and parameter spatial dimensions differ")
    scale = max(np.max(abs(X)), np.max(abs(parameters.intercept))) or 1.0
    arrays = (*_raw(parameters)[:4], parameters.intercept / scale)
    errors, prediction = _innovations(X / scale, arrays, return_fitted=True)
    residuals = _represented(_multiply(errors, scale), "physical innovations")
    mask = np.arange(len(X)) < parameters.conditioning_length
    fitted = _multiply(prediction, scale)
    if parameters.conditioning_length == 0:
        fitted = np.broadcast_to(parameters.intercept, X.shape).copy()
    _represented(fitted[~mask], "conditional fitted values")
    fitted[mask] = np.nan
    likelihood = None
    if parameters.row_covariance is not None:
        # Roots share the data scale, avoiding covariance/data_scale**2.
        root_u = _chol(parameters.row_covariance, "row covariance") / np.sqrt(scale)
        root_v = _chol(parameters.column_covariance, "column covariance") / np.sqrt(
            scale
        )
        loss, _ = _gaussian_loss(errors[~mask], root_u, root_v)
        likelihood = -(loss + X.shape[1] * X.shape[2] * np.log(scale)) * (~mask).sum()
        if not np.isfinite(likelihood):
            raise FloatingPointError(
                "conditional likelihood exceeds floating-point range"
            )
    return MARMAFilterResult(parameters, residuals, fitted, mask, likelihood, X.copy())


class _ParameterChart:
    """Fixed nonzero left pivots and a Cholesky covariance gauge."""

    def __init__(self, parameters, fit_intercept=True, covariance=False):
        parameters = _canonical(parameters)
        self.shape = parameters.intercept.shape
        self.p, self.q = parameters.ar_order, parameters.ma_order
        self.fit_intercept, self.covariance = fit_intercept, covariance
        self.pivots = [
            int(np.argmax(abs(a))) for a in [*parameters.ar_left, *parameters.ma_left]
        ]

    def pack(self, parameters):
        pieces = []
        for pivot, a, b in zip(
            self.pivots,
            [*parameters.ar_left, *parameters.ma_left],
            [*parameters.ar_right, *parameters.ma_right],
        ):
            value = a.flat[pivot]
            if not value:
                raise ValueError("initial coefficient is outside the fixed-pivot chart")
            pieces.extend((np.delete((a / value).ravel(), pivot), (b * value).ravel()))
        if self.fit_intercept:
            pieces.append(parameters.intercept.ravel())
        if self.covariance:
            u = _chol(parameters.row_covariance, "row covariance")
            v = _chol(parameters.column_covariance, "column covariance")
            anchor = u[0, 0]
            u, v = u / anchor, v * anchor
            for mode, root in enumerate((u, v)):
                entries = []
                for i in range(len(root)):
                    for j in range(i + 1):
                        if mode == 0 and i == j == 0:
                            continue
                        entries.append(np.log(root[i, i]) if i == j else root[i, j])
                pieces.append(np.array(entries))
        return np.concatenate(pieces) if pieces else np.empty(0)

    def decode(self, vector):
        m, n = self.shape
        index, left, right = 0, [], []
        for pivot in self.pivots:
            a = np.insert(vector[index : index + m * m - 1], pivot, 1).reshape(m, m)
            index += m * m - 1
            b = vector[index : index + n * n].reshape(n, n)
            index += n * n
            left.append(a)
            right.append(b)
        constant = (
            vector[index : index + m * n].reshape(m, n)
            if self.fit_intercept
            else np.zeros((m, n))
        )
        index += m * n if self.fit_intercept else 0
        roots = []
        if self.covariance:
            for mode, d in enumerate((m, n)):
                root = np.zeros((d, d))
                root[0, 0] = 1
                for i in range(d):
                    for j in range(i + 1):
                        if mode == 0 and i == j == 0:
                            continue
                        root[i, j] = np.exp(vector[index]) if i == j else vector[index]
                        index += 1
                roots.append(root)
        arrays = (
            np.array(left[: self.p]).reshape(self.p, m, m),
            np.array(right[: self.p]).reshape(self.p, n, n),
            np.array(left[self.p :]).reshape(self.q, m, m),
            np.array(right[self.p :]).reshape(self.q, n, n),
            constant,
        )
        return (*arrays, *roots)

    def parameters(self, vector):
        decoded = self.decode(vector)
        covariance = [root @ root.T for root in decoded[5:]]
        return _canonical(MARMAParameters(*decoded[:5], *covariance))

    def gradient(self, coefficients, covariance=()):
        ga, gb, gl, gr, gc = coefficients
        pieces = []
        for pivot, a, b in zip(self.pivots, [*ga, *gl], [*gb, *gr]):
            pieces.extend((np.delete(a.ravel(), pivot), b.ravel()))
        if self.fit_intercept:
            pieces.append(gc.ravel())
        for mode, (gradient, root) in enumerate(covariance):
            values = []
            for i in range(len(root)):
                for j in range(i + 1):
                    if mode == 0 and i == j == 0:
                        continue
                    values.append(gradient[i, j] * (root[i, i] if i == j else 1))
            pieces.append(np.array(values))
        return np.concatenate(pieces) if pieces else np.empty(0)


def _objective_gradient(vector, X, chart, method="ls"):
    """Mean-per-time objective and exact backward-recursion derivative."""
    decoded = chart.decode(vector)
    arrays = decoded[:5]
    a, b, l, r, _ = arrays
    t0, count = max(len(a), len(l)), len(X) - max(len(a), len(l))
    errors = _innovations(X, arrays)
    suffix = errors[t0:]
    covariance_gradients = []
    weights = np.zeros_like(errors)
    if method == "ls":
        loss = float(np.mean(np.sum(suffix * suffix, axis=(1, 2))))
        weights[t0:] = 2 * suffix / count
    else:
        row, col = decoded[5:]
        loss, white = _gaussian_loss(suffix, row, col)
        score = np.linalg.solve(row.T, white)
        score = np.linalg.solve(col.T, score.transpose(0, 2, 1)).transpose(0, 2, 1)
        weights[t0:] = score / count
        # dNLL/dCholesky = dimension*L^-T - L^-T*(mean ZZ') (and column analogue).
        m, n = chart.shape
        ru = np.linalg.solve(
            row.T, n * np.eye(m) - np.einsum("tij,tkj->ik", white, white) / count
        )
        rv = np.linalg.solve(
            col.T, m * np.eye(n) - np.einsum("tji,tjk->ik", white, white) / count
        )
        covariance_gradients = [(ru, row), (rv, col)]
    ga, gb, gl, gr = (np.zeros_like(x) for x in (a, b, l, r))
    gc = np.zeros(chart.shape)
    for t in range(len(X) - 1, t0 - 1, -1):
        current = weights[t]
        gc -= current
        for lag, (left, right) in enumerate(zip(a, b), 1):
            ga[lag - 1] -= current @ right @ X[t - lag].T
            gb[lag - 1] -= current.T @ left @ X[t - lag]
        for lag, (left, right) in enumerate(zip(l, r), 1):
            gl[lag - 1] += current @ right @ errors[t - lag].T
            gr[lag - 1] += current.T @ left @ errors[t - lag]
            if t - lag >= t0:
                weights[t - lag] += left.T @ current @ right
    gradient = chart.gradient((ga, gb, gl, gr, gc), covariance_gradients)
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError("MARMA objective or gradient is nonfinite")
    return loss, gradient


def _covariance_fit(errors, iterations=100):
    count, m, n = errors.shape
    row, col = np.eye(m), np.eye(n)
    for _ in range(iterations):
        weighted = np.linalg.solve(col, errors.transpose(0, 2, 1))
        row = np.einsum("tij,tjk->ik", errors, weighted) / (count * n)
        _chol(row, "initial row residual covariance")
        weighted = np.linalg.solve(row, errors)
        col = np.einsum("tji,tjk->ik", errors, weighted) / (count * m)
        _chol(col, "initial column residual covariance")
        gauge = np.trace(row) / m
        row, col = row / gauge, col * gauge
    return row, col


def _initial(X, p, q, intercept, rng):
    count, m, n = X.shape
    left, right = [], []
    if p:
        flat = X.transpose(0, 2, 1).reshape(count, m * n)
        design = np.concatenate(
            [flat[p - lag : count - lag] for lag in range(1, p + 1)], axis=1
        )
        target = flat[p:]
        if intercept:
            design, target = design - design.mean(axis=0), target - target.mean(axis=0)
        coefficient = np.linalg.lstsq(design, target, rcond=None)[0]
        for lag in range(p):
            a, b = nearest_kronecker_product(
                coefficient[lag * m * n : (lag + 1) * m * n].T, (m, n)
            )
            left.append(a)
            right.append(b)
    # Nonzero MA factors give both mode derivatives a usable starting direction.
    ma_left = np.repeat((np.eye(m) / np.sqrt(m))[None], q, axis=0)
    ma_right = np.array(
        [
            0.05 * np.sqrt(m) * (np.eye(n) + 0.1 * rng.normal(size=(n, n))) / max(q, 1)
            for _ in range(q)
        ]
    ).reshape(q, n, n)
    return MARMAParameters(
        left,
        right,
        ma_left,
        ma_right,
        X[max(p, q) :].mean(axis=0) if intercept else np.zeros((m, n)),
    )


@dataclass(frozen=True)
class MARMAOptimizationRun:
    """One local solve; convergence is numerical, not identification or globality."""

    converged: bool
    n_iter: int
    objective_history: np.ndarray
    gradient_norm: float
    message: str
    failed: bool = False
    initialization_iterations: int = 0
    initialization_converged: bool | None = None
    initialization_objective_history: np.ndarray | None = None
    initialization_message: str | None = None
    constraint_residuals: np.ndarray | None = None
    feasible: bool = False
    optimizer: str = "BFGS"


@dataclass
class MARMAResult:
    """Fitted rank-one MARMA with all local attempts and conditioning retained.

    Objectives refer to numerically centered/scaled data. LS histories contain
    mean per-time Frobenius squared errors; MLE histories contain mean per-time
    Gaussian NLL. Convergence does not imply stability, invertibility, minimal
    orders, left coprimeness or global optimality. Methods inherit Tsay's model
    reference and the extension boundaries of :func:`fit_marma`.
    """

    parameters: MARMAParameters
    filter_result: MARMAFilterResult
    method: str
    runs: tuple
    selected_start: int
    data_scale: float
    location: np.ndarray
    max_dense_dimension: int
    enforce_admissibility: bool = True
    stability_margin: float = 1e-6
    _innovation_covariance: np.ndarray | None = field(default=None, repr=False)

    @property
    def converged(self):
        return self.runs[self.selected_start].converged

    @property
    def n_iter(self):
        return self.runs[self.selected_start].n_iter

    @property
    def objective_history(self):
        return self.runs[self.selected_start].objective_history

    @property
    def objective(self):
        return float(self.objective_history[-1])

    @property
    def residuals(self):
        return self.filter_result.residuals[~self.filter_result.conditioning_mask]

    @property
    def fitted_values(self):
        return self.filter_result.fitted_values[~self.filter_result.conditioning_mask]

    @property
    def log_likelihood(self):
        return self.filter_result.log_likelihood

    @property
    def innovation_covariance(self):
        """Dense plug-in innovation covariance; see forecast_covariance for scope."""
        return self.forecast_covariance(1)[0]

    @property
    def diagnostics(self):
        return marma_diagnostics(
            self.parameters, max_dense_dimension=self.max_dense_dimension
        )

    def forecast(self, steps, history=None):
        """Forecast after refiltering the complete supplied history, without lookahead."""
        filtered = (
            self.filter_result
            if history is None
            else filter_marma(history, self.parameters)
        )
        return filtered.forecast(steps)

    def forecast_covariance(self, steps, *, max_dense_dimension=None):
        """Full horizon covariance; LS uses uncentered residual second moments."""
        maximum = (
            self.max_dense_dimension
            if max_dense_dimension is None
            else max_dense_dimension
        )
        return _forecast_covariance(
            self.parameters, steps, maximum, self._innovation_covariance
        )


def fit_marma(
    X,
    ar_order=1,
    ma_order=1,
    *,
    method="ls",
    intercept=True,
    initial=None,
    n_starts=3,
    max_iter=300,
    tol=1e-8,
    random_state=0,
    max_dense_dimension=256,
    enforce_admissibility=True,
    stability_margin=1e-6,
):
    """Fit recursive conditional LS or separable Gaussian rank-one MARMA.

    Orders are nonnegative, including MA-only, AR-only and white-noise limits.
    The first max(p,q) observations are conditioned on and their innovations
    fixed at zero. Every objective and analytic gradient traverses the full
    residual recursion. SLSQP solves fixed-pivot factor charts subject to both
    full-polynomial companion radii <=1-stability_margin. Numerical constraint
    derivatives are used, not numerical objective derivatives. A finite feasible
    iterate is retained even on solver failure, with convergence false.
    Setting enforce_admissibility=False selects unconstrained BFGS, an explicit
    extension that can return unstable/noninvertible estimates. Root restrictions
    do not establish structural ARMA identification.

    With a free intercept, subtracting a training location is only a numerical
    reparameterization; the physical intercept is restored after fitting.
    Initialization projects an unrestricted VAR(p) and adds small nonzero MA
    pairs; it is NOT the paper's suggested unrestricted VARMA initializer.
    Additional local starts perturb these pairs. MLE first runs at most 100
    coefficient-LS iterations (separately recorded), initializes covariance by
    separable flip-flop residual fitting, and optimizes Cholesky coordinates.
    Explicit initial covariance is used for the first start without this
    warm-up; supplied parameters are not silently discarded.
    max_iter limits each final solve; the separate warm-up is limited to
    min(max_iter,100). An unconverged finite warm-up may seed the final solve.
    Degenerate residual covariance is rejected, not floored. Fixed pivots can
    limit a local chart; use different initial parameters to assess sensitivity.
    Automatic starts are made strictly feasible by multiplying each lag-l right
    factor by c**l, which scales every companion root by c. Infeasible user
    starts are rejected rather than modified. tol is SLSQP's absolute scaled
    objective/optimality tolerance or BFGS's gradient tolerance, not a statistical
    test. max_dense_dimension guards initializer/operator/covariance allocations.

    References
    ----------
    Tsay, R. S. (2024), "Matrix-Variate Time Series Analysis: A Brief Review
    and Some New Developments", equations (4)--(5), Section 2.5,
    https://doi.org/10.1111/insr.12558. Conditional Gaussian NLL corrects the
    missing factor 1/2 in equation (30). LS is the isotropic conditional
    Gaussian coefficient objective. Solver, multistart and gauge choices are
    package implementations; rank-r/seasonal models and inference are absent.
    """
    p, q = positive_int(ar_order, "ar_order", minimum=0), positive_int(
        ma_order, "ma_order", minimum=0
    )
    X = as_series(X, min_samples=max(p, q) + 2)
    m, n = X.shape[1:]
    maximum = positive_int(max_dense_dimension, "max_dense_dimension")
    if m * n * max(p, q, 1) > maximum:
        raise ValueError(
            "model exceeds max_dense_dimension; explicitly increase the allocation guard"
        )
    n_starts, max_iter = positive_int(n_starts, "n_starts"), positive_int(
        max_iter, "max_iter"
    )
    tol = finite_scalar(tol, "tol", minimum=0)
    if (
        not tol
        or method not in ("ls", "mle")
        or not isinstance(intercept, (bool, np.bool_))
    ):
        raise ValueError("require positive tol, method ls/mle and boolean intercept")
    if not isinstance(enforce_admissibility, (bool, np.bool_)):
        raise ValueError("enforce_admissibility must be boolean")
    stability_margin = finite_scalar(stability_margin, "stability_margin", minimum=0)
    if not 0 < stability_margin < 1 or 1 - stability_margin == 1:
        raise ValueError(
            "stability_margin must give a representably interior unit bound"
        )
    bound = float(np.max(abs(X))) or 1.0
    location = (X / bound).mean(axis=0) * bound if intercept else np.zeros((m, n))
    centered = X - location
    scale = float(np.max(abs(centered)))
    if not np.isfinite(scale):
        raise FloatingPointError("centered observations exceed floating-point range")
    if not scale:
        raise ValueError("constant or zero data do not identify MARMA innovations")
    work = centered / scale
    rng = (
        deepcopy(random_state)
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    if initial is not None:
        initial = _canonical(initial)
        if (initial.ar_order, initial.ma_order, initial.intercept.shape) != (
            p,
            q,
            (m, n),
        ):
            raise ValueError(
                "initial parameters do not match requested orders/dimensions"
            )
        if enforce_admissibility and np.any(
            _radii(_raw(initial)) > 1 - stability_margin
        ):
            raise ValueError(
                "initial parameters violate stationarity/invertibility constraints"
            )
        constant = initial.intercept / bound - location / bound
        for a, b in zip(initial.ar_left, initial.ar_right):
            constant += a @ (location / bound) @ b.T
        constant = _represented(
            _multiply(constant, bound / scale), "centered intercept"
        )
        if not intercept and np.any(initial.intercept):
            raise ValueError("initial intercept must be zero when intercept=False")
        covariance = []
        if method == "mle" and initial.row_covariance is not None:
            for value in (initial.row_covariance, initial.column_covariance):
                root = _chol(value, "initial covariance") / np.sqrt(scale)
                covariance.append(
                    _represented(root @ root.T, "scaled initial covariance")
                )
        base = MARMAParameters(*_raw(initial)[:4], constant, *covariance)
    else:
        base = _initial(work, p, q, intercept, rng)
        if enforce_admissibility:
            base = _admissible_start(base, stability_margin)
    runs, candidates = [], []
    for start in range(n_starts):
        try:
            initialization_iterations = 0
            warm = None
            if start:
                arrays = []
                for values in _raw(base)[:4]:
                    arrays.append(values + 0.15 * rng.normal(size=values.shape))
                current = _canonical(MARMAParameters(*arrays, base.intercept))
                if enforce_admissibility:
                    current = _admissible_start(current, stability_margin)
            else:
                current = base
            if method == "mle":
                if current.row_covariance is None:
                    warm = fit_marma(
                        work,
                        p,
                        q,
                        method="ls",
                        intercept=intercept,
                        initial=current,
                        n_starts=1,
                        max_iter=min(max_iter, 100),
                        tol=max(tol, 1e-5),
                        max_dense_dimension=maximum,
                        enforce_admissibility=enforce_admissibility,
                        stability_margin=stability_margin,
                    )
                    initialization_iterations = warm.n_iter
                    current = warm.parameters
                    errors = _innovations(work, _raw(current))[max(p, q) :]
                    u, v = _covariance_fit(errors)
                    current = MARMAParameters(*_raw(current), u, v)
            chart = _ParameterChart(
                current, fit_intercept=intercept, covariance=method == "mle"
            )
            vector = chart.pack(current)
            history = []
            best = [float("inf"), vector.copy()]

            def constraints(value):
                try:
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        result = 1 - stability_margin - _radii(chart.decode(value)[:5])
                    return result if np.isfinite(result).all() else np.full(2, -1e100)
                except (ValueError, np.linalg.LinAlgError):
                    return np.full(2, -1e100)

            def evaluate(value):
                try:
                    with np.errstate(
                        over="ignore", invalid="ignore", divide="ignore", under="ignore"
                    ):
                        loss, gradient = _objective_gradient(value, work, chart, method)
                    if loss < best[0] and (
                        not enforce_admissibility or np.all(constraints(value) >= 0)
                    ):
                        best[:] = loss, value.copy()
                    return loss, gradient
                except (FloatingPointError, np.linalg.LinAlgError, ValueError):
                    return float("inf"), np.zeros_like(value)

            first, _ = evaluate(vector)
            if not np.isfinite(first):
                raise ValueError("nonfinite initial objective")
            history.append(first)

            def callback(value):
                history.append(evaluate(value)[0])

            if vector.size:
                optimizer = "SLSQP" if enforce_admissibility else "BFGS"
                solved = minimize(
                    evaluate,
                    vector,
                    method=optimizer,
                    jac=True,
                    callback=callback,
                    constraints=(
                        {"type": "ineq", "fun": constraints}
                        if enforce_admissibility
                        else ()
                    ),
                    options={
                        "maxiter": max_iter,
                        "ftol" if enforce_admissibility else "gtol": tol,
                    },
                )
                terminal_value, _ = evaluate(solved.x)
                chosen = solved.x
                converged, iterations, message = (
                    bool(solved.success),
                    int(solved.nit),
                    str(solved.message),
                )
                if enforce_admissibility and (
                    np.any(constraints(chosen) < 0) or not np.isfinite(terminal_value)
                ):
                    chosen = best[1]
                    converged = False
                    message += "; returned best strictly feasible iterate"
                value, gradient = evaluate(chosen)
                fitted = chart.parameters(chosen)
            else:
                value, gradient = evaluate(vector)
                fitted = current
                converged, iterations, message = (
                    True,
                    0,
                    "No free coefficient parameters",
                )
                chosen, optimizer = vector, "none"
            if not np.isfinite(value):
                raise ValueError("nonfinite terminal objective")
            if history[-1] != value:
                history.append(value)
            runs.append(
                MARMAOptimizationRun(
                    converged,
                    iterations,
                    np.array(history),
                    float(np.max(abs(gradient), initial=0)),
                    message,
                    initialization_iterations=initialization_iterations,
                    initialization_converged=None if warm is None else warm.converged,
                    initialization_objective_history=(
                        None if warm is None else warm.objective_history.copy()
                    ),
                    initialization_message=(
                        None if warm is None else warm.runs[warm.selected_start].message
                    ),
                    constraint_residuals=constraints(chosen),
                    feasible=bool(np.all(constraints(chosen) >= 0)),
                    optimizer=optimizer,
                )
            )
            candidates.append((value, len(runs) - 1, fitted))
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            runs.append(
                MARMAOptimizationRun(
                    False,
                    0,
                    np.array([]),
                    float("inf"),
                    str(exc),
                    True,
                    initialization_iterations=initialization_iterations,
                    initialization_converged=None if warm is None else warm.converged,
                    initialization_objective_history=(
                        None if warm is None else warm.objective_history.copy()
                    ),
                    initialization_message=(
                        None if warm is None else warm.runs[warm.selected_start].message
                    ),
                    optimizer="SLSQP" if enforce_admissibility else "BFGS",
                )
            )
    if not candidates:
        raise ValueError(
            "all MARMA starts failed: " + "; ".join(run.message for run in runs)
        )
    _, selected, fitted = min(candidates, key=lambda item: item[0])
    constant = location / bound
    for a, b in zip(fitted.ar_left, fitted.ar_right):
        constant = constant - a @ (location / bound) @ b.T
    constant = _represented(
        _multiply(constant, bound) + _multiply(fitted.intercept, scale),
        "physical intercept",
    )
    u = v = None
    if method == "mle":
        # Balanced factors retain representable covariances at very large units.
        u = _represented(
            _multiply(fitted.row_covariance, scale), "physical row covariance"
        )
        v = _represented(
            _multiply(fitted.column_covariance, scale), "physical column covariance"
        )
    physical = MARMAParameters(*_raw(fitted)[:4], constant, u, v)
    filtered = filter_marma(X, physical)
    covariance = None
    if method == "ls":
        errors = _innovations(work, _raw(fitted))[max(p, q) :]
        flat = errors.transpose(0, 2, 1).reshape(len(errors), m * n)
        covariance = _multiply(flat.T @ flat / len(flat), scale, scale)
        if not np.isfinite(covariance).all() or np.any(
            (np.diag(flat.T @ flat) > 0) & (np.diag(covariance) == 0)
        ):
            covariance = None  # Coefficients/forecasts can remain representable.
    return MARMAResult(
        physical,
        filtered,
        method,
        tuple(runs),
        selected,
        scale,
        location,
        maximum,
        bool(enforce_admissibility),
        stability_margin,
        covariance,
    )


@dataclass(frozen=True)
class MARMASimulation:
    """Simulated observations, generating innovations and conditional means."""

    observations: np.ndarray
    innovations: np.ndarray
    conditional_mean: np.ndarray


def simulate_marma(
    n_samples, parameters, *, burnin=300, random_state=None, max_dense_dimension=256
):
    """Simulate a stable MARMA model with independent matrix-normal innovations.

    Omitted covariance factors mean identity covariance. Presample observations
    and innovations are zero; burn-in is a caller-selected transient reduction,
    not exact stationary initialization. Noninvertible MA parameters are allowed
    for simulation, but unstable AR parameters are rejected.

    References
    ----------
    Tsay (2024), "Matrix-Variate Time Series Analysis: A Brief Review and Some
    New Developments", equation (4), https://doi.org/10.1111/insr.12558.
    Gaussian draws and explicit burn-in generate this model family; they do
    not reproduce a particular paper experiment.
    """
    parameters = _canonical(parameters)
    n_samples, burnin = positive_int(n_samples, "n_samples"), positive_int(
        burnin, "burnin", minimum=0
    )
    if not marma_diagnostics(
        parameters, max_dense_dimension=max_dense_dimension
    ).is_stable:
        raise ValueError("simulation requires a stable AR polynomial")
    m, n = parameters.intercept.shape
    u = np.eye(m) if parameters.row_covariance is None else parameters.row_covariance
    v = (
        np.eye(n)
        if parameters.column_covariance is None
        else parameters.column_covariance
    )
    rng = (
        deepcopy(random_state)
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    count = n_samples + burnin
    t0 = parameters.conditioning_length
    X, errors = np.zeros((count + t0, m, n)), np.zeros((count + t0, m, n))
    row, col = _chol(u, "row covariance"), _chol(v, "column covariance")
    errors[t0:] = row @ rng.normal(size=(count, m, n)) @ col.T
    means = np.empty((count, m, n))
    for t in range(t0, count + t0):
        value = parameters.intercept.copy()
        for lag, (a, b) in enumerate(zip(parameters.ar_left, parameters.ar_right), 1):
            value += a @ X[t - lag] @ b.T
        for lag, (a, b) in enumerate(zip(parameters.ma_left, parameters.ma_right), 1):
            value -= a @ errors[t - lag] @ b.T
        means[t - t0] = value
        X[t] = value + errors[t]
    _represented(X, "simulation")
    return MARMASimulation(X[t0 + burnin :], errors[t0 + burnin :], means[burnin:])
