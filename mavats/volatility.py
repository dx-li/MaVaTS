"""Trace-identified matrix GARCH of Yu, Li, Jiang and Zhu (2025).

The full first-order model uses equations (4)--(9), with the zero-state
conditional Gaussian quasi-likelihood (12)--(13), of the accepted manuscript:
https://doi.org/10.1080/01621459.2024.2415719 (online 2024).
The earlier arXiv:2306.05169v1 lacks the accepted version's stationarity theorem.
No QMLE standard errors, portmanteau test, or factor-GARCH inference are supplied.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.optimize import minimize

from ._validation import as_series, finite_scalar, positive_int


def _matrix(value, name, shape=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be a finite real matrix")
    result = np.array(raw, dtype=float, copy=True)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite real matrix")
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    return result


def _positive(value, name):
    value = finite_scalar(value, name, minimum=0)
    if value == 0:
        raise ValueError(f"{name} must be strictly positive")
    return value


@dataclass(frozen=True)
class MatrixGARCHParameters:
    """Parameters of the full first-order trace-normalized matrix GARCH model.

    A0/B0 are lower triangular, have strictly positive diagonal entries and
    first diagonal exactly one. This interior covariance convention ensures
    positive definiteness even with the paper's zero-state initialization.
    A1/B1 multiply the preceding observation scatter; A2/B2 multiply the
    previous *unnormalized* shape state. Dynamic matrices are unrestricted
    in sign; each can be negated independently without changing the model.

    w > 0, alpha >= 0 and beta >= 0 ensure a positive trace process. Positivity
    alone does not imply stationarity. Scalar spatial modes have no identified
    shape dynamics: their intercept must be [[1]] and dynamics [[0]].
    Arrays are copied and made read-only.
    """

    A0: np.ndarray
    A1: np.ndarray
    A2: np.ndarray
    B0: np.ndarray
    B1: np.ndarray
    B2: np.ndarray
    w: float
    alpha: float
    beta: float

    def __post_init__(self):
        for prefix in ("A", "B"):
            first = _matrix(getattr(self, prefix + "0"), prefix + "0")
            if not first.shape[0] or first.shape[0] != first.shape[1]:
                raise ValueError(f"{prefix}0 must be a nonempty square matrix")
            if (
                np.any(np.triu(first, 1) != 0)
                or np.any(np.diag(first) <= 0)
                or first[0, 0] != 1
            ):
                raise ValueError(
                    f"{prefix}0 must be lower triangular with positive diagonal and [0,0]=1"
                )
            for suffix in ("0", "1", "2"):
                name = prefix + suffix
                array = _matrix(getattr(self, name), name, first.shape)
                if first.shape == (1, 1) and suffix != "0" and np.any(array != 0):
                    raise ValueError(
                        "scalar modes must have zero shape dynamics (unidentified otherwise)"
                    )
                array.flags.writeable = False
                object.__setattr__(self, name, array)
        object.__setattr__(self, "w", _positive(self.w, "w"))
        object.__setattr__(self, "alpha", finite_scalar(self.alpha, "alpha", minimum=0))
        object.__setattr__(self, "beta", finite_scalar(self.beta, "beta", minimum=0))

    @property
    def shape(self):
        return (len(self.A0), len(self.B0))

    @property
    def sufficient_stationarity_bound(self):
        """Theorem 1's Frobenius bound; <1 is sufficient for Gaussian noise.

        This condition is not necessary. Its failure must not be called
        nonstationarity. It is stronger than separate BEKK spectral checks.
        """
        return float(
            np.sum(self.A1**2)
            + np.sum(self.B1**2)
            + self.alpha
            + max(np.sum(self.A2**2), np.sum(self.B2**2), self.beta)
        )

    @property
    def spectral_bounds(self):
        """Assumption 3.3(i) row/column radii and alpha+beta, not a proof of stationarity."""
        radii = [
            float(np.max(np.abs(np.linalg.eigvals(np.kron(a, a) + np.kron(b, b)))))
            for a, b in ((self.A1, self.A2), (self.B1, self.B2))
        ]
        return (*radii, self.alpha + self.beta)


@dataclass
class MatrixGARCHState:
    """Post-observation state at t: S1[t], S2[t], y[t], and X[t].

    ``forecast_one`` uses only this state to calculate covariance at t+1.
    Shape states are not trace-normalized. Trace and last observation use
    original data units. A zero state is the paper's conditional initializer.
    """

    row_shape: np.ndarray
    column_shape: np.ndarray
    trace: float
    last_observation: np.ndarray


def _state(state, shape):
    m, n = shape
    if state is None:
        return MatrixGARCHState(
            np.zeros((m, m)), np.zeros((n, n)), 0.0, np.zeros(shape)
        )
    if not isinstance(state, MatrixGARCHState):
        raise ValueError("initial_state must be a MatrixGARCHState")
    matrices = []
    for value, d, name in (
        (state.row_shape, m, "row_shape"),
        (state.column_shape, n, "column_shape"),
    ):
        a = _matrix(value, name, (d, d))
        if not np.allclose(a, a.T, rtol=1e-12, atol=0):
            raise ValueError(f"{name} must be symmetric positive semidefinite")
        if np.linalg.eigvalsh(a)[0] < 0:
            raise ValueError(f"{name} must be symmetric positive semidefinite")
        matrices.append(a)
    return MatrixGARCHState(
        *matrices,
        finite_scalar(state.trace, "state trace", minimum=0),
        _matrix(state.last_observation, "last_observation", shape),
    )


def _next(parameters, row, col, y, past):
    a, b = parameters.A1 @ past, parameters.B1 @ past.T
    row = (
        parameters.A0 @ parameters.A0.T
        + a @ a.T
        + parameters.A2 @ row @ parameters.A2.T
    )
    col = (
        parameters.B0 @ parameters.B0.T
        + b @ b.T
        + parameters.B2 @ col @ parameters.B2.T
    )
    if parameters.alpha == 0:
        energy = 0.0
    else:
        # Keep ordinary-unit arithmetic unchanged. An unweighted squared norm
        # can overflow even when its alpha-weighted value is representable.
        with np.errstate(over="ignore", invalid="ignore"):
            energy = parameters.alpha * np.sum(past**2)
            if not np.isfinite(energy):
                energy = np.sum((np.sqrt(parameters.alpha) * past) ** 2)
    y = parameters.w + energy + parameters.beta * y
    return (row + row.T) / 2, (col + col.T) / 2, float(y)


def _covariance(row, col, y):
    if not np.isfinite(row).all() or not np.isfinite(col).all() or not np.isfinite(y):
        raise FloatingPointError(
            "matrix GARCH state overflowed; check parameters and data units"
        )
    u, v = (row / np.trace(row)) * y, col / np.trace(col)
    if y <= 0 or not np.isfinite(u).all() or not np.isfinite(v).all():
        raise FloatingPointError("matrix GARCH covariance is not finite and positive")
    try:
        lu, lv = np.linalg.cholesky(u), np.linalg.cholesky(v)
    except np.linalg.LinAlgError as exc:
        raise FloatingPointError(
            "matrix GARCH covariance lost positive definiteness"
        ) from exc
    return u, v, lu, lv


def _likelihood(x, lu, lv):
    m, n = x.shape
    white = solve_triangular(lu, x, lower=True, check_finite=False)
    white = solve_triangular(lv, white.T, lower=True, check_finite=False).T
    nll = 0.5 * (
        m * n * np.log(2 * np.pi)
        + 2 * n * np.log(np.diag(lu)).sum()
        + 2 * m * np.log(np.diag(lv)).sum()
        + np.sum(white**2)
    )
    if not np.isfinite(nll):
        raise FloatingPointError("matrix GARCH likelihood is not finite")
    return float(nll), white


def _change_units(p, scale, *, to_work):
    # X_work = X / scale; the shape states are unchanged, A1_work=scale*A1.
    multiplier = scale if to_work else 1 / scale
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        w = (p.w / scale) / scale if to_work else (p.w * scale) * scale
        return MatrixGARCHParameters(
            p.A0,
            p.A1 * multiplier,
            p.A2,
            p.B0,
            p.B1 * multiplier,
            p.B2,
            w,
            p.alpha,
            p.beta,
        )


def _check_original_covariance(u, v, y):
    """Reject original-unit underflow/overflow without adding a covariance floor.

    Accepts a single covariance or a history. The marginal covariance factors
    must remain positive definite, and each represented entry variance must be
    positive. Dense Kronecker materialization has its own definiteness check.
    """
    u, v, y = np.asarray(u), np.asarray(v), np.asarray(y)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        column = v * y[..., None, None]
        variances = (
            np.diagonal(u, axis1=-2, axis2=-1)[..., :, None]
            * np.diagonal(v, axis1=-2, axis2=-1)[..., None, :]
        )
    if (
        not np.isfinite(u).all()
        or not np.isfinite(column).all()
        or not np.isfinite(y).all()
        or np.any(y <= 0)
        or not np.isfinite(variances).all()
        or np.any(variances <= 0)
    ):
        raise FloatingPointError(
            "covariance is not representable in original data units"
        )
    try:
        np.linalg.cholesky(u)
        np.linalg.cholesky(column)
    except np.linalg.LinAlgError as exc:
        raise FloatingPointError(
            "covariance lost positive definiteness in original data units"
        ) from exc


def _dense_covariance(u, v):
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        covariance = np.kron(v, u)
    if not np.isfinite(covariance).all():
        raise FloatingPointError("dense covariance is not representable in data units")
    try:
        np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise FloatingPointError(
            "dense covariance lost positive definiteness in data units"
        ) from exc
    return covariance


@dataclass
class MatrixGARCHForecast:
    """Exact conditional covariance for the next observation, in original units."""

    row_covariance: np.ndarray
    column_factor: np.ndarray
    trace: float

    @property
    def column_covariance(self):
        return self.column_factor * self.trace

    def covariance(self):
        """Materialize Cov(vec_F(X_next)): V_next kron U_next."""
        return _dense_covariance(self.row_covariance, self.column_factor)


@dataclass
class MatrixGARCHFilterResult:
    """Conditional Gaussian likelihood and covariance histories.

    ``row_covariances[t]`` is U[t]; ``column_factors[t]`` is trace-one V[t],
    so the column covariance is y[t]*V[t]. ``standardized_residuals`` uses
    Cholesky whitening, which suffices for the Gaussian likelihood but differs
    from the paper's symmetric-root residual coordinates. No whiteness test
    or inferential calibration is implied. Histories precede their observation:
    the covariance at t cannot use X[t].
    """

    parameters: MatrixGARCHParameters
    row_covariances: np.ndarray
    column_factors: np.ndarray
    traces: np.ndarray
    negative_log_likelihoods: np.ndarray
    standardized_residuals: np.ndarray
    state: MatrixGARCHState

    @property
    def log_likelihood(self):
        return -float(self.negative_log_likelihoods.sum())

    @property
    def column_covariances(self):
        return self.column_factors * self.traces[:, None, None]

    def covariance(self, index):
        """Materialize one conditional covariance in column-major order."""
        return _dense_covariance(
            self.row_covariances[index], self.column_factors[index]
        )

    def forecast_one(self):
        """Exact one-step covariance; multistep expectations do not close."""
        s = self.state
        scale = max(
            float(np.max(np.abs(s.last_observation))),
            np.sqrt(self.parameters.w),
            np.sqrt(s.trace),
            np.finfo(float).tiny,
        )
        p = _change_units(self.parameters, scale, to_work=True)
        row, col, y = _next(
            p,
            s.row_shape,
            s.column_shape,
            (s.trace / scale) / scale,
            s.last_observation / scale,
        )
        u, v, _, _ = _covariance(row, col, y)
        with np.errstate(over="ignore", invalid="ignore"):
            u, y = (u * scale) * scale, (y * scale) * scale
        _check_original_covariance(u, v, y)
        return MatrixGARCHForecast(u, v, float(y))


def filter_matrix_garch(X, parameters, *, initial_state=None):
    """Filter observed zero-mean matrices with fixed matrix GARCH parameters.

    Default is the paper's zero X0, shape states and trace, *before* the first
    covariance update. Thus first covariance uses A0 A0.T, B0 B0.T and w.
    Pass a previous result's ``state`` for chronological continuation. This
    does not refit parameters or estimate a mean. Gaussian constants are
    included in per-observation likelihoods, in the original data units.
    """
    X = as_series(X, min_samples=1)
    if (
        not isinstance(parameters, MatrixGARCHParameters)
        or parameters.shape != X.shape[1:]
    ):
        raise ValueError(
            "parameters must be MatrixGARCHParameters matching observation shape"
        )
    s = _state(initial_state, parameters.shape)
    scale = max(
        float(np.max(np.abs(X))),
        float(np.max(np.abs(s.last_observation))),
        np.sqrt(parameters.w),
        np.sqrt(s.trace),
        np.finfo(float).tiny,
    )
    p = _change_units(parameters, scale, to_work=True)
    row, col, y, past = (
        s.row_shape,
        s.column_shape,
        (s.trace / scale) / scale,
        s.last_observation / scale,
    )
    us, vs, ys, values, residuals = [], [], [], [], []
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for x in X / scale:
            row, col, y = _next(p, row, col, y, past)
            u, v, lu, lv = _covariance(row, col, y)
            nll, white = _likelihood(x, lu, lv)
            us.append((u * scale) * scale)
            vs.append(v)
            ys.append((y * scale) * scale)
            values.append(nll + X.shape[1] * X.shape[2] * np.log(scale))
            residuals.append(white)
            past = x
    _check_original_covariance(us, vs, ys)
    terminal = MatrixGARCHState(row, col, float(ys[-1]), X[-1].copy())
    return MatrixGARCHFilterResult(
        parameters,
        np.asarray(us),
        np.asarray(vs),
        np.asarray(ys),
        np.asarray(values),
        np.asarray(residuals),
        terminal,
    )


@dataclass
class MatrixGARCHSimulation:
    """Gaussian observations and the covariance states actually used to draw them."""

    observations: np.ndarray
    row_covariances: np.ndarray
    column_factors: np.ndarray
    traces: np.ndarray
    initial_state: MatrixGARCHState
    state: MatrixGARCHState


def simulate_matrix_garch(n, parameters, *, burnin=300, random_state=None):
    """Simulate Gaussian matrix GARCH with a finite burn-in from the zero state.

    Burn-in approximates stationarity and does not prove it. Unstable recursions
    are rejected if numerical states become invalid. The returned initial_state
    is the post-burn-in state immediately preceding the retained observations,
    enabling an exact simulation/filter oracle. Innovations are standard matrix
    normal, not elementwise or matrix Student-t.
    """
    n = positive_int(n, "n")
    burnin = positive_int(burnin, "burnin", minimum=0)
    if not isinstance(parameters, MatrixGARCHParameters):
        raise ValueError("parameters must be MatrixGARCHParameters")
    rng = np.random.default_rng(random_state)
    s = _state(None, parameters.shape)
    row, col, y, past = s.row_shape, s.column_shape, 0.0, s.last_observation
    observations, us, vs, ys = [], [], [], []
    initial = s
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(n + burnin):
            if t == burnin:
                initial = MatrixGARCHState(row.copy(), col.copy(), y, past.copy())
            row, col, y = _next(parameters, row, col, y, past)
            u, v, lu, lv = _covariance(row, col, y)
            _check_original_covariance(u, v, y)
            past = lu @ rng.normal(size=parameters.shape) @ lv.T
            if not np.isfinite(past).all():
                raise FloatingPointError(
                    "simulated observation exceeds the finite numeric range"
                )
            if t >= burnin:
                observations.append(past)
                us.append(u)
                vs.append(v)
                ys.append(y)
    return MatrixGARCHSimulation(
        np.asarray(observations),
        np.asarray(us),
        np.asarray(vs),
        np.asarray(ys),
        initial,
        MatrixGARCHState(row, col, y, past),
    )


class _Parameterization:
    """Positive intercept diagonals/log w, direct nonnegative trace dynamics."""

    def __init__(self, shape, dynamics):
        self.shape = shape
        self.entries = []
        for name, d in zip(
            ("A0", "A1", "A2", "B0", "B1", "B2"), (shape[0],) * 3 + (shape[1],) * 3
        ):
            if d == 1:
                continue
            for i in range(d):
                for j in range(d):
                    if name.endswith("0"):
                        if j > i or (i == j == 0):
                            continue
                    elif dynamics == "diagonal" and i != j:
                        continue
                    self.entries.append((name, i, j, name.endswith("0") and i == j))
        self.labels = ("log(w)", "alpha", "beta") + tuple(
            ("log(" if log else "") + f"{name}[{i},{j}]" + (")" if log else "")
            for name, i, j, log in self.entries
        )

    def pack(self, p):
        return np.array(
            [np.log(p.w), p.alpha, p.beta]
            + [
                np.log(getattr(p, name)[i, j]) if log else getattr(p, name)[i, j]
                for name, i, j, log in self.entries
            ]
        )

    def unpack(self, x):
        m, n = self.shape
        arrays = {
            name: np.eye(d) if name.endswith("0") else np.zeros((d, d))
            for name, d in zip(
                ("A0", "A1", "A2", "B0", "B1", "B2"), (m,) * 3 + (n,) * 3
            )
        }
        for value, (name, i, j, log) in zip(x[3:], self.entries):
            arrays[name][i, j] = np.exp(value) if log else value
        return MatrixGARCHParameters(**arrays, w=np.exp(x[0]), alpha=x[1], beta=x[2])

    def derivatives(self, p):
        k = len(self.labels)
        output = {
            name: np.zeros((k, *getattr(p, name).shape))
            for name in ("A0", "A1", "A2", "B0", "B1", "B2")
        }
        for idx, (name, i, j, log) in enumerate(self.entries, 3):
            output[name][idx, i, j] = getattr(p, name)[i, j] if log else 1.0
        return output


def _objective_gradient(X, p, codec, *, data_scale=None):
    """Analytic derivative of the entire zero-state likelihood recursion."""
    m, n = p.shape
    k = len(codec.labels)
    derivative = codec.derivatives(p)
    row, col, y, past = np.zeros((m, m)), np.zeros((n, n)), 0.0, np.zeros((m, n))
    dr, dc, dy = np.zeros((k, m, m)), np.zeros((k, n, n)), np.zeros(k)
    gradient, objective = np.zeros(k), 0.0
    physical_rows, physical_columns, physical_traces = [], [], []
    constants = [
        derivative[name] @ a.T + a @ derivative[name].transpose(0, 2, 1)
        for name, a in (("A0", p.A0), ("B0", p.B0))
    ]
    for x in X:
        next_derivatives = []
        for prefix, state, ds, z, constant in (
            ("A", row, dr, past, constants[0]),
            ("B", col, dc, past.T, constants[1]),
        ):
            a1, a2 = getattr(p, prefix + "1"), getattr(p, prefix + "2")
            d1, d2 = derivative[prefix + "1"], derivative[prefix + "2"]
            shock, dshock = a1 @ z, d1 @ z
            one = dshock @ shock.T
            two = d2 @ state @ a2.T
            next_derivatives.append(
                constant
                + one
                + one.transpose(0, 2, 1)
                + two
                + two.transpose(0, 2, 1)
                + a2 @ ds @ a2.T
            )
        dy = p.beta * dy
        dy[0] += p.w
        dy[1] += np.sum(past**2)
        dy[2] += y
        dr, dc = next_derivatives
        row, col, y = _next(p, row, col, y, past)
        u, v, lu, lv = _covariance(row, col, y)
        if data_scale is not None:
            physical_rows.append((u * data_scale) * data_scale)
            physical_columns.append(v)
            physical_traces.append((y * data_scale) * data_scale)
        value, _ = _likelihood(x, lu, lv)
        objective += value
        tr, tc = np.trace(row), np.trace(col)
        du = (dr - (row / tr) * np.trace(dr, axis1=1, axis2=2)[:, None, None]) * (
            y / tr
        )
        du += (row / tr) * dy[:, None, None]
        dv = (dc - v * np.trace(dc, axis1=1, axis2=2)[:, None, None]) / tc
        ui = cho_solve((lu, True), np.eye(m), check_finite=False)
        vi = cho_solve((lv, True), np.eye(n), check_finite=False)
        ux, vx = ui @ x, vi @ x.T
        gu = n * ui - ux @ vi @ ux.T
        gv = m * vi - vx @ ui @ vx.T
        gradient += 0.5 * (
            np.einsum("ij,kji->k", gu, du) + np.einsum("ij,kji->k", gv, dv)
        )
        past = x
    if not np.isfinite(gradient).all():
        raise FloatingPointError("matrix GARCH likelihood derivative is not finite")
    if data_scale is not None:
        _check_original_covariance(physical_rows, physical_columns, physical_traces)
    return objective / len(X), gradient / len(X)


@dataclass
class MatrixGARCHOptimizationRun:
    """One constrained optimization attempt; gradient norm is not a KKT residual."""

    objective: float
    converged: bool
    status: int
    message: str
    n_iter: int
    n_evaluations: int
    gradient_norm: float
    constraint_violation: float
    active_bounds: tuple
    objective_history: np.ndarray
    invalid_evaluations: int
    retained_earlier_iterate: bool
    terminal_constraint_violation: float
    start_fallback: bool


@dataclass
class MatrixGARCHResult:
    """Local conditional QMLE and all starts' optimization diagnostics.

    ``objective`` includes the Gaussian constant in original data units.
    ``runs`` records optimizer objectives in standardized data units, differing
    by the same constant ``m*n*log(data_scale)``. Optimizer success does not
    establish a global minimum, identification, stationarity, or valid inference.
    Coefficient signs follow the selected start: compare covariance paths,
    not raw signed coefficients. No covariance floor is applied.
    """

    parameters: MatrixGARCHParameters
    filtered: MatrixGARCHFilterResult
    converged: bool
    n_iter: int
    objective: float
    runs: tuple
    selected_start: int
    dynamics: str
    constraint: str
    constraint_margin: float
    data_scale: float

    @property
    def log_likelihood(self):
        return self.filtered.log_likelihood

    @property
    def active_bounds(self):
        return self.runs[self.selected_start].active_bounds

    def forecast_one(self):
        return self.filtered.forecast_one()

    def filter(self, X, *, continue_history=True):
        """Filter new observations without refitting; continuation is chronological."""
        if not isinstance(continue_history, (bool, np.bool_)):
            raise ValueError("continue_history must be boolean")
        return filter_matrix_garch(
            X,
            self.parameters,
            initial_state=self.filtered.state if continue_history else None,
        )


def fit_matrix_garch(
    X,
    *,
    dynamics="full",
    initial=None,
    n_starts=2,
    constraint="spectral",
    constraint_margin=1e-4,
    max_iter=300,
    tol=1e-7,
    parameter_bound=5.0,
    random_state=0,
):
    """Fit the full trace-identified first-order model by conditional Gaussian QMLE.

    ``dynamics='full'`` estimates every entry in A1/A2/B1/B2; ``'diagonal'``
    restricts these four matrices, retaining full triangular A0/B0. Scalar
    spatial modes omit unidentified shape dynamics. Observations must have zero
    conditional mean; no mean subtraction or mean-model fitting is implicit.

    ``constraint='spectral'`` enforces Assumption 3.3(i)'s two BEKK spectral
    bounds and alpha+beta below 1-margin. These do not by themselves establish
    stationarity. ``'sufficient'`` instead imposes the stronger Frobenius bound
    from accepted-manuscript Theorem 1. ``'none'`` keeps only positivity and
    finite optimization boxes. Constraints are evaluated in ORIGINAL data units.
    Failing the sufficient bound is not proof of nonstationarity.

    SLSQP uses analytic recursive objective gradients and numerical constraint
    derivatives. Starts are deterministic given random_state. ``initial`` is a
    MatrixGARCHParameters in original units and becomes the first start; the
    others perturb it. Otherwise use a sample marginal-covariance initialization
    with small nonzero dynamic diagonals. All starts use the paper's zero-state
    conditional likelihood; no burn-in observations are removed from fitting.

    For conditioning, X is divided by its largest absolute entry. In these work
    units, log(w) lies within +/-parameter_bound of log(mean ||X||_F^2), free
    log intercept diagonals lie within +/-parameter_bound, other entries within
    +/-parameter_bound, and alpha/beta within [0,parameter_bound]. These are
    numerical compact-domain choices, not universal paper-prescribed bounds.
    Enlarging the boxes and checking multiple starts can change a local fit.
    Active box bounds and nonlinear constraints are reported. Convergence uses
    the optimizer's success AND feasibility; finite failed starts remain visible.

    No standard errors, estimation-adjusted portmanteau test, matrix factor GARCH,
    or multistep covariance approximation is implemented. See volatility-notes.md.
    """
    X = as_series(X, min_samples=3)
    if dynamics not in ("full", "diagonal"):
        raise ValueError("dynamics must be 'full' or 'diagonal'")
    if constraint not in ("spectral", "sufficient", "none"):
        raise ValueError("constraint must be 'spectral', 'sufficient', or 'none'")
    n_starts = positive_int(n_starts, "n_starts")
    max_iter = positive_int(max_iter, "max_iter")
    tol = _positive(tol, "tol")
    bound = _positive(parameter_bound, "parameter_bound")
    margin = _positive(constraint_margin, "constraint_margin")
    if margin >= 1 or bound > 30:
        raise ValueError("constraint_margin must be <1 and parameter_bound <=30")
    scale = float(np.max(np.abs(X)))
    if scale == 0:
        raise ValueError("zero observations drive the trace likelihood to its boundary")
    work = X / scale
    energy = float(np.mean(np.sum(work**2, axis=(1, 2))))
    with np.errstate(over="ignore", under="ignore"):
        original_energy = (energy * scale) * scale
    if not np.isfinite(original_energy) or original_energy == 0:
        raise ValueError(
            "data units imply a variance outside the finite floating-point range"
        )
    codec = _Parameterization(X.shape[1:], dynamics)
    lower, upper = np.full(len(codec.labels), -bound), np.full(len(codec.labels), bound)
    lower[0], upper[0] = np.log(energy) - bound, np.log(energy) + bound
    lower[1:3] = 0

    def inequalities(x):
        if constraint == "none":
            return np.empty(0)
        size = 3 if constraint == "spectral" else 1
        try:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                p = _change_units(codec.unpack(x), scale, to_work=False)
                bounds = (
                    p.spectral_bounds
                    if constraint == "spectral"
                    else [p.sufficient_stationarity_bound]
                )
                result = 1 - margin - np.asarray(bounds)
            if np.isfinite(result).all():
                return result
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            pass
        # SLSQP requires finite constraint values, including invalid trials.
        return np.full(size, -1e10)

    def feasible_start(x):
        try:
            _change_units(codec.unpack(x), scale, to_work=False)
            return bool(np.all(inequalities(x) >= 0))
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            return False

    if initial is not None:
        if (
            not isinstance(initial, MatrixGARCHParameters)
            or initial.shape != X.shape[1:]
        ):
            raise ValueError("initial must be MatrixGARCHParameters matching X")
        if dynamics == "diagonal" and any(
            np.any(a != np.diag(np.diag(a)))
            for a in (initial.A1, initial.A2, initial.B1, initial.B2)
        ):
            raise ValueError("initial dynamics must be diagonal for diagonal fitting")
        start = codec.pack(_change_units(initial, scale, to_work=True))
        if (
            np.any(start < lower)
            or np.any(start > upper)
            or np.any(inequalities(start) < 0)
        ):
            raise ValueError(
                "initial lies outside the optimization bounds or chosen constraints"
            )
    else:
        arrays = []
        for data in (work, work.transpose(0, 2, 1)):
            d = data.shape[1]
            cov = np.einsum("tij,tkj->ik", data, data) / len(data)
            try:
                a0 = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    "sample marginal covariance is singular; supply a valid initial parameter set"
                ) from exc
            a0 /= a0[0, 0]
            arrays.extend(
                (
                    a0,
                    0.1 * min(1.0, scale) * np.eye(d) if d > 1 else np.zeros((1, 1)),
                    0.25 * np.eye(d) if d > 1 else np.zeros((1, 1)),
                )
            )
        p0 = MatrixGARCHParameters(*arrays, 0.3 * energy, 0.1, 0.6)
        start = np.clip(codec.pack(p0), lower, upper)
        # The conservative start has a feasible scale even in larger modes.
        for _ in range(1024):
            if feasible_start(start):
                break
            for j, (name, _, _, _) in enumerate(codec.entries, 3):
                if not name.endswith("0"):
                    start[j] *= 0.5
            start[1:3] *= 0.5
        else:
            raise ValueError(
                "no representable feasible initialization in these data units"
            )
    rng = np.random.default_rng(random_state)
    runs, solutions = [], []
    for attempt in range(n_starts):
        x0 = start.copy()
        start_fallback = False
        if attempt:
            candidate = np.clip(
                start + rng.normal(scale=0.1, size=len(start)), lower, upper
            )
            # Contract a proposed perturbation toward a known feasible start.
            for _ in range(40):
                if feasible_start(candidate):
                    break
                candidate = (candidate + start) / 2
            if feasible_start(candidate):
                x0 = candidate
            else:
                start_fallback = True
        invalid = [0]
        history = []
        cached = [None, None, None]

        def objective(x):
            # SLSQP, its callback, and the terminal audit often request exactly
            # the same point. Cache only that point (never approximate matches).
            # Own both arrays: optimizers may mutate inputs or returned gradients.
            if cached[0] is None or not np.array_equal(x, cached[0]):
                try:
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        p = codec.unpack(x)
                        _change_units(p, scale, to_work=False)
                        value, gradient = _objective_gradient(
                            work, p, codec, data_scale=scale
                        )
                except (FloatingPointError, ValueError, np.linalg.LinAlgError):
                    invalid[0] += 1
                    value, gradient = 1e30 + float(x @ x), 2 * x
                cached[:] = [x.copy(), value, gradient.copy()]
            return cached[1], cached[2].copy()

        initial_value = objective(x0)[0]
        history.append(initial_value)
        best = [float(initial_value), x0.copy()]

        def callback(x):
            value = float(objective(x)[0])
            history.append(value)
            if (
                value < best[0]
                and np.all(inequalities(x) >= 0)
                and np.all(x >= lower)
                and np.all(x <= upper)
            ):
                best[:] = [value, x.copy()]

        constraints = (
            [] if constraint == "none" else [{"type": "ineq", "fun": inequalities}]
        )
        fit = minimize(
            objective,
            x0,
            jac=True,
            method="SLSQP",
            bounds=list(zip(lower, upper)),
            constraints=constraints,
            options={"maxiter": max_iter, "ftol": tol},
            callback=callback,
        )
        value, gradient = objective(fit.x)
        violations = np.r_[lower - fit.x, fit.x - upper, -inequalities(fit.x), 0.0]
        terminal_violation = float(np.max(violations))
        retained = (
            value >= 1e29
            or terminal_violation > max(1e-8, tol)
            or (not fit.success and value > best[0])
        )
        solution = best[1] if retained else fit.x
        if retained:
            value, gradient = objective(solution)
        violation = float(
            np.max(
                np.r_[lower - solution, solution - upper, -inequalities(solution), 0.0]
            )
        )
        active = [
            label
            for label, x, lo, hi in zip(codec.labels, solution, lower, upper)
            if min(x - lo, hi - x) < 1e-5
        ]
        for j, v in enumerate(inequalities(solution)):
            if v < 1e-5:
                active.append(f"{constraint} constraint {j}")
        converged = bool(
            fit.success
            and not retained
            and violation <= max(1e-8, tol)
            and value < 1e29
        )
        runs.append(
            MatrixGARCHOptimizationRun(
                float(value),
                converged,
                int(fit.status),
                str(fit.message),
                int(fit.nit),
                int(fit.nfev),
                float(np.linalg.norm(gradient)),
                violation,
                tuple(active),
                np.asarray(history),
                invalid[0],
                retained,
                terminal_violation,
                start_fallback,
            )
        )
        solutions.append(solution)
    eligible = [
        i
        for i, run in enumerate(runs)
        if run.objective < 1e29 and run.constraint_violation <= max(1e-8, tol)
    ]
    if not eligible:
        raise RuntimeError(
            "no optimization start produced a finite feasible matrix GARCH fit"
        )
    selected = min(eligible, key=lambda i: runs[i].objective)
    parameters = _change_units(codec.unpack(solutions[selected]), scale, to_work=False)
    filtered = filter_matrix_garch(X, parameters)
    return MatrixGARCHResult(
        parameters,
        filtered,
        runs[selected].converged,
        runs[selected].n_iter,
        -filtered.log_likelihood / len(X),
        tuple(runs),
        selected,
        dynamics,
        constraint,
        margin,
        scale,
    )
