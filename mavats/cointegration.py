"""Fixed-rank cointegrated matrix autoregression (Li and Xiao, 2024).

Alternating reduced-rank regressions implement the structured error-correction
objective, including corrections to the manuscript's printed update formulas.
This is not stationary reduced-rank MAR or a relabeled vector Johansen fit.
Result spaces, likelihoods and forecasts evaluate this same fitted model.

References
----------
Li, Z. and Xiao, H. (2024), "Cointegrated Matrix Autoregression Models",
arXiv:2409.10860v1, https://arxiv.org/abs/2409.10860v1, equations (3)--(10).
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_triangular

from ._validation import as_series, finite_scalar, positive_int, ranks_tuple


def _array(value, shape, name):
    raw = np.asarray(value)
    if raw.shape != shape or np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a real array of shape {shape}")
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result.copy()


def _design(X, lags):
    difference = np.diff(X, axis=0)
    return (
        difference[lags:],
        X[lags:-1],
        [difference[lags - j : len(difference) - j] for j in range(1, lags + 1)],
    )


def _flat(X):
    return X.transpose(0, 2, 1).reshape(len(X), -1)


def _stack_columns(X):
    return X.transpose(0, 2, 1).reshape(-1, X.shape[1])


def _unstack_columns(X, n, p, q):
    return X.reshape(n, q, p).transpose(0, 2, 1)


def _full_design_svd(design, rcond, name):
    """Column equilibration followed by a strict full-column-rank SVD."""
    scales = np.linalg.norm(design, axis=0)
    if np.any(scales == 0):
        raise ValueError(f"{name} has a zero column; parameters are not identified")
    u, singular, vt = np.linalg.svd(design / scales, full_matrices=False)
    ratio = float(singular[-1] / singular[0]) if len(singular) else 0.0
    if len(singular) != design.shape[1] or ratio <= rcond:
        raise ValueError(f"{name} is rank deficient at rcond={rcond:g}")
    return u, singular, vt, scales, ratio


def _svd_solve(decomposition, response):
    u, singular, vt, scales, _ = decomposition
    return (vt.T @ ((u.T @ response) / singular[:, None])) / scales[:, None]


def _chol(covariance, rcond, name):
    # Restore physical covariances without overflowing C+C.T near max_float,
    # or losing a subnormal diagonal by halving it before addition.
    scale = float(np.max(np.abs(covariance)))
    if scale == 0 or not np.isfinite(scale):
        raise ValueError(f"{name} is zero or nonfinite")
    work = covariance / scale
    work = (work + work.T) * 0.5
    spectrum = np.linalg.eigvalsh(work)
    if spectrum[-1] <= 0 or spectrum[0] <= rcond * spectrum[-1]:
        raise ValueError(
            f"{name} is singular at rcond={rcond:g}; no covariance floor is applied"
        )
    return np.linalg.cholesky(work) * np.sqrt(scale)


def _right_whiten(X, cholesky):
    shape = X.shape
    return solve_triangular(cholesky, X.reshape(-1, shape[-1]).T, lower=True).T.reshape(
        shape
    )


def _block_update(
    Y, levels, differences, opposite, short_opposite, rank, intercept, covariance, rcond
):
    """Exact one-mode LS or profiled Gaussian RRR update using stacked data.

    When covariance is supplied it is the *opposite*-mode covariance. The
    monitored-mode covariance is jointly profiled, not frozen during RRR.
    """
    n, p, q = Y.shape
    whiten = (
        np.eye(q)
        if covariance is None
        else _chol(covariance, rcond, "opposite covariance")
    )
    response = _stack_columns(_right_whiten(Y, whiten))
    level = _stack_columns(_right_whiten(levels @ opposite.T, whiten))
    blocks = [
        _stack_columns(_right_whiten(x @ b.T, whiten))
        for x, b in zip(differences, short_opposite)
    ]
    if intercept:
        identity = _right_whiten(np.eye(q), whiten)
        blocks.append(np.tile(identity.T, (n, 1)))
    nuisance = np.concatenate(blocks, axis=1) if blocks else np.empty((n * q, 0))
    minimum_ratio = 1.0
    if nuisance.shape[1]:
        z_svd = _full_design_svd(nuisance, rcond, "short-run/intercept design")
        qz = z_svd[0]
        y_res = response - qz @ (qz.T @ response)
        x_res = level - qz @ (qz.T @ level)
        minimum_ratio = z_svd[-1]
    else:
        z_svd = None
        y_res, x_res = response, level
    x_svd = _full_design_svd(x_res, rcond, "residualized lagged-level design")
    minimum_ratio = min(minimum_ratio, x_svd[-1])
    qx = x_svd[0]
    if covariance is None:
        response_root = np.eye(p)
    else:
        unrestricted_residual = y_res - qx @ (qx.T @ y_res)
        scatter = unrestricted_residual.T @ unrestricted_residual / (n * q)
        response_root = _chol(scatter, rcond, "unrestricted residual covariance")
    whitened = _right_whiten(y_res, response_root)
    _, singular, vt = np.linalg.svd(qx.T @ whitened, full_matrices=False)
    if singular[0] == 0 or singular[rank - 1] <= rcond * singular[0]:
        raise ValueError("reduced-rank coefficient lies on a lower-rank boundary")
    response_basis = vt[:rank].T
    coefficients = _svd_solve(x_svd, whitened) @ response_basis @ response_basis.T
    coefficient = (coefficients @ response_root.T).T
    psi = (
        _svd_solve(z_svd, response - level @ coefficient.T)
        if z_svd
        else np.empty((0, p))
    )
    short = np.array([psi[j * p : (j + 1) * p].T for j in range(len(differences))])
    if not len(differences):
        short = np.empty((0, p, p))
    constant = psi[len(differences) * p :].T if intercept else np.zeros((p, q))
    residual = response - level @ coefficient.T - nuisance @ psi
    fitted_covariance = (
        residual.T @ residual / (n * q) if covariance is not None else None
    )
    if fitted_covariance is not None:
        _chol(fitted_covariance, rcond, "updated covariance")
    return coefficient, short, constant, fitted_covariance, minimum_ratio


def _normalize_pair(left, right):
    scale = float(np.max(np.abs(left)))
    if scale == 0 or not np.isfinite(scale):
        raise ValueError("a zero coefficient pair has no normalized identification")
    work = left / scale
    norm = float(np.linalg.norm(work))
    # Keep reciprocal coefficient gauges such as (1e200*A, 1e-200*B)
    # equivalent without squaring A or forming its unrepresentable norm.
    right_mantissa, right_exponent = np.frexp(right)
    scale_mantissa, scale_exponent = np.frexp(scale)
    norm_mantissa, norm_exponent = np.frexp(norm)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        right = np.ldexp(
            right_mantissa * scale_mantissa * norm_mantissa,
            right_exponent + scale_exponent + norm_exponent,
        )
    if not np.isfinite(right).all():
        raise FloatingPointError(
            "identified coefficient pair exceeds floating-point range"
        )
    left = work / norm
    pivot = np.unravel_index(np.argmax(np.abs(left)), left.shape)
    if left[pivot] < 0:
        left, right = -left, -right
    return left, right


def _normalize(A1, A2, B1, B2):
    A1, A2 = _normalize_pair(A1, A2)
    for j in range(len(B1)):
        B1[j], B2[j] = _normalize_pair(B1[j], B2[j])
    return A1, A2, B1, B2


def _reduced(matrix, rank):
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    return (u[:, :rank] * s[:rank]) @ vt[:rank]


def _nearest_pair(operator, p, q):
    rearranged = (
        operator.reshape(q, p, q, p).transpose(1, 3, 0, 2).reshape(p * p, q * q)
    )
    u, s, vt = np.linalg.svd(rearranged, full_matrices=False)
    return u[:, 0].reshape(p, p), (s[0] * vt[0]).reshape(q, q)


def _initial(Y, levels, differences, ranks, intercept):
    """An unrestricted OLS/Kronecker *initializer*, never the final estimator."""
    n, p, q = Y.shape
    blocks = [_flat(levels)] + [_flat(x) for x in differences]
    if intercept:
        blocks.append(np.ones((n, 1)))
    design = np.concatenate(blocks, axis=1)
    scales = np.linalg.norm(design, axis=0)
    scales[scales == 0] = 1
    coefficient = (
        np.linalg.lstsq(design / scales, _flat(Y), rcond=None)[0] / scales[:, None]
    )
    d = p * q
    A1, A2 = _nearest_pair(coefficient[:d].T, p, q)
    A1, A2 = _reduced(A1, ranks[0]), _reduced(A2, ranks[1])
    pairs = [
        _nearest_pair(coefficient[(j + 1) * d : (j + 2) * d].T, p, q)
        for j in range(len(differences))
    ]
    B1 = np.array([a for a, _ in pairs]).reshape(-1, p, p)
    B2 = np.array([b for _, b in pairs]).reshape(-1, q, q)
    constant = coefficient[-1].reshape(q, p).T if intercept else np.zeros((p, q))
    return (*_normalize(A1, A2, B1, B2), constant)


def _residual(Y, levels, differences, parameters):
    A1, A2, B1, B2, constant = parameters
    predicted = A1 @ levels @ A2.T + constant
    for x, left, right in zip(differences, B1, B2):
        predicted += left @ x @ right.T
    return Y - predicted


def _objective(residual, covariances, rcond):
    n, p, q = residual.shape
    if covariances is None:
        return float(np.sum(residual**2) / n)
    row, column = covariances
    lr, lc = _chol(row, rcond, "row covariance"), _chol(
        column, rcond, "column covariance"
    )
    white = _right_whiten(residual, lc)
    white = _right_whiten(white.transpose(0, 2, 1), lr)
    return float(
        0.5
        * (
            p * q * np.log(2 * np.pi)
            + 2 * q * np.log(np.diag(lr)).sum()
            + 2 * p * np.log(np.diag(lc)).sum()
            + np.sum(white**2) / n
        )
    )


@dataclass(frozen=True)
class CMAROptimizationRun:
    """One start; finite failed/convergence-limited attempts are not hidden."""

    objective_history: np.ndarray
    converged: bool
    n_iter: int
    failure: str | None
    minimum_design_singular_ratio: float | None

    @property
    def objective(self):
        return (
            float(self.objective_history[-1]) if len(self.objective_history) else None
        )


@dataclass(frozen=True)
class CMARI1Diagnostics:
    """Tolerance-dependent checks of fitted coefficients, not a statistical test.

    Companion eigenvalues are inverse polynomial roots. ``compatible`` requires
    exactly dimension-rank(Pi) roots near +1, all other roots strictly inside
    1-tolerance, and a nonsingular full-complement I(1) condition. A near-boundary
    or defective system may require higher precision or substantive analysis.
    """

    compatible: bool
    long_run_rank: int
    unit_roots_expected: int
    unit_roots_observed: int
    eigenvalues: np.ndarray
    stable_roots_radius: float
    unstable_roots_count: int
    boundary_roots_count: int
    complement_singular_values: np.ndarray
    complement_singular_ratio: float
    long_run_impact: np.ndarray | None
    tolerance: float
    rcond: float


def _level_operators(pi, short):
    if not len(short):
        return np.array([np.eye(len(pi)) + pi])
    return np.array(
        [np.eye(len(pi)) + pi + short[0]]
        + [short[j] - short[j - 1] for j in range(1, len(short))]
        + [-short[-1]]
    )


def cmar_i1_diagnostics(
    A1,
    A2,
    short_run_left=None,
    short_run_right=None,
    *,
    tolerance=1e-6,
    rcond=1e-12,
    max_dense_dimension=256,
):
    """Inspect the complete vectorized CMAR companion and I(1) rank condition.

    Implements the coefficient conditions in Li--Xiao Assumption 2/Theorem 1.
    No mode-wise root shortcut is used. The full complements of
    alpha2 kron alpha1 and beta2 kron beta1 have dimension p*q-r1*r2;
    they are NOT products of the separate mode complements. This dense
    diagnostic costs cubic time in p*q*(difference_lags+1). Its companion
    dimension cannot exceed max_dense_dimension (default 256) without an
    explicit caller override.

    References
    ----------
    Li, Z. and Xiao, H. (2024), "Cointegrated Matrix Autoregression Models",
    Assumption 2 and Theorem 1, https://arxiv.org/abs/2409.10860v1.
    The tolerance-based numerical checks implement coefficient conditions;
    they are not a paper-derived hypothesis test or an empirical I(1) proof.
    """
    raw1, raw2 = np.asarray(A1), np.asarray(A2)
    if (
        raw1.ndim != 2
        or raw2.ndim != 2
        or raw1.shape[0] != raw1.shape[1]
        or raw2.shape[0] != raw2.shape[1]
    ):
        raise ValueError("A1 and A2 must be square matrices")
    p, q = len(raw1), len(raw2)
    if not p or not q:
        raise ValueError("coefficient modes must be nonempty")
    A1, A2 = _array(A1, (p, p), "A1"), _array(A2, (q, q), "A2")
    if (short_run_left is None) != (short_run_right is None):
        raise ValueError("supply both short-run coefficient arrays")
    if short_run_left is not None and np.asarray(short_run_left).ndim != 3:
        raise ValueError("short-run coefficient arrays must have a leading lag axis")
    k = 0 if short_run_left is None else len(short_run_left)
    max_dense_dimension = positive_int(max_dense_dimension, "max_dense_dimension")
    if p * q * (k + 1) > max_dense_dimension:
        raise ValueError(
            "dense companion exceeds max_dense_dimension; explicitly increase the limit if appropriate"
        )
    B1 = (
        np.empty((0, p, p))
        if short_run_left is None
        else _array(short_run_left, (k, p, p), "short_run_left")
    )
    B2 = (
        np.empty((0, q, q))
        if short_run_right is None
        else _array(short_run_right, (k, q, q), "short_run_right")
    )
    tolerance = finite_scalar(tolerance, "tolerance", minimum=0)
    rcond = finite_scalar(rcond, "rcond", minimum=0)
    if not 0 < tolerance < 1 or not 0 < rcond < 1:
        raise ValueError("tolerance and rcond must be strictly between zero and one")
    pi = np.kron(A2, A1)
    short = np.array([np.kron(b2, b1) for b1, b2 in zip(B1, B2)])
    coefficients = _level_operators(pi, short)
    d, order = p * q, k + 1
    companion = np.zeros((d * order, d * order))
    companion[:d] = np.concatenate(coefficients, axis=1)
    if order > 1:
        companion[d:, :-d] = np.eye(d * (order - 1))
    if not np.isfinite(companion).all():
        raise FloatingPointError("dense CMAR operator exceeds floating-point range")
    eigenvalues = np.linalg.eigvals(companion)
    u, singular, vt = np.linalg.svd(pi)
    rank = int(np.sum(singular > rcond * singular[0])) if singular[0] else 0
    alpha_perp, beta_perp = u[:, rank:], vt[rank:].T
    gamma = np.eye(d) - (short.sum(axis=0) if k else 0)
    complement = alpha_perp.T @ gamma @ beta_perp
    singular_complement = np.linalg.svd(complement, compute_uv=False)
    ratio = (
        float(
            singular_complement[-1]
            / max(np.linalg.norm(gamma, 2), np.finfo(float).tiny)
        )
        if len(singular_complement)
        else 0.0
    )
    full = ratio > rcond
    impact = beta_perp @ np.linalg.solve(complement, alpha_perp.T) if full else None
    unit = np.abs(eigenvalues - 1) <= tolerance
    others = np.abs(eigenvalues[~unit])
    radius = float(np.max(others)) if len(others) else 0.0
    return CMARI1Diagnostics(
        compatible=bool(
            d - rank > 0
            and np.sum(unit) == d - rank
            and radius < 1 - tolerance
            and full
        ),
        long_run_rank=rank,
        unit_roots_expected=d - rank,
        unit_roots_observed=int(np.sum(unit)),
        eigenvalues=eigenvalues,
        stable_roots_radius=radius,
        unstable_roots_count=int(np.sum(others > 1 + tolerance)),
        boundary_roots_count=int(
            np.sum((others >= 1 - tolerance) & (others <= 1 + tolerance))
        ),
        complement_singular_values=singular_complement,
        complement_singular_ratio=ratio,
        long_run_impact=impact,
        tolerance=tolerance,
        rcond=rcond,
    )


@dataclass
class CMARResult:
    """Structured error-correction estimates with fixed positive mode ranks.

    A_j=alpha_j beta_j.T; beta_j has orthonormal columns spanning the RIGHT
    singular space. Identification fixes ||A1||_F and each ||B_i1||_F to one
    and a deterministic paired sign. Cointegration spaces, not individual
    rotated vectors, are invariant estimands.

    Histories/objective are scaled mean Frobenius loss for LS, or mean Gaussian
    negative log likelihood in X/data_scale units for MLE. ``log_likelihood``
    restores physical units and includes Gaussian constants. No ridge,
    covariance floor, stationarity projection, rank test or standard errors
    are silently applied. A converged local optimizer is not an I(1) certificate.
    Spaces, diagnostics and forecasts inherit the model reference documented
    in :func:`fit_cmar` and this module.
    """

    A1: np.ndarray
    A2: np.ndarray
    short_run_left: np.ndarray
    short_run_right: np.ndarray
    intercept: np.ndarray
    alpha1: np.ndarray
    alpha2: np.ndarray
    beta1: np.ndarray
    beta2: np.ndarray
    fitted_differences: np.ndarray
    residuals: np.ndarray
    history: np.ndarray
    method: str
    data_scale: float
    row_covariance: np.ndarray | None
    column_covariance: np.ndarray | None
    runs: tuple
    selected_start: int
    rcond: float
    max_dense_dimension: int

    @property
    def ranks(self):
        return self.beta1.shape[1], self.beta2.shape[1]

    @property
    def difference_lags(self):
        return len(self.short_run_left)

    @property
    def converged(self):
        return self.runs[self.selected_start].converged

    @property
    def n_iter(self):
        return self.runs[self.selected_start].n_iter

    @property
    def objective(self):
        return self.runs[self.selected_start].objective

    @property
    def objective_history(self):
        return self.runs[self.selected_start].objective_history

    @property
    def cointegration_rank(self):
        return self.ranks[0] * self.ranks[1]

    @property
    def cointegrating_vectors(self):
        self._guard_dense()
        return np.kron(self.beta2, self.beta1)

    @property
    def cointegration_projectors(self):
        return self.beta1 @ self.beta1.T, self.beta2 @ self.beta2.T

    @property
    def long_run_operator(self):
        self._guard_dense()
        return np.kron(self.A2, self.A1)

    @property
    def short_run_operators(self):
        self._guard_dense(max(1, self.difference_lags))
        d = self.A1.shape[0] * self.A2.shape[0]
        return np.array(
            [
                np.kron(b2, b1)
                for b1, b2 in zip(self.short_run_left, self.short_run_right)
            ]
        ).reshape(-1, d, d)

    @property
    def level_coefficients(self):
        self._guard_dense(self.difference_lags + 1)
        return _level_operators(self.long_run_operator, self.short_run_operators)

    def _guard_dense(self, order=1):
        if self.A1.shape[0] * self.A2.shape[0] * order > self.max_dense_dimension:
            raise ValueError(
                "dense representation exceeds max_dense_dimension; increase the explicit result limit if appropriate"
            )

    @property
    def log_likelihood(self):
        if self.method != "mle":
            return None
        d = self.A1.shape[0] * self.A2.shape[0]
        return -len(self.residuals) * (self.objective + d * np.log(self.data_scale))

    def cointegrating_scores(self, X):
        """Return beta1.T X_t beta2, without centering or statistical testing."""
        X = as_series(X, min_samples=1)
        if X.shape[1:] != self.intercept.shape:
            raise ValueError("observations must match fitted matrix dimensions")
        scale = float(np.max(np.abs(X)))
        if scale == 0:
            return np.zeros((len(X), *self.ranks))
        with np.errstate(over="ignore", invalid="ignore"):
            scores = (self.beta1.T @ (X / scale) @ self.beta2) * scale
        if not np.isfinite(scores).all():
            raise FloatingPointError("cointegrating scores exceed physical data units")
        return scores

    def i1_diagnostics(self, *, tolerance=1e-6, max_dense_dimension=None):
        return cmar_i1_diagnostics(
            self.A1,
            self.A2,
            self.short_run_left,
            self.short_run_right,
            tolerance=tolerance,
            rcond=self.rcond,
            max_dense_dimension=(
                self.max_dense_dimension
                if max_dense_dimension is None
                else max_dense_dimension
            ),
        )

    def forecast(self, steps, history=None):
        """Recursive LEVEL forecasts; only supplied past observations are used.

        Need difference_lags+1 chronological levels. Parameters are never
        refitted, and future innovation means are zero. An unrestricted
        intercept is carried in the difference equation, not added just once.
        """
        steps = positive_int(steps, "steps")
        past = (
            self.history
            if history is None
            else as_series(history, min_samples=self.difference_lags + 1)
        )
        if past.shape[1:] != self.intercept.shape:
            raise ValueError("history must match fitted matrix dimensions")
        states = list(past[-self.difference_lags - 1 :].copy())
        output = []
        for _ in range(steps):
            scale = (
                max(
                    float(np.max(np.abs(states[-self.difference_lags - 1 :]))),
                    float(np.max(np.abs(self.intercept))),
                )
                or 1.0
            )
            work = np.array(states[-self.difference_lags - 1 :]) / scale
            difference = self.A1 @ work[-1] @ self.A2.T + self.intercept / scale
            for j, (left, right) in enumerate(
                zip(self.short_run_left, self.short_run_right), start=1
            ):
                difference += left @ (work[-j] - work[-j - 1]) @ right.T
            with np.errstate(over="ignore", invalid="ignore"):
                prediction = (work[-1] + difference) * scale
            if not np.isfinite(prediction).all():
                raise FloatingPointError("CMAR forecast exceeds floating-point range")
            states.append(prediction)
            output.append(prediction)
        return np.array(output)


def fit_cmar(
    X,
    ranks,
    *,
    difference_lags=0,
    intercept=True,
    method="ls",
    initial=None,
    n_starts=3,
    max_iter=200,
    tol=1e-8,
    rcond=1e-12,
    random_state=0,
    init="ols",
    max_dense_dimension=256,
):
    """Fit Li--Xiao's fixed-rank matrix VECM by alternating exact block updates.

    ``difference_lags=k`` means a level VAR order of k+1; observations are
    time-first and the first usable response has index k+2 (one-based).
    ``intercept=True`` estimates an unrestricted matrix D in differences.
    Restricted cointegration intercepts and deterministic time trends are not
    implemented; no centering or detrending is implicit.

    Positive ``ranks=(r1,r2)`` are fixed before fitting, rj<=dimension_j and
    r1*r2<p*q. One full mode, including the vector limit, is permitted; zero
    ranks and the full-rank stationary boundary are outside this interface.
    Rank selection/Johansen tests and asymptotic inference are NOT provided.

    ``method='ls'`` minimizes Frobenius residual loss with unrestricted
    innovation covariance. ``'mle'`` profiles separable Gaussian covariance
    (Sigma2 kron Sigma1), with trace(Sigma1)=p for identification. Its finite
    positive-definite conditional MLE need not exist for degenerate samples.
    Rank-deficient block designs and singular residual covariance are rejected
    at the relative ``rcond`` tolerance, without generalized-inverse or ridge
    substitution. See notes for manuscript corrections and assumptions.

    Default init='ols' is dense unrestricted VECM OLS followed by nearest-Kronecker
    and mode-rank truncation; this is ONLY a start for structured optimization.
    Its predictor dimension p*q*(k+1) is limited by max_dense_dimension=256.
    Set init='random' for dimension-sized rank-projector starts without a dense
    vector fit, or increase the explicit allocation limit. The limit also
    guards result dense operators and I(1) companion diagnostics.
    Additional starts perturb the base using an independent local RNG. ``initial``
    may instead be a mapping with A1/A2 and, for k>0, short_run_left/right;
    optional intercept is in physical data units. All arrays are copied.
    Every start and failed attempt is retained. The smallest finite completed
    objective is selected even if its optimizer did not converge. Tolerance is
    relative mean loss for LS, absolute mean NLL increment per scalar entry for
    MLE, so additive likelihood unit constants do not alter stopping. Together
    with max_iter it governs local convergence, not global optimality.

    Global nonzero rescaling uses X/max(abs(X)); dimensionless coefficients and
    subspaces are unchanged up to numerical error. MLE covariance arrays must
    remain representable in physical units; otherwise rescale the input.

    References
    ----------
    Li, Z. and Xiao, H. (2024), "Cointegrated Matrix Autoregression Models",
    Sections 3.1--3.2, https://arxiv.org/abs/2409.10860v1.
    The implementation follows the structured residual objectives with the
    algebraic corrections recorded in docs/cointegration-notes.md; numerical
    initialization, multistart selection and stopping rules are explicit
    package choices, not additional rank-selection or inference results.
    """
    k = positive_int(difference_lags, "difference_lags", minimum=0)
    X = as_series(X, min_samples=k + 3)
    p, q = X.shape[1:]
    ranks = ranks_tuple(ranks, (p, q))
    if ranks is None or ranks[0] * ranks[1] >= p * q:
        raise ValueError("supply positive mode ranks with product below p*q")
    if not isinstance(intercept, (bool, np.bool_)):
        raise ValueError("intercept must be boolean")
    if not isinstance(method, str) or method not in ("ls", "mle"):
        raise ValueError("method must be 'ls' or 'mle'")
    n_starts = positive_int(n_starts, "n_starts")
    max_iter = positive_int(max_iter, "max_iter")
    max_dense_dimension = positive_int(max_dense_dimension, "max_dense_dimension")
    if not isinstance(init, str) or init not in ("ols", "random"):
        raise ValueError("init must be 'ols' or 'random'")
    tol = finite_scalar(tol, "tol", minimum=0)
    rcond = finite_scalar(rcond, "rcond", minimum=0)
    if tol == 0 or not 0 < rcond < 1:
        raise ValueError("tol must be positive and rcond must lie in (0,1)")
    scale = float(np.max(np.abs(X)))
    if scale == 0:
        raise ValueError("zero data have no identifiable cointegration parameters")
    Y, levels, differences = _design(X / scale, k)
    rng = (
        deepcopy(random_state)
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )
    if initial is None:
        if init == "ols":
            if p * q * (k + 1) > max_dense_dimension:
                raise ValueError(
                    "dense initialization exceeds max_dense_dimension; use init='random', explicit initial coefficients, or increase the limit"
                )
            base = _initial(Y, levels, differences, ranks, intercept)
        else:
            q1 = np.linalg.qr(rng.normal(size=(p, ranks[0])))[0]
            q2 = np.linalg.qr(rng.normal(size=(q, ranks[1])))[0]
            b1 = np.repeat(np.eye(p)[None], k, axis=0)
            b2 = np.repeat((0.05 * np.eye(q))[None], k, axis=0)
            base = (
                *_normalize(-q1 @ q1.T, 0.2 * q2 @ q2.T, b1, b2),
                Y.mean(axis=0) if intercept else np.zeros((p, q)),
            )
    else:
        if not isinstance(initial, dict):
            raise ValueError("initial must be a coefficient mapping")
        unknown = set(initial) - {
            "A1",
            "A2",
            "short_run_left",
            "short_run_right",
            "intercept",
        }
        if unknown:
            raise ValueError(f"unknown initialization fields: {sorted(unknown)}")
        try:
            a1 = _array(initial["A1"], (p, p), "initial A1")
            a2 = _array(initial["A2"], (q, q), "initial A2")
            b1 = _array(
                initial.get("short_run_left", np.empty((0, p, p))),
                (k, p, p),
                "initial short_run_left",
            )
            b2 = _array(
                initial.get("short_run_right", np.empty((0, q, q))),
                (k, q, q),
                "initial short_run_right",
            )
            constant = (
                _array(
                    initial.get("intercept", np.zeros((p, q))),
                    (p, q),
                    "initial intercept",
                )
                / scale
            )
        except KeyError as exc:
            raise ValueError("initial must supply A1 and A2") from exc
        if not intercept and np.any(constant):
            raise ValueError("initial intercept must be zero when intercept=False")
        for a, r in zip((a1, a2), ranks):
            entry_scale = float(np.max(np.abs(a)))
            s = (
                np.linalg.svd(a / entry_scale, compute_uv=False)
                if entry_scale
                else np.zeros(len(a))
            )
            if np.sum(s > rcond * s[0]) != r:
                raise ValueError("initial coefficient ranks must equal the fixed ranks")
        base = (*_normalize(a1, a2, b1, b2), constant)
    runs, solutions = [], []
    for start in range(n_starts):
        parameters = tuple(a.copy() for a in base)
        if start:
            a1, a2, b1, b2, constant = parameters
            a1 = _reduced(
                a1 + 0.3 * np.linalg.norm(a1) * rng.normal(size=a1.shape) / np.sqrt(p),
                ranks[0],
            )
            a2 = _reduced(
                a2 + 0.3 * np.linalg.norm(a2) * rng.normal(size=a2.shape) / np.sqrt(q),
                ranks[1],
            )
            for b, dim in ((b1, p), (b2, q)):
                for j in range(k):
                    b[j] += (
                        0.3
                        * np.linalg.norm(b[j])
                        * rng.normal(size=b[j].shape)
                        / np.sqrt(dim)
                    )
            parameters = (*_normalize(a1, a2, b1, b2), constant)
        covariances = (
            None
            if method == "ls"
            else (
                np.eye(p),
                np.eye(q) * max(float(np.mean(Y**2)), np.finfo(float).tiny),
            )
        )
        history, completed, converged, failure, ratio = [], 0, False, None, 1.0
        try:
            history.append(
                _objective(
                    _residual(Y, levels, differences, parameters), covariances, rcond
                )
            )
            if not np.isfinite(history[-1]):
                raise FloatingPointError("initial objective is not finite")
            for iteration in range(1, max_iter + 1):
                a1, a2, b1, b2, constant = (a.copy() for a in parameters)
                row, column = covariances if covariances is not None else (None, None)
                a1, b1, constant, row, ratio1 = _block_update(
                    Y, levels, differences, a2, b2, ranks[0], intercept, column, rcond
                )
                a2, b2, constant_t, column, ratio2 = _block_update(
                    Y.transpose(0, 2, 1),
                    levels.transpose(0, 2, 1),
                    [x.transpose(0, 2, 1) for x in differences],
                    a1,
                    b1,
                    ranks[1],
                    intercept,
                    row,
                    rcond,
                )
                candidate = (*_normalize(a1, a2, b1, b2), constant_t.T)
                next_covariances = None
                if method == "mle":
                    normalization = np.trace(row) / p
                    next_covariances = row / normalization, column * normalization
                objective = _objective(
                    _residual(Y, levels, differences, candidate),
                    next_covariances,
                    rcond,
                )
                # LS uses relative loss, never an additive 1 that turns tiny
                # scaled residuals into an absolute early-stop tolerance.
                # Gaussian NLL increments per entry are invariant to additive
                # unit/Jacobian constants; relative NLL increments are not.
                stopping_scale = (
                    max(abs(history[-1]), np.finfo(float).tiny)
                    if method == "ls"
                    else p * q
                )
                if (
                    not np.isfinite(objective)
                    or objective > history[-1] + 1e-8 * stopping_scale
                ):
                    raise FloatingPointError(
                        "alternating objective increased or became nonfinite"
                    )
                parameters, covariances = candidate, next_covariances
                completed = iteration
                ratio = min(ratio, ratio1, ratio2)
                history.append(objective)
                if abs(history[-2] - objective) <= tol * stopping_scale:
                    converged = True
                    break
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            failure = str(exc)
        runs.append(
            CMAROptimizationRun(
                np.asarray(history),
                converged,
                completed,
                failure,
                ratio if completed else None,
            )
        )
        solutions.append((parameters, covariances) if completed else None)
    eligible = [i for i, solution in enumerate(solutions) if solution is not None]
    if not eligible:
        reasons = "; ".join(
            dict.fromkeys(run.failure or "no completed update" for run in runs)
        )
        raise ValueError(
            f"no CMAR start completed a finite identified update: {reasons}"
        )
    selected = min(eligible, key=lambda i: runs[i].objective)
    parameters, covariances = solutions[selected]
    a1, a2, b1, b2, constant = parameters
    beta1 = np.linalg.svd(a1, full_matrices=False)[2][: ranks[0]].T
    beta2 = np.linalg.svd(a2, full_matrices=False)[2][: ranks[1]].T
    residual = _residual(Y, levels, differences, parameters)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        physical_residual, fitted, constant = (
            residual * scale,
            (Y - residual) * scale,
            constant * scale,
        )
        row, column = (
            (None, None)
            if covariances is None
            else (covariances[0], (covariances[1] * scale) * scale)
        )
    if not all(np.isfinite(a).all() for a in (physical_residual, fitted, constant)):
        raise FloatingPointError(
            "CMAR residuals or intercept exceed physical data units"
        )
    if column is not None:
        if not np.isfinite(column).all():
            raise FloatingPointError(
                "CMAR physical covariance overflows; rescale input"
            )
        _chol(column, rcond, "physical column covariance")
    return CMARResult(
        a1,
        a2,
        b1,
        b2,
        constant,
        a1 @ beta1,
        a2 @ beta2,
        beta1,
        beta2,
        fitted,
        physical_residual,
        X[-k - 1 :].copy(),
        method,
        scale,
        row,
        column,
        tuple(runs),
        selected,
        rcond,
        max_dense_dimension,
    )
