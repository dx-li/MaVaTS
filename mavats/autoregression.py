"""Bilinear matrix autoregression with column-major vectorization.

The MAR(1) projection, least-squares and separable Gaussian likelihood
estimators follow Chen, Xiao and Yang (2021), *Journal of Econometrics* 222,
539--560, https://doi.org/10.1016/j.jeconom.2020.07.015 (Section 3).
Multiple lags, intercept profiling and the optional ridge penalty are
extensions implemented here; the paper's asymptotic results concern MAR(1).
"""

from dataclasses import dataclass, field

import numpy as np

from ._validation import as_series, positive_int, random_generator, ranks_tuple


def _nonnegative(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be a finite nonnegative number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite nonnegative number") from exc
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _normalize(A, B):
    """Fix scale and sign without changing B kron A or mutating inputs."""
    scale = np.max(np.abs(A))
    if scale == 0:
        canonical = np.zeros_like(A)
        canonical[0, 0] = 1
        return canonical, np.zeros_like(B)
    norm = np.linalg.norm(A / scale)
    left, right = (A / scale) / norm, (B * scale) * norm
    if left.flat[np.argmax(np.abs(left))] < 0:
        left, right = -left, -right
    return left, right


def _rearrange_phi(Phi, m, n):
    # (column output, row output, column input, row input) -> vec(A), vec(B).
    return Phi.reshape(n, m, n, m).transpose(3, 1, 2, 0).reshape(m * m, n * n)


def nearest_kronecker_product(coefficient, shape):
    """Return A, B minimizing ``||coefficient - kron(B, A)||_F``.

    ``shape=(m,n)`` gives square factors of size m and n. A has unit
    Frobenius norm, with its largest-magnitude entry nonnegative. The
    approximation is obtained by the leading singular triplet after the
    Van Loan rearrangement; repeated leading singular values imply a
    nonunique answer. No explicit inverse or normal equations are used.
    """
    if len(shape) != 2:
        raise ValueError("shape must contain row and column dimensions")
    m, n = (positive_int(v, "dimension") for v in shape)
    raw = np.asarray(coefficient)
    if np.iscomplexobj(raw):
        raise ValueError("coefficient must be real-valued")
    Phi = np.asarray(raw, dtype=float)
    if Phi.shape != (m * n, m * n) or not np.isfinite(Phi).all():
        raise ValueError("coefficient must be finite with shape (m*n, m*n)")
    u, s, vh = np.linalg.svd(_rearrange_phi(Phi, m, n), full_matrices=False)
    return _normalize(
        u[:, 0].reshape(m, m, order="F"), (s[0] * vh[0]).reshape(n, n, order="F")
    )


def _lstsq(design, response, ridge=0.0, rank=None):
    if ridge:
        design = np.concatenate((design, np.sqrt(ridge) * np.eye(design.shape[1])))
        response = np.concatenate(
            (response, np.zeros((design.shape[1], response.shape[1])))
        )
    coefficient = np.linalg.lstsq(design, response, rcond=None)[0]
    if rank is not None and rank < min(coefficient.shape):
        # Rank-constrained regression truncates fitted responses, NOT the
        # coefficient matrix. The latter is incorrect for nonwhite designs.
        _, _, vh = np.linalg.svd(design @ coefficient, full_matrices=False)
        basis = vh[:rank].T
        coefficient = coefficient @ basis @ basis.T
    return coefficient


def _left_update(Y, Z, B, ridge=0.0, whitening=None, rank=None):
    D = Z @ B.T
    if whitening is not None:
        D, Y = D @ whitening, Y @ whitening
    m = Y.shape[1]
    return _lstsq(
        D.transpose(0, 2, 1).reshape(-1, m),
        Y.transpose(0, 2, 1).reshape(-1, m),
        ridge,
        rank,
    ).T


def _right_update(Y, Z, A, ridge=0.0, whitening=None, rank=None):
    return _left_update(
        Y.transpose(0, 2, 1), Z.transpose(0, 2, 1), A, ridge, whitening, rank
    )


def _whitener(covariance):
    values, vectors = np.linalg.eigh(covariance)
    if values[0] <= 0:
        raise ValueError(
            "separable covariance is singular; increase covariance_floor or use ALS"
        )
    return (vectors / np.sqrt(values)) @ vectors.T


def _floor_covariance(covariance, floor):
    covariance = (covariance + covariance.T) / 2
    values, vectors = np.linalg.eigh(covariance)
    scale = float(np.max(values))
    if scale <= 0:
        raise ValueError(
            "separable Gaussian MLE is undefined for zero residual variance"
        )
    threshold = floor * scale
    active = bool(values[0] < threshold)
    if values[0] <= 0 and not floor:
        raise ValueError(
            "separable covariance is singular; increase covariance_floor or use ALS"
        )
    if active:
        covariance = (vectors * np.maximum(values, threshold)) @ vectors.T
    return covariance, active


def _covariance_update(residuals, row_cov, col_cov, floor):
    N, m, n = residuals.shape
    white = _whitener(row_cov) @ residuals
    col_cov, c_active = _floor_covariance(
        np.einsum("tij,tik->jk", white, white) / (N * m), floor
    )
    white = residuals @ _whitener(col_cov)
    row_cov, r_active = _floor_covariance(
        np.einsum("tij,tkj->ik", white, white) / (N * n), floor
    )
    scale = np.linalg.norm(row_cov)
    return row_cov / scale, col_cov * scale, c_active or r_active


def _negative_loglike(residuals, row_cov, col_cov):
    N, m, n = residuals.shape
    white = _whitener(row_cov) @ residuals @ _whitener(col_cov)
    return float(
        0.5
        * (
            N
            * (
                m * n * np.log(2 * np.pi)
                + n * np.linalg.slogdet(row_cov)[1]
                + m * np.linalg.slogdet(col_cov)[1]
            )
            + np.sum(white**2)
        )
    )


def _prediction(lags, left, right):
    fitted = np.zeros_like(lags[0])
    for Z, A, B in zip(lags, left, right):
        fitted += A @ Z @ B.T
    return fitted


@dataclass
class MARResult:
    """Fitted ``X[t] = intercept + sum(A[l] X[t-l-1] B[l].T) + E[t]``.

    ``left`` and ``right`` retain a leading lag axis, including at order one.
    Fitted values and residuals correspond to times ``order:``. Stability is
    diagnosed, not enforced. Iterative estimators find a stationary local
    solution; ``converged`` does not certify global optimality.

    Objective histories refer to observations divided by ``data_scale``.
    For ALS/projection multiply by ``data_scale**2`` for the physical SSE
    plus penalty, when representable. For MLE add
    ``residuals.size * log(data_scale)`` for the physical negative log
    likelihood. MLE covariance factors are in physical units with the scale
    shared between factors: ``||row_covariance||_F = data_scale``. Their
    Kronecker product is the physical innovation covariance; this product
    need not itself fit floating-point range at extreme data scales.
    """

    left: np.ndarray
    right: np.ndarray
    intercept: np.ndarray
    fitted_values: np.ndarray
    residuals: np.ndarray
    method: str
    converged: bool
    n_iter: int
    objective_history: np.ndarray
    row_covariance: np.ndarray | None = None
    column_covariance: np.ndarray | None = None
    covariance_regularized: bool = False
    ridge: float = 0.0
    ranks: tuple | None = None
    data_scale: float = 1.0
    _history: np.ndarray = field(default_factory=lambda: np.empty(0), repr=False)

    @property
    def order(self):
        return len(self.left)

    @property
    def A(self):
        """Left coefficient for MAR(1); use ``left`` for higher orders."""
        if self.order != 1:
            raise ValueError("A is only defined for order one; use left")
        return self.left[0]

    @property
    def B(self):
        """Right coefficient for MAR(1); use ``right`` for higher orders."""
        if self.order != 1:
            raise ValueError("B is only defined for order one; use right")
        return self.right[0]

    @property
    def coefficients(self):
        """Dense column-major VAR coefficients; allocates p*(m*n)**2 entries."""
        return np.stack([np.kron(B, A) for A, B in zip(self.left, self.right)])

    @property
    def spectral_radius(self):
        """Companion spectral radius (dense companion for order > 1)."""
        if self.order == 1:
            return float(
                np.max(np.abs(np.linalg.eigvals(self.A)))
                * np.max(np.abs(np.linalg.eigvals(self.B)))
            )
        phi = self.coefficients
        d = phi.shape[1]
        companion = np.zeros((self.order * d, self.order * d))
        companion[:d] = np.concatenate(phi, axis=1)
        companion[d:, :-d] = np.eye((self.order - 1) * d)
        return float(np.max(np.abs(np.linalg.eigvals(companion))))

    @property
    def is_stable(self):
        return self.spectral_radius < 1.0

    @property
    def log_likelihood(self):
        """Conditional Gaussian log likelihood, available for method='mle'."""
        if self.row_covariance is None:
            return None
        return -_negative_loglike(
            self.residuals, self.row_covariance, self.column_covariance
        )

    def residual_covariance(self, *, ddof=0):
        """Centered covariance of column-major residuals (dense mn by mn)."""
        if (
            isinstance(ddof, (bool, np.bool_))
            or not isinstance(ddof, (int, np.integer))
            or ddof < 0
        ):
            raise ValueError("ddof must be a nonnegative integer")
        if len(self.residuals) <= ddof:
            raise ValueError("ddof must be less than the number of residuals")
        flat = self.residuals.transpose(0, 2, 1).reshape(len(self.residuals), -1)
        flat = flat - flat.mean(axis=0)
        return flat.T @ flat / (len(flat) - ddof)

    def forecast(self, steps, history=None):
        """Recursively forecast future matrices using the most recent p values.

        A supplied history must have shape ``(T,m,n)`` in chronological
        order and contain at least ``order`` observations. The result has
        shape ``(steps,m,n)``. These are conditional mean forecasts.
        """
        steps = positive_int(steps, "steps")
        observed = (
            self._history
            if history is None
            else as_series(history, min_samples=self.order)
        )
        if observed.shape[1:] != self.intercept.shape:
            raise ValueError("history matrix dimensions do not match fitted model")
        values = [x.copy() for x in observed[-self.order :]]
        forecasts = []
        for _ in range(steps):
            scale = (
                max(
                    np.max(np.abs(self.intercept)),
                    max(np.max(np.abs(x)) for x in values[-self.order :]),
                )
                or 1.0
            )
            with np.errstate(over="ignore", invalid="ignore"):
                prediction = self.intercept / scale
                for lag, (A, B) in enumerate(zip(self.left, self.right), 1):
                    prediction += A @ (values[-lag] / scale) @ B.T
                prediction *= scale
            if not np.isfinite(prediction).all():
                raise FloatingPointError("MAR forecast exceeds floating-point range")
            forecasts.append(prediction)
            values.append(prediction)
        return np.stack(forecasts)


def fit_mar(
    X,
    *,
    method="als",
    order=1,
    fit_intercept=False,
    max_iter=200,
    tol=1e-8,
    ridge=0.0,
    init="auto",
    initial=None,
    random_state=None,
    covariance_floor=1e-8,
    initial_covariance=None,
    ranks=None,
):
    """Fit matrix autoregression with one bilinear term per lag.

    Parameters
    ----------
    X : array_like, shape (T, m, n)
        Finite real observations, with time on the first axis.
    method : {'als', 'projection', 'mle'}
        ALS minimizes squared prediction error. Projection fits a dense
        unrestricted VAR, then projects each lag onto a Kronecker product.
        MLE uses alternating weighted regressions and row/column covariance
        updates for Gaussian innovations with separable covariance.
    order : int, default 1
        Number of lags, with a distinct pair of factors for each lag.
    fit_intercept : bool, default False
        Profile a free matrix intercept by separately centering the response
        and each lag predictor over the regression sample.
    max_iter : int, default 200
        Maximum complete coordinate sweeps for iterative methods.
    tol : float, default 1e-8
        Relative objective-change tolerance. Check ``result.converged``.
    ridge : float, default 0
        ALS objective adds ``ridge * sum(||A_l||_F**2 * ||B_l||_F**2)``.
        This equals a ridge penalty on the structured VAR coefficients and
        is invariant to factor rescaling. Unsupported for MLE/projection.
    init : {'auto', 'projection', 'identity', 'random'}, default 'auto'
        Auto uses projection when m*n <= 256, identity otherwise. Identity
        and random avoid allocating a dense unrestricted VAR. Initialization
        affects local solutions; compare starts on difficult problems.
    initial : pair of arrays, optional
        Left/right factors with a leading lag axis (2-D accepted for order
        one), copied before fitting. Overrides ``init``.
    random_state : int or numpy.random.Generator, optional
        Random initializer seed; never changes NumPy's global RNG.
    covariance_floor : float, default 1e-8
        MLE covariance eigenvalues are floored relative to their largest
        eigenvalue. If flooring occurs, ``covariance_regularized`` is True:
        that result is a stabilized likelihood estimate, not an unregularized
        MLE. Set zero for strict positive-definite covariance updates.
    initial_covariance : (row_covariance, column_covariance), optional
        Positive-definite starting covariances for MLE.
    ranks : (row_rank, column_rank), optional
        Upper bounds on the ranks of A and B for every lag. Supported by
        ALS; each block uses exact reduced-rank regression, including the
        augmented-design ridge objective when requested. For order one and
        ridge zero this is RR.LS, Section 3.1 of Xiao, Han, Chen and Liu,
        "Reduced Rank Autoregressive Models for Matrix Time Series",
        https://yuefenghan.github.io/papers/Reduced_Rank_MAR.pdf.
        The paper's distinct RR.CC estimator is not implemented here.

    Returns
    -------
    MARResult
        Coefficients, residuals, recursive forecasting and fit diagnostics.
        Objective histories use data divided by ``result.data_scale``:
        ALS/projection records scaled residual sum of squares plus the
        equivalently scaled ridge penalty; MLE records negative conditional
        Gaussian log likelihood of the scaled data. ``log_likelihood`` is
        in original data units. See MARResult for conversion formulas.

    Notes
    -----
    Regressions use SVD least squares on the design matrix, including an
    augmented design for ridge, avoiding squared conditioning from normal
    equations. ALS works with matrix-sized designs. Projection explicitly
    allocates a dense VAR coefficient of size p*(m*n)**2, so is unsuitable
    for large matrices. No estimator constrains stationarity or handles
    missing observations. Finite separable MLEs need not exist for small or
    degenerate samples; singular updates raise when flooring is disabled.

    Examples
    --------
    >>> rng = np.random.default_rng(7)
    >>> data = rng.normal(size=(100, 2, 3))
    >>> result = fit_mar(data, fit_intercept=True)
    >>> result.forecast(5).shape
    (5, 2, 3)
    """
    order = positive_int(order, "order")
    X = as_series(X, min_samples=order + 1)
    max_iter = positive_int(max_iter, "max_iter")
    tol, ridge = _nonnegative(tol, "tol"), _nonnegative(ridge, "ridge")
    covariance_floor = _nonnegative(covariance_floor, "covariance_floor")
    if covariance_floor >= 1:
        raise ValueError("covariance_floor must be less than one")
    if not isinstance(fit_intercept, (bool, np.bool_)):
        raise ValueError("fit_intercept must be boolean")
    if method not in ("als", "projection", "mle"):
        raise ValueError("method must be 'als', 'projection', or 'mle'")
    if ridge and method != "als":
        raise ValueError("ridge is supported only for method='als'")
    if init not in ("auto", "projection", "identity", "random"):
        raise ValueError("init must be 'auto', 'projection', 'identity', or 'random'")
    if initial_covariance is not None and method != "mle":
        raise ValueError("initial_covariance is supported only for method='mle'")
    T, m, n = X.shape
    original = X
    original_ridge = ridge
    data_scale = float(max(np.max(np.abs(X)), np.sqrt(ridge))) or 1.0
    X = X / data_scale
    ridge = (np.sqrt(ridge) / data_scale) ** 2
    ranks = ranks_tuple(ranks, (m, n))
    if ranks is not None and method != "als":
        raise ValueError("rank constraints are supported only for method='als'")
    row_rank, col_rank = ranks if ranks is not None else (None, None)
    raw_lags = [X[order - lag : T - lag] for lag in range(1, order + 1)]
    response_mean = X[order:].mean(axis=0) if fit_intercept else np.zeros((m, n))
    lag_means = [
        z.mean(axis=0) if fit_intercept else np.zeros((m, n)) for z in raw_lags
    ]
    Y = X[order:] - response_mean
    lags = [z - mean for z, mean in zip(raw_lags, lag_means)]
    use_projection = method == "projection" or (
        initial is None and (init == "projection" or (init == "auto" and m * n <= 256))
    )
    if use_projection:
        design = np.concatenate(
            [z.transpose(0, 2, 1).reshape(T - order, -1) for z in lags], axis=1
        )
        Phi = _lstsq(design, Y.transpose(0, 2, 1).reshape(T - order, -1)).T
        pairs = [
            nearest_kronecker_product(Phi[:, j * m * n : (j + 1) * m * n], (m, n))
            for j in range(order)
        ]
        left, right = (np.stack(part) for part in zip(*pairs))
    elif initial is not None:
        if len(initial) != 2:
            raise ValueError("initial must contain left and right coefficients")
        arrays = []
        for raw, dim in zip(initial, (m, n)):
            if np.iscomplexobj(raw):
                raise ValueError("initial coefficients must be real-valued")
            value = np.array(raw, dtype=float, copy=True)
            if order == 1 and value.ndim == 2:
                value = value[None]
            if value.shape != (order, dim, dim) or not np.isfinite(value).all():
                raise ValueError(
                    "initial coefficient dimensions do not match order and X"
                )
            arrays.append(value)
        left, right = arrays
    elif init == "random":
        rng = random_generator(random_state)
        left, right = rng.normal(size=(order, m, m)), rng.normal(size=(order, n, n))
    else:
        left = np.repeat(np.eye(m)[None], order, axis=0)
        right = np.repeat(np.eye(n)[None], order, axis=0) / order
    for j in range(order):
        if ranks is not None:
            for coefficients, rank in ((left, row_rank), (right, col_rank)):
                u, s, vh = np.linalg.svd(coefficients[j], full_matrices=False)
                coefficients[j] = (u[:, :rank] * s[:rank]) @ vh[:rank]
        left[j], right[j] = _normalize(left[j], right[j])
    row_cov, col_cov = np.eye(m), np.eye(n)
    if initial_covariance is not None:
        if len(initial_covariance) != 2:
            raise ValueError(
                "initial_covariance must contain row and column covariances"
            )
        covs = []
        for raw, dim in zip(initial_covariance, (m, n)):
            if np.iscomplexobj(raw):
                raise ValueError("initial covariances must be real-valued")
            cov = np.array(raw, dtype=float, copy=True)
            if (
                cov.shape != (dim, dim)
                or not np.isfinite(cov).all()
                or not np.allclose(cov, cov.T)
            ):
                raise ValueError(
                    "initial covariance must be finite, symmetric and dimensionally compatible"
                )
            cov = cov / data_scale
            cov = cov / 2 + cov.T / 2
            _whitener(cov)
            covs.append(cov)
        row_cov, col_cov = covs
    residuals = Y - _prediction(lags, left, right)

    def objective():
        if method == "mle":
            return _negative_loglike(residuals, row_cov, col_cov)
        penalty = ridge * sum(
            np.sum(A * A) * np.sum(B * B) for A, B in zip(left, right)
        )
        return float(np.sum(residuals**2) + penalty)

    objectives = [objective()]
    if not np.isfinite(objectives[0]):
        raise FloatingPointError(
            "initial MAR objective is not finite; check starting parameters"
        )
    converged, n_iter, regularized = method == "projection", 0, False
    if method != "projection":
        for n_iter in range(1, max_iter + 1):
            for j, Z in enumerate(lags):
                target = residuals + left[j] @ Z @ right[j].T
                left[j] = _left_update(
                    target,
                    Z,
                    right[j],
                    ridge * np.sum(right[j] ** 2),
                    _whitener(col_cov) if method == "mle" else None,
                    row_rank,
                )
                # A zero block can be represented by nonzero A and zero B,
                # allowing the subsequent B update to leave that boundary.
                left[j], right[j] = _normalize(left[j], right[j])
                right[j] = _right_update(
                    target,
                    Z,
                    left[j],
                    ridge * np.sum(left[j] ** 2),
                    _whitener(row_cov) if method == "mle" else None,
                    col_rank,
                )
                left[j], right[j] = _normalize(left[j], right[j])
                residuals = target - left[j] @ Z @ right[j].T
            if method == "mle":
                row_cov, col_cov, active = _covariance_update(
                    residuals, row_cov, col_cov, covariance_floor
                )
                regularized |= active
            current = objective()
            if not np.isfinite(current):
                raise FloatingPointError(
                    "MAR objective is not finite; rescale the observations"
                )
            objectives.append(current)
            if abs(objectives[-2] - current) <= tol * max(
                abs(current), abs(objectives[-2]), np.finfo(float).tiny
            ):
                converged = True
                break
    intercept = response_mean.copy()
    for A, mean, B in zip(left, lag_means, right):
        intercept -= A @ mean @ B.T
    with np.errstate(over="ignore", invalid="ignore"):
        fitted = (intercept + _prediction(raw_lags, left, right)) * data_scale
        intercept = intercept * data_scale
        residuals = original[order:] - fitted
        row_cov = row_cov * data_scale
        col_cov = col_cov * data_scale
    if not all(
        np.isfinite(value).all()
        for value in (fitted, intercept, residuals, row_cov, col_cov)
    ):
        raise FloatingPointError(
            "MAR fitted values or covariance factors exceed floating-point range"
        )
    return MARResult(
        left=left,
        right=right,
        intercept=intercept,
        fitted_values=fitted,
        residuals=residuals,
        method=method,
        converged=converged,
        n_iter=n_iter,
        objective_history=np.asarray(objectives),
        row_covariance=row_cov if method == "mle" else None,
        column_covariance=col_cov if method == "mle" else None,
        covariance_regularized=regularized,
        ridge=original_ridge,
        ranks=ranks,
        data_scale=data_scale,
        _history=original[-order:].copy(),
    )


@dataclass
class MARRankSelection:
    """Joint RR.LS EBIC grid, indexed by ``[row_rank-1, column_rank-1]``."""

    ranks: tuple
    result: MARResult
    scores: np.ndarray
    converged: np.ndarray


def select_mar_rank(X, *, max_ranks=None, max_iter=200, tol=1e-8, init="auto"):
    """Select MAR(1) coefficient ranks with the published joint RR.LS EBIC.

    Implements Equation (13), Section 5 of Xiao, Han, Chen and Liu,
    *Reduced Rank Autoregressive Models for Matrix Time Series*,
    https://yuefenghan.github.io/papers/Reduced_Rank_MAR.pdf.

    Every candidate fits a zero-intercept, unpenalized RR.LS model. The score
    uses T total observations, exactly as in the paper:
    ``log(SSE/(T*m*n)) + (log(T*n)*r*(2*m-r) + log(T*m)*s*(2*n-s))/(T*m*n)``.
    Center or difference data before selection when the zero-mean model is
    inappropriate. The grid includes positive ranks up to ``max_ranks``
    (default matrix dimensions). Candidate convergence is exposed; local
    minima and unconverged fits can affect selection. The separate rank
    search approximation and the likelihood RR.CC criterion are not used.
    """
    X = as_series(X)
    T, m, n = X.shape
    maximum = ranks_tuple(max_ranks, (m, n)) or (m, n)
    scores = np.empty(maximum)
    converged = np.empty(maximum, dtype=bool)
    best, best_rank, best_score = None, None, np.inf
    for r in range(1, maximum[0] + 1):
        for s in range(1, maximum[1] + 1):
            result = fit_mar(X, ranks=(r, s), max_iter=max_iter, tol=tol, init=init)
            sse = np.sum((result.residuals / result.data_scale) ** 2)
            if sse <= 0:
                raise ValueError(
                    "rank EBIC is undefined for a perfect zero-variance fit"
                )
            score = (
                np.log(sse / (T * m * n))
                + 2 * np.log(result.data_scale)
                + (np.log(T * n) * r * (2 * m - r) + np.log(T * m) * s * (2 * n - s))
                / (T * m * n)
            )
            scores[r - 1, s - 1], converged[r - 1, s - 1] = score, result.converged
            if score < best_score:
                best, best_rank, best_score = result, (r, s), score
    return MARRankSelection(best_rank, best, scores, converged)
