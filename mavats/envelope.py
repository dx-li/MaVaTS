"""Envelope matrix autoregression with shared response reducing spaces.

References
----------
Samadi, S. Y. and De Alwis, T. P. (2026), "Envelope Matrix Autoregressive
Models", JBES 44(2), 397--412, https://doi.org/10.1080/07350015.2025.2537404.
Equations (13), (15)--(16) and Algorithm 1 specify the model and estimator.
See docs/envelope-notes.md for source corrections and numerical choices.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_triangular

from ._envelope_optimization import _fit_envelope, _logdet, _spd
from ._validation import (
    as_series,
    finite_scalar,
    positive_int,
    random_generator,
    ranks_tuple,
)
from .autoregression import MARResult, _normalize, _prediction, fit_mar


def _right_whiten(values, covariance):
    _, factor = _spd(covariance, "whitening covariance")
    count, rows, columns = values.shape
    white = solve_triangular(
        factor[0],
        values.transpose(2, 0, 1).reshape(columns, -1),
        lower=True,
        check_finite=False,
    )
    return white.reshape(columns, count, rows).transpose(1, 2, 0)


def _moments(response, lags, opposite, covariance):
    """Joint-lag weighted LS and Eq. (15) moments, using the same N rows."""
    design = np.concatenate([z @ b.T for z, b in zip(lags, opposite)], axis=1)
    wy = _right_whiten(response, covariance)
    wf = _right_whiten(design, covariance)
    d = wf.transpose(0, 2, 1).reshape(-1, wf.shape[1])
    y = wy.transpose(0, 2, 1).reshape(-1, wy.shape[1])
    coefficient, _, rank, _ = np.linalg.lstsq(d, y, rcond=None)
    if rank < d.shape[1]:
        raise ValueError(
            "envelope weighted lag design is rank deficient; use more data or different starts"
        )
    error = y - d @ coefficient
    residual = error.T @ error / len(y)
    response_moment = y.T @ y / len(y)
    # A saturated regression can leave roundoff-sized, apparently SPD errors.
    # Such profiles cannot be evaluated reliably as unregularized likelihoods.
    threshold = 10 * np.finfo(float).eps * np.linalg.norm(response_moment, 2)
    if np.linalg.eigvalsh(residual)[0] <= threshold:
        raise ValueError(
            "envelope residual moment is numerically singular; unregularized MLE is unavailable"
        )
    _spd(response_moment, "response moment")
    return coefficient.T, residual, response_moment


def _block_update(response, lags, opposite, covariance, rank, **options):
    unrestricted, residual, moment = _moments(response, lags, opposite, covariance)
    basis, diagnostic = _fit_envelope(residual, moment, rank, **options)
    projector = basis @ basis.T
    complement = np.eye(len(moment)) - projector
    coefficient = projector @ unrestricted
    estimate, _ = _spd(
        projector @ residual @ projector + complement @ moment @ complement,
        "envelope covariance",
    )
    return coefficient, estimate, basis, diagnostic


def _nll(residuals, row, column):
    white = _right_whiten(residuals, column)
    white = _right_whiten(white.transpose(0, 2, 1), row)
    count, m, n = residuals.shape
    return float(
        0.5
        * (
            count
            * (
                m * n * np.log(2 * np.pi)
                + n * _logdet(_spd(row, "row covariance")[1])
                + m * _logdet(_spd(column, "column covariance")[1])
            )
            + np.sum(white * white)
        )
    )


def _basis(value, dimension, rank, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a real orthonormal basis")
    basis = np.array(raw, dtype=float, copy=True)
    if (
        basis.shape != (dimension, rank)
        or not np.isfinite(basis).all()
        or not np.allclose(basis.T @ basis, np.eye(rank), rtol=1e-9, atol=1e-10)
    ):
        raise ValueError(
            f"{name} must be an orthonormal basis with shape {(dimension, rank)}"
        )
    return basis


@dataclass
class EnvelopeMARResult(MARResult):
    """Fixed-dimension EMAR estimate and local-optimization diagnostics.

    The model paper and references of :func:`fit_envelope_mar` also apply to
    all inherited forecast/covariance methods. ``row_envelope`` and
    ``column_envelope`` are orthonormal bases; compare their projectors, not
    their coordinates. Every lag's output lies in these spaces and each
    innovation covariance is reduced by the corresponding space.

    ``optimization_history`` records every attempted sweep, with row/column
    selected-start and per-start objectives, gradient norms and stop reasons.
    ``converged`` requires both outer likelihood and selected inner gradient
    criteria, not global optimality or stationarity of the time series.
    ``warmup_converged`` concerns only the unrestricted initialization.
    ``objective_history`` contains accepted complete EMAR sweeps on scaled
    data, excluding the unrestricted warmup. Covariance scaling and physical
    likelihood conversion follow MARResult. Dense diagnostic allocations are
    guarded by ``max_dense_elements``; matrix-form forecasts avoid them.
    """

    envelope_dims: tuple = (0, 0)
    row_envelope: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    column_envelope: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    optimization_history: tuple = ()
    warmup_converged: bool | None = None
    stop_reason: str = "max_iter"
    likelihood_converged: bool = False
    inner_converged: bool = False
    max_dense_elements: int = 10_000_000

    def _guard_dense(self, count):
        if count > self.max_dense_elements:
            raise ValueError(
                "dense diagnostic exceeds max_dense_elements; matrix-form forecasting is still available"
            )

    @property
    def coefficients(self):
        self._guard_dense(self.order * self.intercept.size**2)
        return super().coefficients

    @property
    def spectral_radius(self):
        if self.order > 1:
            self._guard_dense((self.order * self.intercept.size) ** 2)
        return super().spectral_radius

    @property
    def log_likelihood(self):
        """Physical-unit conditional Gaussian likelihood, without squaring scale."""
        return float(
            -self.objective_history[-1] - self.residuals.size * np.log(self.data_scale)
        )

    def residual_covariance(self, *, ddof=0):
        self._guard_dense(self.intercept.size**2)
        return super().residual_covariance(ddof=ddof)


def fit_envelope_mar(
    X,
    envelope_dims,
    *,
    order=1,
    fit_intercept=True,
    max_iter=100,
    tol=1e-8,
    inner_max_iter=200,
    inner_tol=1e-6,
    inner_starts=3,
    random_state=0,
    initial=None,
    initial_covariance=None,
    initial_envelopes=None,
    warmup_max_iter=100,
    max_dense_elements=10_000_000,
):
    """Fit Gaussian envelope MAR(p) with fixed positive envelope dimensions.

    Parameters
    ----------
    X : array_like, shape (T, m, n)
        Finite real time-first observations. All lags share the same N=T-p
        regression rows. Missing observations are not supported.
    envelope_dims : (u, v)
        Response envelope dimensions, 1 <= u <= m and 1 <= v <= n.
        A_l = R eta_l and B_l = C xi_l; past inputs are NOT projected onto
        R and C. Covariances have material and orthogonal-complement blocks.
        Full dimensions give the unrestricted separable MAR likelihood.
    order : int, default 1
        Number of distinct bilinear lag terms with shared output envelopes.
    fit_intercept : bool, default True
        Profile a free intercept by centering response and each lag over the
        same regression sample; restore the intercept in original units.
    max_iter, tol : int, float
        Complete row/column sweeps and relative negative-loglikelihood
        tolerance on data divided by its maximum absolute entry.
    inner_max_iter, inner_tol, inner_starts : int, float, int
        Grassmann iterations, absolute gradient tolerance, and fresh starts
        per block. Starts are the smallest residual eigenvectors, smallest
        response eigenvectors, then random orthonormal bases. The previous
        envelope is always an additional start. No global optimum guarantee.
    random_state : int or numpy.random.Generator, default 0
        Local random starts; does not change NumPy's global RNG.
    initial : (left, right), optional
        Finite factors shaped (p,m,m), (p,n,n); 2-D accepted for p=1.
        Bypasses unrestricted MAR-MLE warmup. Without this, the paper's
        unrestricted warmup uses up to warmup_max_iter sweeps.
    initial_covariance : (row, column), optional
        SPD covariance factors in physical units, divided by data_scale
        individually for computation. With no initial coefficients these
        initialize the unrestricted warmup. Otherwise default to identities
        in scaled-data units.
    initial_envelopes : (R, C), optional
        Orthonormal bases shaped (m,u), (n,v), used as extra first starts.
        Without supplied bases, an unrestricted warmup also supplies leading
        left singular vectors of its concatenated row/column coefficients.
        These are additional numerical starts, not fixed envelope estimates.
    warmup_max_iter : int, default 100
        Unrestricted MAR-MLE initialization budget, separately diagnosed.
    max_dense_elements : int, default 10000000
        Guard for dense result diagnostics and automatic projection warmup.
        Larger warmups use matrix-sized identity initialization instead.

    Returns
    -------
    EnvelopeMARResult
        Bilinear factors, reducing spaces, separable covariance, forecasts
        and inner/outer optimization diagnostics. Inspect ``converged``.

    Notes
    -----
    Implements equations (15)--(16) using joint-lag, whitened SVD regressions
    and the actual logdet Grassmann profile, not PCA or reduced-rank ALS.
    Singular weighted designs and numerically singular residual moments
    raise: no ridge, covariance flooring or pseudodeterminant is substituted.
    Each nonzero lag pair is scale/sign normalized separately. Stability is
    diagnosed, never imposed. Only fixed-dimension point estimation is
    implemented: no dimension/order selection, sparse SEMAR, standard errors
    or the separate one-sided identity-coefficient model.

    References
    ----------
    Samadi, S. Y. and De Alwis, T. P. (2026), "Envelope Matrix Autoregressive
    Models", Journal of Business & Economic Statistics 44(2), 397--412,
    equations (13), (15)--(16), Algorithm 1,
    https://doi.org/10.1080/07350015.2025.2537404.
    Numerical solver and source-equation corrections are documented in
    docs/envelope-notes.md; this is not an exact replication of paper code.
    """
    order = positive_int(order, "order")
    original = as_series(X, min_samples=order + 2)
    m, n = original.shape[1:]
    dims = ranks_tuple(envelope_dims, (m, n))
    if dims is None:
        raise ValueError("envelope_dims must specify two positive dimensions")
    max_iter = positive_int(max_iter, "max_iter")
    inner_max_iter = positive_int(inner_max_iter, "inner_max_iter")
    inner_starts = positive_int(inner_starts, "inner_starts")
    warmup_max_iter = positive_int(warmup_max_iter, "warmup_max_iter")
    max_dense_elements = positive_int(max_dense_elements, "max_dense_elements")
    tol = finite_scalar(tol, "tol", minimum=0)
    inner_tol = finite_scalar(inner_tol, "inner_tol", minimum=0)
    if not isinstance(fit_intercept, (bool, np.bool_)):
        raise ValueError("fit_intercept must be boolean")
    rng = random_generator(random_state)
    scale = float(np.max(np.abs(original))) or 1.0
    data = original / scale
    raw_lags = [data[order - lag : len(data) - lag] for lag in range(1, order + 1)]
    mean = data[order:].mean(axis=0) if fit_intercept else np.zeros((m, n))
    lag_means = [
        z.mean(axis=0) if fit_intercept else np.zeros((m, n)) for z in raw_lags
    ]
    response = data[order:] - mean
    lags = [z - mu for z, mu in zip(raw_lags, lag_means)]
    bases = [None, None]
    if initial_envelopes is not None:
        if len(initial_envelopes) != 2:
            raise ValueError("initial_envelopes must contain row and column bases")
        bases = [
            _basis(b, d, r, "initial envelope")
            for b, d, r in zip(initial_envelopes, (m, n), dims)
        ]
    covariances = [np.eye(m), np.eye(n)]
    if initial_covariance is not None:
        if len(initial_covariance) != 2:
            raise ValueError("initial_covariance must contain two matrices")
        for i, (value, d) in enumerate(zip(initial_covariance, (m, n))):
            raw = np.asarray(value)
            if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
                raise ValueError("initial covariance must be real numeric")
            cov = np.array(raw, dtype=float, copy=True) / scale
            if cov.shape != (d, d) or not np.allclose(cov, cov.T, rtol=1e-10, atol=0):
                raise ValueError(
                    "initial covariance must be symmetric and dimensionally compatible"
                )
            covariances[i] = _spd(cov, "initial covariance")[0]
    warmup_converged = None
    if initial is None:
        warmup = fit_mar(
            data,
            method="mle",
            order=order,
            fit_intercept=fit_intercept,
            max_iter=warmup_max_iter,
            tol=tol,
            covariance_floor=0,
            initial_covariance=tuple(covariances),
            init="auto" if order * (m * n) ** 2 <= max_dense_elements else "identity",
        )
        left, right = warmup.left.copy(), warmup.right.copy()
        covariances = [warmup.row_covariance.copy(), warmup.column_covariance.copy()]
        warmup_converged = warmup.converged
        if initial_envelopes is None:
            # Covariance-eigenvector starts alone can be trapped in immaterial
            # directions when material variance is high. Use the training-only
            # warmup's coefficient spaces as an additional first start.
            bases = [
                np.linalg.svd(np.concatenate(coef, axis=1), full_matrices=False)[0][
                    :, :rank
                ]
                for coef, rank in zip((left, right), dims)
            ]
    else:
        if len(initial) != 2:
            raise ValueError("initial must contain left and right coefficients")
        coefficients = []
        for value, d in zip(initial, (m, n)):
            raw = np.asarray(value)
            if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
                raise ValueError("initial coefficients must be real numeric")
            array = np.array(raw, dtype=float, copy=True)
            if order == 1 and array.ndim == 2:
                array = array[None]
            if array.shape != (order, d, d) or not np.isfinite(array).all():
                raise ValueError(
                    "initial coefficient dimensions must match order and X"
                )
            coefficients.append(array)
        left, right = coefficients
    row, column = covariances
    history, optimization = [], []
    likelihood_converged = inner_converged = False
    reason = "max_iter"
    options = dict(starts=inner_starts, rng=rng, max_iter=inner_max_iter, tol=inner_tol)
    transposed_lags = [z.transpose(0, 2, 1) for z in lags]
    for iteration in range(1, max_iter + 1):
        a, new_row, r, rd = _block_update(
            response, lags, right, column, dims[0], initial=bases[0], **options
        )
        new_left = np.stack(np.split(a, order, axis=1))
        b, new_column, c, cd = _block_update(
            response.transpose(0, 2, 1),
            transposed_lags,
            new_left,
            new_row,
            dims[1],
            initial=bases[1],
            **options,
        )
        new_right = np.stack(np.split(b, order, axis=1))
        for lag in range(order):
            # Never replace a zero A with an out-of-envelope coordinate vector.
            if np.any(new_left[lag]):
                new_left[lag], new_right[lag] = _normalize(
                    new_left[lag], new_right[lag]
                )
        norm = np.linalg.norm(new_row)
        new_row, new_column = new_row / norm, new_column * norm
        residual = response - _prediction(lags, new_left, new_right)
        value = _nll(residual, new_row, new_column)
        if not np.isfinite(value):
            raise FloatingPointError("envelope likelihood exceeds floating-point range")
        accepted = not history or value <= history[-1] + 1e-10 * max(
            1, abs(history[-1])
        )
        optimization.append(
            dict(
                iteration=iteration,
                row=rd,
                column=cd,
                accepted=accepted,
                objective=value,
            )
        )
        if not accepted:
            reason = "likelihood_increase"
            likelihood_converged = False
            break
        likelihood_converged = bool(
            history and abs(history[-1] - value) <= tol * max(1, abs(history[-1]))
        )
        inner_converged = rd["converged"] and cd["converged"]
        history.append(value)
        left, right, row, column, bases = (
            new_left,
            new_right,
            new_row,
            new_column,
            [r, c],
        )
        if likelihood_converged and inner_converged:
            reason = "likelihood_and_gradient_tolerance"
            break
    intercept = mean.copy()
    for a, mu, b in zip(left, lag_means, right):
        intercept -= a @ mu @ b.T
    fitted = mean + _prediction(lags, left, right)
    with np.errstate(over="ignore", invalid="ignore"):
        physical = [
            intercept * scale,
            fitted * scale,
            (data[order:] - fitted) * scale,
            row * scale,
            column * scale,
        ]
    if not all(np.isfinite(value).all() for value in physical):
        raise FloatingPointError(
            "envelope fit exceeds floating-point range in physical units"
        )
    return EnvelopeMARResult(
        left=left,
        right=right,
        intercept=physical[0],
        fitted_values=physical[1],
        residuals=physical[2],
        method="envelope-mle",
        converged=likelihood_converged and inner_converged,
        n_iter=len(history),
        objective_history=np.asarray(history),
        row_covariance=physical[3],
        column_covariance=physical[4],
        data_scale=scale,
        _history=original[-order:].copy(),
        envelope_dims=dims,
        row_envelope=bases[0],
        column_envelope=bases[1],
        optimization_history=tuple(optimization),
        warmup_converged=warmup_converged,
        stop_reason=reason,
        likelihood_converged=likelihood_converged,
        inner_converged=inner_converged,
        max_dense_elements=max_dense_elements,
    )
