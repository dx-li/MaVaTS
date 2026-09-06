"""Multilinear TenAR(p,R) of Li and Xiao, arXiv:2110.00928v1.

The transition at each lag is a sum of Kronecker products of square mode
matrices, NOT a low-Tucker-rank transition tensor. Sections 3.1--3.3 supply
alternating LS, separable Gaussian likelihood, and projection initialization.
See docs/tensor-autoregression-notes.md for indexing corrections and limits.
Result transitions, covariance accessors and forecasts evaluate this model.

References
----------
Li, Z. and Xiao, H. (2021), "Multi-Linear Tensor Autoregressive Models",
arXiv:2110.00928v1, https://arxiv.org/abs/2110.00928v1.
"""

from dataclasses import dataclass, field
from math import prod

import numpy as np

from ._validation import as_series, finite_scalar, positive_int
from .factors import _mode_product


def _norm(x):
    scale = float(np.max(np.abs(x)))
    return scale * float(np.linalg.norm(x / scale)) if scale else 0.0


def _unfold(x, mode):
    """Time-first mode unfolding; other spatial indices run in Fortran order."""
    axes = [0, mode + 1] + [i for i in range(x.ndim - 1, 0, -1) if i != mode + 1]
    return x.transpose(axes).reshape(len(x), x.shape[mode + 1], -1)


def _flat(x):
    return x.transpose([0] + list(range(x.ndim - 1, 0, -1))).reshape(len(x), -1)


def _apply(x, matrices, skip=None):
    for k, matrix in enumerate(matrices):
        if k != skip:
            x = _mode_product(x, matrix, k)
    return x


def _copy_coefficients(coefficients):
    return [[list(a.copy() for a in term) for term in lag] for lag in coefficients]


def _balanced_rescale(matrix, factors):
    """Multiply scalar gauges without losing intermediate compensation."""
    mantissa, exponent = np.frexp(matrix)
    exponent = exponent.astype(np.int64)
    for factor in factors:
        factor_mantissa, factor_exponent = np.frexp(factor)
        mantissa = mantissa * factor_mantissa
        exponent += int(factor_exponent)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.ldexp(mantissa, exponent)
    if not np.isfinite(result).all() or np.any((matrix != 0) & (result == 0)):
        raise FloatingPointError(
            "normalized factor cannot be represented without overflow/underflow"
        )
    return result


def _normalize_term(term):
    term = [a.copy() for a in term]
    factors = []
    for k in range(len(term) - 1):
        scale = float(np.max(np.abs(term[k])))
        if scale == 0:
            for j in range(len(term) - 1):
                term[j] = np.zeros_like(term[j])
                term[j][0, 0] = 1
            term[-1] = np.zeros_like(term[-1])
            return term
        norm = float(np.linalg.norm(term[k] / scale))
        sign = -1 if term[k].flat[np.argmax(np.abs(term[k]))] < 0 else 1
        term[k] = (term[k] / scale) / norm * sign
        factors.extend((scale, norm * sign))
    term[-1] = _balanced_rescale(term[-1], factors)
    if not all(np.isfinite(a).all() for a in term):
        raise FloatingPointError("coefficient normalization overflowed")
    return term


def _canonicalize(coefficients):
    """Matrix multiterm SVD via skinny QR; never construct the transition."""
    result = []
    for lag in coefficients:
        lag = [_normalize_term(term) for term in lag]
        if len(lag[0]) == 2 and len(lag) > 1:
            u = np.column_stack([term[0].ravel(order="F") for term in lag])
            v = np.column_stack([term[1].ravel(order="F") for term in lag])
            qu, ru = np.linalg.qr(u, mode="reduced")
            qv, rv = np.linalg.qr(v, mode="reduced")
            left, values, right = np.linalg.svd(ru @ rv.T, full_matrices=False)
            u, v = qu @ left, (qv @ right.T) * values
            lag = [
                _normalize_term(
                    [
                        u[:, r].reshape(lag[0][0].shape, order="F"),
                        v[:, r].reshape(lag[0][1].shape, order="F"),
                    ]
                )
                for r in range(len(lag))
            ]
        else:
            lag.sort(key=lambda term: _norm(term[-1]), reverse=True)
        result.append(lag)
    return result


def _prediction(lagged, coefficients):
    result = np.zeros_like(lagged[0])
    for x, lag in zip(lagged, coefficients):
        for term in lag:
            result += _apply(x, term)
    if not np.isfinite(result).all():
        raise FloatingPointError("multilinear prediction overflowed")
    return result


def _whitener(covariance):
    values, vectors = np.linalg.eigh(covariance)
    if values[0] <= 0 or not np.isfinite(values).all():
        raise ValueError("separable covariance must be positive definite")
    return (vectors / np.sqrt(values)) @ vectors.T


def _whiten(x, covariances, skip=None):
    for mode, covariance in enumerate(covariances):
        if mode != skip:
            x = _mode_product(x, _whitener(covariance), mode)
    return x


def _coefficient_block(response, predictor, mode, covariances=None):
    """Exact LS/GLS block solve without squared-condition normal equations."""
    if covariances is not None:
        response = _whiten(response, covariances, skip=mode)
        predictor = _whiten(predictor, covariances, skip=mode)
    d = response.shape[mode + 1]
    design = _unfold(predictor, mode).transpose(0, 2, 1).reshape(-1, d)
    target = _unfold(response, mode).transpose(0, 2, 1).reshape(-1, d)
    coefficient, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
    if rank < d:
        raise ValueError(
            f"mode {mode} coefficient design is rank deficient ({rank} < {d})"
        )
    condition = float(singular[0] / singular[-1])
    return coefficient.T, condition


def _spd(covariance, floor=0.0):
    covariance = covariance * 0.5 + covariance.T * 0.5
    values, vectors = np.linalg.eigh(covariance)
    if not np.isfinite(values).all() or values[-1] <= 0:
        raise ValueError("Gaussian likelihood is undefined at zero residual variance")
    threshold = floor * values[-1]
    active = values[0] < threshold
    if values[0] <= 0 and floor == 0:
        raise ValueError(
            "separable covariance update is singular; no floor is implicit"
        )
    if active:
        covariance = (vectors * np.maximum(values, threshold)) @ vectors.T
    return covariance, bool(active)


def _covariance_block(residuals, covariances, mode, floor=0.0):
    white = _unfold(_whiten(residuals, covariances, skip=mode), mode)
    d = white.shape[1]
    rows = white.transpose(0, 2, 1).reshape(-1, d)
    singular = np.linalg.svd(rows, compute_uv=False)
    cutoff = singular[0] * max(rows.shape) * np.finfo(float).eps
    rank = int(np.sum(singular > cutoff))
    if rank < d and floor == 0:
        raise ValueError(
            f"mode {mode} residual covariance is rank deficient ({rank} < {d})"
        )
    covariance, active = _spd(rows.T @ rows / len(rows), floor)
    return covariance, active or rank < d


def _normalize_covariances(covariances):
    covariances = [s.copy() for s in covariances]
    factors = []
    for k in range(len(covariances) - 1):
        scale = float(np.max(np.abs(covariances[k])))
        if scale <= 0 or not np.isfinite(scale):
            raise ValueError("invalid covariance scale")
        norm = float(np.linalg.norm(covariances[k] / scale))
        covariances[k] = (covariances[k] / scale) / norm
        factors.extend((scale, norm))
    covariances[-1] = _balanced_rescale(covariances[-1], factors)
    if not all(np.isfinite(s).all() for s in covariances):
        raise FloatingPointError("covariance normalization overflowed")
    return covariances


def _negative_loglike(residuals, covariances):
    dimension = prod(residuals.shape[1:])
    logdet = sum((dimension // len(s)) * np.linalg.slogdet(s)[1] for s in covariances)
    white = _whiten(residuals, covariances)
    value = 0.5 * (
        len(residuals) * (dimension * np.log(2 * np.pi) + logdet) + np.sum(white**2)
    )
    if not np.isfinite(value):
        raise FloatingPointError("tensor Gaussian likelihood is not finite")
    return float(value)


def _rearrange(transition, shape):
    """vec_F output/input pairs -> vec_F mode matrices, Eq. (13)."""
    k = len(shape)
    paired = transition.reshape(tuple(shape) * 2, order="F")
    return paired.transpose([j for i in range(k) for j in (i, i + k)]).reshape(
        tuple(d * d for d in shape), order="F"
    )


def _cp_tensor(factors):
    shape = tuple(len(f) for f in factors)
    result = np.zeros(shape)
    for r in range(factors[0].shape[1]):
        term = factors[0][:, r]
        for factor in factors[1:]:
            term = np.multiply.outer(term, factor[:, r])
        result += term
    return result


@dataclass
class TensorARProjection:
    """Projection solve, not a guarantee of the best CP approximation."""

    method: str
    converged: bool
    n_iter: int
    objective_history: np.ndarray
    rank_deficient_blocks: int = 0
    cancellation_ratio: float = 1.0


def _cp_project(tensor, rank, rng, max_iter, tol):
    """Local CP-ALS, using complete mode blocks, on a bounded dense tensor."""
    factors = []
    for k, d in enumerate(tensor.shape):
        matrix = np.moveaxis(tensor, k, 0).reshape(d, -1)
        u = np.linalg.svd(matrix, full_matrices=False)[0]
        if rank == 1:
            factors.append(u[:, :1])
        else:
            # Random directions avoid pairing unrelated mode singular vectors.
            factors.append(rng.normal(size=(d, rank)))
    for k in range(len(factors) - 1):
        norms = np.linalg.norm(factors[k], axis=0)
        factors[k] /= norms
    history = [float(np.sum((tensor - _cp_tensor(factors)) ** 2))]
    deficient, converged, cancellation = 0, False, 1.0
    for iteration in range(1, max_iter + 1):
        old = [f.copy() for f in factors]
        for k in range(len(factors)):
            columns = []
            for r in range(rank):
                column = np.ones(1)
                for j in range(len(factors) - 1, -1, -1):
                    if j != k:
                        column = np.kron(column, factors[j][:, r])
                columns.append(column)
            design = np.column_stack(columns)
            target = _unfold(tensor[None], k)[0].T
            solution, _, block_rank, _ = np.linalg.lstsq(design, target, rcond=None)
            factors[k] = solution.T
            deficient += block_rank < rank
        for k in range(len(factors) - 1):
            norms = np.linalg.norm(factors[k], axis=0)
            if np.any(norms == 0):
                raise ValueError(
                    "CP projection produced a zero component; change initialization"
                )
            factors[k] /= norms
            factors[-1] *= norms
        estimate = _cp_tensor(factors)
        value = float(np.sum((tensor - estimate) ** 2))
        cancellation = float(
            np.linalg.norm(factors[-1], axis=0).sum()
            / max(_norm(estimate), np.finfo(float).tiny)
        )
        if not np.isfinite(value) or cancellation > 1e8:
            raise ValueError(
                "CP projection is numerically degenerate (component cancellation)"
            )
        if value > history[-1] + 1e-10 * max(1.0, history[-1]):
            factors = old
            break
        history.append(value)
        if abs(history[-2] - value) <= tol * max(1.0, abs(history[-2])):
            converged = True
            break
    return factors, TensorARProjection(
        "local_cp_als",
        converged,
        len(history) - 1,
        np.asarray(history),
        int(deficient),
        cancellation,
    )


def _projection(lagged, response, terms, rng, max_iter, tol, max_dimension):
    shape, dimension = response.shape[1:], prod(response.shape[1:])
    if dimension * len(lagged) > max_dimension:
        raise ValueError(
            "dense VAR projection exceeds max_dense_dimension; use init='random' or initial"
        )
    design = np.concatenate([_flat(x) for x in lagged], axis=1)
    target = _flat(response)
    coefficient, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < design.shape[1]:
        raise ValueError(
            "unrestricted VAR projection design is rank deficient; use init='random' or initial"
        )
    result, diagnostics = [], []
    for i, count in enumerate(terms):
        tensor = _rearrange(coefficient[i * dimension : (i + 1) * dimension].T, shape)
        if len(shape) == 2:
            u, s, vh = np.linalg.svd(tensor, full_matrices=False)
            factors = [u[:, :count], vh[:count].T * s[:count]]
            error = float(np.sum(s[count:] ** 2))
            diagnostic = TensorARProjection("matrix_svd", True, 0, np.array([error]))
        else:
            factors, diagnostic = _cp_project(tensor, count, rng, max_iter, tol)
        result.append(
            [
                [f[:, r].reshape((d, d), order="F") for f, d in zip(factors, shape)]
                for r in range(count)
            ]
        )
        diagnostics.append(diagnostic)
    return _canonicalize(result), tuple(diagnostics), target - design @ coefficient


def _covariance_projection(residual_vectors, shape, floor):
    """Hierarchical nearest-Kronecker SVD covariance initializer, Sec. 3.3."""
    remainder = residual_vectors.T @ residual_vectors / len(residual_vectors)
    covariances, active = [], False
    for k, d in enumerate(shape[:-1]):
        remaining = prod(shape[k + 1 :])
        arranged = _rearrange(remainder, (d, remaining))
        u, s, vh = np.linalg.svd(arranged, full_matrices=False)
        first = u[:, 0].reshape((d, d), order="F")
        remainder = (s[0] * vh[0]).reshape((remaining, remaining), order="F")
        if np.trace(first) < 0:
            first, remainder = -first, -remainder
        first, floored = _spd(first, floor)
        covariances.append(first)
        active |= floored
    last, floored = _spd(remainder, floor)
    return _normalize_covariances(covariances + [last]), active or floored


def _coefficient_diagnostics(coefficients):
    diagnostics = []
    for lag in coefficients:
        ranks = []
        for k in range(len(lag[0])):
            columns = []
            for term in lag:
                scale = float(np.max(np.abs(term[k])))
                column = term[k].ravel() / scale if scale else term[k].ravel()
                columns.append(column)
            ranks.append(int(np.linalg.matrix_rank(np.column_stack(columns))))
        ranks = tuple(ranks)
        strengths = np.array([_norm(term[-1]) for term in lag])
        final_scale = max(float(np.max(np.abs(term[-1]))) for term in lag)
        scaled_strengths = np.array(
            [_norm(term[-1] / final_scale) if final_scale else 0.0 for term in lag]
        )
        if np.any(strengths == 0):
            identification = "zero_component_not_identified"
        elif len(lag) == 1:
            identification = "single_nonzero_term"
        elif len(lag[0]) == 2:
            separated = np.all(
                np.abs(np.diff(scaled_strengths)) > 1e-8 * scaled_strengths[0]
            )
            identification = (
                "distinct_matrix_singular_values"
                if separated
                else "unseparated_matrix_singular_values"
            )
        elif all(rank == len(lag) for rank in ranks):
            identification = "full_column_rank_sufficient"
        else:
            identification = "kruskal_condition_not_certified"
        # First modes are normalized. Scale the final mode collectively before
        # Gram products so cancellation is invariant to transition amplitude.
        gram = np.ones((len(lag), len(lag)))
        for k in range(len(lag[0])):
            vectors = np.column_stack([term[k].ravel() for term in lag])
            if k == len(lag[0]) - 1 and final_scale:
                vectors = vectors / final_scale
            gram *= vectors.T @ vectors
        combined = np.sqrt(max(0.0, float(gram.sum())))
        cancellation = float(
            scaled_strengths.sum() / max(combined, np.finfo(float).tiny)
        )
        diagnostics.append(
            {
                "identification": identification,
                "mode_column_ranks": ranks,
                "term_norms": strengths,
                "cancellation_ratio": cancellation,
            }
        )
    return tuple(diagnostics)


@dataclass
class TensorARRun:
    """One start's local objective convergence and failures, in working units."""

    converged: bool
    n_iter: int
    objective_history: np.ndarray
    status: str
    message: str
    covariance_regularized: bool = False
    maximum_block_condition: float = 1.0
    maximum_cancellation: float = 1.0
    projection: tuple = ()

    @property
    def objective(self):
        return (
            float(self.objective_history[-1]) if len(self.objective_history) else np.inf
        )


@dataclass
class TensorARResult:
    """Time-first TenAR fit; coefficients[lag-index][term][mode].

    Histories/objective are SSE or Gaussian NLL after dividing observations
    by data_scale. Coefficients do not change units. MLE covariance factors
    have unit Frobenius norm in all but the final mode, which carries physical
    variance units. Stability is diagnosed, not enforced. Convergence refers
    to objective change, not global optimality, parameter identification or
    valid asymptotic inference. center=True is a fixed sample-mean extension.
    All model-derived accessors inherit the reference and scope in
    :func:`fit_tensor_ar` and this module.
    """

    coefficients: tuple
    lags: tuple
    terms: tuple
    method: str
    fitted_values: np.ndarray
    residuals: np.ndarray
    mean: np.ndarray
    data_scale: float
    covariance_factors: tuple | None
    runs: tuple
    selected_start: int
    identification_diagnostics: tuple
    max_dense_dimension: int
    _history: np.ndarray = field(repr=False)

    @property
    def order(self):
        return max(self.lags)

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
        return self.runs[self.selected_start].objective

    @property
    def covariance_regularized(self):
        return self.runs[self.selected_start].covariance_regularized

    @property
    def log_likelihood(self):
        if self.method != "mle":
            return None
        return -float(self.objective + self.residuals.size * np.log(self.data_scale))

    def transition_matrices(self, *, max_dimension=None):
        """Dense vec_F transitions for the specified lags, with an allocation guard."""
        limit = (
            self.max_dense_dimension
            if max_dimension is None
            else positive_int(max_dimension, "max_dimension")
        )
        dimension = prod(self.mean.shape)
        if dimension > limit:
            raise ValueError("dense transition exceeds max_dimension")
        result = []
        for lag in self.coefficients:
            transition = np.zeros((dimension, dimension))
            for term in lag:
                product = np.ones((1, 1))
                for matrix in reversed(term):
                    product = np.kron(product, matrix)
                transition += product
            result.append(transition)
        if not np.isfinite(result).all():
            raise FloatingPointError("dense tensor transition overflowed")
        return np.asarray(result)

    def companion_matrix(self, *, max_dimension=None):
        """Full-sum VAR companion including zero blocks for omitted lags."""
        limit = (
            self.max_dense_dimension
            if max_dimension is None
            else positive_int(max_dimension, "max_dimension")
        )
        dimension = prod(self.mean.shape)
        if dimension * self.order > limit:
            raise ValueError(
                "dense companion exceeds max_dimension; stability is not certified"
            )
        companion = np.zeros((dimension * self.order,) * 2)
        for lag, transition in zip(
            self.lags, self.transition_matrices(max_dimension=limit)
        ):
            companion[:dimension, (lag - 1) * dimension : lag * dimension] = transition
        companion[dimension:, :-dimension] = np.eye(dimension * (self.order - 1))
        return companion

    @property
    def spectral_radius(self):
        if self.lags == (1,) and self.terms == (1,):
            radii = [
                float(np.max(np.abs(np.linalg.eigvals(a))))
                for a in self.coefficients[0][0]
            ]
            if not np.isfinite(radii).all():
                raise FloatingPointError("mode spectral radius is not finite")
            if any(radius == 0 for radius in radii):
                return 0.0
            with np.errstate(over="ignore", under="ignore"):
                radius = float(np.exp(sum(np.log(radii))))
            if not np.isfinite(radius):
                raise FloatingPointError("tensor transition spectral radius overflowed")
            return radius
        return float(np.max(np.abs(np.linalg.eigvals(self.companion_matrix()))))

    @property
    def is_stable(self):
        return self.spectral_radius < 1

    def innovation_covariance(self, *, max_dimension=None):
        if self.covariance_factors is None:
            raise ValueError(
                "separable innovation covariance is available only for MLE"
            )
        limit = (
            self.max_dense_dimension
            if max_dimension is None
            else positive_int(max_dimension, "max_dimension")
        )
        if prod(self.mean.shape) > limit:
            raise ValueError("dense covariance exceeds max_dimension")
        covariance = np.ones((1, 1))
        for factor in reversed(self.covariance_factors):
            covariance = np.kron(covariance, factor)
        if not np.isfinite(covariance).all():
            raise FloatingPointError(
                "innovation covariance exceeds physical data units"
            )
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise FloatingPointError(
                "innovation covariance is not representably positive definite"
            ) from exc
        return covariance

    def forecast(self, steps, history=None):
        """Recursive linear forecast; conditional mean under martingale errors.

        Serially uncorrelated noise alone does not establish a conditional
        mean interpretation. Parameter uncertainty is not included.
        """
        steps = positive_int(steps, "steps")
        observed = (
            self._history
            if history is None
            else as_series(history, min_samples=self.order, ndim=self.mean.ndim + 1)
        )
        if observed.shape[1:] != self.mean.shape:
            raise ValueError("forecast history dimensions do not match fit")
        values = [x.copy() for x in observed[-self.order :]]
        output = []
        for _ in range(steps):
            scale = max(
                float(np.max(np.abs(self.mean))),
                max(float(np.max(np.abs(x))) for x in values[-self.order :]),
                np.finfo(float).tiny,
            )
            lagged = [
                (values[-lag] / scale - self.mean / scale)[None] for lag in self.lags
            ]
            with np.errstate(over="ignore", invalid="ignore"):
                prediction = (
                    _prediction(lagged, self.coefficients)[0] + self.mean / scale
                ) * scale
            if not np.isfinite(prediction).all():
                raise FloatingPointError(
                    "tensor AR forecast exceeds floating-point range"
                )
            output.append(prediction)
            values.append(prediction)
        return np.asarray(output)


def fit_tensor_ar(
    X,
    terms=(1,),
    *,
    lags=None,
    method="ls",
    init="projection",
    initial=None,
    covariance_initial=None,
    n_starts=1,
    max_iter=100,
    tol=1e-7,
    projection_max_iter=100,
    projection_tol=1e-8,
    max_dense_dimension=256,
    covariance_floor=0.0,
    center=False,
    random_state=0,
):
    """Fit Li--Xiao multilag, multiterm tensor autoregression.

    terms gives positive component counts for corresponding positive increasing
    lags (default 1,...,len(terms)); gaps are allowed through explicit lags.
    X has shape (T,d1,...,dK), K>=2. method is 'ls', 'mle', or 'projection'.
    initial has nested [lag][term][mode] square matrices in public mode order.
    It replaces init, and later starts perturb it without using true parameters.
    init='projection' requires an identified unrestricted VAR design and a
    bounded dense allocation; it never silently falls back to another method.
    K=2 projection uses exact SVD, K>=3 uses explicitly local CP-ALS. Projection
    method reports its own convergence, not convergence of the LS objective.

    MLE assumes iid Gaussian innovations with separable covariance. Supplied
    covariance_initial factors are in physical units; otherwise projection
    initialization uses hierarchical SVD of unrestricted VAR residual moments.
    Random/user coefficient starts default to isotropic residual covariance,
    an explicit computational alternative. covariance_floor is a relative
    mode eigenvalue floor (default zero); singular unregularized updates fail.
    Mean centering is optional, not fitted-intercept profiling. All starts and
    failed sweeps remain visible; increasing objectives are rolled back.
    tol controls relative SSE change for LS and change in mean negative log
    likelihood per observed scalar for MLE (invariant to unit log constants).
    Automatic order/rank selection and inference are not implemented.

    References
    ----------
    Li, Z. and Xiao, H. (2021), "Multi-Linear Tensor Autoregressive Models",
    Sections 3.1--3.3, https://arxiv.org/abs/2110.00928v1.
    Fixed sample-mean centering, local CP-ALS projection, multistart choices
    and optional covariance flooring are documented computational choices;
    a local projection is not a certified best CP approximation. See
    docs/tensor-autoregression-notes.md for manuscript indexing corrections.
    """
    X = as_series(X, min_samples=3, ndim=None)
    if X.ndim < 3:
        raise ValueError("X needs at least two spatial modes")
    try:
        terms = tuple(positive_int(r, "terms") for r in terms)
        lags = (
            tuple(range(1, len(terms) + 1))
            if lags is None
            else tuple(positive_int(lag, "lag") for lag in lags)
        )
    except TypeError as exc:
        raise ValueError("terms and lags must be sequences") from exc
    if not terms or len(terms) != len(lags) or tuple(sorted(set(lags))) != lags:
        raise ValueError(
            "terms and increasing unique lags must have equal nonzero length"
        )
    order, shape = max(lags), X.shape[1:]
    if len(X) <= order:
        raise ValueError("insufficient observations for the largest lag")
    if len(shape) == 2 and max(terms) > min(d * d for d in shape):
        raise ValueError(
            "matrix term counts cannot exceed the maximum rearranged matrix rank"
        )
    if method not in ("ls", "mle", "projection") or init not in (
        "projection",
        "random",
    ):
        raise ValueError(
            "method must be ls/mle/projection; init must be projection/random"
        )
    if method == "projection" and (initial is not None or init != "projection"):
        raise ValueError(
            "projection estimator requires projection initialization without initial"
        )
    if not isinstance(center, (bool, np.bool_)):
        raise ValueError("center must be boolean")
    n_starts = positive_int(n_starts, "n_starts")
    max_iter = positive_int(max_iter, "max_iter")
    projection_max_iter = positive_int(projection_max_iter, "projection_max_iter")
    max_dense_dimension = positive_int(max_dense_dimension, "max_dense_dimension")
    tol, projection_tol = finite_scalar(tol, "tol", minimum=0), finite_scalar(
        projection_tol, "projection_tol", minimum=0
    )
    if not tol or not projection_tol:
        raise ValueError("tolerances must be positive")
    floor = finite_scalar(covariance_floor, "covariance_floor", minimum=0)
    if floor >= 1 or (method != "mle" and (floor or covariance_initial is not None)):
        raise ValueError("covariance options require mle and covariance_floor < 1")
    scale = float(np.max(np.abs(X)))
    if scale == 0:
        raise ValueError("zero observations do not identify tensor AR coefficients")
    work = X / scale
    mean = work.mean(axis=0) if center else np.zeros(shape)
    work = work - mean
    if not np.any(work):
        raise ValueError(
            "constant centered observations do not identify tensor AR coefficients"
        )
    response = work[order:]
    lagged = [work[order - lag : len(work) - lag] for lag in lags]
    rng = np.random.default_rng(random_state)
    supplied = None
    if initial is not None:
        try:
            if len(initial) != len(terms) or any(
                len(lag) != count for lag, count in zip(initial, terms)
            ):
                raise ValueError("initial lag/term structure must match terms")
            supplied = []
            for lag in initial:
                one = []
                for term in lag:
                    if len(term) != len(shape):
                        raise ValueError("initial must contain one matrix per mode")
                    matrices = []
                    for a, d in zip(term, shape):
                        raw = np.asarray(a)
                        if (
                            raw.dtype.kind not in "iuf"
                            or raw.shape != (d, d)
                            or not np.isfinite(raw).all()
                        ):
                            raise ValueError(
                                "initial coefficient matrices must be finite real square arrays"
                            )
                        matrices.append(np.array(raw, dtype=float, copy=True))
                    one.append(matrices)
                supplied.append(one)
            supplied = _canonicalize(supplied)
        except TypeError as exc:
            raise ValueError("initial must be nested [lag][term][mode]") from exc
    supplied_covariance = None
    if covariance_initial is not None:
        if len(covariance_initial) != len(shape):
            raise ValueError("one covariance_initial factor is required per mode")
        supplied_covariance = []
        for s, d in zip(covariance_initial, shape):
            raw = np.asarray(s)
            if (
                raw.dtype.kind not in "iuf"
                or raw.shape != (d, d)
                or not np.isfinite(raw).all()
                or not np.allclose(raw, raw.T, rtol=1e-12, atol=0)
            ):
                raise ValueError(
                    "covariance_initial factors must be finite real symmetric matrices"
                )
            supplied_covariance.append(_spd(np.array(raw, dtype=float), 0)[0])
        supplied_covariance = _normalize_covariances(supplied_covariance)
        supplied_covariance[-1] = (supplied_covariance[-1] / scale) / scale
        _whitener(supplied_covariance[-1])
    runs, solutions = [], []
    for attempt in range(n_starts):
        history, diagnostics = [], ()
        regularized, max_condition, max_cancellation = False, 1.0, 1.0
        coefficients, covariances, residuals = None, None, None
        status, message, converged = "iteration_limit", "maximum sweeps reached", False
        try:
            if supplied is not None:
                coefficients = _copy_coefficients(supplied)
                if attempt:
                    for lag in coefficients:
                        for term in lag:
                            for k in range(len(term)):
                                term[k] += rng.normal(size=term[k].shape) * (
                                    0.05
                                    * max(_norm(term[k]), 0.1)
                                    / np.sqrt(term[k].size)
                                )
                    coefficients = _canonicalize(coefficients)
                var_residual = None
            elif init == "projection":
                coefficients, diagnostics, var_residual = _projection(
                    lagged,
                    response,
                    terms,
                    rng,
                    projection_max_iter,
                    projection_tol,
                    max_dense_dimension,
                )
                if attempt and method != "projection":
                    for lag in coefficients:
                        for term in lag:
                            for k in range(len(term)):
                                term[k] += rng.normal(size=term[k].shape) * (
                                    0.05
                                    * max(_norm(term[k]), 0.1)
                                    / np.sqrt(term[k].size)
                                )
                    coefficients = _canonicalize(coefficients)
            else:
                coefficients = _canonicalize(
                    [
                        [
                            [rng.normal(size=(d, d)) / np.sqrt(d) for d in shape]
                            for _ in range(count)
                        ]
                        for count in terms
                    ]
                )
                for lag in coefficients:
                    for term in lag:
                        term[-1] *= (
                            0.2
                            / max(_norm(term[-1]), np.finfo(float).tiny)
                            / sum(terms)
                        )
                var_residual = None
            residuals = response - _prediction(lagged, coefficients)
            if method == "mle":
                if supplied_covariance is not None:
                    covariances = [s.copy() for s in supplied_covariance]
                elif var_residual is not None:
                    covariances, regularized = _covariance_projection(
                        var_residual, shape, floor
                    )
                else:
                    energy = float(np.mean(residuals**2))
                    if energy <= 0:
                        raise ValueError(
                            "Gaussian likelihood is undefined at zero residual variance"
                        )
                    covariances = [np.eye(d) for d in shape]
                    covariances[-1] *= energy
                    covariances = _normalize_covariances(covariances)
            objective = (
                _negative_loglike(residuals, covariances)
                if method == "mle"
                else float(np.sum(residuals**2))
            )
            history.append(objective)
            if method == "projection":
                converged = all(d.converged for d in diagnostics)
                status, message = (
                    ("projection_converged", "projection objective converged")
                    if converged
                    else (
                        "projection_iteration_limit",
                        "local CP projection reached iteration limit",
                    )
                )
            else:
                for iteration in range(1, max_iter + 1):
                    old_coefficients = _copy_coefficients(coefficients)
                    old_covariances = (
                        None if covariances is None else [s.copy() for s in covariances]
                    )
                    old_residuals = residuals.copy()
                    try:
                        for i, lag in enumerate(coefficients):
                            for r, term in enumerate(lag):
                                for k in range(len(shape)):
                                    partial = residuals + _apply(lagged[i], term)
                                    predictor = _apply(lagged[i], term, skip=k)
                                    term[k], condition = _coefficient_block(
                                        partial, predictor, k, covariances
                                    )
                                    max_condition = max(max_condition, condition)
                                    term = _normalize_term(term)
                                    coefficients[i][r] = term
                                    residuals = partial - _apply(lagged[i], term)
                        coefficients = _canonicalize(coefficients)
                        residuals = response - _prediction(lagged, coefficients)
                        if method == "mle":
                            for k in range(len(shape) - 1, -1, -1):
                                covariances[k], active = _covariance_block(
                                    residuals, covariances, k, floor
                                )
                                regularized |= active
                            covariances = _normalize_covariances(covariances)
                        value = (
                            _negative_loglike(residuals, covariances)
                            if method == "mle"
                            else float(np.sum(residuals**2))
                        )
                        cancellation = max(
                            d["cancellation_ratio"]
                            for d in _coefficient_diagnostics(coefficients)
                        )
                        max_cancellation = max(max_cancellation, cancellation)
                        if not np.isfinite(value) or cancellation > 1e8:
                            raise ValueError(
                                "numerically degenerate fit: nonfinite objective or component cancellation"
                            )
                        objective_scale = (
                            residuals.size
                            if method == "mle"
                            else max(abs(history[-1]), abs(value), np.finfo(float).tiny)
                        )
                        if value > history[-1] + 1e-9 * objective_scale:
                            raise ValueError(
                                "objective increased; last complete sweep retained"
                            )
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                        coefficients, covariances, residuals = (
                            old_coefficients,
                            old_covariances,
                            old_residuals,
                        )
                        raise
                    history.append(value)
                    if abs(history[-2] - value) <= tol * objective_scale:
                        converged, status, message = (
                            True,
                            "converged",
                            (
                                "mean likelihood change below tolerance"
                                if method == "mle"
                                else "relative SSE change below tolerance"
                            ),
                        )
                        break
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            status, message, converged = "failed", str(exc), False
        runs.append(
            TensorARRun(
                converged,
                max(0, len(history) - 1),
                np.asarray(history),
                status,
                message,
                regularized,
                max_condition,
                max_cancellation,
                diagnostics,
            )
        )
        solutions.append((coefficients, covariances, residuals))
    eligible = [i for i, run in enumerate(runs) if np.isfinite(run.objective)]
    if not eligible:
        raise ValueError(
            "no finite tensor AR start: " + "; ".join(run.message for run in runs)
        )
    selected = min(
        eligible,
        key=lambda i: (
            sum(d.objective_history[-1] for d in runs[i].projection)
            if method == "projection"
            else runs[i].objective
        ),
    )
    coefficients, covariances, residuals = solutions[selected]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        fitted = (_prediction(lagged, coefficients) + mean) * scale
        physical_residuals = residuals * scale
        physical_mean = mean * scale
        if covariances is not None:
            covariances[-1] = (covariances[-1] * scale) * scale
            for covariance in covariances:
                _whitener(covariance)
    if not np.isfinite(fitted).all() or not np.isfinite(physical_residuals).all():
        raise FloatingPointError("tensor AR output exceeds physical data units")
    return TensorARResult(
        tuple(
            tuple(tuple(a.copy() for a in term) for term in lag) for lag in coefficients
        ),
        lags,
        terms,
        method,
        fitted,
        physical_residuals,
        physical_mean,
        scale,
        None if covariances is None else tuple(covariances),
        tuple(runs),
        selected,
        _coefficient_diagnostics(coefficients),
        max_dense_dimension,
        X[-order:].copy(),
    )
