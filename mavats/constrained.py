"""Matrix factor estimation with full, partial and multi-term constraints.

References
----------
Chen, Tsay and Chen (2020), Constrained Factor Models for High-Dimensional
Matrix-Variate Time Series. https://doi.org/10.1080/01621459.2019.1584899
Algorithm version: https://arxiv.org/html/1710.06075v3, Sections 3.1--3.4.
"""

from dataclasses import dataclass, field

import numpy as np

from ._validation import as_series, positive_int
from .factors import (
    _canonical_signs,
    _eigenspace,
    _factor_ranks,
    _lagged_moment,
    _lags,
    _prepare,
    _result,
    eigenvalue_ratio,
)


def _constraint_basis(constraint, dimension, name):
    if constraint is None:
        return np.eye(dimension)
    raw = np.asarray(constraint)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    matrix = np.asarray(raw, dtype=float)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != dimension
        or not 1 <= matrix.shape[1] <= dimension
    ):
        raise ValueError(
            f"{name} must have shape ({dimension}, q), 1 <= q <= {dimension}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    # Column scales have no meaning for a span; normalize them before checking
    # numerical rank so arbitrary units do not turn a valid basis into a failure.
    scales = np.max(np.abs(matrix), axis=0)
    if np.any(scales == 0):
        raise ValueError(f"{name} must have full column rank")
    left, singular, _ = np.linalg.svd(matrix / scales, full_matrices=False)
    cutoff = np.finfo(float).eps * max(matrix.shape) * singular[0]
    if singular[-1] <= cutoff:
        raise ValueError(f"{name} must have full numerical column rank")
    return _canonical_signs(left)


def fit_constrained_factor(
    X,
    ranks=None,
    *,
    row_constraints=None,
    column_constraints=None,
    lags=1,
    center=False,
):
    """Fit a single-term constrained lagged matrix factor model.

    ``row_constraints`` and ``column_constraints`` are full-column-rank bases
    for the permitted loading spaces, with shapes ``(m, q_row)`` and
    ``(n, q_col)``. ``None`` leaves that side unrestricted. Ranks cannot exceed
    the corresponding constraint dimensions. Constraint columns are
    orthonormalized; their units and invertible reparameterizations do not
    change the estimated spaces.

    Implements Chen, Tsay and Chen (2020), Sections 3.1--3.2: project into the
    constraint spaces, estimate lagged factors, then lift the loadings. The
    Partial and multi-term models have separate fitting functions. White
    measurement noise and temporally informative factors are required.

    Returns a ``FactorResult`` with orthonormal loadings in the original
    coordinates. ``center=True`` adds an unrestricted training mean; constraints
    then apply to deviations from that mean. No entrywise standardization is
    performed. ``lags`` and automatic ranks follow ``fit_lagged_factor``.

    References
    ----------
    Chen, Tsay and Chen (2020), Constrained Factor Models for High-Dimensional
    Matrix-Variate Time Series, Sections 3.1--3.2.
    https://doi.org/10.1080/01621459.2019.1584899
    """
    X, work, mean, scale = _prepare(X, center)
    row = _constraint_basis(row_constraints, X.shape[1], "row_constraints")
    column = _constraint_basis(column_constraints, X.shape[2], "column_constraints")
    ranks = _factor_ranks(ranks, (row.shape[1], column.shape[1]))
    lags = _lags(lags, len(X))
    reduced = row.T @ work @ column
    moments = (
        _lagged_moment(reduced, lags, "topup"),
        _lagged_moment(reduced.transpose(0, 2, 1), lags, "topup"),
    )
    pairs = [_eigenspace(moment, rank) for moment, rank in zip(moments, ranks)]
    loadings = [
        _canonical_signs(basis @ pair[0]) for basis, pair in zip((row, column), pairs)
    ]
    return _result(
        X, loadings, mean, [pair[1] for pair in pairs], "constrained-lagged", scale
    )


def _split_constraints(constraint, dimension, name):
    if constraint is not None and np.shape(constraint) == (dimension, 0):
        raw = np.asarray(constraint)
        if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
            raise ValueError(f"{name} must contain real numeric values")
        return np.empty((dimension, 0)), np.eye(dimension)
    basis = _constraint_basis(constraint, dimension, name)
    if basis.shape[1] == dimension:
        return basis, np.empty((dimension, 0))
    complement = np.linalg.svd(basis, full_matrices=True)[0][:, basis.shape[1] :]
    return basis, complement


def _group_ranks(ranks, dimensions, name):
    if ranks is None:
        ranks = (None,) * len(dimensions)
    try:
        ranks = tuple(ranks)
    except TypeError as exc:
        raise ValueError(f"{name} must have one rank per loading group") from exc
    if len(ranks) != len(dimensions):
        raise ValueError(f"{name} must have one rank per loading group")
    result = []
    for rank, dimension in zip(ranks, dimensions):
        rank = None if rank is None else positive_int(rank, name, minimum=0)
        if rank is not None and rank > dimension:
            raise ValueError(f"{name} exceeds its constraint or complement dimension")
        result.append(0 if dimension == 0 else rank)
    return tuple(result)


def _moment_space(moment, rank, name):
    """Scale-safe eigensolve with explicit numerical identification diagnostics."""
    dimension = len(moment)
    scale = float(np.max(np.abs(moment))) if moment.size else 0.0
    if scale:
        normalized = moment / scale
        values, vectors = np.linalg.eigh((normalized + normalized.T) / 2)
        values, vectors = np.maximum(values[::-1], 0), vectors[:, ::-1]
        cutoff = values[0] * dimension * np.finfo(float).eps
        numerical_rank = int(np.count_nonzero(values > cutoff))
    else:
        values, vectors, numerical_rank = np.zeros(dimension), np.eye(dimension), 0
    bound = min(max(1, dimension // 2), dimension - 1) if dimension else 0
    automatic = rank is None
    if automatic:
        rank = eigenvalue_ratio(values) if numerical_rank else 0
    if rank > numerical_rank:
        raise ValueError(
            f"{name} is unidentified: requested rank {rank} exceeds "
            f"numerical lag-moment rank {numerical_rank}"
        )
    loading = _canonical_signs(vectors[:, :rank]) if rank else np.empty((dimension, 0))
    return (
        loading,
        values * scale,
        dict(
            rank=rank,
            automatic=automatic,
            numerical_moment_rank=numerical_rank,
            search_bound=bound if automatic else None,
            zero_moment=not bool(scale),
        ),
    )


def _finite_output(value, name):
    if not np.isfinite(value).all():
        raise FloatingPointError(f"{name} exceeds representable original data units")
    return value


def _new_work(X, mean):
    X = as_series(X, min_samples=1)
    if X.shape[1:] != mean.shape:
        raise ValueError("observation dimensions must match fitted constraints")
    scales = np.maximum(np.max(np.abs(X), axis=(1, 2)), np.max(np.abs(mean)))
    scales = np.maximum(scales, np.finfo(float).tiny)
    return X / scales[:, None, None] - mean / scales[:, None, None], scales


def _core_array(value, ranks, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain finite real numeric values")
    value = np.asarray(raw, dtype=float)
    if value.ndim != 3 or len(value) < 1 or value.shape[1:] != ranks:
        raise ValueError(f"{name} must have shape (T, {ranks[0]}, {ranks[1]})")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite values")
    return value


def _reconstruct_parts(cores, loadings, mean):
    n = len(cores[0])
    scales = np.full(n, max(float(np.max(np.abs(mean))), np.finfo(float).tiny))
    for core in cores:
        if core.size:
            scales = np.maximum(scales, np.max(np.abs(core), axis=(1, 2)))
    work = np.broadcast_to(mean / scales[:, None, None], (n, *mean.shape)).copy()
    for core, (row, column) in zip(cores, loadings):
        work += row @ (core / scales[:, None, None]) @ column.T
    with np.errstate(over="ignore", invalid="ignore"):
        return _finite_output(work * scales[:, None, None], "reconstruction")


@dataclass
class PartialConstrainedFactorResult:
    """Four-block partial model with shared orthonormal row/column groups.

    ``ranks`` are total ranks; ``row_ranks`` and ``column_ranks`` distinguish
    supplied constraints from their complements. ``factor_blocks`` and
    ``signal_blocks`` are 2-by-2 tuples, excluding the optional training mean.
    Eigenvalues and block moments use the single original ``X/data_scale``
    normalization; fourth-power physical moment units need not be representable.
    ``converged=True`` describes direct eigensolves, not statistical validity.

    References
    ----------
    Chen, Tsay and Chen (2020), Constrained Factor Models for High-Dimensional
    Matrix-Variate Time Series. https://doi.org/10.1080/01621459.2019.1584899
    """

    loadings: tuple
    factors: np.ndarray
    signal: np.ndarray
    residuals: np.ndarray
    mean: np.ndarray
    row_ranks: tuple
    column_ranks: tuple
    constraint_bases: tuple
    eigenvalues: tuple
    block_moments: tuple
    rank_diagnostics: tuple
    data_scale: float
    interactions: bool
    lags: tuple
    method: str = "partial-constrained-lagged"
    converged: bool = True
    n_iter: int = 0

    @property
    def ranks(self):
        return sum(self.row_ranks), sum(self.column_ranks)

    @property
    def factor_blocks(self):
        r, c = self.row_ranks[0], self.column_ranks[0]
        return tuple(
            tuple(self.factors[:, rs, cs] for cs in (slice(0, c), slice(c, None)))
            for rs in (slice(0, r), slice(r, None))
        )

    @property
    def signal_blocks(self):
        rows = np.split(self.loadings[0], [self.row_ranks[0]], axis=1)
        columns = np.split(self.loadings[1], [self.column_ranks[0]], axis=1)
        return tuple(
            tuple(
                _reconstruct_parts(
                    [self.factor_blocks[l][k]],
                    [(rows[l], columns[k])],
                    np.zeros_like(self.mean),
                )
                for k in range(2)
            )
            for l in range(2)
        )

    def transform(self, X):
        """Project new observations with fixed shared loading spaces.

        References
        ----------
        Chen, Tsay and Chen (2020), Constrained Factor Models for High-Dimensional
        Matrix-Variate Time Series, equations (5)--(6).
        https://doi.org/10.1080/01621459.2019.1584899
        This contemporaneous projection is not a forecast or loading refit.
        """
        work, scales = _new_work(X, self.mean)
        cores = self.loadings[0].T @ work @ self.loadings[1]
        if not self.interactions:
            r, c = self.row_ranks[0], self.column_ranks[0]
            cores[:, :r, c:] = 0
            cores[:, r:, :c] = 0
        with np.errstate(over="ignore", invalid="ignore"):
            return _finite_output(cores * scales[:, None, None], "factor scores")

    def inverse_transform(self, factors):
        """Reconstruct the fitted block model and restore its training mean.

        References
        ----------
        Chen, Tsay and Chen (2020), Constrained Factor Models for High-Dimensional
        Matrix-Variate Time Series. https://doi.org/10.1080/01621459.2019.1584899
        Mean restoration is an explicit implementation convention.
        """
        factors = _core_array(factors, self.ranks, "factors")
        if not self.interactions:
            r, c = self.row_ranks[0], self.column_ranks[0]
            if np.any(factors[:, :r, c:] != 0) or np.any(factors[:, r:, :c] != 0):
                raise ValueError(
                    "diagonal interaction model requires zero cross blocks"
                )
        return _reconstruct_parts([factors], [self.loadings], self.mean)


def fit_partial_constrained_factor(
    X,
    *,
    row_constraints=None,
    column_constraints=None,
    row_ranks=(None, None),
    column_ranks=(None, None),
    interactions=True,
    lags=1,
    center=False,
):
    """Fit the partial matrix factor model with shared four-block loadings.

    Constraint matrices have shapes (p,m1), (q,m2). Each rank pair gives the
    rank inside the supplied space followed by the rank in its orthogonal
    complement, bounded by (m1,p-m1) or (m2,q-m2). None constraints mean the
    full space; an explicit (dimension,0) matrix means an empty supplied space.
    Empty groups have rank zero; explicit zero ranks suppress a loading group.
    Missing ranks use a numerical eigenvalue ratio, not a no-factor test;
    exactly zero moments return zero. Positive requested ranks must have
    numerically informative lag moments. Full population identification and
    constraint correctness cannot be certified by these finite-sample checks.

    For each l,k, form X_lk=H_Rl' X H_Ck. Equation (14) adds the SEPARATE
    lag-moment Gram matrices over k for row group l, and over l for column
    group k. Cross-block lag products are not included. Data are normalized
    once before all blocks; blockwise rescaling would change the estimator.
    ``interactions=False`` imposes equation (6)'s zero cross-factor blocks;
    only diagonal blocks then enter loading estimation and reconstruction.
    This restriction must be scientifically justified, not selected implicitly.

    ``center=True`` removes/restores an ordinary training mean. No entrywise
    standardization, forecast dynamics, rank tests, or uncertainty estimates
    are supplied. See ``eigenvalue_ratio``: the implementation uses half the
    block dimension as its search bound, not the paper's dimensionally
    ambiguous ambient-p bound. Full block ranks require explicit specification.

    References
    ----------
    Chen, Tsay and Chen (2020), Constrained Factor Models for High-Dimensional
    Matrix-Variate Time Series, equations (5), (6), (12), (14), Section 3.4.
    https://doi.org/10.1080/01621459.2019.1584899
    Algorithm text: https://arxiv.org/html/1710.06075v3#S3.SS4
    """
    X, work, mean, scale = _prepare(X, center)
    if not isinstance(interactions, (bool, np.bool_)):
        raise ValueError("interactions must be boolean")
    lags = _lags(lags, len(X))
    rows = _split_constraints(row_constraints, X.shape[1], "row_constraints")
    columns = _split_constraints(column_constraints, X.shape[2], "column_constraints")
    requested = (
        _group_ranks(row_ranks, [h.shape[1] for h in rows], "row_ranks"),
        _group_ranks(column_ranks, [h.shape[1] for h in columns], "column_ranks"),
    )
    moments = [[], []]
    for l in range(2):
        rm, cm = [], []
        for k in range(2):
            block = rows[l].T @ work @ columns[k]
            active = interactions or l == k
            rm.append(
                _lagged_moment(block, lags, "topup")
                if active and block.size
                else np.zeros((rows[l].shape[1],) * 2)
            )
            cm.append(
                _lagged_moment(block.transpose(0, 2, 1), lags, "topup")
                if active and block.size
                else np.zeros((columns[k].shape[1],) * 2)
            )
        moments[0].append(tuple(rm))
        moments[1].append(tuple(cm))
    pairs = [[], []]
    for l in range(2):
        pairs[0].append(
            _moment_space(sum(moments[0][l]), requested[0][l], f"row group {l}")
        )
        pairs[1].append(
            _moment_space(
                sum(moments[1][k][l] for k in range(2)),
                requested[1][l],
                f"column group {l}",
            )
        )
    loading = tuple(
        np.column_stack([h @ entry[0] for h, entry in zip(bases, pair)])
        for bases, pair in zip((rows, columns), pairs)
    )
    result = PartialConstrainedFactorResult(
        loading,
        np.empty((0, 0, 0)),
        np.empty_like(X),
        np.empty_like(X),
        mean,
        tuple(p[0].shape[1] for p in pairs[0]),
        tuple(p[0].shape[1] for p in pairs[1]),
        (rows, columns),
        tuple(tuple(p[1] for p in pair) for pair in pairs),
        tuple(tuple(part) for part in moments),
        tuple(tuple(p[2] for p in pair) for pair in pairs),
        scale,
        bool(interactions),
        lags,
    )
    result.factors = result.transform(X)
    result.signal = result.inverse_transform(result.factors)
    with np.errstate(over="ignore", invalid="ignore"):
        result.residuals = _finite_output(X - result.signal, "residuals")
    return result


def _orthogonal(first, second):
    return np.max(np.abs(first.T @ second), initial=0.0) <= (
        10 * np.finfo(float).eps * max(first.shape[0], second.shape[0])
    )


def _complement_union(bases, dimension):
    """Orthonormal complement of a possibly overlapping union of spans."""
    if not bases:
        return np.eye(dimension)
    matrix = np.column_stack(bases)
    u, singular, _ = np.linalg.svd(matrix, full_matrices=True)
    rank = np.count_nonzero(
        singular > singular[0] * max(matrix.shape) * np.finfo(float).eps
    )
    return u[:, rank:]


def _survival(complement, basis, loading, name):
    projected_constraint = complement.T @ basis
    projected_loading = projected_constraint @ loading
    # Absolute cutoff matters here: an exact lost direction may survive only
    # as O(eps) projector error. A relative cutoff on that error would accept it.
    cutoff = 10 * np.finfo(float).eps * max(basis.shape)
    cs = np.linalg.svd(projected_constraint, compute_uv=False)
    ls = np.linalg.svd(projected_loading, compute_uv=False)
    retained = int(np.count_nonzero(ls > cutoff))
    if retained < loading.shape[1]:
        raise ValueError(
            f"{name} is unidentified: competing projection loses loading rank"
        )
    return dict(
        constraint_rank=int(np.count_nonzero(cs > cutoff)),
        loading_rank=retained,
        smallest_loading_singular_value=float(ls[-1]) if len(ls) else None,
    )


@dataclass
class MultiTermConstrainedFactorResult:
    """Sum of separately constrained components with joint least-squares scores.

    ``loadings[j]`` is a (row,column) pair and ``factors[j]`` has shape
    (T,rj,cj). Components exclude ``mean``. Overlapping components are generally
    not orthogonal: their energies do not obey a Pythagorean decomposition.
    ``identification_diagnostics`` records the surviving projected ranks, not
    a proof of population identification. ``converged=True`` means direct
    linear algebra completed, not statistical consistency or rank correctness.

    References
    ----------
    Chen, Tsay and Chen (2020), Constrained Factor Models for High-Dimensional
    Matrix-Variate Time Series. https://doi.org/10.1080/01621459.2019.1584899
    """

    loadings: tuple
    factors: tuple
    signal: np.ndarray
    residuals: np.ndarray
    mean: np.ndarray
    constraint_bases: tuple
    eigenvalues: tuple
    rank_diagnostics: tuple
    identification_diagnostics: tuple
    score_condition: float
    score_design_rank: int
    data_scale: float
    lags: tuple
    _score_inverse: np.ndarray = field(repr=False)
    method: str = "multiterm-constrained-lagged"
    converged: bool = True
    n_iter: int = 0

    @property
    def ranks(self):
        return tuple((row.shape[1], column.shape[1]) for row, column in self.loadings)

    @property
    def component_signals(self):
        return tuple(
            _reconstruct_parts([core], [loading], np.zeros_like(self.mean))
            for core, loading in zip(self.factors, self.loadings)
        )

    def transform(self, X):
        """Jointly solve new contemporaneous component scores in vec_F order.

        References
        ----------
        Chen, Tsay and Chen (2020), Constrained Factor Models for High-Dimensional
        Matrix-Variate Time Series. https://doi.org/10.1080/01621459.2019.1584899
        Joint score regression completes the reconstruction of Section 3.3's
        loading estimator; it is an implementation extension, not an explicit
        paper factor-update equation or a forecast. Loadings remain fixed.
        """
        work, scales = _new_work(X, self.mean)
        flat = work.transpose(0, 2, 1).reshape(len(work), -1)
        with np.errstate(over="ignore", invalid="ignore"):
            scores = _finite_output(
                (flat @ self._score_inverse.T) * scales[:, None], "joint factor scores"
            )
        cores, start = [], 0
        for r, c in self.ranks:
            cores.append(
                scores[:, start : start + r * c]
                .reshape(len(work), c, r)
                .transpose(0, 2, 1)
            )
            start += r * c
        return tuple(cores)

    def inverse_transform(self, factors):
        """Sum component signals and restore the optional training mean.

        References
        ----------
        Chen, Tsay and Chen (2020), Constrained Factor Models for High-Dimensional
        Matrix-Variate Time Series, equation (3).
        https://doi.org/10.1080/01621459.2019.1584899
        Mean restoration is an implementation convention.
        """
        try:
            factors = tuple(factors)
        except TypeError as exc:
            raise ValueError("factors must contain one core series per term") from exc
        if len(factors) != len(self.ranks):
            raise ValueError("factors must contain one core series per term")
        cores = tuple(
            _core_array(core, rank, f"factors[{j}]")
            for j, (core, rank) in enumerate(zip(factors, self.ranks))
        )
        if len({len(core) for core in cores}) != 1:
            raise ValueError("component factor series must have equal sample counts")
        return _reconstruct_parts(cores, self.loadings, self.mean)


def fit_multiterm_constrained_factor(
    X,
    constraints,
    ranks=None,
    *,
    lags=1,
    center=False,
    max_design_elements=10_000_000,
):
    """Fit a sum of constrained matrix factor terms, including overlapping spans.

    ``constraints`` is a nonempty sequence of (row_basis,column_basis) pairs;
    each basis is full column rank, or None for the full corresponding space.
    ``ranks`` supplies one (row_rank,column_rank) pair per term. None ranks use
    the half-block-dimension numerical ``eigenvalue_ratio`` convention, not a
    calibrated absence-of-factors test; explicit (0,0) omits a term and its competing
    projections. A term must have both ranks positive or both zero.

    Orthogonal product spaces are isolated by own-space projections (Section
    3.3); either row OR column orthogonality removes a competing term. Otherwise
    Remark 3 estimates row loadings after annihilating competing column spans,
    and column loadings after annihilating competing row spans. Complement
    coordinates are used instead of dense projectors. Already annihilated
    orthogonal competitors are excluded, avoiding unnecessary information loss.
    For more than two terms, removing the union of competing spans is the
    explicit algebraic extension of the paper's two-term construction.

    Both the projected constraints and estimated loadings must retain the
    requested numerical ranks, and the joint vec_F Kronecker score design must
    be full column rank. These are necessary numerical checks, not guarantees
    that unknown population factors survive or have informative lag dynamics.
    Near-overlap can strongly amplify noise; survival singular values and the
    joint score condition number are returned, with no implicit ridge.

    Scores jointly minimize the original-coordinate reconstruction error.
    This is an explicit reconstruction extension for overlapping terms, not a
    factor-update equation specified in Remark 3. The dense score design is
    guarded by ``max_design_elements``. Fitting uses one original-data scale;
    centering is optional and no entrywise standardization is performed.
    White measurement noise independent of temporally informative factors is
    required. No forecasting, inference or calibrated rank test is supplied.

    References
    ----------
    Chen, Tsay and Chen (2020), Constrained Factor Models for High-Dimensional
    Matrix-Variate Time Series, equation (3), Section 3.3 and Remark 3.
    https://doi.org/10.1080/01621459.2019.1584899
    Algorithm text: https://arxiv.org/html/1710.06075v3#S3.SS3
    """
    X, work, mean, scale = _prepare(X, center)
    lags = _lags(lags, len(X))
    max_design_elements = positive_int(max_design_elements, "max_design_elements")
    try:
        constraints = tuple(tuple(pair) for pair in constraints)
    except TypeError as exc:
        raise ValueError("constraints must contain (row,column) basis pairs") from exc
    if not constraints or any(len(pair) != 2 for pair in constraints):
        raise ValueError("constraints must contain nonempty (row,column) basis pairs")
    bases = tuple(
        tuple(
            _constraint_basis(h, X.shape[k + 1], f"constraints[{j}][{k}]")
            for k, h in enumerate(pair)
        )
        for j, pair in enumerate(constraints)
    )
    if ranks is None:
        ranks = (None,) * len(bases)
    try:
        ranks = tuple(ranks)
    except TypeError as exc:
        raise ValueError("ranks must contain one pair per term") from exc
    if len(ranks) != len(bases):
        raise ValueError("ranks must contain one pair per term")
    requested = tuple(
        _group_ranks(pair, [h.shape[1] for h in hs], f"ranks[{j}]")
        for j, (pair, hs) in enumerate(zip(ranks, bases))
    )
    for pair in requested:
        if 0 in pair and pair != (0, 0):
            raise ValueError("an omitted term must have ranks (0,0)")
    active = [j for j, pair in enumerate(requested) if pair != (0, 0)]
    loadings, spectra, rank_info, identification = [], [], [], []
    for j, (row, column) in enumerate(bases):
        others = [i for i in active if i != j]
        direct = all(
            _orthogonal(row, bases[i][0]) or _orthogonal(column, bases[i][1])
            for i in others
        )
        if requested[j] == (0, 0):
            loadings.append((row[:, :0], column[:, :0]))
            spectra.append((np.zeros(row.shape[1]), np.zeros(column.shape[1])))
            rank_info.append(
                tuple(
                    dict(
                        rank=0,
                        automatic=False,
                        numerical_moment_rank=None,
                        search_bound=None,
                        zero_moment=None,
                    )
                    for _ in range(2)
                )
            )
            identification.append(
                dict(estimator="omitted", row_survival=None, column_survival=None)
            )
            continue
        if direct:
            reduced = row.T @ work @ column
            series = (reduced, reduced.transpose(0, 2, 1))
            row_complement, column_complement = np.eye(len(row)), np.eye(len(column))
        else:
            column_competitors = [
                bases[i][1] for i in others if not _orthogonal(row, bases[i][0])
            ]
            row_competitors = [
                bases[i][0] for i in others if not _orthogonal(column, bases[i][1])
            ]
            row_complement = _complement_union(row_competitors, len(row))
            column_complement = _complement_union(column_competitors, len(column))
            if not row_complement.shape[1] or not column_complement.shape[1]:
                raise ValueError(
                    f"term {j} is unidentified: competing constraints exhaust an ambient space"
                )
            series = (
                row.T @ work @ column_complement,
                column.T @ work.transpose(0, 2, 1) @ row_complement,
            )
        pairs = tuple(
            _moment_space(_lagged_moment(z, lags, "topup"), rank, f"term {j} mode {k}")
            for k, (z, rank) in enumerate(zip(series, requested[j]))
        )
        if any(p[0].shape[1] == 0 for p in pairs):
            raise ValueError(
                f"term {j} has no informative projected dynamics; explicitly omit it with (0,0)"
            )
        survived = (
            _survival(row_complement, row, pairs[0][0], f"term {j} row"),
            _survival(column_complement, column, pairs[1][0], f"term {j} column"),
        )
        loadings.append((row @ pairs[0][0], column @ pairs[1][0]))
        spectra.append(tuple(p[1] for p in pairs))
        rank_info.append(tuple(p[2] for p in pairs))
        identification.append(
            dict(
                estimator="direct" if direct else "competing-complements",
                row_survival=survived[0],
                column_survival=survived[1],
            )
        )
    score_dimension = sum(row.shape[1] * column.shape[1] for row, column in loadings)
    if X.shape[1] * X.shape[2] * score_dimension > max_design_elements:
        raise ValueError("joint score design exceeds max_design_elements")
    design = np.column_stack([np.kron(column, row) for row, column in loadings])
    if score_dimension:
        u, s, vh = np.linalg.svd(design, full_matrices=False)
        rank = int(np.count_nonzero(s > s[0] * max(design.shape) * np.finfo(float).eps))
        if rank != score_dimension:
            raise ValueError(
                "joint score design is rank deficient: components are not identifiable"
            )
        inverse, condition = (vh.T / s) @ u.T, float(s[0] / s[-1])
    else:
        inverse, condition, rank = np.empty((0, design.shape[0])), 1.0, 0
    result = MultiTermConstrainedFactorResult(
        tuple(loadings),
        (),
        np.empty_like(X),
        np.empty_like(X),
        mean,
        bases,
        tuple(spectra),
        tuple(rank_info),
        tuple(identification),
        condition,
        rank,
        scale,
        lags,
        inverse,
    )
    result.factors = result.transform(X)
    result.signal = result.inverse_transform(result.factors)
    with np.errstate(over="ignore", invalid="ignore"):
        result.residuals = _finite_output(X - result.signal, "residuals")
    return result
