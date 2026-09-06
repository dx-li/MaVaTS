"""Matrix time-series factor estimation within known loading subspaces."""

import numpy as np

from .factors import (
    _canonical_signs,
    _eigenspace,
    _factor_ranks,
    _lagged_moment,
    _lags,
    _prepare,
    _result,
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
    paper's partial and multi-term models are not implemented here. White
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
