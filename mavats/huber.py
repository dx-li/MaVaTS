"""Matrixwise Huber factor estimation with a checked descent safeguard."""

from dataclasses import dataclass

import numpy as np

from ._validation import finite_scalar, positive_int, ranks_tuple
from .factors import (
    FactorResult,
    _canonical_signs,
    _eigenspace,
    _matrix_moments,
    _prepare,
    _result,
    _space_distance,
)


@dataclass
class HuberFactorResult(FactorResult):
    """Robust loading estimate and fixed-threshold optimization diagnostics.

    ``threshold`` is in original matrix Frobenius-norm units. ``weights`` are
    final observation weights. ``objective_history`` includes initialization
    and accepted sweeps, reporting mean Huber loss after dividing observations
    and threshold by ``data_scale``. Multiply by ``data_scale**2`` for original
    loss units when representable. The Huber convention is quadratic loss below
    threshold and ``2 * threshold * norm - threshold**2`` above it.

    ``safeguard_steps`` counts simultaneous updates replaced with sequential
    weighted projection. ``converged`` concerns this local optimization only.
    Factor scores use ordinary projection and are not individually robust.
    """

    threshold: float = 0.0
    weights: np.ndarray | None = None
    objective_history: np.ndarray | None = None
    safeguard_steps: int = 0
    stopping_reason: str = "max_iter"


def _residual_norms(work, loadings):
    row, column = loadings
    core = row.T @ work @ column
    residual = work - row @ core @ column.T
    # Explicit residuals avoid catastrophic cancellation in ||X||² - ||F||².
    return np.sqrt(np.einsum("tij,tij->t", residual, residual, optimize=True))


def _weights_and_loss(norms, threshold):
    weights = np.ones_like(norms)
    large = norms > threshold
    weights[large] = threshold / norms[large]
    losses = norms**2
    losses[large] = threshold * (2 * norms[large] - threshold)
    return weights, float(losses.mean())


def _weighted_moment(work, opposite, weights):
    projected = work @ opposite
    return np.einsum(
        "t,tij,tkj->ik", weights, projected, projected, optimize=True
    ) / len(work)


def _initial_spaces(initial, work, ranks):
    if initial is None:
        return [
            _eigenspace(moment, rank)[0]
            for moment, rank in zip(_matrix_moments(work), ranks)
        ]
    try:
        initial = tuple(initial)
    except TypeError as exc:
        raise ValueError("initial must be a pair of loading matrices") from exc
    if len(initial) != 2:
        raise ValueError("initial must be a pair of loading matrices")
    spaces = []
    for matrix, dimension, rank in zip(initial, work.shape[1:], ranks):
        raw = np.asarray(matrix)
        if (
            np.iscomplexobj(raw)
            or raw.dtype.kind not in "biuf"
            or raw.shape != (dimension, rank)
        ):
            raise ValueError("initial loading shapes must match dimensions and ranks")
        matrix = np.asarray(raw, dtype=float)
        if not np.isfinite(matrix).all():
            raise ValueError("initial loadings must be finite")
        scales = np.max(np.abs(matrix), axis=0)
        if np.any(scales == 0):
            raise ValueError("initial loadings must have full column rank")
        vectors, singular, _ = np.linalg.svd(matrix / scales, full_matrices=False)
        if singular[-1] <= np.finfo(float).eps * max(matrix.shape) * singular[0]:
            raise ValueError("initial loadings must have full numerical column rank")
        spaces.append(_canonical_signs(vectors))
    return spaces


def fit_huber_factor(
    X, ranks, *, threshold=None, max_iter=200, tol=1e-8, center=False, initial=None
):
    """Estimate factors by Huber loss on each matrix's residual Frobenius norm.

    Minimizes the mean matrixwise Huber objective of He et al., equation (4.1).
    ``ranks`` must contain two explicit positive ranks. ``threshold=None`` fixes
    the threshold at the median initial residual norm, following the paper's
    simulation rule; a zero median requires an explicit positive threshold.
    An explicit threshold is in original Frobenius-norm units, not per-entry
    standard deviations. Scaling data requires scaling the threshold equally.

    Initialization is alpha=0 PCA unless ``initial=(row, column)`` supplies
    full-rank loading bases. Each sweep recomputes matrix weights and proposes
    the simultaneous weighted projection of Algorithm 2. If actual loss rises,
    sequential weighted updates replace that proposal to ensure descent.
    This safeguard is an implementation extension. Convergence requires both
    relative loss change <= ``tol`` and projector change <= ``sqrt(tol)``.

    Returns ``HuberFactorResult`` with orthonormal loadings. This is not the
    entrywise iterative Huber regression estimator or Algorithm 3 rank selection.
    With ``center=True``, a fixed ordinary sample mean is removed; that location
    estimate is not robust. Prefer a justified known offset for contaminated data.

    Reference: https://arxiv.org/html/2112.04186v3, Sections 4.1 and 5.1.
    """
    X, work, mean, scale = _prepare(X, center)
    ranks = ranks_tuple(ranks, X.shape[1:])
    if ranks is None:
        raise ValueError("ranks must contain two explicit positive integers")
    max_iter = positive_int(max_iter, "max_iter")
    tol = finite_scalar(tol, "tol", minimum=0)
    if tol <= 0:
        raise ValueError("tol must be positive")
    loadings = _initial_spaces(initial, work, ranks)
    norms = _residual_norms(work, loadings)
    # Any orthogonal projection residual is bounded by the observation norm.
    # Capping the normalized threshold at this bound preserves the quadratic
    # regime while preventing overflow when explicit threshold / scale is huge.
    maximum_norm = float(np.sqrt(np.einsum("tij,tij->t", work, work)).max())
    threshold_cap = max(maximum_norm * (1 + 1e-12), np.finfo(float).tiny)
    if threshold is None:
        normalized_threshold = float(np.median(norms))
        if normalized_threshold <= np.finfo(float).eps * max(maximum_norm, 1.0):
            raise ValueError(
                "initial median residual is numerically zero; set a positive threshold"
            )
        threshold = normalized_threshold * scale
        if not np.isfinite(threshold):
            raise ValueError(
                "automatic threshold exceeds floating-point range; rescale data"
            )
    else:
        threshold = finite_scalar(threshold, "threshold", minimum=0)
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        normalized_threshold = min(threshold / scale, threshold_cap)
        if normalized_threshold == 0:
            raise ValueError("threshold is too small relative to the data scale")
    weights, objective = _weights_and_loss(norms, normalized_threshold)
    history = [objective]
    safeguards = 0
    converged = False
    reason = "max_iter"
    iteration = 0
    for iteration in range(1, max_iter + 1):
        row_moment = _weighted_moment(work, loadings[1], weights)
        column_moment = _weighted_moment(work.transpose(0, 2, 1), loadings[0], weights)
        row = _eigenspace(row_moment, ranks[0])[0]
        column = _eigenspace(column_moment, ranks[1])[0]
        candidate = [row, column]
        next_norms = _residual_norms(work, candidate)
        next_weights, next_objective = _weights_and_loss(
            next_norms, normalized_threshold
        )
        if next_objective > objective:
            safeguards += 1
            # Frozen IRLS weights define a quadratic majorizer. Updating the
            # column space using the new row space decreases that majorizer.
            column_moment = _weighted_moment(work.transpose(0, 2, 1), row, weights)
            candidate[1] = _eigenspace(column_moment, ranks[1])[0]
            next_norms = _residual_norms(work, candidate)
            next_weights, next_objective = _weights_and_loss(
                next_norms, normalized_threshold
            )
        distance = _space_distance(loadings, candidate)
        if next_objective > objective:
            # Never publish an increasing history by clipping its values.
            roundoff = 64 * np.finfo(float).eps * max(objective, np.finfo(float).tiny)
            converged = next_objective - objective <= roundoff and distance <= np.sqrt(
                tol
            )
            reason = "numerical_tolerance" if converged else "descent_stalled"
            iteration -= 1
            break
        improvement = objective - next_objective
        relative = improvement / max(objective, np.finfo(float).tiny)
        loadings, weights, objective = candidate, next_weights, next_objective
        history.append(objective)
        if relative <= tol and distance <= np.sqrt(tol):
            converged = True
            reason = "converged"
            break
    # Spectra describe final robust weighted projected moments in normalized units.
    moments = (
        _weighted_moment(work, loadings[1], weights),
        _weighted_moment(work.transpose(0, 2, 1), loadings[0], weights),
    )
    spectra = [_eigenspace(moment, rank)[1] for moment, rank in zip(moments, ranks)]
    result = _result(
        X, loadings, mean, spectra, "matrix-huber", scale, iteration, converged
    )
    return HuberFactorResult(
        **result.__dict__,
        threshold=threshold,
        weights=weights,
        objective_history=np.asarray(history),
        safeguard_steps=safeguards,
        stopping_reason=reason,
    )
