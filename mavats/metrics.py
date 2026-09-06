"""Scale- and rotation-aware errors for matrix and tensor methods.

References are attached to each metric. These are evaluation utilities, not
new statistical estimators; their explicit formulas define the conventions.
"""

import numpy as np


def _pair(actual, predicted):
    if np.iscomplexobj(actual) or np.iscomplexobj(predicted):
        raise ValueError("inputs must be real")
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(
        predicted, dtype=float
    )
    if actual.shape != predicted.shape or not actual.size:
        raise ValueError("inputs must have the same nonempty shape")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("inputs must be finite")
    return actual, predicted


def mean_squared_error(actual, predicted):
    """Mean squared error per scalar entry (not per time point).

    Scaling avoids overflow in intermediate squaring and reduction. Returns
    infinity only when the resulting MSE exceeds floating-point range.

    References
    ----------
    Hyndman and Koehler (2006), *Another Look at Measures of Forecast Accuracy*,
    https://doi.org/10.1016/j.ijforecast.2006.03.001, reviews squared forecast
    errors. Here all supplied scalar entries are pooled; the utility does not
    implement the paper's distinct MASE proposal.
    """
    actual, predicted = _pair(actual, predicted)
    with np.errstate(over="ignore"):
        difference = actual - predicted
    if np.isfinite(difference).all():
        scale = np.max(np.abs(difference))
        if scale == 0:
            return 0.0
        difference = difference / scale
    else:
        scale = max(np.max(np.abs(actual)), np.max(np.abs(predicted)))
        difference = actual / scale - predicted / scale
    # Apply the normalized mean before the second scale factor, so a
    # representable MSE survives even if the unnormalized sum would overflow.
    with np.errstate(over="ignore", under="ignore"):
        return float((scale * np.mean(difference * difference)) * scale)


def relative_frobenius_error(actual, predicted):
    """Frobenius error divided by truth norm; inf for nonzero error at zero truth.

    References
    ----------
    Chen, Yang and Zhang (2022), *Factor Models for High-Dimensional Tensor
    Time Series*, https://doi.org/10.1080/01621459.2021.1912757, provides the
    tensor reconstruction setting. This is the conventional relative norm
    utility, with the zero-truth convention stated above, not a paper-specific
    estimator or a claim to reproduce all of that paper's evaluation losses.
    """
    actual, predicted = _pair(actual, predicted)
    scale = max(np.max(np.abs(actual)), np.max(np.abs(predicted)))
    if scale == 0:
        return 0.0
    with np.errstate(over="ignore"):
        difference = actual - predicted
    # Preserve cancellation accuracy with direct subtraction where possible.
    difference = (
        difference / scale
        if np.isfinite(difference).all()
        else actual / scale - predicted / scale
    )
    # hypot reduction also avoids squaring tiny normalized denominators.
    denominator = np.hypot.reduce((actual / scale).ravel())
    numerator = np.hypot.reduce(difference.ravel())
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    with np.errstate(over="ignore"):
        return float(numerator / denominator)


def subspace_distance(actual, estimated):
    """Normalized projector distance, invariant to loading scale and rotation.

    Returns ``||P_actual - P_estimated||_F / sqrt(r_actual + r_estimated)``.
    The result lies in [0, 1]; unequal-rank subspaces are allowed. Input bases
    must have full column rank. SVD orthogonalization handles non-unit loadings.

    References
    ----------
    Wang, Liu and Chen (2019), *Factor Models for Matrix-Valued
    High-Dimensional Time Series*, https://doi.org/10.1016/j.jeconom.2018.09.013,
    uses rotation-invariant loading-space comparisons. The displayed symmetric
    normalization is this library's convention; it agrees with the usual
    normalized projector loss at equal ranks and explicitly extends it to
    unequal ranks. It is not an inferential distance test.
    """
    bases = []
    for basis in (actual, estimated):
        if np.iscomplexobj(basis):
            raise ValueError("loadings must be real")
        basis = np.asarray(basis, dtype=float)
        if basis.ndim != 2 or not all(basis.shape) or not np.isfinite(basis).all():
            raise ValueError("loadings must be nonempty finite matrices")
        scales = np.max(np.abs(basis), axis=0)
        if np.any(scales == 0):
            raise ValueError("loading matrices must have full column rank")
        u, s, _ = np.linalg.svd(basis / scales, full_matrices=False)
        rank = np.count_nonzero(s > s[0] * max(basis.shape) * np.finfo(float).eps)
        if rank != basis.shape[1]:
            raise ValueError("loading matrices must have full column rank")
        bases.append(u)
    u, v = bases
    if len(u) != len(v):
        raise ValueError("loading matrices must have the same ambient dimension")
    total = u.shape[1] + v.shape[1]
    cross = u.T @ v
    distance_squared = (
        np.sum((v - u @ cross) ** 2) + np.sum((u - v @ cross.T) ** 2)
    ) / total
    return float(np.sqrt(np.clip(distance_squared, 0, 1)))
