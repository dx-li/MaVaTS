"""Matrix Kendall's tau and robust loading-space estimation.

References
----------
He, Wang, Yu, Zhou and Zhou (2025), A New Non-Parametric Kendall's Tau for
Matrix-Valued Elliptical Observations, *Bernoulli* 31, 3331--3355.
Algorithm source: https://arxiv.org/abs/2207.09633, equations (2.3)--(2.4).
"""

import warnings

import numpy as np

from ._validation import as_series, positive_int, random_generator
from .factors import _eigenspace, _factor_ranks, _result


def matrix_kendall(X, *, max_pairs=None, batch_size=256, random_state=None):
    """Return row and column sample matrix Kendall matrices.

    Average ``D @ D.T / ||D||_F**2`` and ``D.T @ D / ||D||_F**2`` over
    unordered temporal pairs, D = X[t] - X[s]. Exact evaluation is quadratic
    in sample count but batch memory is linear in ``batch_size``. Duplicate
    observations contribute zero; all-identical observations are rejected.

    ``max_pairs`` opts into a Monte Carlo approximation: that many independent
    uniform unordered pairs are sampled *with replacement*. This is unbiased
    for the exact pair average, but it is not the paper's exact U-statistic.
    ``random_state`` controls only this approximation.

    References
    ----------
    He, Wang, Yu, Zhou and Zhou (2025), A New Non-Parametric Kendall's Tau for
    Matrix-Valued Elliptical Observations, equations (2.3)--(2.4).
    https://arxiv.org/abs/2207.09633
    Earlier manuscript title: Matrix Kendall's Tau in High-Dimensions:
    A Robust Statistic for Matrix Factor Model.
    """
    X = as_series(X)
    batch_size = positive_int(batch_size, "batch_size")
    t, m, n = X.shape
    total_pairs = t * (t - 1) // 2
    if max_pairs is not None:
        max_pairs = positive_int(max_pairs, "max_pairs")
    # Scaling before subtraction prevents overflow on opposite huge values.
    scale = max(float(np.max(np.abs(X))), np.finfo(float).tiny)
    work = X / scale
    row, col = np.zeros((m, m)), np.zeros((n, n))
    nonzero, pairs = 0, 0

    def accumulate(differences):
        nonlocal row, col, nonzero, pairs
        pairs += len(differences)
        pair_scale = np.max(np.abs(differences), axis=(1, 2))
        valid = pair_scale > 0
        nonzero += int(valid.sum())
        differences = differences[valid] / pair_scale[valid, None, None]
        if not len(differences):
            return
        differences /= np.sqrt(np.sum(differences**2, axis=(1, 2)))[:, None, None]
        row += np.einsum("tij,tkj->ik", differences, differences, optimize=True)
        col += np.einsum("tij,tik->jk", differences, differences, optimize=True)

    if max_pairs is not None and max_pairs < total_pairs:
        rng = random_generator(random_state)
        for start in range(0, max_pairs, batch_size):
            count = min(batch_size, max_pairs - start)
            first = rng.integers(t, size=count)
            second = rng.integers(t - 1, size=count)
            second += second >= first
            accumulate(work[first] - work[second])
    else:
        for first in range(t - 1):
            for start in range(first + 1, t, batch_size):
                accumulate(work[first] - work[start : start + batch_size])
    if not nonzero:
        raise ValueError(
            "no distinct sampled observations; Kendall loading spaces are undefined"
        )
    if nonzero < pairs:
        warnings.warn(
            "duplicate observations contribute zero to matrix Kendall's tau",
            RuntimeWarning,
            stacklevel=2,
        )
    return row / pairs, col / pairs


def fit_matrix_kendall(
    X, ranks=None, *, max_pairs=None, batch_size=256, random_state=None
):
    """Fit the matrix robust two-step (MRTS) factor estimator.

    Leading eigenvectors of row/column matrix Kendall's tau give robust
    loading spaces, then least-squares projection gives factor scores. Modern
    MaVaTS uses orthonormal loadings; this rescales scores relative to the paper
    but leaves signal estimates identical. The factor scores themselves are
    not robust against individual corrupted observations. No temporal mean is
    subtracted. The paper's guarantees require its matrix elliptical model;
    arbitrary contamination or serial dependence need separate justification.

    ``ranks=None`` uses eigenvalue ratios with a relative machine-precision
    floor and a half-dimension search bound, not the paper's optional tuned
    statistical ridge. Explicit ranks are recommended in comparative studies.

    References
    ----------
    He, Wang, Yu, Zhou and Zhou (2025), A New Non-Parametric Kendall's Tau for
    Matrix-Valued Elliptical Observations. https://arxiv.org/abs/2207.09633
    The implementation's optional sampled-pair approximation is described in
    ``matrix_kendall``; it is not the exact U-statistic used in the theory.
    """
    X = as_series(X)
    ranks = _factor_ranks(ranks, X.shape[1:])
    moments = matrix_kendall(
        X, max_pairs=max_pairs, batch_size=batch_size, random_state=random_state
    )
    pairs = [_eigenspace(moment, rank) for moment, rank in zip(moments, ranks)]
    return _result(
        X,
        [pair[0] for pair in pairs],
        np.zeros(X.shape[1:]),
        [pair[1] for pair in pairs],
        "matrix-kendall",
        1.0,
    )
