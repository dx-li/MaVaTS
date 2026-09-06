"""Bilinear segmentation of matrix time series by lagged cross-covariances.

The transformation follows Han, Chen, Zhang and Yao, *Simultaneous
Decorrelation of Matrix Time Series*, equations (5)--(10),
https://arxiv.org/abs/2103.09411. No components are discarded. Graph edges
are selected by an explicit correlation threshold or the ratio heuristic
(11), with its search bound resolved from the authors' journal supplement;
see ``docs/decorrelation-notes.md`` for scope and conventions.

References
----------
Han, Y., Chen, R., Zhang, C.-H. and Yao, Q. (2024).
Simultaneous Decorrelation of Matrix Time Series. JASA, 119, 957-969.
https://doi.org/10.1080/01621459.2022.2151448
Open manuscript: https://arxiv.org/abs/2103.09411
Equations (5)-(11); the ratio search bound follows the authors'
journal supplement, https://doi.org/10.6084/m9.figshare.21641763.v2.
The fixed correlation-threshold default is an uncalibrated convenience.
"""

from dataclasses import dataclass

import numpy as np

from ._validation import as_series, finite_scalar, positive_int
from .factors import _canonical_signs


def _marginal_roots(work, rank_tol, name):
    """Roots of mean(X.T X)/rows, computed without a covariance inverse."""
    design = work.reshape(-1, work.shape[-1])
    _, singular, right = np.linalg.svd(design, full_matrices=False)
    if (
        len(singular) != work.shape[-1]
        or singular[0] == 0
        or singular[-1] <= rank_tol * singular[0]
    ):
        raise ValueError(f"{name} marginal covariance must have full numerical rank")
    root_values = singular / np.sqrt(len(design))
    root = (right.T * root_values) @ right
    inverse_root = (right.T / root_values) @ right
    return root, inverse_root


def _lag_moment(partial, lags):
    """Equation (7), streaming the cross-row products of X Sigma^-1/2."""
    n_time, other_dim, dimension = partial.shape
    moment = np.zeros((dimension, dimension))
    for lag in range(lags + 1):
        future = partial[lag:]
        past = partial[: n_time - lag]
        for index in range(other_dim):
            # cross[j] = V_{lag,index,j}/other_dim. Keeping only one index
            # at a time avoids allocating the full (p*q)-squared covariance.
            cross = (
                (future[:, index].T @ past.reshape(len(past), -1))
                .reshape(dimension, other_dim, dimension)
                .transpose(1, 0, 2)
            )
            cross /= len(past) * other_dim
            moment += np.einsum("jab,jcb->ac", cross, cross)
            if lag:
                # V_{-lag,i,j} = V_{lag,j,i}.T; include every negative lag.
                moment += np.einsum("jba,jbc->ac", cross, cross)
    return (moment + moment.T) * 0.5


def _maximum_correlations(partial, lags):
    """Equation (10), with full-series variances and T-|lag| covariances."""
    scale = np.max(np.abs(partial), axis=0)
    standardized = partial / np.where(scale == 0, 1, scale)
    deviations = np.sqrt(np.mean(standardized**2, axis=0))
    standardized /= np.where(deviations == 0, 1, deviations)
    n_time, _, dimension = partial.shape
    correlations = np.zeros((dimension, dimension))
    for lag in range(lags + 1):
        future = standardized[lag:]
        past = standardized[: n_time - lag]
        for index in range(dimension):
            cross = future[:, :, index].T @ past.reshape(len(past), -1)
            cross = cross.reshape(partial.shape[1], partial.shape[1], dimension)
            correlations[index] = np.maximum(
                correlations[index], np.max(np.abs(cross), axis=(0, 1)) / len(past)
            )
    # Transposing the complete covariance gives the negative-lag contribution.
    correlations = np.maximum(correlations, correlations.T)
    np.fill_diagonal(correlations, 0)
    return correlations


def _connected_groups(correlations, threshold):
    pending = set(range(len(correlations)))
    groups = []
    while pending:
        first = min(pending)
        pending.remove(first)
        group, queue = [first], [first]
        while queue:
            current = queue.pop()
            neighbors = sorted(
                j for j in pending if correlations[current, j] > threshold
            )
            pending.difference_update(neighbors)
            group.extend(neighbors)
            queue.extend(neighbors)
        groups.append(tuple(sorted(group)))
    return tuple(groups)


def _ratio_groups(correlations, delta, max_edges):
    """Equation (11), restricted to the authors' half-pair search by default.

    Independently implemented from the ratio criterion. The search endpoint
    is documented by real data analysis.R in the authors' supplement,
    https://doi.org/10.6084/m9.figshare.21641763.v2. No reference code is used.
    """
    dimension = len(correlations)
    if dimension == 1:
        if max_edges is not None:
            raise ValueError("ratio_max_edges must be None for a singleton mode")
        return ((0,),), 0, 0
    first, second = np.triu_indices(dimension, k=1)
    values = correlations[first, second]
    n_pairs = len(values)
    if n_pairs < 2:
        raise ValueError(
            "ratio grouping needs at least three coordinates in each nontrivial "
            "mode; use threshold grouping for a mode of dimension two"
        )
    max_edges = n_pairs // 2 if max_edges is None else max_edges
    if not 1 <= max_edges < n_pairs:
        raise ValueError(
            "ratio_max_edges must lie between 1 and number of pairs minus 1"
        )
    order = np.argsort(-values, kind="stable")
    ordered = values[order]
    # Rescaling protects addition of a large smoothing delta, and logarithms
    # preserve ratio order without overflow for widely separated correlations.
    scale = max(ordered[0], delta)
    smoothed = ordered / scale + delta / scale if scale > 0 else np.zeros_like(ordered)
    numerator, denominator = smoothed[:max_edges], smoothed[1 : max_edges + 1]
    scores = np.zeros(max_edges)
    positive = denominator > 0
    scores[positive] = np.log(numerator[positive]) - np.log(denominator[positive])
    scores[(denominator == 0) & (numerator > 0)] = np.inf
    # With delta=0, the value of 0/0 is the delta->0+ limit, one. This
    # continuous-ridge convention is explicit, including the all-zero case.
    n_edges = int(np.argmax(scores)) + 1
    adjacency = np.zeros_like(correlations)
    chosen = order[:n_edges]
    adjacency[first[chosen], second[chosen]] = 1
    adjacency[second[chosen], first[chosen]] = 1
    return _connected_groups(adjacency, 0.5), n_edges, max_edges


def _ratio_bounds(value):
    if value is None:
        return (None, None)
    if np.ndim(value) == 0:
        values = (value, value)
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValueError("ratio_max_edges must be an integer or pair") from exc
        if len(values) != 2:
            raise ValueError("ratio_max_edges must be an integer or pair")
    return tuple(
        None if bound is None else positive_int(bound, "ratio_max_edges")
        for bound in values
    )


def _thresholds(value):
    if np.ndim(value) == 0:
        values = (value, value)
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValueError("correlation_threshold must be a scalar or pair") from exc
        if len(values) != 2:
            raise ValueError("correlation_threshold must be a scalar or pair")
    return tuple(finite_scalar(v, "correlation_threshold", minimum=0) for v in values)


def _contiguous_groups(groups):
    offset = 0
    result = []
    for group in groups:
        result.append(tuple(range(offset, offset + len(group))))
        offset += len(group)
    return tuple(result)


@dataclass
class MatrixDecorrelationResult:
    """Invertible matrix segmentation, with contiguous groups in latent space.

    ``series`` has the same shape as the training data. Its rectangular blocks
    are indexed by ``row_groups`` and ``column_groups``. Estimated separation
    is approximate and is not an independence test or an all-lag guarantee.
    ``row_rotation`` and ``column_rotation`` are the reordered orthogonal
    eigenvectors of equation (7); the corresponding spectra are reordered to
    match them. ``row_moment`` and ``column_moment`` remain in the original
    marginally whitened coordinates. ``*_correlations`` follow the new order.

    The symmetric ``*_root_scaled`` and ``*_whitener_scaled`` matrices refer
    to data divided by ``data_scale``. This representation avoids squaring the
    units of the data. Physical-unit transforms and mixing matrices are exposed
    as properties. ``row_transform @ (X-mean) @ column_transform`` equals
    ``series``; ``row_mixing @ series @ column_mixing.T + mean`` inverts it.
    Under positive global scaling by c, latent values scale by 1/c, as required
    by the two marginal normalizations in equation (8).

    References
    ----------
    Han, Y., Chen, R., Zhang, C.-H. and Yao, Q. (2024).
    Simultaneous Decorrelation of Matrix Time Series. JASA, 119, 957-969.
    https://doi.org/10.1080/01621459.2022.2151448
    Open manuscript: https://arxiv.org/abs/2103.09411
    Equations (5)-(11); the ratio search bound follows the authors'
    journal supplement, https://doi.org/10.6084/m9.figshare.21641763.v2.
    The fixed correlation-threshold default is an uncalibrated convenience.
    """

    series: np.ndarray
    mean: np.ndarray
    data_scale: float
    row_rotation: np.ndarray
    column_rotation: np.ndarray
    row_root_scaled: np.ndarray
    column_root_scaled: np.ndarray
    row_whitener_scaled: np.ndarray
    column_whitener_scaled: np.ndarray
    row_moment: np.ndarray
    column_moment: np.ndarray
    row_eigenvalues: np.ndarray
    column_eigenvalues: np.ndarray
    row_correlations: np.ndarray
    column_correlations: np.ndarray
    row_groups: tuple
    column_groups: tuple
    row_n_edges: int
    column_n_edges: int
    lags: int
    correlation_lags: int
    correlation_threshold: tuple | None
    grouping: str
    ratio_delta: float
    ratio_max_edges: tuple
    center: bool
    rank_tol: float
    _mean_scaled: np.ndarray
    method: str = "matrix-decorrelation-threshold"

    @property
    def row_transform(self):
        """Left transformation B_*.T Sigma_row^-1/2 in original data units."""
        return (self.row_rotation.T @ self.row_whitener_scaled) / self.data_scale

    @property
    def column_transform(self):
        """Right transformation Sigma_column^-1/2 A_* in original units."""
        return (self.column_whitener_scaled @ self.column_rotation) / self.data_scale

    @property
    def row_mixing(self):
        """B = Sigma_row^1/2 B_* in original units."""
        return (self.row_root_scaled @ self.row_rotation) * self.data_scale

    @property
    def column_mixing(self):
        """A = Sigma_column^1/2 A_* in original units."""
        return (self.column_root_scaled @ self.column_rotation) * self.data_scale

    def transform(self, X):
        """Transform new data with the fitted mean and both fitted marginals.

        References
        ----------
        Han, Y., Chen, R., Zhang, C.-H. and Yao, Q. (2024).
        Simultaneous Decorrelation of Matrix Time Series. JASA, 119, 957-969.
        https://doi.org/10.1080/01621459.2022.2151448
        Open manuscript: https://arxiv.org/abs/2103.09411
        Equations (5)-(11); the ratio search bound follows the authors'
        journal supplement, https://doi.org/10.6084/m9.figshare.21641763.v2.
        The fixed correlation-threshold default is an uncalibrated convenience.
        """
        X = as_series(X, min_samples=1)
        if X.shape[1:] != self.mean.shape:
            raise ValueError("observation dimensions must match the fitted model")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            work = X / self.data_scale - self._mean_scaled
            output = (
                self.row_rotation.T
                @ self.row_whitener_scaled
                @ work
                @ self.column_whitener_scaled
                @ self.column_rotation
            ) / self.data_scale
        if not np.isfinite(output).all():
            raise ValueError(
                "transformed values exceed floating-point range; rescale data"
            )
        return output

    def inverse_transform(self, series):
        """Invert complete latent matrices, adding the training mean.

        References
        ----------
        Han, Y., Chen, R., Zhang, C.-H. and Yao, Q. (2024).
        Simultaneous Decorrelation of Matrix Time Series. JASA, 119, 957-969.
        https://doi.org/10.1080/01621459.2022.2151448
        Open manuscript: https://arxiv.org/abs/2103.09411
        Equations (5)-(11); the ratio search bound follows the authors'
        journal supplement, https://doi.org/10.6084/m9.figshare.21641763.v2.
        The fixed correlation-threshold default is an uncalibrated convenience.
        """
        series = as_series(series, min_samples=1)
        if series.shape[1:] != self.mean.shape:
            raise ValueError("latent dimensions must match the fitted model")
        with np.errstate(over="ignore", invalid="ignore"):
            normalized = (
                self.row_root_scaled
                @ self.row_rotation
                @ (series * self.data_scale)
                @ self.column_rotation.T
                @ self.column_root_scaled
            )
            output = (normalized + self._mean_scaled) * self.data_scale
        if not np.isfinite(output).all():
            raise ValueError("reconstructed values exceed floating-point range")
        return output

    def blocks(self, X=None):
        """Return nested row/column tuples of latent block time series.

        With no argument, return the training blocks. Otherwise transform X
        with the fitted parameters and then extract its blocks.

        References
        ----------
        Han, Y., Chen, R., Zhang, C.-H. and Yao, Q. (2024).
        Simultaneous Decorrelation of Matrix Time Series. JASA, 119, 957-969.
        https://doi.org/10.1080/01621459.2022.2151448
        Open manuscript: https://arxiv.org/abs/2103.09411
        Equations (5)-(11); the ratio search bound follows the authors'
        journal supplement, https://doi.org/10.6084/m9.figshare.21641763.v2.
        The fixed correlation-threshold default is an uncalibrated convenience.
        """
        series = self.series if X is None else self.transform(X)
        return tuple(
            tuple(series[:, rows, :][:, :, columns] for columns in self.column_groups)
            for rows in self.row_groups
        )

    def inverse_blocks(self, blocks):
        """Reassemble and invert nested block predictions from ``blocks()``.

        References
        ----------
        Han, Y., Chen, R., Zhang, C.-H. and Yao, Q. (2024).
        Simultaneous Decorrelation of Matrix Time Series. JASA, 119, 957-969.
        https://doi.org/10.1080/01621459.2022.2151448
        Open manuscript: https://arxiv.org/abs/2103.09411
        Equations (5)-(11); the ratio search bound follows the authors'
        journal supplement, https://doi.org/10.6084/m9.figshare.21641763.v2.
        The fixed correlation-threshold default is an uncalibrated convenience.
        """
        try:
            block_rows = tuple(tuple(row) for row in blocks)
        except TypeError as exc:
            raise ValueError("blocks must be nested row/column sequences") from exc
        if len(block_rows) != len(self.row_groups) or any(
            len(row) != len(self.column_groups) for row in block_rows
        ):
            raise ValueError("block partition must match the fitted groups")
        assembled_rows = []
        n_time = None
        for row, rows in zip(block_rows, self.row_groups):
            arrays = []
            for block, columns in zip(row, self.column_groups):
                array = as_series(block, min_samples=1)
                if array.shape[1:] != (len(rows), len(columns)):
                    raise ValueError("block dimensions must match the fitted groups")
                if n_time is None:
                    n_time = len(array)
                if len(array) != n_time:
                    raise ValueError(
                        "all blocks must have the same number of observations"
                    )
                arrays.append(array)
            assembled_rows.append(np.concatenate(arrays, axis=2))
        return self.inverse_transform(np.concatenate(assembled_rows, axis=1))


def fit_matrix_decorrelation(
    X,
    *,
    lags=1,
    correlation_lags=10,
    correlation_threshold=0.2,
    grouping="threshold",
    ratio_delta=0.0,
    ratio_max_edges=None,
    center=True,
    rank_tol=None,
):
    """Fit Han et al.'s bilinear transform and group dependent coordinates.

    Parameters
    ----------
    X : array_like, shape (time, rows, columns)
        Finite real, weakly stationary observations. Both empirical marginal
        covariance matrices must be numerically positive definite.
    lags : int, default 1
        Include all signed lags from -lags through +lags, including zero, in
        the moment matrices. Must be positive and smaller than sample size.
    correlation_lags : int, default 10
        Maximum signed lag for grouping, also positive and below sample size.
    correlation_threshold : float or pair, default 0.2
        Connect two rows/columns when their equation (10) maximum absolute
        cross-correlation exceeds this value. A pair specifies separate row
        and column thresholds. The default is a convenience, not a calibrated
        significance level or consistent automatic tuning rule. Inspect the
        returned correlation matrices and assess sensitivity to the threshold.
        All nonnegative finite thresholds are allowed: equation (10) can
        exceed one in small samples because its denominators use all times.
        Used only with ``grouping='threshold'``.
    grouping : {"threshold", "ratio"}, default "threshold"
        ``ratio`` selects the first maximizer of consecutive descending
        correlation ratios (11), then connects that many strongest pairs.
        The default search ends at floor(number_of_pairs/2), as in the authors'
        journal supplement. It always selects at least one pair for each mode
        of dimension at least three, so cannot select all singleton groups.
        A scalar mode needs no selection; dimension two requires threshold
        grouping because it has no adjacent pair of ordered correlations.
    ratio_delta : float, default 0
        Nonnegative smoothing constant added to both correlations in each
        ratio. Zero matches the reference script; positive values implement
        the ridge in (11). At zero, 0/0 uses its continuous-ridge limit one,
        and positive/0 is infinite. Thus even all-zero scores select one edge.
        Equal edge strengths are ordered lexicographically by coordinate pair.
    ratio_max_edges : int, pair, or None
        Optional upper bound on the ratio search, independently for rows and
        columns. Bounds must be below the number of pairs. None uses the
        reference half-pair bound. A singleton mode requires a None bound.
    center : bool, default True
        Subtract and retain the training mean. False requires zero-mean data.
    rank_tol : float, optional
        Relative singular-value cutoff for each marginal data matricization.
        Defaults to sqrt(machine epsilon). No ridge or pseudoinverse is used.
        Smaller values permit less stable marginal whitening.

    Notes
    -----
    Each side's eigenanalysis uses *all* cross-row (respectively cross-column)
    covariance matrices, not only the summed matrix autocovariance. No PCA
    truncation is performed. Grouping also uses separate one-sided transforms
    (9), not correlations of the fully transformed matrix. Connected graph
    components define the row and column groups, contiguous after reordering.
    Zero-variance scalar coordinates contribute zero to grouping correlations.

    Repeated population moment eigenvalues across distinct blocks prevent
    their identification (paper Remark 3). Finite samples, weak dependence, or
    an unsuitable threshold can likewise merge or split true blocks. The
    optional VAR prewhitening, recursive irregular segmentation and alternative
    spectral functions are not implemented. The authors' reference script
    uses nonnegative moment lags and trimmed variance estimates; here both
    grouping modes use the signed-lag and full-variance equations (7) and (10).
    Covariance computation streams one opposite-mode index at a time, using
    O(max(rows*columns**2, columns*rows**2)) temporary storage in addition to
    the data, with no (rows*columns)-square eigenproblem.

    References
    ----------
    Han, Y., Chen, R., Zhang, C.-H. and Yao, Q. (2024).
    Simultaneous Decorrelation of Matrix Time Series. JASA, 119, 957-969.
    https://doi.org/10.1080/01621459.2022.2151448
    Open manuscript: https://arxiv.org/abs/2103.09411
    Equations (5)-(11); the ratio search bound follows the authors'
    journal supplement, https://doi.org/10.6084/m9.figshare.21641763.v2.
    The fixed correlation-threshold default is an uncalibrated convenience.
    """
    X = as_series(X)
    lags = positive_int(lags, "lags")
    correlation_lags = positive_int(correlation_lags, "correlation_lags")
    if max(lags, correlation_lags) >= len(X):
        raise ValueError("lags and correlation_lags must be smaller than sample size")
    thresholds = _thresholds(correlation_threshold)
    if not isinstance(grouping, str) or grouping not in ("threshold", "ratio"):
        raise ValueError("grouping must be 'threshold' or 'ratio'")
    ratio_delta = finite_scalar(ratio_delta, "ratio_delta", minimum=0)
    ratio_bounds = _ratio_bounds(ratio_max_edges)
    if not isinstance(center, (bool, np.bool_)):
        raise ValueError("center must be boolean")
    rank_tol = (
        np.sqrt(np.finfo(float).eps)
        if rank_tol is None
        else finite_scalar(rank_tol, "rank_tol", minimum=0)
    )
    if not 0 < rank_tol < 1:
        raise ValueError("rank_tol must lie strictly between zero and one")
    data_scale = float(np.max(np.abs(X)))
    if data_scale == 0:
        raise ValueError("marginal covariances must have full numerical rank")
    normalized = X / data_scale
    mean_scaled = normalized.mean(axis=0) if center else np.zeros(X.shape[1:])
    work = normalized - mean_scaled
    roots, whiteners, moments, rotations, spectra, correlations, groups = (
        [] for _ in range(7)
    )
    edge_counts, search_bounds = [], []
    for oriented, name, threshold, bound in zip(
        (work.transpose(0, 2, 1), work), ("row", "column"), thresholds, ratio_bounds
    ):
        root, whitener = _marginal_roots(oriented, rank_tol, name)
        partial = oriented @ whitener
        moment = _lag_moment(partial, lags)
        values, vectors = np.linalg.eigh(moment)
        values = np.maximum(values[::-1], 0)
        vectors = _canonical_signs(vectors[:, ::-1])
        correlation = _maximum_correlations(partial @ vectors, correlation_lags)
        if grouping == "threshold":
            original_groups = _connected_groups(correlation, threshold)
            edge_count = int(np.count_nonzero(np.triu(correlation > threshold, k=1)))
            search_bound = None
        else:
            original_groups, edge_count, search_bound = _ratio_groups(
                correlation, ratio_delta, bound
            )
        edge_counts.append(edge_count)
        search_bounds.append(search_bound)
        permutation = np.array([index for group in original_groups for index in group])
        roots.append(root)
        whiteners.append(whitener)
        moments.append(moment)
        rotations.append(vectors[:, permutation])
        spectra.append(values[permutation])
        correlations.append(correlation[np.ix_(permutation, permutation)])
        groups.append(_contiguous_groups(original_groups))
    result = MatrixDecorrelationResult(
        series=np.empty((0, *X.shape[1:])),
        mean=mean_scaled * data_scale,
        data_scale=data_scale,
        row_rotation=rotations[0],
        column_rotation=rotations[1],
        row_root_scaled=roots[0],
        column_root_scaled=roots[1],
        row_whitener_scaled=whiteners[0],
        column_whitener_scaled=whiteners[1],
        row_moment=moments[0],
        column_moment=moments[1],
        row_eigenvalues=spectra[0],
        column_eigenvalues=spectra[1],
        row_correlations=correlations[0],
        column_correlations=correlations[1],
        row_groups=groups[0],
        column_groups=groups[1],
        row_n_edges=edge_counts[0],
        column_n_edges=edge_counts[1],
        lags=lags,
        correlation_lags=correlation_lags,
        correlation_threshold=thresholds if grouping == "threshold" else None,
        grouping=grouping,
        ratio_delta=ratio_delta,
        ratio_max_edges=tuple(search_bounds),
        center=bool(center),
        rank_tol=rank_tol,
        _mean_scaled=mean_scaled,
        method=f"matrix-decorrelation-{grouping}",
    )
    result.series = result.transform(X)
    return result
