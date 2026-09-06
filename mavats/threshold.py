"""Threshold matrix factors of Liu and Chen (2022).

This implements the *published* origin-regime cross moments, rather than the
four origin/destination partitions in the authors' earlier 2019 preprint.
Observations and the observed threshold variable are aligned at time ``t``.
"""

from dataclasses import dataclass

import numpy as np

from ._validation import as_series, finite_scalar, positive_int
from .factors import _canonical_signs, _factor_ranks, _project, eigenvalue_ratio


def _threshold_variable(z, n):
    raw = np.asarray(z)
    if raw.dtype.kind not in "iuf" or np.iscomplexobj(raw):
        raise ValueError("z must be a finite real numeric vector")
    z = np.asarray(raw, dtype=float)
    if z.shape != (n,) or not np.isfinite(z).all():
        raise ValueError("z must be a finite vector with one entry per observation")
    return z


def _regime_ranks(ranks, shape):
    if ranks is None:
        return ((None, None), (None, None))
    try:
        ranks = tuple(ranks)
    except TypeError as exc:
        raise ValueError("ranks must contain a (row, column) pair per regime") from exc
    if len(ranks) != 2:
        raise ValueError("ranks must contain a (row, column) pair per regime")
    return tuple(_factor_ranks(pair, shape) for pair in ranks)


def _moments(X, masks, max_lag):
    """Equations (9)-(10), denominator T, origin mask only.

    Work one source column at a time to avoid a (p*q)-squared cross moment.
    Temporal order is preserved: masking never compresses the time series.
    """
    n = len(X)
    result = []
    for mask in masks:
        mode_moments = []
        for work in (X, X.transpose(0, 2, 1)):
            p, q = work.shape[1:]
            moment = np.zeros((p, p))
            for lag in range(1, max_lag + 1):
                keep = mask[:-lag]
                future = work[lag:][keep].reshape((-1, p * q))
                past = work[:-lag][keep]
                for column in range(q):
                    omega = past[:, :, column].T @ future / n
                    moment += omega @ omega.T
            mode_moments.append((moment + moment.T) / 2)
        result.append(tuple(mode_moments))
    return tuple(result)


def _spaces(moments, ranks, n):
    loadings, spectra, complements = [], [], []
    for regime, pair in enumerate(moments):
        regime_q, regime_values, regime_b = [], [], []
        for mode, moment in enumerate(pair):
            values, vectors = np.linalg.eigh(moment)
            values = np.maximum(values[::-1], 0)
            vectors = vectors[:, ::-1]
            if values[0] <= 0:
                raise ValueError("regime has no nonzero lagged cross moment")
            rank = ranks[regime][mode]
            if rank is None:
                rank = eigenvalue_ratio(
                    values, max_rank=max(1, min(len(values), n) // 2)
                )
            if values[rank - 1] <= values[0] * np.finfo(float).eps * len(values):
                raise ValueError("requested rank exceeds numerical lag-moment rank")
            regime_q.append(_canonical_signs(vectors[:, :rank]))
            regime_values.append(values)
            regime_b.append(vectors[:, rank:])
        loadings.append(tuple(regime_q))
        spectra.append(tuple(regime_values))
        complements.append(tuple(regime_b))
    return tuple(loadings), tuple(spectra), tuple(complements)


def _score(moments, complements):
    score = 0.0
    for pair, bs in zip(moments, complements):
        for moment, b in zip(pair, bs):
            if b.shape[1]:
                projected = b.T @ moment @ b
                score += max(float(np.linalg.eigvalsh(projected)[-1]), 0.0)
    return score


def _stable_project(X, loadings, inverse=False):
    scale = max(float(np.max(np.abs(X))) if X.size else 1.0, np.finfo(float).tiny)
    with np.errstate(over="ignore", invalid="ignore"):
        projected = _project(X / scale, loadings, inverse=inverse) * scale
    if not np.isfinite(projected).all():
        raise ValueError("factor or signal values exceed the finite numeric range")
    return projected


@dataclass
class ThresholdFactorResult:
    """Two-regime loading spaces and an auditable threshold search.

    ``loadings[i]`` and ``eigenvalues[i]`` contain row and column quantities
    for regime ``i``. ``factors[i]`` contains only that regime's observations,
    in time order, allowing different ranks in each regime; ``indices[i]``
    maps them to the original time axis. ``signal`` and ``residuals`` retain
    the full input shape. Factors and signal are in original data units.

    ``candidate_scores`` is equation (11) applied to ``X / data_scale``;
    original-unit scores would be these values times ``data_scale**4``.
    ``reference_loadings`` are estimated once from the two extreme subsets,
    and are held fixed throughout the profile. No forecast law is fitted.
    """

    threshold: float
    loadings: tuple
    factors: tuple
    signal: np.ndarray
    residuals: np.ndarray
    regimes: np.ndarray
    eigenvalues: tuple
    candidates: np.ndarray
    candidate_scores: np.ndarray
    threshold_bounds: tuple | None
    reference_loadings: tuple | None
    data_scale: float
    max_lag: int
    method: str = "threshold-factor"

    @property
    def ranks(self):
        return tuple(tuple(q.shape[1] for q in pair) for pair in self.loadings)

    @property
    def indices(self):
        return tuple(np.flatnonzero(self.regimes == i) for i in range(2))

    def classify(self, z):
        """Assign observed ``z`` values; equality belongs to regime one (upper).

        References
        ----------
        Liu and Chen (2022), Identification and Estimation of Threshold
        Matrix-Variate Factor Models. https://doi.org/10.1111/sjos.12576
        This applies the fitted threshold; it does not predict future z.
        """
        raw = np.asarray(z)
        if raw.ndim != 1:
            raise ValueError("z must be a one-dimensional vector")
        return (_threshold_variable(z, len(raw)) >= self.threshold).astype(int)

    def transform(self, X, z):
        """Project aligned new observations into regime-specific core arrays.

        Returns ``(lower_regime_cores, upper_regime_cores)`` in each regime's
        temporal order. The caller supplies contemporaneously available ``z``;
        this method never estimates a threshold or refits loading spaces.

        References
        ----------
        Liu and Chen (2022), Identification and Estimation of Threshold
        Matrix-Variate Factor Models. https://doi.org/10.1111/sjos.12576
        """
        X = as_series(X, min_samples=1)
        shape = tuple(q.shape[0] for q in self.loadings[0])
        if X.shape[1:] != shape:
            raise ValueError("observation dimensions must match fitted loadings")
        regimes = self.classify(_threshold_variable(z, len(X)))
        return tuple(
            _stable_project(X[regimes == i], self.loadings[i]) for i in range(2)
        )

    def inverse_transform(self, factors, z):
        """Reconstruct regime-specific cores into the time order specified by z.

        References
        ----------
        Liu and Chen (2022), Identification and Estimation of Threshold
        Matrix-Variate Factor Models. https://doi.org/10.1111/sjos.12576
        This is fitted model reconstruction, not threshold-variable forecasting.
        """
        regimes = self.classify(z)
        try:
            factors = tuple(factors)
        except TypeError as exc:
            raise ValueError("factors must contain one core array per regime") from exc
        if len(factors) != 2:
            raise ValueError("factors must contain one core array per regime")
        shape = tuple(q.shape[0] for q in self.loadings[0])
        signal = np.empty((len(regimes), *shape))
        for i, core in enumerate(factors):
            core = as_series(core, min_samples=0)
            if core.shape != (int(np.sum(regimes == i)), *self.ranks[i]):
                raise ValueError("core shape must match regime count and fitted ranks")
            signal[regimes == i] = _stable_project(core, self.loadings[i], inverse=True)
        return signal

    def reconstruct(self, X, z):
        """Return fitted common components for aligned new observations.

        References
        ----------
        Liu and Chen (2022), Identification and Estimation of Threshold
        Matrix-Variate Factor Models. https://doi.org/10.1111/sjos.12576
        Contemporaneous projection uses observed X and z, not a forecast law.
        """
        return self.inverse_transform(self.transform(X, z), z)


def fit_threshold_factors(
    X,
    z,
    ranks=None,
    *,
    threshold=None,
    max_lag=1,
    trim=(0.1, 0.9),
    threshold_bounds=None,
    candidates=None,
):
    """Fit Liu and Chen's two-regime matrix-variate factor model (2022).

    Parameters
    ----------
    X : array_like, shape (T, p, q)
        Finite observations. The published estimator uses uncentered cross
        moments and assumes serially uncorrelated idiosyncratic noise, and
        sufficiently informative nonzero lagged factor cross moments.
    z : array_like, shape (T,)
        Observed threshold variable aligned with ``X[t]``. Regime zero means
        ``z[t] < threshold``; equality belongs to regime one. For a lagged
        threshold variable, align and truncate both arrays before calling.
        In forecasting applications z must be available at the forecast
        origin; the function cannot validate the provenance of supplied z.
    ranks : pair of (row_rank, column_rank), optional
        One rank pair per regime, e.g. ``((1, 2), (2, 1))``. Missing entries
        use the equation (12) eigenvalue ratio, searching to half the smaller
        of T and the mode dimension. In threshold search these ranks are
        selected from extreme subsets and fixed for the final fit.
    threshold : float, optional
        Known threshold. Otherwise estimate it by a trimmed profile search.
    max_lag : int, default 1
        Include all temporal lags from one through this value, less than T.
    trim : pair of float, default (0.1, 0.9)
        Quantiles defining the two extreme subsets and open search interval.
        The true threshold must lie inside this interval; this is an assumption
        of the method, not something the data can guarantee.
    threshold_bounds : pair of float, optional
        Explicit extreme cutoffs instead of quantiles. Uses ``z < lower`` and
        ``z >= upper``. Both must have lagged origins represented in the data.
    candidates : vector of float, optional
        Search these thresholds inside the open bounds. By default search
        every unique observed z inside the bounds, as in the paper. Exact
        score ties select the smallest candidate. A supplied grid is a
        deliberate approximation to the paper's complete observed-value search.

    Notes
    -----
    Implements equations (9)-(12) and Section 2.3 of the author manuscript
    https://par.nsf.gov/servlets/purl/10351724 corresponding to Liu and Chen,
    *Identification and estimation of threshold matrix-variate factor models*,
    Scandinavian Journal of Statistics 49, 1383-1417 (2022),
    https://doi.org/10.1111/sjos.12576.

    References
    ----------
    Liu and Chen (2022), Identification and Estimation of Threshold
    Matrix-Variate Factor Models. https://doi.org/10.1111/sjos.12576

    For every source regime, temporal lag and pair of matrix columns, form
    ``Omega = sum_t X[t,:,u] X[t+h,:,v].T I(z[t] in regime) / T``.
    The row moment sums ``Omega @ Omega.T``; the column moment repeats this
    calculation on transposes. Unlike the earlier 2019 preprint, the *future*
    threshold value does not partition these moments. For unknown threshold,
    estimate complementary loading spaces from the two extreme subsets, then
    minimize the sum of the four spectral norms ``||B.T M(r) B||_2``.

    This API fits one threshold and two regimes, including unequal ranks. The
    paper's multi-threshold and threshold-variable-selection extensions are
    not implemented. No confidence interval or statistical threshold-existence
    test is implied by the profile. Weak dynamics or nested regime spaces can
    make the threshold poorly identified. Storage per moment is O(p*q*max(p,q)),
    with an exhaustive search costing one moment evaluation per candidate.
    """
    X = as_series(X, min_samples=3)
    z = _threshold_variable(z, len(X))
    max_lag = positive_int(max_lag, "max_lag")
    if max_lag >= len(X):
        raise ValueError("max_lag must be smaller than the observation count")
    ranks = _regime_ranks(ranks, X.shape[1:])
    scale = max(float(np.max(np.abs(X))), np.finfo(float).tiny)
    work = X / scale
    reference = None
    bounds = None
    grid = np.empty(0)
    scores = np.empty(0)
    if threshold is not None:
        threshold = finite_scalar(threshold, "threshold")
        if threshold_bounds is not None or candidates is not None:
            raise ValueError("search options cannot be combined with known threshold")
    else:
        if threshold_bounds is None:
            trim = _threshold_variable(trim, 2)
            if not 0 < trim[0] < trim[1] < 1:
                raise ValueError("trim must satisfy 0 < lower < upper < 1")
            # Quantiles on a scaled variable avoid overflow in interpolation.
            z_scale = max(float(np.max(np.abs(z))), np.finfo(float).tiny)
            bounds = tuple(np.quantile(z / z_scale, trim) * z_scale)
        else:
            bounds = tuple(_threshold_variable(threshold_bounds, 2))
        if not bounds[0] < bounds[1]:
            raise ValueError("threshold bounds must be strictly increasing")
        extreme_masks = (z < bounds[0], z >= bounds[1])
        if any(not np.any(mask[:-1]) for mask in extreme_masks):
            raise ValueError("each extreme regime needs a lagged origin observation")
        moments = _moments(work, extreme_masks, max_lag)
        reference, _, complements = _spaces(moments, ranks, len(X))
        ranks = tuple(tuple(q.shape[1] for q in pair) for pair in reference)
        if any(all(b.shape[1] == 0 for b in pair) for pair in complements):
            raise ValueError(
                "threshold search needs a loading complement in each regime"
            )
        if candidates is None:
            grid = np.unique(z[(z > bounds[0]) & (z < bounds[1])])
        else:
            raw_grid = np.asarray(candidates)
            if raw_grid.ndim != 1:
                raise ValueError("candidates must be a finite nonempty vector")
            grid = np.unique(_threshold_variable(raw_grid, len(raw_grid)))
            if np.any((grid <= bounds[0]) | (grid >= bounds[1])):
                raise ValueError("candidates must lie strictly inside threshold bounds")
        if not len(grid):
            raise ValueError("no candidate thresholds inside the trimmed interval")
        scores = np.array(
            [
                _score(_moments(work, (z < r, z >= r), max_lag), complements)
                for r in grid
            ]
        )
        threshold = float(grid[np.argmin(scores)])
    regimes = (z >= threshold).astype(int)
    masks = tuple(regimes == i for i in range(2))
    if any(not np.any(mask[:-1]) for mask in masks):
        raise ValueError("each regime needs a lagged origin observation")
    moments = _moments(work, masks, max_lag)
    loadings, spectra, _ = _spaces(moments, ranks, len(X))
    factors = tuple(
        _stable_project(X[mask], pair) for mask, pair in zip(masks, loadings)
    )
    signal = np.empty_like(X)
    for i in range(2):
        signal[masks[i]] = _stable_project(factors[i], loadings[i], inverse=True)
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = X - signal
    if not np.isfinite(residuals).all():
        raise ValueError("residuals exceed the finite numeric range")
    return ThresholdFactorResult(
        threshold,
        loadings,
        factors,
        signal,
        residuals,
        regimes,
        spectra,
        grid,
        scores,
        bounds,
        reference,
        scale,
        max_lag,
    )
