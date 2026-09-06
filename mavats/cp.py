"""Matrix CP factor estimation using refined generalized eigenanalysis."""

from dataclasses import dataclass

import numpy as np

from ._validation import as_series, finite_scalar, positive_int
from .factors import _canonical_signs, _eigenspace, _lags, _prepare


class CPIdentificationError(ValueError):
    """The selected lag moments do not identify a stable, real CP decomposition."""


def _scalar_proxy(values, length, name):
    raw = np.asarray(values)
    if raw.shape != (length,) or np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
        raise ValueError(
            f"{name} must be a finite real vector with one value per observation"
        )
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    scale = np.max(np.abs(result))
    if scale == 0:
        raise CPIdentificationError(
            f"{name} is constant and cannot identify lagged factors"
        )
    result = result / scale
    result = result - result.mean()
    if np.max(np.abs(result)) <= np.finfo(float).eps:
        raise CPIdentificationError(
            f"{name} is constant and cannot identify lagged factors"
        )
    return result


def _pca_proxy(X, variance):
    matrix = X.reshape(len(X), -1)
    matrix = matrix - matrix.mean(axis=0)
    # The loading sign convention makes the averaged scores reproducible.
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    if singular[0] == 0:
        raise CPIdentificationError(
            "constant observations cannot identify dynamic CP factors"
        )
    energy = (singular / singular[0]) ** 2
    count = min(
        len(energy),
        int(np.searchsorted(np.cumsum(energy) / energy.sum(), variance)) + 1,
    )
    signs = np.where(
        right[np.arange(count), np.argmax(np.abs(right[:count]), axis=1)] < 0, -1, 1
    )
    return (left[:, :count] * singular[:count] * signs).mean(axis=1)


def _cross_moment(centered, proxy, lag):
    # Full-sample means, not separate means of each lag-truncated sample: eq14.
    return np.einsum("tij,t->ij", centered[lag:], proxy[:-lag], optimize=True) / (
        len(centered) - lag
    )


def _condition(matrix, limit, name):
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[0] == 0 or singular[-1] <= singular[0] / limit:
        raise CPIdentificationError(
            f"{name} is singular or ill-conditioned; try another scalar proxy or rank"
        )
    return float(singular[0] / singular[-1])


def _normalized_columns(matrix):
    scale = np.max(np.abs(matrix), axis=0)
    if np.any(scale == 0):
        raise CPIdentificationError("a CP loading has zero norm")
    matrix = matrix / scale
    return matrix / np.linalg.norm(matrix, axis=0)


def _refined_loadings(H1, H2, condition_limit, eigenvalue_tol):
    condition = _condition(H1, condition_limit, "lag-one projected moment")
    _condition(H2, condition_limit, "lag-two projected moment")
    # Cancelling H1.T from the normal equation avoids squaring its condition
    # number. The solve is exactly J1 in eq27 whenever H1 is nonsingular.
    eigenvalues, vectors = np.linalg.eig(np.linalg.solve(H1, H2))
    if np.any(np.imag(eigenvalues) != 0) or np.any(np.imag(vectors) != 0):
        raise CPIdentificationError(
            "projected lag moments have non-real eigenvalues; a real CP fit is not identified"
        )
    eigenvalues, vectors = np.real(eigenvalues), np.real(vectors)
    if not np.isfinite(eigenvalues).all():
        raise CPIdentificationError("projected lag moments have nonfinite eigenvalues")
    order = np.argsort(eigenvalues)
    eigenvalues, vectors = eigenvalues[order], vectors[:, order]
    if len(eigenvalues) > 1:
        scale = np.maximum(
            1.0, np.maximum(np.abs(eigenvalues[1:]), np.abs(eigenvalues[:-1]))
        )
        if np.any(np.diff(eigenvalues) <= eigenvalue_tol * scale):
            raise CPIdentificationError(
                "generalized eigenvalues are not separated; individual CP components are not identified"
            )
    U = _normalized_columns(H1 @ vectors)
    _condition(U, condition_limit, "projected row loadings")
    # H1.T @ U^{-T} = solve(U, H1).T; no explicit matrix inverse.
    V = _normalized_columns(np.linalg.solve(U, H1).T)
    _condition(V, condition_limit, "projected column loadings")
    return U, V, eigenvalues, condition


def _design(loadings):
    A, B = loadings
    # Row-major vectorization consistently permutes the paper's b kron a
    # design and the observations, giving identical least-squares scores.
    return np.einsum("ir,jr->ijr", A, B).reshape(A.shape[0] * B.shape[0], A.shape[1])


@dataclass
class CPResult:
    """A fitted CP model ``Y[t] = sum_r scores[t,r] A[:,r] B[:,r].T``.

    Loading columns have unit length but generally are not orthogonal. They
    are identifiable only up to paired permutations and signs. Columns are
    ordered by ascending generalized eigenvalue and their largest entries are
    made positive. Scores absorb signs and original data scale. ``signal``
    includes the unrestricted training mean when centering was requested.
    """

    scores: np.ndarray
    loadings: tuple
    signal: np.ndarray
    residuals: np.ndarray
    mean: np.ndarray
    eigenvalues: np.ndarray
    condition_number: float
    xi: np.ndarray
    eta: np.ndarray
    method: str = "cp-refined"

    @property
    def rank(self):
        return self.scores.shape[1]

    @property
    def factors(self):
        return self.scores

    @property
    def A(self):
        return self.loadings[0]

    @property
    def B(self):
        return self.loadings[1]

    def transform(self, X):
        """Estimate scalar factors by joint least squares in the CP basis.

        References
        ----------
        Chang, He, Yang and Yao (2023), Modelling Matrix Time Series via a
        Tensor CP-Decomposition. https://arxiv.org/abs/2112.15423
        Uses fixed fitted loadings; this does not estimate a forecast law.
        """
        X = as_series(X, min_samples=1)
        if X.shape[1:] != self.mean.shape:
            raise ValueError("observation dimensions must match the fitted CP model")
        centered = X - self.mean
        scale = max(float(np.max(np.abs(centered))), np.finfo(float).tiny)
        scores = np.linalg.lstsq(
            _design(self.loadings), (centered / scale).reshape(len(X), -1).T, rcond=None
        )[0].T
        return scores * scale

    def inverse_transform(self, scores):
        """Reconstruct matrices from scalar factors, adding the training mean.

        References
        ----------
        Chang, He, Yang and Yao (2023), Modelling Matrix Time Series via a
        Tensor CP-Decomposition. https://arxiv.org/abs/2112.15423
        Restoring the optional training mean is an implementation convention.
        """
        scores = as_series(scores, min_samples=1, ndim=2)
        if scores.shape[1] != self.rank:
            raise ValueError("score dimensions must match the fitted CP rank")
        return (
            np.einsum("tr,ir,jr->tij", scores, self.A, self.B, optimize=True)
            + self.mean
        )


def fit_cp_factor(
    X,
    rank,
    *,
    lags=3,
    xi=None,
    eta=None,
    center=False,
    proxy_variance=0.99,
    condition_limit=1e10,
    eigenvalue_tol=1e-8,
):
    """Fit Chang, He, Yang and Yao's refined matrix CP factor estimator.

    Parameters
    ----------
    X : array_like, shape (T, p, q)
        Finite real matrix time series with at least four observations.
    rank : int
        Number of CP components, at most min(p, q). Row and column loadings
        must each have this rank; overcomplete CP models are not supported.
    lags : int or sequence of int, default 3
        Lags used to estimate the initial row and column spaces. An integer K
        includes lags 1,...,K. The refinement always uses lags one and two.
    xi, eta : array_like, shape (T,), optional
        Scalar proxies for the initial and projected stages. Each must have
        informative lag covariance with every component. The model assumes
        these are contemporaneous linear combinations of observations (of the
        projected observations for eta); arbitrary external proxies require
        corresponding noise exogeneity. By default, each proxy averages PCA
        scores accounting for at least ``proxy_variance`` of total variance,
        following the paper's simulation procedure. Proxies are centered and
        scaled internally, and returned in that convention.
    center : bool, default False
        Remove and restore an unrestricted temporal mean when computing scores
        and signal. Moment estimation always subtracts full-sample means.
    proxy_variance : float, default 0.99
        Cumulative variance fraction in (0, 1] used for data-derived proxies.
    condition_limit : float, default 1e10
        Reject projected lag moments/loadings with larger condition numbers.
    eigenvalue_tol : float, default 1e-8
        Minimum separation relative to max(1, magnitudes) of adjacent sorted
        generalized eigenvalues. Complex eigenvalues are rejected explicitly.

    Returns
    -------
    CPResult
        Unit-norm nonorthogonal loadings, scalar factors, signal and diagnostics.

    Notes
    -----
    Implements equations (29)--(34) with delta1=delta2=0, the unthresholded
    finite-dimensional estimator. It does not implement the paper's thresholded
    high-dimensional covariance estimator or automatic rank selection.
    Identification requires full-rank projected moments at lags one and two,
    distinct real generalized eigenvalues and appropriately white errors. A
    noisy finite sample may violate these requirements and raises
    ``CPIdentificationError`` instead of returning complex or unstable loadings.

    References
    ----------
    Chang, He, Yang and Yao (2023), Modelling Matrix Time Series via a Tensor
    CP-Decomposition. https://arxiv.org/abs/2112.15423
    """
    X = as_series(X, min_samples=4)
    rank = positive_int(rank, "rank")
    if rank > min(X.shape[1:]):
        raise ValueError("rank cannot exceed either matrix dimension")
    lags = _lags(lags, len(X))
    variance = finite_scalar(proxy_variance, "proxy_variance", minimum=0)
    if not 0 < variance <= 1:
        raise ValueError("proxy_variance must be in (0, 1]")
    condition_limit = finite_scalar(condition_limit, "condition_limit", minimum=1)
    if condition_limit <= 1:
        raise ValueError("condition_limit must exceed one")
    eigenvalue_tol = finite_scalar(eigenvalue_tol, "eigenvalue_tol", minimum=0)
    if eigenvalue_tol == 0:
        raise ValueError("eigenvalue_tol must be positive")
    X, normalized, mean, _ = _prepare(X, center)
    centered = normalized - normalized.mean(axis=0)
    xi = _scalar_proxy(
        _pca_proxy(centered, variance) if xi is None else xi, len(X), "xi"
    )
    moments = [_cross_moment(centered, xi, lag) for lag in lags]
    row = sum(moment @ moment.T for moment in moments)
    col = sum(moment.T @ moment for moment in moments)
    P, row_values = _eigenspace(row, rank)
    Q, col_values = _eigenspace(col, rank)
    for values in (row_values, col_values):
        if values[0] == 0 or values[rank - 1] <= values[0] / condition_limit**2:
            raise CPIdentificationError(
                "initial lag moments do not identify the requested row and column ranks"
            )
    Z = P.T @ centered @ Q
    eta = _scalar_proxy(_pca_proxy(Z, variance) if eta is None else eta, len(X), "eta")
    H1, H2 = (_cross_moment(Z, eta, lag) for lag in (1, 2))
    U, V, eigenvalues, condition = _refined_loadings(
        H1, H2, condition_limit, eigenvalue_tol
    )
    loadings = (_canonical_signs(P @ U), _canonical_signs(Q @ V))
    _condition(_design(loadings), condition_limit, "CP least-squares design")
    result = CPResult(
        np.empty((len(X), rank)),
        loadings,
        np.empty_like(X),
        np.empty_like(X),
        mean,
        eigenvalues,
        condition,
        xi,
        eta,
    )
    result.scores = result.transform(X)
    result.signal = result.inverse_transform(result.scores)
    result.residuals = X - result.signal
    return result
