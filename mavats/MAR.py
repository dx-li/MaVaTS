"""Compatibility entry points for the original MAR(1) tuple-returning API.

New code should use :func:`mavats.autoregression.fit_mar`, which also provides
forecasting, intercepts, multiple lags and numerical diagnostics.
"""

import numpy as np

from ._validation import as_series
from .autoregression import (
    _covariance_update,
    _left_update,
    _normalize,
    _rearrange_phi,
    _right_update,
    _whitener,
    fit_mar,
)


def estimate_mar1(X, method="least_square", **kwargs):
    """Estimate ``X[t] = A @ X[t-1] @ B.T + E[t]``.

    Methods are ``'proj'``, ``'least_square'`` and ``'mle'``. Returns ``A,B``
    or, for MLE, ``A,B,Sigc,Sigr`` (column covariance before row covariance,
    preserving the legacy order). Covariances have shapes ``(n,n)`` and
    ``(m,m)``. ``niter`` sets the maximum sweeps; ``tol`` controls stopping.
    ``A_init``, ``B_init`` and initial covariances are copied, never mutated.
    MLE's default covariance eigenvalue floor is 1e-8; use ``fit_mar`` to
    inspect whether stabilization was active and whether fitting converged.
    """
    mapping = {"proj": "projection", "least_square": "als", "mle": "mle"}
    if method not in mapping:
        raise ValueError("method must be 'proj', 'least_square', or 'mle'")
    accepted = {
        "niter",
        "A_init",
        "B_init",
        "init_method",
        "Sigc_init",
        "Sigr_init",
        "tol",
        "ridge",
        "covariance_floor",
        "random_state",
    }
    unknown = set(kwargs) - accepted
    if unknown:
        raise TypeError(f"unexpected keyword argument(s): {', '.join(sorted(unknown))}")
    A, B = kwargs.get("A_init"), kwargs.get("B_init")
    if (A is None) != (B is None):
        raise ValueError("A_init and B_init must be supplied together")
    row, col = kwargs.get("Sigr_init"), kwargs.get("Sigc_init")
    if (row is None) != (col is None):
        raise ValueError("Sigr_init and Sigc_init must be supplied together")
    init = kwargs.get("init_method", "proj")
    init = "projection" if init == "proj" else init
    result = fit_mar(
        X,
        method=mapping[method],
        max_iter=kwargs.get("niter", 50),
        initial=None if A is None else (A, B),
        init=init,
        initial_covariance=None if row is None else (row, col),
        **{
            key: kwargs[key]
            for key in ("tol", "ridge", "covariance_floor", "random_state")
            if key in kwargs
        },
    )
    if method == "mle":
        # Preserve legacy unit-Frobenius row covariance when its companion
        # column covariance remains representable. At extreme scales retain
        # the balanced physical covariance factors from the modern result.
        with np.errstate(over="ignore", invalid="ignore"):
            row, col = _normalize(result.row_covariance, result.column_covariance)
        if not np.isfinite(col).all() or not np.any(col):
            row, col = result.row_covariance, result.column_covariance
        return result.A, result.B, col, row
    return result.A, result.B


def estimate_residual_cov(X, A, B):
    """Centered sample covariance (ddof=1), stacking residual columns."""
    X = as_series(X, min_samples=3)
    for coefficient, dim, name in ((A, X.shape[1], "A"), (B, X.shape[2], "B")):
        if np.iscomplexobj(coefficient) or np.shape(coefficient) != (dim, dim):
            raise ValueError(f"{name} must be a real square matrix matching X")
        if not np.isfinite(coefficient).all():
            raise ValueError(f"{name} must be finite")
    residuals = X[1:] - np.asarray(A) @ X[:-1] @ np.asarray(B).T
    flat = residuals.transpose(0, 2, 1).reshape(len(residuals), -1)
    flat -= flat.mean(axis=0)
    return flat.T @ flat / (len(flat) - 1)


def _proj_estimate(X):
    return estimate_mar1(X, "proj")


def _initialize_AB(X, A_init, B_init, init_method="proj"):
    if (A_init is None) != (B_init is None):
        raise ValueError("A_init and B_init must be supplied together")
    if A_init is not None:
        return np.array(A_init, dtype=float, copy=True), np.array(
            B_init, dtype=float, copy=True
        )
    if init_method == "proj":
        return _proj_estimate(X)
    if init_method == "random":
        X = as_series(X)
        rng = np.random.default_rng()
        return _normalize(
            rng.normal(size=(X.shape[1], X.shape[1])),
            rng.normal(size=(X.shape[2], X.shape[2])),
        )
    raise ValueError("init_method must be 'proj' or 'random'")


def _least_square_estimate(X, niter=50, A_init=None, B_init=None, init_method="proj"):
    return estimate_mar1(
        X,
        "least_square",
        niter=niter,
        A_init=A_init,
        B_init=B_init,
        init_method=init_method,
    )


def _mle_estimate(
    X,
    niter=50,
    A_init=None,
    B_init=None,
    init_method="proj",
    Sigc_init=None,
    Sigr_init=None,
):
    return estimate_mar1(
        X,
        "mle",
        niter=niter,
        A_init=A_init,
        B_init=B_init,
        init_method=init_method,
        Sigc_init=Sigc_init,
        Sigr_init=Sigr_init,
    )


def _update_lse(X_tp1, X_t, A, B):
    A = _left_update(X_tp1, X_t, B)
    A, B = _normalize(A, B)
    B = _right_update(X_tp1, X_t, A)
    return _normalize(A, B)


def _update_A_mle(X_tp1, X_t, A, B, Sigc, Sigr):
    return _left_update(X_tp1, X_t, B, whitening=_whitener(Sigc))


def _update_B_mle(X_tp1, X_t, A, B, Sigc, Sigr):
    return _right_update(X_tp1, X_t, A, whitening=_whitener(Sigr))


def _compute_R(X_tp1, X_t, A, B):
    return X_tp1 - A @ X_t @ B.T


def _update_Sigc_mle(X_tp1, X_t, A, B, R, Sigc, Sigr):
    white = _whitener(Sigr) @ R
    return np.einsum("tij,tik->jk", white, white) / (len(R) * R.shape[1])


def _update_Sigr_mle(X_tp1, X_t, A, B, R, Sigc, Sigr):
    white = R @ _whitener(Sigc)
    return np.einsum("tij,tkj->ik", white, white) / (len(R) * R.shape[2])


def _update_mle(X_tp1, X_t, A, B, Sigc, Sigr):
    A = _update_A_mle(X_tp1, X_t, A, B, Sigc, Sigr)
    A, B = _normalize(A, B)
    B = _update_B_mle(X_tp1, X_t, A, B, Sigc, Sigr)
    A, B = _normalize(A, B)
    residuals = _compute_R(X_tp1, X_t, A, B)
    Sigr, Sigc, _ = _covariance_update(residuals, Sigr, Sigc, 1e-8)
    return A, B, Sigc, Sigr
