"""Reproducible matrix/tensor data generators with known ground truth.

Time is always axis zero. Matrix vectorization uses column-major order, so
``vec(A @ X @ B.T) == kron(B, A) @ vec(X)``.

References are given on each generator. These simulate the cited model
families with documented user-selected designs, not exact paper experiments.
"""

from dataclasses import dataclass

import numpy as np

from ._validation import finite_scalar, positive_int, random_generator, ranks_tuple


def _square_matrix(x, name):
    raw = np.asarray(x)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be real numeric")
    x = np.asarray(raw, dtype=float)
    if x.ndim != 2 or x.shape[0] != x.shape[1] or not x.shape[0]:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} must be finite")
    return x


def _covariance_root(cov, dimension, name):
    if cov is None:
        return np.eye(dimension)
    cov = _square_matrix(cov, name)
    if cov.shape != (dimension, dimension):
        raise ValueError(f"{name} has the wrong dimensions")
    scale = float(np.max(np.abs(cov))) or 1.0
    scaled = cov / scale
    if not np.allclose(scaled, scaled.T, rtol=1e-12, atol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    eig, vectors = np.linalg.eigh((scaled + scaled.T) / 2)
    if eig[0] < -1e-12 * max(float(eig[-1]), 1.0):
        raise ValueError(f"{name} must be positive semidefinite")
    return vectors * (np.sqrt(np.maximum(eig, 0)) * np.sqrt(scale))


def _real_array(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    return np.asarray(raw, dtype=float)


def matrix_normal(size, row_cov, column_cov, *, mean=None, random_state=None):
    """Draw matrices with ``Cov(vec(E)) = column_cov ⊗ row_cov``.

    Positive semidefinite (including singular) covariances are supported.
    The return shape is ``(size, rows, columns)``.

    References
    ----------
    Dawid (1981), *Some Matrix-Variate Distribution Theory: Notational
    Considerations and a Bayesian Application*,
    https://doi.org/10.1093/biomet/68.1.265, develops matrix-normal notation.
    The singular-covariance eigen-root extension is explicitly supported here.
    """
    size = positive_int(size, "size")
    row_cov = _square_matrix(row_cov, "row_cov")
    column_cov = _square_matrix(column_cov, "column_cov")
    m, n = len(row_cov), len(column_cov)
    row_root = _covariance_root(row_cov, m, "row_cov")
    col_root = _covariance_root(column_cov, n, "column_cov")
    noise = random_generator(random_state).standard_normal((size, m, n))
    with np.errstate(over="ignore", invalid="ignore"):
        result = row_root @ noise @ col_root.T
    if mean is not None:
        mean = _real_array(mean, "mean")
        if mean.shape != (m, n) or not np.isfinite(mean).all():
            raise ValueError(
                "mean must be a finite matrix matching covariance dimensions"
            )
        with np.errstate(over="ignore", invalid="ignore"):
            result += mean
    if not np.isfinite(result).all():
        raise FloatingPointError("matrix-normal draws exceed floating-point range")
    return result


def _mar_coefficients(left, right):
    left, right = _real_array(left, "left"), _real_array(right, "right")
    if left.ndim == 2:
        left = left[None]
    if right.ndim == 2:
        right = right[None]
    if left.ndim != 3 or right.ndim != 3 or len(left) != len(right) or not len(left):
        raise ValueError("left and right need matching nonzero lag counts")
    for a in left:
        _square_matrix(a, "left")
    for b in right:
        _square_matrix(b, "right")
    return left, right


def mar_spectral_radius(left, right):
    """Exact VAR companion spectral radius; < 1 implies stationary MAR.

    For one lag the Kronecker identity avoids constructing the full operator.
    Multiple lags require a dense ``(order*rows*columns)`` square companion.

    References
    ----------
    Chen, Xiao and Yang (2021), *Autoregressive Models for Matrix-Valued Time
    Series*, https://doi.org/10.1016/j.jeconom.2020.07.015, gives the MAR
    Kronecker stability condition. Multiple lags use the standard vectorized
    companion extension; this is a coefficient diagnostic, not a unit-root test.
    """
    left, right = _mar_coefficients(left, right)
    if len(left) == 1:
        return float(
            max(abs(np.linalg.eigvals(left[0]))) * max(abs(np.linalg.eigvals(right[0])))
        )
    d = left.shape[1] * right.shape[1]
    p = len(left)
    companion = np.zeros((p * d, p * d))
    companion[:d] = np.concatenate([np.kron(b, a) for a, b in zip(left, right)], axis=1)
    companion[d:, :-d] = np.eye((p - 1) * d)
    return float(max(abs(np.linalg.eigvals(companion))))


def simulate_mar(
    n_samples,
    left,
    right,
    *,
    burnin=200,
    row_cov=None,
    column_cov=None,
    intercept=None,
    initial=None,
    random_state=None,
    check_stationarity=True,
):
    """Simulate ``X[t] = intercept + sum(A[l] X[t-l-1] B[l].T) + E[t]``.

    ``initial`` contains ``order`` matrices, oldest first; zeros by default.
    Burn-in reduces initialization effects but is not an exact stationary draw.
    Set ``check_stationarity=False`` deliberately to simulate explosive models.

    References
    ----------
    Chen, Xiao and Yang (2021), *Autoregressive Models for Matrix-Valued Time
    Series*, https://doi.org/10.1016/j.jeconom.2020.07.015. Multiple lags,
    supplied intercept/initial conditions and burn-in are explicit simulation
    design choices; this does not claim to reproduce a numbered paper design.
    """
    n_samples = positive_int(n_samples, "n_samples")
    burnin = positive_int(burnin, "burnin", minimum=0)
    if not isinstance(check_stationarity, (bool, np.bool_)):
        raise ValueError("check_stationarity must be boolean")
    left, right = _mar_coefficients(left, right)
    p, m, _ = left.shape
    n = right.shape[1]
    if check_stationarity and mar_spectral_radius(left, right) >= 1:
        raise ValueError("MAR coefficients are not stationary (spectral radius >= 1)")
    constant = (
        np.zeros((m, n)) if intercept is None else _real_array(intercept, "intercept")
    )
    if constant.shape != (m, n) or not np.isfinite(constant).all():
        raise ValueError("intercept must be a finite matrix of shape (rows, columns)")
    history = np.zeros((p + burnin + n_samples, m, n))
    if initial is not None:
        initial = _real_array(initial, "initial")
        if initial.shape != (p, m, n) or not np.isfinite(initial).all():
            raise ValueError(
                "initial must have shape (order, rows, columns) and be finite"
            )
        history[:p] = initial
    rr = _covariance_root(row_cov, m, "row_cov")
    cr = _covariance_root(column_cov, n, "column_cov")
    rng = random_generator(random_state)
    # Equivalent, balanced factors prevent overflow from the arbitrary
    # identifiability transformation (A, B) -> (c*A, B/c).
    balanced = []
    for a, b in zip(left, right):
        amax, bmax = np.max(np.abs(a)), np.max(np.abs(b))
        if amax == 0 or bmax == 0:
            balanced.append((np.zeros_like(a), np.zeros_like(b)))
        else:
            common = np.sqrt(amax) * np.sqrt(bmax)
            balanced.append(((a / amax) * common, (b / bmax) * common))
    with np.errstate(over="ignore", invalid="ignore"):
        innovations = rr @ rng.standard_normal((n_samples + burnin, m, n)) @ cr.T
        for t in range(p, len(history)):
            history[t] = constant + innovations[t - p]
            for lag, (a, b) in enumerate(balanced, 1):
                history[t] += a @ history[t - lag] @ b.T
    if not np.isfinite(history).all():
        raise FloatingPointError(
            "simulation overflowed; reduce coefficient magnitudes or horizon"
        )
    return history[p + burnin :]


@dataclass
class FactorSimulation:
    """Observed series and identifiable reconstruction/subspace ground truth."""

    observations: np.ndarray
    signal: np.ndarray
    factors: np.ndarray
    loadings: tuple
    noise: np.ndarray


def simulate_factor(
    n_samples,
    shape,
    ranks,
    *,
    ar=0.6,
    noise_std=1.0,
    signal_scale=1.0,
    burnin=200,
    random_state=None,
):
    """Simulate a Tucker factor time series of any positive tensor order.

    Loading columns are orthogonal with squared norm equal to the mode size.
    Core entries are independent, stationary unit-variance Gaussian AR(1)
    processes. Thus signal strength follows the usual pervasive-factor scaling.
    ``ar=0`` is useful for checking the limitations of lag-only estimators.

    References
    ----------
    Chen, Yang and Zhang (2022), *Factor Models for High-Dimensional Tensor
    Time Series*, https://doi.org/10.1080/01621459.2021.1912757, supplies the
    multilinear factor model. Independent Gaussian AR(1) cores and this exact
    loading/noise design are library simulation choices, not a claim of exact
    reproduction of the paper's experiments.
    """
    n_samples = positive_int(n_samples, "n_samples")
    burnin = positive_int(burnin, "burnin", minimum=0)
    shape = tuple(positive_int(d, "dimension") for d in shape)
    if not shape:
        raise ValueError("shape must contain at least one spatial dimension")
    ranks = ranks_tuple(ranks, shape)
    if ranks is None:
        raise ValueError("simulation requires explicit ranks")
    ar = finite_scalar(ar, "ar")
    if abs(ar) >= 1:
        raise ValueError("abs(ar) must be < 1")
    noise_std = finite_scalar(noise_std, "noise_std", minimum=0)
    signal_scale = finite_scalar(signal_scale, "signal_scale", minimum=0)
    rng = random_generator(random_state)
    loadings = tuple(
        np.linalg.qr(rng.standard_normal((d, r)), mode="reduced")[0] * np.sqrt(d)
        for d, r in zip(shape, ranks)
    )
    factors = np.empty((n_samples + burnin,) + ranks)
    previous = rng.standard_normal(ranks)
    for t in range(len(factors)):
        previous = ar * previous + np.sqrt(1 - ar * ar) * rng.standard_normal(ranks)
        factors[t] = previous
    with np.errstate(over="ignore", invalid="ignore"):
        factors = factors[burnin:] * signal_scale
        signal = factors.copy()
        for mode, loading in enumerate(loadings, 1):
            signal = np.moveaxis(
                np.tensordot(signal, loading, axes=(mode, 1)), -1, mode
            )
        noise = rng.normal(scale=noise_std, size=(n_samples,) + shape)
        observations = signal + noise
    if not all(
        np.isfinite(value).all() for value in (observations, signal, factors, noise)
    ):
        raise FloatingPointError("factor simulation exceeds floating-point range")
    return FactorSimulation(observations, signal, factors, loadings, noise)
