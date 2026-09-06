"""Dense, independent simulation oracles for structured autoregressions.

These deliberately construct small vectorized transitions rather than reuse
the estimators' mode contractions. They are benchmark designs, not public
general-purpose simulators. Every spatial tensor is vectorized in Fortran order.
"""

from dataclasses import dataclass

import numpy as np


def _kronecker(matrices):
    result = np.ones((1, 1))
    for matrix in reversed(matrices):
        result = np.kron(result, matrix)
    return result


def _correlation(dimension, rho):
    indices = np.arange(dimension)
    return rho ** np.abs(indices[:, None] - indices[None, :])


@dataclass
class CointegratedData:
    observations: np.ndarray
    conditional_mean: np.ndarray
    innovations: np.ndarray
    A1: np.ndarray
    A2: np.ndarray
    short_run_left: np.ndarray
    short_run_right: np.ndarray
    intercept: np.ndarray
    beta1: np.ndarray
    beta2: np.ndarray
    row_covariance: np.ndarray
    column_covariance: np.ndarray
    companion: np.ndarray


def cointegrated_data(n, *, shape=(3, 4), ranks=(1, 2), regime="separable", seed=1193):
    """I(1) CMAR with one difference lag and known cointegrating spaces.

    Orthogonal mode bases diagonalize both long- and short-run coefficients.
    Each common-trend coordinate has roots 1 and g, 0<g<1. Cointegrating
    coordinates have stable AR(2) roots. Burn-in reduces stationary-coordinate
    transients; the levels themselves have no stationary distribution.
    """
    if regime not in ("isotropic", "separable", "weak-adjustment"):
        raise ValueError("unknown cointegration regime")
    m, q = shape
    r, s = ranks
    if n < 3 or not 0 < r < m or not 0 < s < q:
        raise ValueError("need n>=3 and positive ranks below both dimensions")
    rng = np.random.default_rng(seed)
    row = np.linalg.qr(rng.normal(size=(m, m)))[0]
    column = np.linalg.qr(rng.normal(size=(q, q)))[0]
    beta1, beta2 = row[:, :r], column[:, :s]
    adjustment = 0.04 if regime == "weak-adjustment" else 0.4
    A1 = -(beta1 * np.linspace(adjustment, 1.2 * adjustment, r)) @ beta1.T
    A2 = (beta2 * np.linspace(0.8, 1.1, s)) @ beta2.T
    left = (row * np.linspace(0.3, 0.15, m)) @ row.T
    right = (column * np.linspace(0.4, 0.65, q)) @ column.T
    intercept = 0.025 * np.outer(row[:, -1], column[:, -1])
    rho = 0 if regime == "isotropic" else 0.45
    row_cov, column_cov = _correlation(m, rho), _correlation(q, rho)
    d = m * q
    pi, gamma = np.kron(A2, A1), np.kron(right, left)
    transitions = (np.eye(d) + pi + gamma, -gamma)
    companion = np.block([[*transitions], [np.eye(d), np.zeros((d, d))]])
    burnin = 200
    state = np.zeros((n + burnin + 2, d))
    means = np.zeros_like(state)
    noise = (
        rng.normal(size=(n + burnin, d))
        @ np.linalg.cholesky(np.kron(column_cov, row_cov)).T
    )
    for t in range(2, len(state)):
        means[t] = (
            transitions[0] @ state[t - 1]
            + transitions[1] @ state[t - 2]
            + intercept.ravel(order="F")
        )
        state[t] = means[t] + noise[t - 2]
    reshape = lambda rows: np.stack([x.reshape(shape, order="F") for x in rows])
    return CointegratedData(
        reshape(state[-n:]),
        reshape(means[-n:]),
        reshape(noise[-n:]),
        A1,
        A2,
        left[None],
        right[None],
        intercept,
        beta1,
        beta2,
        row_cov,
        column_cov,
        companion,
    )


@dataclass
class TensorARData:
    observations: np.ndarray
    conditional_mean: np.ndarray
    innovations: np.ndarray
    coefficients: tuple
    transitions: np.ndarray
    covariance: np.ndarray
    covariance_factors: tuple
    companion: np.ndarray


@dataclass
class ContaminatedFactorData:
    observations: np.ndarray
    signal: np.ndarray
    loadings: tuple
    contamination_mask: np.ndarray


def contaminated_factor_data(n, *, shape=(8, 10), regime="gaussian", seed=1693):
    """Paired rank-(2,2) factors with Gaussian, entry or matrix contamination.

    Contamination affects observations, not the latent reconstruction target.
    Matrix outliers corrupt every entry at 5% of times; entry outliers corrupt
    5% of entries independently. Both use additive N(0,8²) corruption.
    """
    if regime not in ("gaussian", "entry-outliers", "matrix-outliers"):
        raise ValueError("unknown factor contamination regime")
    if n < 3 or min(shape) < 3:
        raise ValueError("need n>=3 and spatial dimensions>=3")
    rng = np.random.default_rng(seed)
    p, q = shape
    row = np.linalg.qr(rng.normal(size=(p, 2)))[0] * np.sqrt(p)
    column = np.linalg.qr(rng.normal(size=(q, 2)))[0] * np.sqrt(q)
    factors = np.zeros((n + 200, 2, 2))
    scale = np.sqrt(np.array([[1.2, 0.8], [0.6, 0.4]]) * (1 - 0.4**2))
    for t in range(1, len(factors)):
        factors[t] = 0.4 * factors[t - 1] + rng.normal(size=(2, 2)) * scale
    signal = row @ factors[200:] @ column.T
    clean = signal + 0.5 * rng.normal(size=signal.shape)
    entry_mask = rng.uniform(size=signal.shape) < 0.05
    matrix_mask = np.broadcast_to(rng.uniform(size=(n, 1, 1)) < 0.05, signal.shape)
    corruption = 8 * rng.normal(size=signal.shape)
    mask = (
        entry_mask
        if regime == "entry-outliers"
        else (
            matrix_mask
            if regime == "matrix-outliers"
            else np.zeros(signal.shape, dtype=bool)
        )
    )
    return ContaminatedFactorData(
        clean + mask * corruption, signal, (row, column), mask.copy()
    )


def tensor_ar_data(n, *, shape=(2, 3, 2), regime="separable", seed=1493):
    """TenAR(2) with term counts (2,1), including the matrix special case.

    Nonsymmetric mode matrices are independently generated. Each term's
    operator norm is bounded by its assigned weight; total weight .70<1
    supplies a conservative stationarity bound for the two-lag recurrence.
    Nonseparable innovations are explicitly a misspecified-MLE comparison.
    """
    if regime not in ("isotropic", "separable", "nonseparable"):
        raise ValueError("unknown tensor autoregression regime")
    if n < 3 or len(shape) < 2 or min(shape) < 2 or np.prod(shape) > 256:
        raise ValueError("need n>=3, at least two nontrivial modes and size<=256")
    rng = np.random.default_rng(seed)
    coefficients = []
    transitions = []
    for weights in ((0.32, 0.23), (0.15,)):
        terms = []
        for weight in weights:
            matrices = []
            for dimension in shape:
                matrix = rng.normal(scale=0.4, size=(dimension, dimension))
                matrix += np.eye(dimension)
                matrix /= np.linalg.norm(matrix, ord=2)
                matrices.append(matrix)
            matrices[-1] = matrices[-1] * weight
            terms.append(tuple(matrices))
        coefficients.append(tuple(terms))
        transitions.append(sum(_kronecker(term) for term in terms))
    transitions = np.asarray(transitions)
    factors = tuple(
        _correlation(d, 0 if regime == "isotropic" else 0.45) for d in shape
    )
    covariance = _kronecker(factors)
    d = int(np.prod(shape))
    if regime == "nonseparable":
        direction = rng.normal(size=d)
        direction /= np.linalg.norm(direction)
        covariance = 0.7 * covariance + 0.3 * d * np.outer(direction, direction)
    companion = np.block([[*transitions], [np.eye(d), np.zeros((d, d))]])
    burnin = 200
    state = np.zeros((n + burnin + 2, d))
    means = np.zeros_like(state)
    noise = rng.normal(size=(n + burnin, d)) @ np.linalg.cholesky(covariance).T
    for t in range(2, len(state)):
        means[t] = transitions[0] @ state[t - 1] + transitions[1] @ state[t - 2]
        state[t] = means[t] + noise[t - 2]
    reshape = lambda rows: np.stack([x.reshape(shape, order="F") for x in rows])
    return TensorARData(
        reshape(state[-n:]),
        reshape(means[-n:]),
        reshape(noise[-n:]),
        tuple(coefficients),
        transitions,
        covariance,
        factors,
        companion,
    )
