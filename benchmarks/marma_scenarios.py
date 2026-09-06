"""Independent dense designs for MARMA and constrained-factor comparisons.

These simulate the model families of Tsay (2024),
https://doi.org/10.1111/insr.12558, and Chen, Tsay and Chen (2020),
https://doi.org/10.1080/01621459.2019.1584899. Coefficients, noise and stress
regimes below are library integration designs, not exact paper replications.
No estimator or public simulator is used to generate the observations.
"""

from dataclasses import dataclass

import numpy as np


def _matrices(vectors, shape):
    return np.stack([v.reshape(shape, order="F") for v in vectors])


def _correlation(dimension, rho):
    indices = np.arange(dimension)
    return rho ** np.abs(indices[:, None] - indices)


@dataclass
class MARMAData:
    observations: np.ndarray
    conditional_mean: np.ndarray
    innovations: np.ndarray
    ar_left: np.ndarray
    ar_right: np.ndarray
    ma_left: np.ndarray
    ma_right: np.ndarray
    intercept: np.ndarray
    row_covariance: np.ndarray
    column_covariance: np.ndarray
    covariance: np.ndarray
    ar_transition: np.ndarray
    ma_transition: np.ndarray


def marma_data(n, *, regime="separable", seed=1973):
    """Stationary/invertible nonsymmetric 2x3 MARMA(1,1), minus-MA convention.

    Near-cancellation moves the MA operator close to the AR operator without
    making them identical. Nonseparable noise violates the fitted MLE model.
    A 300-step burn-in approximates stationary initialization; returned true
    innovations are an inaccessible latent-history oracle, not fitted residuals.
    """
    if regime not in ("isotropic", "separable", "near-cancellation", "nonseparable"):
        raise ValueError("unknown MARMA regime")
    if n < 3:
        raise ValueError("need at least three observations")
    rng = np.random.default_rng(seed)
    shape = (2, 3)
    ar_left = np.array([[0.55, 0.12], [-0.06, 0.4]])
    ar_right = np.array([[0.7, 0.13, 0], [-0.08, 0.6, 0.1], [0.04, 0, 0.5]])
    ma_left = np.array([[-0.7, 0.11], [0.08, -0.5]])
    ma_right = np.array([[0.8, -0.1, 0.06], [0.03, 0.65, 0], [0.1, 0.04, 0.7]])
    if regime == "near-cancellation":
        ma_left = 0.95 * ar_left + np.array([[0, 0.012], [-0.006, 0]])
        ma_right = ar_right.copy()
    ar_transition = np.kron(ar_right, ar_left)
    ma_transition = np.kron(ma_right, ma_left)
    rho = 0 if regime == "isotropic" else 0.45
    row_covariance = _correlation(2, rho)
    column_covariance = _correlation(3, rho)
    covariance = np.kron(column_covariance, row_covariance)
    if regime == "nonseparable":
        direction = rng.normal(size=6)
        direction /= np.linalg.norm(direction)
        covariance = 0.7 * covariance + 0.3 * 6 * np.outer(direction, direction)
    intercept = np.array([[0.08, -0.03, 0.05], [-0.04, 0.06, 0.02]])
    burnin = 300
    innovations = (
        rng.normal(size=(n + burnin + 1, 6)) @ np.linalg.cholesky(covariance).T
    )
    state = np.zeros_like(innovations)
    means = np.zeros_like(innovations)
    for t in range(1, len(state)):
        means[t] = intercept.ravel(order="F") + ar_transition @ state[t - 1]
        means[t] -= ma_transition @ innovations[t - 1]
        state[t] = means[t] + innovations[t]
    return MARMAData(
        _matrices(state[-n:], shape),
        _matrices(means[-n:], shape),
        _matrices(innovations[-n:], shape),
        ar_left[None],
        ar_right[None],
        ma_left[None],
        ma_right[None],
        intercept,
        row_covariance,
        column_covariance,
        covariance,
        ar_transition,
        ma_transition,
    )


@dataclass
class PartialFactorData:
    observations: np.ndarray
    signal: np.ndarray
    factors: np.ndarray
    loadings: tuple
    row_constraints: np.ndarray
    column_constraints: np.ndarray
    row_ranks: tuple
    column_ranks: tuple
    signal_blocks: tuple


def partial_factor_data(n, *, shape=(8, 10), regime="full-interactions", seed=2173):
    """Four-block partial-factor model with shared loading spaces.

    Known mode ranks are (1,2) for row groups and (2,1) for column groups.
    Misspecified constraints rotate a signal direction into a pure-noise
    direction, so they genuinely exclude part of the true constrained loading.
    """
    if regime not in (
        "full-interactions",
        "diagonal-only",
        "weak-complement",
        "misspecified-constraints",
    ):
        raise ValueError("unknown partial-factor regime")
    p, q = shape
    if n < 3 or p < 6 or q < 8:
        raise ValueError("need n>=3 and dimensions at least 6x8")
    rng = np.random.default_rng(seed)
    row_basis = np.linalg.qr(rng.normal(size=(p, p)))[0]
    column_basis = np.linalg.qr(rng.normal(size=(q, q)))[0]
    mr, mc = p // 2 - 1, q // 2 - 1
    row_constraints, column_constraints = (
        row_basis[:, :mr].copy(),
        column_basis[:, :mc].copy(),
    )
    row = row_basis[:, [0, mr, mr + 1]] * np.sqrt(p)
    column = column_basis[:, [0, 1, mc]] * np.sqrt(q)
    if regime == "misspecified-constraints":
        row_constraints[:, 0] = (
            np.cos(0.8) * row_basis[:, 0] + np.sin(0.8) * row_basis[:, -1]
        )
        column_constraints[:, 0] = (
            np.cos(0.8) * column_basis[:, 0] + np.sin(0.8) * column_basis[:, -1]
        )
    factors = np.zeros((n + 200, 3, 3))
    amplitude = np.sqrt(np.array([[1.5, 0.7, 0.9], [0.8, 1.2, 0.6], [1.0, 0.5, 0.75]]))
    if regime == "diagonal-only":
        amplitude[0, 2] = 0
        amplitude[1:, :2] = 0
    elif regime == "weak-complement":
        amplitude[1:] *= 0.15
        amplitude[:, 2] *= 0.15
    for t in range(1, len(factors)):
        factors[t] = (
            0.55 * factors[t - 1]
            + np.sqrt(1 - 0.55**2) * rng.normal(size=(3, 3)) * amplitude
        )
    factors = factors[200:]
    # Independently vectorized multilinear signal, not a fitted contraction.
    design = np.kron(column, row)
    signal = _matrices(np.stack([design @ f.ravel(order="F") for f in factors]), shape)
    row_groups, column_groups = (slice(0, 1), slice(1, 3)), (slice(0, 2), slice(2, 3))
    blocks = tuple(
        tuple(row[:, r] @ factors[:, r, c] @ column[:, c].T for c in column_groups)
        for r in row_groups
    )
    observations = signal + 0.8 * rng.normal(size=signal.shape)
    return PartialFactorData(
        observations,
        signal,
        factors,
        (row, column),
        row_constraints,
        column_constraints,
        (1, 2),
        (2, 1),
        blocks,
    )


@dataclass
class MultiTermFactorData:
    observations: np.ndarray
    signal: np.ndarray
    factors: tuple
    loadings: tuple
    constraints: tuple
    component_signals: tuple


def multiterm_factor_data(n, *, shape=(8, 10), regime="orthogonal", seed=2373):
    """Two rank-(1,1) constrained terms with controlled subspace overlap.

    Orthonormal bases within each constraint span are supplied. Nonorthogonal
    terms have principal-angle cosine .7, or .98 for the near-overlap stress.
    The joint signal design remains full rank; competing projections weaken
    the signal as overlap increases without silently changing the true ranks.
    """
    if regime not in ("orthogonal", "overlapping", "near-overlap"):
        raise ValueError("unknown multi-term-factor regime")
    p, q = shape
    if n < 3 or min(shape) < 6:
        raise ValueError("need n>=3 and dimensions at least 6")
    rng = np.random.default_rng(seed)
    u = np.linalg.qr(rng.normal(size=(p, p)))[0]
    v = np.linalg.qr(rng.normal(size=(q, q)))[0]
    cosine = {"orthogonal": 0.0, "overlapping": 0.7, "near-overlap": 0.98}[regime]
    sine = np.sqrt(1 - cosine**2)
    rows = (u[:, :2], cosine * u[:, :2] + sine * u[:, 2:4])
    columns = (v[:, :2], cosine * v[:, :2] + sine * v[:, 2:4])
    # The same local loading direction makes nearly overlapping signal terms
    # genuinely ill-conditioned, rather than merely overlapping unused spans.
    local_r, local_c = np.array([0.8, 0.6]), np.array([0.6, -0.8])
    loadings = tuple(
        ((r @ local_r)[:, None] * np.sqrt(p), (c @ local_c)[:, None] * np.sqrt(q))
        for r, c in zip(rows, columns)
    )
    factors = np.zeros((n + 200, 2))
    persistence = np.array([0.65, -0.4])
    for t in range(1, len(factors)):
        factors[t] = persistence * factors[t - 1] + np.sqrt(
            1 - persistence**2
        ) * rng.normal(size=2)
    factors = factors[200:]
    components = tuple(
        _matrices(factors[:, j, None] @ np.kron(c, r).T, shape)
        for j, (r, c) in enumerate(loadings)
    )
    signal = components[0] + components[1]
    observations = signal + 0.8 * rng.normal(size=signal.shape)
    return MultiTermFactorData(
        observations,
        signal,
        tuple(factors[:, j, None, None] for j in range(2)),
        loadings,
        tuple(zip(rows, columns)),
        components,
    )
