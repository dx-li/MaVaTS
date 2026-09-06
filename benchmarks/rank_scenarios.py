"""Independent matrix-factor designs for published rank-criterion comparisons.

Han, Chen and Zhang (2022), Rank Determination in Tensor Factor Model,
https://doi.org/10.1214/22-EJS1991, motivates the model, weak loading directions,
white-noise assumptions and TIPUP cancellation checks. These fixed synthetic
designs are integration experiments, not copies of the paper's simulation tables.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class RankFactorData:
    observations: np.ndarray
    signal: np.ndarray
    noise: np.ndarray
    factors: np.ndarray
    loadings: tuple
    ranks: tuple
    factor_ar: np.ndarray
    noise_ar: float
    regime: str


def rank_factor_data(n, *, shape=(10, 12), regime="strong", seed=2573):
    """Matrix series with explicit loading strength and signed core dynamics.

    All regimes share Gaussian draws at a given seed/shape. Weak factors alter
    only the second loading direction in each mode. Cancellation uses four
    independent equal-variance AR components with signs [[+,-],[-,+]], making
    BOTH population TIPUP lag-one moments exactly zero. At lag two they are
    nonzero. Dense Kronecker reconstruction is independent of package helpers.
    """
    regimes = {
        "strong",
        "weak",
        "cancellation",
        "white-factors",
        "no-factors",
        "serial-noise",
    }
    if regime not in regimes:
        raise ValueError(f"regime must be one of {sorted(regimes)}")
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer of at least two")
    if len(shape) != 2 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 4
        for v in shape
    ):
        raise ValueError("shape must contain two integer dimensions of at least four")
    rng = np.random.default_rng(seed)
    p, q = shape
    row = np.linalg.qr(rng.normal(size=(p, 2)), mode="reduced")[0] * np.sqrt(p)
    column = np.linalg.qr(rng.normal(size=(q, 2)), mode="reduced")[0] * np.sqrt(q)
    if regime == "weak":
        # Weakest loading norm d_k**((1-delta)/2), with delta=.5.
        row[:, 1] *= p ** (-0.25)
        column[:, 1] *= q ** (-0.25)
    phi = np.full((2, 2), 0.65)
    if regime == "cancellation":
        phi *= np.array([[1, -1], [-1, 1]])
    elif regime == "white-factors":
        phi[:] = 0
    burnin = 200
    factors = np.empty((n + burnin, 2, 2))
    state = rng.normal(size=(2, 2))  # Exact stationary marginal initialization.
    factor_innovations = rng.normal(size=factors.shape)
    for t, innovation in enumerate(factor_innovations):
        state = phi * state + np.sqrt(1 - phi**2) * innovation
        factors[t] = state
    factors = factors[burnin:]
    design = np.kron(column, row)
    signal = np.array(
        [(design @ f.ravel(order="F")).reshape(shape, order="F") for f in factors]
    )
    noise_ar = 0.5 if regime == "serial-noise" else 0.0
    noise_innovations = rng.normal(size=(n + burnin, p, q))
    noise = np.empty_like(noise_innovations)
    state = rng.normal(size=(p, q))
    for t, innovation in enumerate(noise_innovations):
        state = noise_ar * state + np.sqrt(1 - noise_ar**2) * innovation
        noise[t] = state
    noise = noise[burnin:]
    ranks = (2, 2)
    if regime == "no-factors":
        signal = np.zeros_like(signal)
        ranks = (0, 0)
        row, column = row[:, :0], column[:, :0]
        factors = factors[:, :0, :0]
    return RankFactorData(
        signal + noise,
        signal,
        noise,
        factors,
        (row, column),
        ranks,
        phi,
        noise_ar,
        regime,
    )


def population_factor_lag_moment(data, lag, mode, method):
    """Dense population Gram oracle for independent stationary unit-variance cores.

    Returns the signal-only moment. Temporally white measurement noise adds
    nothing at positive lags; serial-noise experiments violate that premise.
    The outer tensor is deliberately materialized only for these small designs.

    References
    ----------
    Han, Chen and Zhang (2022), https://doi.org/10.1214/22-EJS1991.
    This is a signal-only population oracle for our AR-core designs, not a
    sample estimator or an exact reproduction of the paper's experiments.
    """
    if lag < 1 or mode not in (0, 1) or method not in ("topup", "tipup"):
        raise ValueError("require positive lag, mode 0/1 and topup/tipup")
    row, column = data.loadings
    p, q = data.signal.shape[1:]
    if not all(data.ranks):
        dim = (p, q)[mode]
        return np.zeros((dim, dim))
    covariance = np.zeros((p * q, p * q))
    for a in range(2):
        for b in range(2):
            loading = np.kron(column[:, b], row[:, a])
            covariance += data.factor_ar[a, b] ** lag * np.outer(loading, loading)
    # Map vec_F index (row + p*column) explicitly before selecting mode fibers.
    outer = np.empty((p, q, p, q))
    for a in range(p):
        for b in range(q):
            for c in range(p):
                for e in range(q):
                    outer[a, b, c, e] = covariance[a + p * b, c + p * e]
    if mode == 1:
        outer = outer.transpose(1, 0, 3, 2)
    dim, other = outer.shape[:2]
    if method == "tipup":
        moment = sum(outer[:, j, :, j] for j in range(other))
    else:
        moment = outer.reshape(dim, -1)
    return moment @ moment.T
