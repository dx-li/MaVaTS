"""Independent synthetic designs for additive dynamics and online monitoring.

These generate observations directly from their defining recurrences, without
calling the corresponding fitted estimators or their internal kernels.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class AdditiveDynamicData:
    observations: np.ndarray
    signal: np.ndarray
    conditional_mean: np.ndarray
    row_loadings: np.ndarray
    column_loadings: np.ndarray
    F: np.ndarray
    G: np.ndarray
    row_ar: np.ndarray
    column_ar: np.ndarray


def additive_dynamic_data(
    n,
    *,
    shape=(8, 10),
    ranks=(1, 2),
    seed=893,
    noise_std=0.5,
    regime="diagonal-dynamics",
):
    """Stationary Gaussian additive model with separate row/column effects.

    ranks=(row rank c,column rank r); F has shape (time,rows,r), G has
    shape (time,columns,c). Conditional means use latent histories and are
    oracle comparators unavailable to a fitted predictor.
    """
    if regime not in ("diagonal-dynamics", "coupled-dynamics"):
        raise ValueError("unknown additive dynamic regime")
    rng = np.random.default_rng(seed)
    p, q = shape
    c, r = ranks
    row = np.linalg.qr(rng.normal(size=(p, c)))[0] * np.sqrt(p)
    column = np.linalg.qr(rng.normal(size=(q, r)))[0] * np.sqrt(q)
    row_ar = np.diag(np.linspace(0.75, 0.4, c))
    column_ar = np.diag(np.linspace(0.8, 0.45, r))
    if regime == "coupled-dynamics" and r > 1:
        column_ar[0, 1] = 0.25  # Explicit violation of diagonal factor dynamics.
    F = np.zeros((n + 200, p, r))
    G = np.zeros((n + 200, q, c))
    mean = np.zeros((n + 200, p, q))
    # Distinct marginal variances identify separate columns under the null.
    row_scale = np.sqrt(np.linspace(1.6, 0.7, c) * (1 - np.diag(row_ar) ** 2))
    column_scale = np.sqrt(np.linspace(1.8, 0.8, r) * (1 - np.diag(column_ar) ** 2))
    for t in range(1, len(F)):
        fmean = F[t - 1] @ column_ar.T
        gmean = G[t - 1] @ row_ar.T
        mean[t] = fmean @ column.T + row @ gmean.T
        F[t] = fmean + rng.normal(size=(p, r)) * column_scale
        G[t] = gmean + rng.normal(size=(q, c)) * row_scale
    F, G = F[200:], G[200:]
    signal = F @ column.T + row @ G.transpose(0, 2, 1)
    observations = signal + noise_std * rng.normal(size=signal.shape)
    return AdditiveDynamicData(
        observations, signal, mean[200:], row, column, F, G, row_ar, column_ar
    )


def monitoring_data(
    training,
    horizon,
    *,
    shape=(20, 15),
    regime="null",
    seed=672,
    change_step=None,
    noise_std=0.5,
):
    """Rank-one strong matrix factors with a prespecified monitoring event.

    change_step is one-based within monitoring: data[:training+change_step-1]
    are pre-event. Null and same-space-volatility do not change row spaces.
    The volatility scenario intentionally violates constant second moments;
    it is a negative control, not an exact stationary-null calibration design.
    """
    if regime not in (
        "null",
        "space-switch",
        "factor-increase",
        "same-space-volatility",
    ):
        raise ValueError("unknown monitoring regime")
    if change_step is None:
        change_step = horizon // 3 + 1
    if not 1 <= change_step <= horizon:
        raise ValueError("change_step must lie within horizon")
    rng = np.random.default_rng(seed)
    p, q = shape
    total = training + horizon
    row = np.linalg.qr(rng.normal(size=(p, 2)))[0] * np.sqrt(p)
    column = np.linalg.qr(rng.normal(size=(q, 1)))[0] * np.sqrt(q)
    factor = np.zeros((total + 200, 2))
    for t in range(1, len(factor)):
        factor[t] = 0.4 * factor[t - 1] + rng.normal(size=2) * np.sqrt(1 - 0.4**2)
    factor = factor[200:]
    signal = np.einsum("t,i,j->tij", factor[:, 0], row[:, 0], column[:, 0])
    index = training + change_step - 1
    if regime == "space-switch":
        signal[index:] = np.einsum(
            "t,i,j->tij", factor[index:, 0], row[:, 1], column[:, 0]
        )
    elif regime == "factor-increase":
        signal[index:] += np.einsum(
            "t,i,j->tij", factor[index:, 1], row[:, 1], column[:, 0]
        )
    elif regime == "same-space-volatility":
        signal[index:] *= 2
    return signal + noise_std * rng.normal(size=signal.shape), change_step
