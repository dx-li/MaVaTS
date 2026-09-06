"""Additive two-way dynamic factors and held-out one-step predictions.

Run ``python examples/two_way_dynamic.py`` after installing MaVaTS.
"""

import numpy as np

from mavats.dynamic import fit_two_way_dynamic


def main():
    rng = np.random.default_rng(10)
    n, m = 6, 5
    Lam = np.sqrt(n) * np.linalg.qr(rng.normal(size=(n, 1)))[0]
    L = np.sqrt(m) * np.linalg.qr(rng.normal(size=(m, 2)))[0]
    F = np.zeros((650, n, 2))
    G = np.zeros((650, m, 1))
    for t in range(1, len(F)):
        F[t] = F[t - 1] * [0.8, 0.3] + rng.normal(size=(n, 2)) * [0.7, 0.5]
        G[t] = G[t - 1] * 0.6 + rng.normal(size=(m, 1)) * 0.5
    signal = F @ L.T + Lam @ G.transpose(0, 2, 1)
    X = signal + 0.3 * rng.normal(size=signal.shape)
    train, test = X[100:600], X[600:]
    fit = fit_two_way_dynamic(train, orders=(1, 1))
    print("Selected (row, column) ranks:", fit.ranks)
    print("Rank selector converged:", fit.rank_converged)
    print("Quasi-likelihood converged:", fit.converged, "iterations:", fit.n_iter)
    print("Column-factor AR coefficients:", fit.column_ar)
    print("Row-factor AR coefficients:", fit.row_ar)
    print("Working noise variance:", fit.noise_variance)
    print("One-step forecast:\n", fit.forecast(1)[0])
    # Every predictor uses the immediately preceding observed matrix. Neither
    # factor loadings nor AR parameters are refitted using held-out data.
    history = np.concatenate((train[-1:], test[:-1]))
    predictions = np.concatenate(
        [fit.forecast(1, history=one[None]) for one in history]
    )
    print("Held-out one-step MSE:", np.mean((test - predictions) ** 2))
    print("Zero forecast MSE:", np.mean(test**2))
    print("Training signal MSE:", np.mean((fit.signal - signal[100:600]) ** 2))


if __name__ == "__main__":
    main()
