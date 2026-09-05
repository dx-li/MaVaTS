"""Reproducible sparse MAR(1) EMVS fit and conditional-mean forecast.

Run with ``python examples/sparse_mar.py`` after installing MaVaTS.
"""

import numpy as np

from mavats.sparse import fit_sparse_mar


def main():
    rng = np.random.default_rng(4)
    A, B = np.diag([0.8, 0.7, 0.65]), np.diag([0.7, 0.8])
    X = np.zeros((1300, 3, 2))
    for t in range(1, len(X)):
        X[t] = A @ X[t - 1] @ B.T + rng.normal(size=(3, 2))
    train, test = X[100:1200], X[1200:]
    fit = fit_sparse_mar(train, spike_variance=0.001, inclusion_prior=(1, 1))
    # These probabilities condition on the fitted mode; they are not MCMC
    # marginal inclusion probabilities, nor frequentist significance tests.
    print("Converged:", fit.converged, "iterations:", fit.n_iter)
    print("Row support:\n", fit.row_support)
    print("Column support:\n", fit.column_support)
    print("Conditional row slab probabilities:\n", fit.row_inclusion)
    print("One-step conditional forecast:\n", fit.forecast(1)[0])
    history = np.concatenate((train[-1:], test[:-1]))
    predictions = fit.A @ history @ fit.B.T
    print("Held-out rolling one-step MSE:", np.mean((test - predictions) ** 2))
    print("VAR coefficient error:", np.linalg.norm(fit.coefficients[0] - np.kron(B, A)))


if __name__ == "__main__":
    main()
