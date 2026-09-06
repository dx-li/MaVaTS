"""Multiterm TenAR(2): independent dense simulation, LS/MLE and held-out forecasts.

Run ``python examples/tensor_autoregression.py`` after installing MaVaTS.
The data generator uses full vectorized transitions, not estimator contractions.
"""

import numpy as np

from mavats.tensor_autoregression import fit_tensor_ar


def main():
    rng = np.random.default_rng(130)
    shape = (2, 3, 2)
    dimension = int(np.prod(shape))
    transitions = []
    for weights in ((0.45, 0.25), (0.15,)):
        transition = np.zeros((dimension, dimension))
        for weight in weights:
            matrices = []
            for d in shape:
                a = np.eye(d) + 0.15 * rng.normal(size=(d, d))
                matrices.append(a / np.linalg.norm(a, ord=2))
            operator = np.ones((1, 1))
            for a in reversed(matrices):
                operator = np.kron(operator, a)
            transition += weight * operator
        transitions.append(transition)
    covariance = np.ones((1, 1))
    for d in reversed(shape):
        correlation = 0.4 ** np.abs(np.arange(d)[:, None] - np.arange(d)[None, :])
        covariance = np.kron(covariance, correlation)
    # Sum of operator-norm bounds .85 < 1 ensures this chosen DGP is stable.
    values = np.zeros((1122, dimension))
    innovations = rng.normal(size=values.shape) @ np.linalg.cholesky(covariance).T
    for t in range(2, len(values)):
        values[t] = (
            transitions[0] @ values[t - 1]
            + transitions[1] @ values[t - 2]
            + innovations[t]
        )
    X = np.stack([v.reshape(shape, order="F") for v in values[122:]])
    train, test = X[:800], X[800:]
    for method in ("ls", "mle"):
        fit = fit_tensor_ar(
            train,
            terms=(2, 1),
            method=method,
            n_starts=2,
            max_iter=150,
            random_state=31,
        )
        print(method, "fit converged:", fit.converged, "sweeps:", fit.n_iter)
        print(
            "  selected start:",
            fit.selected_start,
            "companion radius:",
            fit.spectral_radius,
        )
        projection = fit.runs[fit.selected_start].projection
        print("  separate local projection flags:", [p.converged for p in projection])
        print(
            "  identification diagnostics:",
            [d["identification"] for d in fit.identification_diagnostics],
        )
        history = np.concatenate((train[-2:], test))
        predictions = np.concatenate(
            [fit.forecast(1, history=history[t : t + 2]) for t in range(len(test))]
        )
        print("  held-out one-step MSE:", np.mean((test - predictions) ** 2))
        print("  zero forecast MSE:", np.mean(test**2))
        print("  four-step origin forecast shape:", fit.forecast(4).shape)
        if method == "mle":
            print("  conditional Gaussian log likelihood:", fit.log_likelihood)
            print("  covariance regularized:", fit.covariance_regularized)


if __name__ == "__main__":
    main()
