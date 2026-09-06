"""Entrywise IHR on cellwise contamination; no oracle loading initialization."""

import numpy as np

from mavats.factors import fit_projected_pca
from mavats.ihr import fit_ihr_factor, select_ihr_ranks


def main():
    rng = np.random.default_rng(105)
    row = rng.normal(size=(10, 2))
    column = rng.normal(size=(8, 2))
    factors = rng.normal(size=(100, 2, 2))
    signal = row @ factors @ column.T
    X = signal + 0.2 * rng.standard_t(3, size=signal.shape)
    mask = rng.random(X.shape) < 0.01
    X[mask] += rng.choice([-1.0, 1.0], mask.sum()) * 10
    train, future = X[:80], X[80:]
    fit = fit_ihr_factor(train, (2, 2), inner_max_iter=300, inner_tol=1e-8)
    ordinary = fit_projected_pca(train, (2, 2), max_iter=100)
    print(f"Method: {fit.method}; converged: {fit.converged}; sweeps: {fit.n_iter}")
    print(
        f"Stopping reason: {fit.stopping_reason}; frozen entry threshold: {fit.threshold:.4f}"
    )
    for name, estimate in (("IHR", fit.signal), ("Least squares", ordinary.signal)):
        error = np.linalg.norm(estimate - signal[:80]) / np.linalg.norm(signal[:80])
        print(f"{name} relative signal error: {error:.4f}")
    future_core, diagnostics = fit.transform(future, return_diagnostics=True)
    predicted_signal = fit.inverse_transform(future_core)
    print(
        f"Robust held-out reconstruction error: {np.linalg.norm(predicted_signal-signal[80:])/np.linalg.norm(signal[80:]):.4f}"
    )
    print(
        f"Unfinished transform regressions: {sum(not d.converged for d in diagnostics)}"
    )
    selection = select_ihr_ranks(
        train,
        (3, 3),
        method="threshold",
        threshold=fit.threshold,
        inner_max_iter=300,
        inner_tol=1e-8,
        max_iter=60,
    )
    print(
        f"Overfit rank-threshold selection: {selection.ranks}; pilot converged: {selection.pilot.converged}"
    )
    print(
        "Reconstruction uses each observed future matrix; it is not a time-series forecast."
    )


if __name__ == "__main__":
    main()
