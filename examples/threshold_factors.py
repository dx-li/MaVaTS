"""Two regimes with unequal row/column ranks and a searched threshold.

Run from the repository root: ``python examples/threshold_factors.py``.
The threshold variable here is observed exogenously. A forecasting application
must supply a threshold variable already available at each forecast origin.
"""

import numpy as np

from mavats.threshold import fit_threshold_factors


def main():
    rng = np.random.default_rng(48)
    n = 400
    z = rng.uniform(-1, 1, n)
    row = np.linalg.qr(rng.normal(size=(6, 3)))[0]
    column = np.linalg.qr(rng.normal(size=(5, 3)))[0]
    loadings = ((row[:, :1], column[:, :2]), (row[:, 1:3], column[:, 2:3]))
    cores = np.zeros((n, 2, 2))
    X = np.empty((n, 6, 5))
    for t in range(n):
        if t:
            cores[t] = 0.8 * cores[t - 1] + rng.normal(size=(2, 2))
        a, b = loadings[int(z[t] >= 0)]
        X[t] = a @ cores[t, : a.shape[1], : b.shape[1]] @ b.T
    X += 0.04 * rng.normal(size=X.shape)

    # Estimate all four ranks from extreme-subset lag moments and then hold
    # them fixed while profiling the threshold on training observations only.
    train = 320
    fitted = fit_threshold_factors(X[:train], z[:train], max_lag=2)
    held_out_signal = fitted.reconstruct(X[train:], z[train:])
    print(f"Estimated threshold: {fitted.threshold:.4f} (true: 0)")
    print(f"Regime ranks: {fitted.ranks}")
    print(f"Profile candidates: {len(fitted.candidates)}")
    print(f"Held-out projection MSE: {np.mean((X[train:] - held_out_signal)**2):.6f}")

    # Known thresholds use the full corresponding regimes to estimate spaces.
    known = fit_threshold_factors(
        X[:train], z[:train], ((1, 2), (2, 1)), threshold=0, max_lag=2
    )
    assert known.ranks == ((1, 2), (2, 1))
    assert np.isfinite(held_out_signal).all()


if __name__ == "__main__":
    main()
