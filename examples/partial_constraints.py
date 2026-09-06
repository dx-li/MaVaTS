"""Partial and overlapping multi-term factors; run from the repository root.

Reference: Chen, Tsay and Chen (2020), Constrained Factor Models for
High-Dimensional Matrix-Variate Time Series,
https://doi.org/10.1080/01621459.2019.1584899.
This seeded demonstration is not a replication of the paper's experiments.
"""

import numpy as np

from mavats.constrained import (
    fit_constrained_factor,
    fit_multiterm_constrained_factor,
    fit_partial_constrained_factor,
)
from mavats.metrics import relative_frobenius_error


def main():
    rng = np.random.default_rng(710)
    n, p, q = 260, 7, 6
    row = np.linalg.qr(rng.normal(size=(p, p)))[0]
    col = np.linalg.qr(rng.normal(size=(q, q)))[0]
    load_row, load_col = row[:, [0, 1, 3]], col[:, [0, 2, 3]]
    factors = rng.normal(size=(n, 3, 3))
    for t in range(1, n):
        factors[t] += 0.7 * factors[t - 1]
    truth = load_row @ factors @ load_col.T
    X = truth + 0.15 * rng.normal(size=truth.shape)
    train = 200
    options = dict(row_constraints=row[:, :3], column_constraints=col[:, :2])
    partial = fit_partial_constrained_factor(
        X[:train],
        **options,
        row_ranks=(2, 1),
        column_ranks=(1, 2),
        lags=2,
    )
    full = fit_constrained_factor(X[:train], (2, 1), **options, lags=2)
    heldout = partial.inverse_transform(partial.transform(X[train:]))
    print(
        "Partial model training signal error:",
        round(relative_frobenius_error(truth[:train], partial.signal), 4),
    )
    print(
        "Fully constrained training signal error:",
        round(relative_frobenius_error(truth[:train], full.signal), 4),
    )
    print(
        "Held-out contemporaneous denoising error (not forecasting):",
        round(relative_frobenius_error(truth[train:], heldout), 4),
    )

    # Nonorthogonal spans, not a sum of independent signal projections.
    r1, c1 = row[:, :2], col[:, :2]
    r2 = 0.7 * r1 + np.sqrt(1 - 0.7**2) * row[:, 2:4]
    c2 = 0.7 * c1 + np.sqrt(1 - 0.7**2) * col[:, 2:4]
    constraints = [(r1, c1), (r2, c2)]
    terms = [
        r1[:, :1] @ factors[:, :1, :1] @ c1[:, :1].T,
        r2[:, :1] @ factors[:, 1:2, 1:2] @ c2[:, :1].T,
    ]
    truth = sum(terms)
    X = truth + 0.08 * rng.normal(size=truth.shape)
    multi = fit_multiterm_constrained_factor(X[:train], constraints, [(1, 1), (1, 1)])
    print(
        "Overlapping multi-term training signal error:",
        round(relative_frobenius_error(truth[:train], multi.signal), 4),
    )
    print("Joint score design condition:", round(multi.score_condition, 3))
    print(
        "Term projection methods:",
        [d["estimator"] for d in multi.identification_diagnostics],
    )


if __name__ == "__main__":
    main()
