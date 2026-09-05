"""Executable coverage of individual matrix estimators and model selection.

Run ``python -m examples.method_gallery``. The synthetic examples use known
models to explain assumptions; ranks supplied from the generator are oracle
choices, not automatic model-selection claims.
"""

import numpy as np

from mavats import (
    fit_alpha_pca,
    fit_constrained_factor,
    fit_cp_factor,
    fit_huber_factor,
    fit_lagged_factor,
    fit_mar,
    fit_matrix_kendall,
    fit_projected_pca,
    fit_tensor_factor,
    select_mar_rank,
)
from mavats.metrics import relative_frobenius_error
from mavats.simulation import simulate_factor, simulate_mar


def main():
    a, b = np.diag([0.7, 0.4]), np.diag([0.7, 0.5, 0.3])
    series = simulate_mar(400, a, b, row_cov=[[1.0, 0.4], [0.4, 1.0]], random_state=91)
    for method in ("projection", "als", "mle"):
        result = fit_mar(series, method=method, fit_intercept=True)
        print(
            method, "converged:", result.converged, "next mean:", result.forecast(1)[0]
        )
        if method == "mle":
            print(
                "Estimated row/column covariance shapes:",
                result.row_covariance.shape,
                result.column_covariance.shape,
            )

    left = np.stack([a * 0.6, a * 0.2])
    right = np.stack([b, b])
    multilag = simulate_mar(400, left, right, random_state=91)
    result = fit_mar(multilag, order=2)
    print("MAR(2) forecast shape:", result.forecast(3).shape)

    low_rank = simulate_mar(
        500, np.diag([0.7, 0.0]), np.diag([0.8, 0.4, 0.0]), random_state=91
    )
    reduced = fit_mar(low_rank, ranks=(1, 2))
    selection = select_mar_rank(low_rank, max_ranks=(2, 3))
    print(
        "Reduced-rank forecast shape:",
        reduced.forecast(2).shape,
        "EBIC selected:",
        selection.ranks,
    )

    data = simulate_factor(200, (8, 10), (2, 2), random_state=91)
    for alpha in (-1.0, 0.0, 1.0):
        result = fit_alpha_pca(data.observations, (2, 2), alpha=alpha)
        print(
            "alpha:",
            alpha,
            "signal error:",
            relative_frobenius_error(data.signal, result.signal),
        )
    for estimator in (fit_projected_pca, fit_lagged_factor, fit_matrix_kendall):
        result = estimator(data.observations, (2, 2))
        print(
            estimator.__name__,
            "signal error:",
            relative_frobenius_error(data.signal, result.signal),
        )

    contaminated = data.observations.copy()
    contaminated[::20] += np.random.default_rng(91).normal(
        scale=30, size=contaminated[::20].shape
    )
    huber = fit_huber_factor(contaminated, (2, 2))
    print("Huber threshold / converged:", huber.threshold, huber.converged)

    # Known constraint spans provide additional information; this is an oracle
    # illustration, not an equally informed comparison to unconstrained fitting.
    constrained = fit_constrained_factor(
        data.observations,
        (2, 2),
        row_constraints=data.loadings[0],
        column_constraints=data.loadings[1],
    )
    print(
        "Constrained signal error:",
        relative_frobenius_error(data.signal, constrained.signal),
    )

    # CP models share one scalar time series per rank-one matrix component.
    # Distinct persistence gives an informative generalized eigenproblem.
    rng = np.random.default_rng(14)
    A, B = rng.normal(size=(6, 3)), rng.normal(size=(5, 3))
    A /= np.linalg.norm(A, axis=0)
    B /= np.linalg.norm(B, axis=0)
    scores = rng.normal(size=(200, 3))
    for t in range(1, len(scores)):
        scores[t] += np.array([-0.8, 0.4, 0.9]) * scores[t - 1]
    cp_series = np.einsum("tr,ir,jr->tij", scores, A, B)
    cp = fit_cp_factor(cp_series, rank=3)
    print("CP signal error:", relative_frobenius_error(cp_series, cp.signal))

    tensor = simulate_factor(200, (5, 6, 4), (2, 2, 2), random_state=91)
    for method in ("topup", "tipup"):
        for iterative in (False, True):
            result = fit_tensor_factor(
                tensor.observations, (2, 2, 2), method=method, iterative=iterative
            )
            print(
                "i" * iterative + method,
                "converged:",
                result.converged,
                "signal error:",
                relative_frobenius_error(tensor.signal, result.signal),
            )


if __name__ == "__main__":
    main()
