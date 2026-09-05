"""Forecast matrices, recover factor spaces and fit tensor time series."""

import numpy as np

from mavats import (
    fit_alpha_pca,
    fit_mar,
    fit_matrix_kendall,
    fit_projected_pca,
    fit_tensor_factor,
)
from mavats.metrics import relative_frobenius_error
from mavats.model_selection import rolling_forecast
from mavats.simulation import simulate_factor, simulate_mar


def main():
    x = simulate_mar(
        250, np.diag([0.6, 0.8]), np.diag([0.7, 0.5, 0.6]), random_state=42
    )
    model = fit_mar(x, method="als")
    print("MAR forecast shape:", model.forecast(5).shape)
    print(
        "MAR converged / spectral radius:",
        model.converged,
        round(model.spectral_radius, 3),
    )
    evaluation = rolling_forecast(
        x, lambda train: fit_mar(train), initial_train_size=200, step=10
    )
    print("Rolling forecast MSE:", round(evaluation.mse, 3))
    data = simulate_factor(150, (8, 10), (2, 3), random_state=42)
    for estimator in (fit_alpha_pca, fit_projected_pca, fit_matrix_kendall):
        result = estimator(data.observations, ranks=(2, 3))
        np.testing.assert_allclose(
            result.inverse_transform(result.transform(data.observations)), result.signal
        )
        print(
            estimator.__name__,
            "signal error:",
            round(relative_frobenius_error(data.signal, result.signal), 3),
        )
    tensor = simulate_factor(150, (5, 6, 4), (2, 2, 2), random_state=42)
    result = fit_tensor_factor(
        tensor.observations, ranks=(2, 2, 2), method="tipup", iterative=True
    )
    print("Tensor core shape:", result.factors.shape)


if __name__ == "__main__":
    main()
