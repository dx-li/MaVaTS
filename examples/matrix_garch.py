"""Full Matrix GARCH: covariance simulation, local QMLE and held-out filtering.

Run from the repository root with ``python -m examples.matrix_garch``.
This example scores exact one-step conditional covariances chronologically.
It does not label projections of expected states as multistep forecasts.
"""

import numpy as np

from mavats.volatility import (
    MatrixGARCHParameters,
    fit_matrix_garch,
    simulate_matrix_garch,
)


def main():
    truth = MatrixGARCHParameters(
        A0=np.array([[1.0, 0], [0.25, 0.8]]),
        A1=np.array([[0.25, 0.04], [-0.02, 0.18]]),
        A2=np.array([[0.4, 0.03], [0.01, 0.3]]),
        B0=np.array([[1.0, 0], [-0.15, 0.9]]),
        B1=np.array([[0.18, -0.03], [0.02, 0.22]]),
        B2=np.array([[0.35, 0.02], [-0.03, 0.4]]),
        w=0.3,
        alpha=0.15,
        beta=0.6,
    )
    sample = simulate_matrix_garch(320, truth, random_state=105)
    train = 250
    fitted = fit_matrix_garch(
        sample.observations[:train],
        dynamics="full",
        n_starts=2,
        max_iter=180,
        random_state=11,
    )
    # This forecast uses only the terminal training observation and its state.
    first_forecast = fitted.forecast_one()
    held_out = fitted.filter(sample.observations[train:])
    assert np.allclose(first_forecast.row_covariance, held_out.row_covariances[0])
    print(f"Optimizer converged: {fitted.converged}; iterations: {fitted.n_iter}")
    print(
        f"Selected start: {fitted.selected_start}; active bounds: {fitted.active_bounds}"
    )
    print(f"Mean held-out Gaussian NLL: {-held_out.log_likelihood / (320 - train):.4f}")
    print(f"Next trace forecast at training origin: {first_forecast.trace:.4f}")
    print(
        f"Sufficient stationarity bound: {fitted.parameters.sufficient_stationarity_bound:.4f}"
    )
    print(f"Separate spectral bounds: {fitted.parameters.spectral_bounds}")
    print("Sufficient-bound failure does not establish nonstationarity.")


if __name__ == "__main__":
    main()
