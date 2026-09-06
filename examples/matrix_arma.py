"""Conditional rank-one MARMA with honest local-optimization diagnostics.

Reference: Tsay (2024), Matrix-Variate Time Series Analysis: A Brief Review
and Some New Developments, https://doi.org/10.1111/insr.12558, Eq. (4), (30).
"""

import numpy as np

from mavats.marma import MARMAParameters, fit_marma, simulate_marma


def main():
    parameters = MARMAParameters(
        ar_left=[[[0.65, 0.12], [0.0, 0.45]]],
        ar_right=[[[0.7, 0.0], [-0.1, 0.5]]],
        ma_left=[[[0.6, 0.08], [0.0, 0.4]]],
        ma_right=[[[0.7, 0.0], [0.1, 0.5]]],
        intercept=[[0.1, -0.1], [0.0, 0.15]],
        row_covariance=[[1.0, 0.2], [0.2, 0.7]],
        column_covariance=[[0.8, -0.15], [-0.15, 1.2]],
    )
    data = simulate_marma(370, parameters, burnin=300, random_state=2026).observations
    for method in ("ls", "mle"):
        fit = fit_marma(data[:350], method=method, n_starts=1, max_iter=250)
        run = fit.runs[fit.selected_start]
        predictions = np.array(
            [
                fit.forecast(1, history=data[:origin])[0]
                for origin in range(350, len(data))
            ]
        )
        mse = np.mean((predictions - data[350:]) ** 2)
        print(
            f"{method.upper()}: converged={fit.converged}, feasible={run.feasible}, "
            f"iterations={fit.n_iter}, warm-up={run.initialization_iterations}, "
            f"AR radius={fit.diagnostics.ar_spectral_radius:.3f}, "
            f"MA radius={fit.diagnostics.ma_spectral_radius:.3f}, "
            f"held-out entrywise MSE={mse:.3f}"
        )
        assert fit.forecast_covariance(3).shape == (3, 4, 4)
    print("Local fits only; gauges and admissibility do not certify identification.")


if __name__ == "__main__":
    main()
