"""Fit a matrix error-correction model and forecast held-out levels.

Run: python -m examples.cointegrated_mar
The inline generator uses a direct recurrence, independently of the fitter.
"""

import numpy as np

from mavats.cointegration import fit_cmar


def main():
    rng = np.random.default_rng(32)
    p, q, training, test = 3, 2, 400, 30
    row = np.linalg.qr(rng.normal(size=(p, 1)))[0]
    column = np.linalg.qr(rng.normal(size=(q, 1)))[0]
    a1, a2 = -0.65 * row @ row.T, column @ column.T
    constant = rng.normal(size=(p, q)) * 0.015
    x = np.zeros((training + test + 50, p, q))
    for t in range(2, len(x)):
        difference = a1 @ x[t - 1] @ a2.T
        difference += 0.08 * (x[t - 1] - x[t - 2]) + constant
        x[t] = x[t - 1] + difference + 0.3 * rng.normal(size=(p, q))
    x = x[50:]
    # The true ranks are known in this illustration, not selected on test data.
    for method in ("ls", "mle"):
        fit = fit_cmar(
            x[:training],
            (1, 1),
            difference_lags=1,
            method=method,
            n_starts=2,
            random_state=84,
        )
        predictions = np.concatenate(
            [
                fit.forecast(1, history=x[t - 2 : t])
                for t in range(training, training + test)
            ]
        )
        diagnostics = fit.i1_diagnostics()
        print(method.upper(), "converged:", fit.converged, "iterations:", fit.n_iter)
        print(
            "  Held-out one-step level MSE:",
            round(float(np.mean((x[training:] - predictions) ** 2)), 5),
        )
        print("  Cointegration rank:", fit.cointegration_rank)
        print("  Roots near +1:", diagnostics.unit_roots_observed)
        print(
            "  Fitted I(1) coefficient checks:",
            diagnostics.compatible,
            "(not a statistical test of integration)",
        )
    print(
        "Random-walk forecast MSE:",
        round(float(np.mean((x[training:] - x[training - 1 : -1]) ** 2)), 5),
    )


if __name__ == "__main__":
    main()
