"""Fit response reducing spaces, not just low-rank AR coefficients.

Samadi and De Alwis (2026), Envelope Matrix Autoregressive Models,
https://doi.org/10.1080/07350015.2025.2537404, equations (13), (15)--(16).
This small synthetic illustration is not a replication of the paper's studies.
"""

import numpy as np

from mavats import fit_envelope_mar
from mavats.simulation import simulate_mar


def main():
    rng = np.random.default_rng(23)
    r = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    c = np.linalg.qr(rng.normal(size=(4, 4)))[0]
    # Both past material and past immaterial coordinates can be predictors.
    a = r[:, :2] @ np.array([[0.7, 0.1, 0.2], [0.0, 0.5, -0.1]]) @ r.T
    b = c[:, :2] @ np.array([[0.6, 0.0, 0.15, 0.1], [0.1, 0.5, 0.0, -0.1]]) @ c.T
    data = simulate_mar(
        350,
        a,
        b,
        row_cov=r @ np.diag([0.5, 1.0, 3.0]) @ r.T,
        column_cov=c @ np.diag([0.4, 0.8, 2.0, 3.0]) @ c.T,
        intercept=np.full((3, 4), 0.1),
        random_state=24,
    )
    model = fit_envelope_mar(data[:300], (2, 2))
    forecasts = np.array(
        [model.forecast(1, history=data[:t])[0] for t in range(300, 350)]
    )
    print(
        f"EMAR: converged={model.converged}, warmup={model.warmup_converged}, "
        f"sweeps={model.n_iter}, stop={model.stop_reason}"
    )
    print(
        f"Spectral radius={model.spectral_radius:.3f}; "
        f"held-out entrywise MSE={np.mean((data[300:] - forecasts)**2):.3f}"
    )
    for name, basis, covariance in (
        ("row", model.row_envelope, model.row_covariance),
        ("column", model.column_envelope, model.column_covariance),
    ):
        projector = basis @ basis.T
        leakage = np.linalg.norm(
            projector @ covariance @ (np.eye(len(basis)) - projector)
        )
        print(f"{name} material/complement covariance leakage={leakage:.2e}")
    assert model.forecast(3).shape == (3, 3, 4)
    print(
        "Dimensions were supplied, not selected. Local convergence is not global optimality."
    )


if __name__ == "__main__":
    main()
