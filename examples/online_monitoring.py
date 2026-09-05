"""Incremental monitoring with a planted new row factor.

Run from the repository root: python -m examples.online_monitoring
This one seeded illustration is not a false-alarm or power study.
"""

import json

import numpy as np

from mavats.monitoring import MatrixFactorMonitor


def main():
    rng = np.random.default_rng(120)
    m, horizon, p, q = 30, 40, 20, 12
    row = np.linalg.qr(rng.normal(size=(p, 2)))[0] * np.sqrt(p)
    column = np.linalg.qr(rng.normal(size=(q, 2)))[0] * np.sqrt(q)
    factors = rng.normal(size=(m + horizon, 2))
    observations = np.einsum("i,t,j->tij", row[:, 0], factors[:, 0], column[:, 0])
    observations += 0.2 * rng.normal(size=(m + horizon, p, q))
    # Known ranks in this simulation. Real-data tuning must use training only.
    monitor = MatrixFactorMonitor(
        observations[:m],
        rank=1,
        projection_rank=2,
        horizon=horizon,
        random_state=22,
    )
    # First ten new observations remain in the stable regime.
    monitor.update_many(observations[m : m + 10])
    # A JSON checkpoint contains the window and local RNG, not future data.
    monitor = MatrixFactorMonitor.from_state(
        json.loads(json.dumps(monitor.state_dict(), allow_nan=False))
    )
    print("Calibration:", monitor.calibration, "(asymptotic, not finite-sample exact)")
    for t in range(m + 10, m + horizon):
        matrix = observations[t] + np.outer(row[:, 1], column[:, 0]) * factors[t, 1]
        step = monitor.update(matrix)
        if step.alarm:
            print("First changed monitoring step:", 11)
            print("Alarm monitoring step:", step.monitoring_index)
            print("Alarm absolute observation:", step.observation_index)
            print("Normalized statistic:", round(step.statistic, 3))
            break
    print("Stopped:", monitor.stop_reason)


if __name__ == "__main__":
    main()
