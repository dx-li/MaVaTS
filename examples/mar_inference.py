"""Intervals for MAR transition coefficients and a structural specification test."""

from mavats.inference import mar_inference, mar_specification_test
from mavats.simulation import simulate_mar


def main():
    X = simulate_mar(
        800, [[0.7, 0.2], [-0.1, 0.5]], [[0.8, -0.15], [0.1, 0.4]], random_state=7
    )
    for method in ("projection", "als", "mle"):
        result = mar_inference(X, method=method)
        lower, upper = result.confidence_interval()
        print(
            method,
            "first transition entry and marginal 95% interval:",
            result.operator[0, 0],
            (lower[0, 0], upper[0, 0]),
        )
    result = mar_specification_test(X)
    print("Kronecker specification test:", result.statistic, "p-value:", result.pvalue)
    print("Intervals assume stationary zero-mean MAR(1) with iid innovations.")
    print(
        "This seeded null example rejects: a true null can be rejected. See the repeated size/coverage study."
    )


if __name__ == "__main__":
    main()
