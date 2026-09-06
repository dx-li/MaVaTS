"""Paired EMAR forecasts and identified operator/covariance errors.

Samadi and De Alwis (2026), https://doi.org/10.1080/07350015.2025.2537404,
equation (13), supplies the envelope model. These controlled simulation
designs, variance reversals and covariance violations are package experiments,
not replications of the paper's tables. Tashman (2000),
https://doi.org/10.1016/S0169-2070(00)00065-0, motivates rolling-origin scores.
All estimation uses the initial training prefix; no test-set tuning occurs.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.run import _measure, environment, source_provenance, source_snapshot
from mavats import fit_envelope_mar, fit_mar, fit_var
from mavats.metrics import relative_frobenius_error, subspace_distance
from mavats.simulation import simulate_mar


def _scenario(count, *, order, regime, seed, shape):
    rng = np.random.default_rng(seed)
    m, n = shape
    qr, qc = [np.linalg.qr(rng.normal(size=(d, d)))[0] for d in shape]
    left, right = [], []
    for q, coefficients in ((qr, left), (qc, right)):
        for _ in range(order):
            eta = rng.normal(size=(2, len(q)))
            eta[:, 2:] *= 0.25
            coefficients.append(q[:, :2] @ eta @ q.T)
    left, right = np.array(left), np.array(right)
    contraction = sum(
        np.linalg.norm(a, 2) * np.linalg.norm(b, 2) for a, b in zip(left, right)
    )
    right *= 0.7 / contraction
    covariances = []
    for q in (qr, qc):
        d = len(q)
        variances = np.r_[0.5, 1.5, np.linspace(4, 6, d - 2)]
        if regime == "high-material":
            variances = 1 / variances
        elif regime == "isotropic":
            variances = np.ones(d)
        variances *= d / variances.sum()  # constant total innovation variance
        block = np.diag(variances)
        if regime == "covariance-coupled":
            cross = rng.normal(size=(2, d - 2))
            cross *= 0.3 / np.linalg.norm(cross, 2)
            correlation = np.eye(d)
            correlation[:2, 2:], correlation[2:, :2] = cross, cross.T
            block = (
                np.sqrt(variances)[:, None] * correlation * np.sqrt(variances)[None, :]
            )
        covariances.append(q @ block @ q.T)
    intercept = rng.normal(scale=0.1, size=shape)
    observations = simulate_mar(
        count,
        left,
        right,
        row_cov=covariances[0],
        column_cov=covariances[1],
        intercept=intercept,
        burnin=300,
        random_state=seed + 1_000_000,
    )
    return dict(
        observations=observations,
        left=left,
        right=right,
        intercept=intercept,
        row=covariances[0],
        column=covariances[1],
        bases=(qr[:, :2], qc[:, :2]),
    )


class _KnownEMAR:
    """Known-parameter conditional means and exact forecast noise covariance.

    References
    ----------
    Samadi and De Alwis (2026), equation (13),
    https://doi.org/10.1080/07350015.2025.2537404.
    Dense companion/impulse evaluation of the specified model, not estimation.
    """

    converged = True
    n_iter = 0

    def __init__(self, data):
        self.intercept = data["intercept"]
        self.coefficients = np.array(
            [np.kron(b, a) for a, b in zip(data["left"], data["right"])]
        )
        self.covariance = np.kron(data["column"], data["row"])

    def forecast(self, steps, history):
        state = [x.ravel(order="F") for x in history[-len(self.coefficients) :]]
        result = []
        for _ in range(steps):
            value = self.intercept.ravel(order="F").copy()
            for lag, phi in enumerate(self.coefficients, 1):
                value += phi @ state[-lag]
            state.append(value)
            result.append(value.reshape(self.intercept.shape, order="F"))
        return np.array(result)

    def noise_floor(self, steps):
        impulses = [np.eye(self.intercept.size)]
        covariance = self.covariance.copy()
        result = [np.trace(covariance) / self.intercept.size]
        for h in range(1, steps):
            impulse = sum(
                self.coefficients[lag - 1] @ impulses[h - lag]
                for lag in range(1, min(h, len(self.coefficients)) + 1)
            )
            impulses.append(impulse)
            covariance += impulse @ self.covariance @ impulse.T
            result.append(np.trace(covariance) / self.intercept.size)
        return np.asarray(result)


def _scores(fit, data, train_count, oracle):
    values = data["observations"]
    origins = range(train_count, len(values) - 4)
    predictions = np.array([fit.forecast(5, history=values[:t]) for t in origins])
    truths = np.array([oracle.forecast(5, history=values[:t]) for t in origins])
    observed = np.array([values[t : t + 5] for t in origins])
    scores = dict(origins=len(origins))
    floor = oracle.noise_floor(5)
    for h in (1, 3, 5):
        scores[f"h{h}_mse"] = float(
            np.mean((observed[:, h - 1] - predictions[:, h - 1]) ** 2)
        )
        scores[f"h{h}_conditional_mean_mse"] = float(
            np.mean((truths[:, h - 1] - predictions[:, h - 1]) ** 2)
        )
        scores[f"h{h}_noise_floor"] = float(floor[h - 1])
    scores["operator_error"] = relative_frobenius_error(
        oracle.coefficients, fit.coefficients
    )
    if getattr(fit, "row_covariance", None) is not None:
        scores["covariance_error"] = relative_frobenius_error(
            oracle.covariance, np.kron(fit.column_covariance, fit.row_covariance)
        )
        scores["log_likelihood"] = fit.log_likelihood
    if hasattr(fit, "envelope_dims"):
        scores.update(
            envelope_dims=list(fit.envelope_dims),
            stop_reason=fit.stop_reason,
            warmup_converged=fit.warmup_converged,
            inner_converged=fit.inner_converged,
            likelihood_converged=fit.likelihood_converged,
            row_space_error=subspace_distance(data["bases"][0], fit.row_envelope),
            column_space_error=subspace_distance(data["bases"][1], fit.column_envelope),
            last_row_gradient=fit.optimization_history[-1]["row"]["gradient_norm"],
            last_column_gradient=fit.optimization_history[-1]["column"][
                "gradient_norm"
            ],
        )
    return scores


def run_suite(*, quick=False, repeats=10, seed=20260905):
    rows = []
    shape, train_count = ((3, 4), 120) if quick else ((4, 5), 350)
    regimes = (
        ("high-immaterial", "covariance-coupled")
        if quick
        else ("high-immaterial", "high-material", "isotropic", "covariance-coupled")
    )
    for replicate in range(repeats):
        for order in (1, 2):
            for regime in regimes:
                current = seed + 100 * replicate + order
                data = _scenario(
                    train_count + 35,
                    order=order,
                    regime=regime,
                    seed=current,
                    shape=shape,
                )
                train = data["observations"][:train_count]
                oracle = _KnownEMAR(data)
                metadata = dict(
                    replicate=replicate,
                    seed=current,
                    order=order,
                    regime=regime,
                    shape=list(shape),
                    n_train=train_count,
                    n_test=35,
                    nominal_dims=[2, 2],
                    nominal_spaces_reduce_covariance=regime != "covariance-coupled",
                    envelope_space_errors_target="nominal-output-spaces-not-necessarily-true-envelopes",
                    design="controlled-variance-reversal-and-coupling-not-paper-replication",
                )
                methods = []
                for name, dims in (
                    ("nominal", (2, 2)),
                    ("under", (1, 2)),
                    ("over", (shape[0] - 1, shape[1] - 1)),
                    ("full", shape),
                ):
                    options = dict(
                        envelope_dims=dims,
                        order=order,
                        max_iter=40 if quick else 100,
                        tol=1e-8,
                        inner_max_iter=200,
                        inner_tol=1e-6,
                        inner_starts=3,
                        random_state=current,
                        fit_intercept=True,
                        warmup_max_iter=100,
                    )
                    methods.append(
                        (
                            f"emar-{name}",
                            lambda options=options: fit_envelope_mar(train, **options),
                            options,
                        )
                    )
                for name, method, ranks in (
                    ("mar-mle", "mle", None),
                    ("mar-als", "als", None),
                    ("rr-mar-als", "als", (2, 2)),
                ):
                    options = dict(
                        method=method,
                        ranks=ranks,
                        order=order,
                        max_iter=200,
                        fit_intercept=True,
                        covariance_floor=0,
                        tol=1e-8,
                    )
                    methods.append(
                        (
                            name,
                            lambda options=options: fit_mar(train, **options),
                            options,
                        )
                    )
                methods += [
                    (
                        "var",
                        lambda: fit_var(train, order=order),
                        dict(order=order, intercept=True),
                    ),
                    (
                        "known-parameter-oracle",
                        lambda: oracle,
                        dict(extra_information="true-parameters"),
                    ),
                ]
                for name, method, options in methods:
                    rows.append(
                        _measure(
                            name,
                            method,
                            lambda fit: _scores(fit, data, train_count, oracle),
                            dict(metadata, fit_options=options),
                        )
                    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--output", type=Path, default=Path("benchmark-envelope.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    snapshot = source_snapshot(__file__)
    rows = run_suite(quick=args.quick, repeats=args.repeats, seed=args.seed)
    report = dict(
        schema_version=1,
        configuration=vars(args) | {"output": str(args.output)},
        environment=environment(),
        results=rows,
        **source_provenance(snapshot),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    errors = sum(row["status"] != "ok" for row in rows)
    unconverged = sum(
        not row.get("converged", False) for row in rows if row["status"] == "ok"
    )
    print(f"Saved {len(rows)} runs; {errors} errors; {unconverged} unconverged")
    if errors or report["source_changed_during_run"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
