"""Held-out additive-dynamic forecasts and matrix volatility covariance scoring."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.run import _measure, environment, source_provenance, source_snapshot
from benchmarks.scenarios import additive_dynamic_data
from mavats.autoregression import fit_mar
from mavats.baselines import fit_var
from mavats.dynamic import fit_two_way_dynamic
from mavats.metrics import (
    mean_squared_error,
    relative_frobenius_error,
    subspace_distance,
)
from mavats.volatility import (
    MatrixGARCHParameters,
    fit_matrix_garch,
    simulate_matrix_garch,
)


def _garch_parameters():
    return MatrixGARCHParameters(
        np.array([[1.0, 0], [0.2, 0.9]]),
        np.array([[0.14, 0.04], [-0.02, 0.12]]),
        np.array([[0.55, 0.04], [0.02, 0.5]]),
        np.array([[1.0, 0], [0.1, 1.1]]),
        np.array([[0.12, -0.03], [0.04, 0.15]]),
        np.array([[0.5, -0.02], [0.03, 0.55]]),
        w=0.5,
        alpha=0.15,
        beta=0.65,
    )


def _gaussian_score(observations, covariances):
    """Dense negative Gaussian log score including constants; lower is better.

    References
    ----------
    Tsay (2024), https://doi.org/10.1111/insr.12558. Conventional Gaussian
    density evaluation, including its required quadratic one-half factor;
    not a separately proposed matrix estimator.
    """
    observations = np.asarray(observations)
    covariances = np.asarray(covariances)
    if observations.ndim != 3 or not len(observations):
        raise ValueError("score observations must be a nonempty matrix series")
    flat = observations.transpose(0, 2, 1).reshape(len(observations), -1)
    if covariances.shape != (len(flat), flat.shape[1], flat.shape[1]):
        raise ValueError("score requires one matching covariance per observation")
    if not np.isfinite(flat).all() or not np.isfinite(covariances).all():
        raise ValueError("score observations and covariances must be finite")
    scores = []
    for x, cov in zip(flat, covariances):
        scale = np.max(np.abs(cov))
        if scale == 0 or not np.allclose(
            cov / scale, cov.T / scale, rtol=1e-12, atol=1e-12
        ):
            raise ValueError(
                "covariance score requires symmetric positive definiteness"
            )
        try:
            chol = np.linalg.cholesky((cov / scale + cov.T / scale) / 2)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "covariance score requires positive definiteness"
            ) from error
        logdet = 2 * np.log(np.diag(chol)).sum() + len(x) * np.log(scale)
        standardized = np.linalg.solve(chol, x / np.sqrt(scale))
        scores.append(
            0.5 * (len(x) * np.log(2 * np.pi) + logdet + standardized @ standardized)
        )
    if not np.isfinite(scores).all():
        raise ValueError("Gaussian score is not representable in float64")
    return float(np.mean(scores))


def run_suite(*, quick=False, repeats=5, seed=893):
    rows = []
    for replicate in range(repeats):
        current = seed + replicate
        for regime in ("diagonal-dynamics", "coupled-dynamics"):
            n, shape = (150, (5, 6)) if quick else (500, (8, 10))
            data = additive_dynamic_data(
                n + 50, shape=shape, ranks=(1, 2), seed=current, regime=regime
            )
            X = data.observations
            train = X[:n]
            actual = X[n:]

            def score(fit):
                predicted = np.concatenate(
                    [fit.forecast(1, history=one[None]) for one in X[n - 1 : -1]]
                )
                result = dict(forecast_mse=mean_squared_error(actual, predicted))
                if hasattr(fit, "F"):
                    result.update(
                        signal_error=relative_frobenius_error(
                            data.signal[:n], fit.signal
                        ),
                        subspace_error=float(
                            np.mean(
                                [
                                    subspace_distance(a, b)
                                    for a, b in zip(
                                        (data.row_loadings, data.column_loadings),
                                        fit.loadings,
                                    )
                                ]
                            )
                        ),
                        ranks=list(fit.ranks),
                        rank_converged=fit.rank_converged,
                        final_inner_converged=bool(np.all(fit.inner_converged[-1])),
                        variance_regularized=fit.variance_regularized,
                        is_stable=fit.is_stable,
                    )
                return result

            metadata = dict(
                task="additive-dynamic-forecast",
                regime=regime,
                replicate=replicate,
                seed=current,
                shape=list(shape),
                n_train=n,
                n_test=50,
                diagonal_dynamics_assumption=regime == "diagonal-dynamics",
            )
            methods = {
                "two-way-known-ranks": (
                    lambda: fit_two_way_dynamic(train, (1, 2)),
                    {"ranks": [1, 2], "orders": [1, 1]},
                ),
                "two-way-auto-ranks": (
                    lambda: fit_two_way_dynamic(train),
                    {"ranks": None, "orders": [1, 1], "ratio_ridge": 0.01},
                ),
                "mar-als": (lambda: fit_mar(train), {}),
                "var-ridge": (
                    lambda: fit_var(train, ridge=1, intercept=False),
                    {"ridge": 1.0, "intercept": False},
                ),
            }
            for name, (factory, options) in methods.items():
                rows.append(
                    _measure(name, factory, score, dict(metadata, options=options))
                )
            for name, prediction in (
                ("zero", np.zeros_like(actual)),
                ("oracle-latent-conditional-mean", data.conditional_mean[n:]),
            ):
                rows.append(
                    dict(
                        metadata,
                        method=name,
                        status="ok",
                        converged=True,
                        fit_seconds=0.0,
                        iterations=0,
                        warnings=[],
                        forecast_mse=mean_squared_error(actual, prediction),
                        options={},
                    )
                )

        n = 120 if quick else 600
        parameters = _garch_parameters()
        data = simulate_matrix_garch(
            n + 50, parameters, burnin=300, random_state=current
        )
        train, actual = data.observations[:n], data.observations[n:]
        truth = np.array(
            [
                np.kron(v, u)
                for u, v in zip(data.row_covariances[n:], data.column_factors[n:])
            ]
        )

        def volatility_score(fit):
            if hasattr(fit, "filter"):
                held_out = fit.filter(actual)
                covariance = np.array(
                    [held_out.covariance(i) for i in range(len(actual))]
                )
                extras = dict(
                    constraint_violation=fit.runs[
                        fit.selected_start
                    ].constraint_violation,
                    active_bounds=list(fit.active_bounds),
                    n_starts=len(fit.runs),
                    successful_starts=sum(run.converged for run in fit.runs),
                )
            else:
                covariance = fit.covariances
                extras = {}
            return dict(
                negative_gaussian_log_score=_gaussian_score(actual, covariance),
                covariance_error=relative_frobenius_error(truth, covariance),
                **extras,
            )

        metadata = dict(
            task="volatility-covariance",
            regime="full-matrix-garch",
            replicate=replicate,
            seed=current,
            shape=[2, 2],
            n_train=n,
            n_test=50,
            true_sufficient_stationarity_bound=parameters.sufficient_stationarity_bound,
        )
        max_iter = 100 if quick else 300
        for dynamics in ("diagonal", "full"):
            options = dict(
                dynamics=dynamics,
                n_starts=1 if quick else 2,
                max_iter=max_iter,
                tol=1e-7,
                constraint="spectral",
                random_state=current,
            )
            rows.append(
                _measure(
                    "matrix-garch-" + dynamics,
                    lambda: fit_matrix_garch(train, **options),
                    volatility_score,
                    dict(metadata, options=options),
                )
            )
        flat = train.transpose(0, 2, 1).reshape(n, -1)
        constant = flat.T @ flat / n
        for name, covariance in (
            ("constant-covariance", np.repeat(constant[None], len(actual), axis=0)),
            ("oracle-conditional-covariance", truth),
        ):
            rows.append(
                _measure(
                    name,
                    lambda: SimpleNamespace(covariances=covariance),
                    volatility_score,
                    dict(metadata, options={}),
                )
            )
        print(
            f"Completed advanced matrix replicate {replicate+1}/{repeats}", flush=True
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=893)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/advanced-matrix.json")
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    snapshot = source_snapshot(__file__, Path(__file__).with_name("scenarios.py"))
    rows = run_suite(quick=args.quick, repeats=args.repeats, seed=args.seed)
    report = dict(
        schema_version=1,
        configuration=dict(quick=args.quick, repeats=args.repeats, seed=args.seed),
        environment=environment(),
        **source_provenance(snapshot),
        results=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    failures = sum(row["status"] != "ok" for row in rows)
    print(
        f"Saved {len(rows)} runs; {failures} errors; {sum(not row.get('converged',True) for row in rows)} unconverged"
    )
    if failures or report["source_changed_during_run"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
