"""Paired cointegrated-matrix and multi-term tensor autoregression study."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from benchmarks.run import _measure, environment, source_provenance, source_snapshot
from benchmarks.structured_scenarios import cointegrated_data, tensor_ar_data
from mavats.baselines import fit_var
from mavats.metrics import (
    mean_squared_error,
    relative_frobenius_error,
    subspace_distance,
)


def _flatten(X):
    return np.stack([x.ravel(order="F") for x in X])[:, :, None]


def _reshape(X, shape):
    return np.stack([x.reshape(shape, order="F") for x in X])


@dataclass
class _VectorForecast:
    fitted: object
    shape: tuple

    def forecast(self, steps, history):
        return _reshape(
            self.fitted.forecast(steps, history=_flatten(history)), self.shape
        )


@dataclass
class _OracleForecast:
    transitions: np.ndarray
    intercept: np.ndarray

    def forecast(self, steps, history):
        state = [x.ravel(order="F").copy() for x in history]
        predicted = []
        for _ in range(steps):
            value = self.intercept.ravel(order="F").copy()
            for lag, transition in enumerate(self.transitions, 1):
                value += transition @ state[-lag]
            state.append(value)
            predicted.append(value)
        return _reshape(predicted, self.intercept.shape)


class _RandomWalk:
    def forecast(self, steps, history):
        return np.repeat(history[-1:], steps, axis=0)


@dataclass
class _VECMForecast(_OracleForecast):
    cointegrating_vectors: np.ndarray
    long_run_operator: np.ndarray


def _fit_vector_vecm(X, rank):
    """Gaussian reduced-rank VECM(1 difference lag), with unrestricted intercept.

    This comparison receives the same total cointegration rank as CMAR, but
    no Kronecker restriction. Partial out short-run terms, whiten the full-rank
    regression residual covariance, and take the rank-r canonical projection.
    No rank test, stability restriction, ridge or standard errors are supplied.

    References
    ----------
    Johansen (1991), Estimation and Hypothesis Testing of Cointegration Vectors
    in Gaussian Vector Autoregressive Models, Econometrica 59, 1551--1580,
    https://doi.org/10.2307/2938278. This implements the conditional Gaussian
    reduced-rank estimation problem with the stated deterministic/lag choices,
    not the paper's hypothesis tests or all deterministic specifications.
    """
    shape = X.shape[1:]
    flat = _flatten(X)[:, :, 0]
    scale = float(np.max(np.abs(flat))) or 1.0
    flat = flat / scale
    changes = np.diff(flat, axis=0)
    target, level = changes[1:], flat[1:-1]
    nuisance = np.column_stack((changes[:-1], np.ones(len(target))))
    qz, sz, _ = np.linalg.svd(nuisance, full_matrices=False)
    if sz[-1] <= np.finfo(float).eps * max(nuisance.shape) * sz[0]:
        raise ValueError("VECM nuisance design is rank deficient")
    residual_level = level - qz @ (qz.T @ level)
    residual_target = target - qz @ (qz.T @ target)
    coefficient, _, design_rank, _ = np.linalg.lstsq(
        residual_level, residual_target, rcond=None
    )
    dimension = level.shape[1]
    if design_rank != dimension or not 0 < rank <= dimension:
        raise ValueError("VECM level design or requested rank is invalid")
    residual = residual_target - residual_level @ coefficient
    root = np.linalg.cholesky(residual.T @ residual / len(residual))
    white = np.linalg.solve(root, residual_target.T).T
    qx = np.linalg.svd(residual_level, full_matrices=False)[0]
    _, _, right = np.linalg.svd(qx.T @ white, full_matrices=False)
    projection = right[:rank].T @ right[:rank]
    coefficient = (
        np.linalg.lstsq(residual_level, white @ projection, rcond=None)[0] @ root.T
    )
    nuisance_coefficient = np.linalg.lstsq(
        nuisance, target - level @ coefficient, rcond=None
    )[0]
    pi, gamma = coefficient.T, nuisance_coefficient[:-1].T
    constant = (nuisance_coefficient[-1] * scale).reshape(shape, order="F")
    vectors = np.linalg.svd(pi, full_matrices=False)[2][:rank].T
    return _VECMForecast(
        np.array([np.eye(dimension) + pi + gamma, -gamma]), constant, vectors, pi
    )


def forecast_scores(fit, X, n_train, *, order=2, horizon=5):
    """Common-origin h=1/h=5 errors with no held-out refitting or future data.

    Origins whose full five-step outcome is unavailable are excluded for both
    scores. Overlapping origins are not independent Monte Carlo replications.

    References
    ----------
    Tashman (2000), Out-of-Sample Tests of Forecasting Accuracy: An Analysis
    and Review, https://doi.org/10.1016/S0169-2070(00)00065-0. This is a
    fixed-fit, common-origin evaluation design, not a new forecasting method.
    """
    predicted, actual = [], []
    for origin in range(n_train, len(X) - horizon + 1):
        predicted.append(fit.forecast(horizon, history=X[origin - order : origin]))
        actual.append(X[origin : origin + horizon])
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    return dict(
        one_step_mse=mean_squared_error(actual[:, 0], predicted[:, 0]),
        five_step_mse=mean_squared_error(actual[:, -1], predicted[:, -1]),
        origins=len(actual),
        forecast_horizon=horizon,
    )


def run_suite(*, quick=False, repeats=10, seed=1193):
    from mavats.cointegration import fit_cmar
    from mavats.tensor_autoregression import fit_tensor_ar

    rows = []
    for replicate in range(repeats):
        current = seed + replicate
        n = 120 if quick else 600
        for regime in ("isotropic", "separable", "weak-adjustment"):
            data = cointegrated_data(n + 50, regime=regime, seed=current)
            X, train = data.observations, data.observations[:n]
            metadata = dict(
                task="cointegrated-matrix-forecast",
                regime=regime,
                replicate=replicate,
                seed=current,
                shape=list(X.shape[1:]),
                n_train=n,
                n_test=50,
                difference_lags=1,
                true_ranks=[1, 2],
            )

            def score(fit):
                result = forecast_scores(fit, X, n)
                if hasattr(fit, "beta1"):
                    diagnostic = fit.i1_diagnostics()
                    result.update(
                        fitted_i1_compatible=diagnostic.compatible,
                        fitted_unit_roots=diagnostic.unit_roots_observed,
                        fitted_stable_roots_radius=diagnostic.stable_roots_radius,
                        successful_starts=sum(run.converged for run in fit.runs),
                        n_starts=len(fit.runs),
                        cointegrating_space_error=subspace_distance(
                            np.kron(data.beta2, data.beta1),
                            np.kron(fit.beta2, fit.beta1),
                        ),
                        long_run_operator_error=relative_frobenius_error(
                            np.kron(data.A2, data.A1), np.kron(fit.A2, fit.A1)
                        ),
                    )
                elif hasattr(fit, "cointegrating_vectors"):
                    result.update(
                        cointegrating_space_error=subspace_distance(
                            np.kron(data.beta2, data.beta1), fit.cointegrating_vectors
                        ),
                        long_run_operator_error=relative_frobenius_error(
                            np.kron(data.A2, data.A1), fit.long_run_operator
                        ),
                    )
                return result

            for method in ("ls", "mle"):
                options = dict(
                    ranks=[1, 2],
                    difference_lags=1,
                    intercept=True,
                    method=method,
                    n_starts=1 if quick else 3,
                    max_iter=200,
                    random_state=current,
                )
                rows.append(
                    _measure(
                        "cmar-" + method + "-known-ranks",
                        lambda: fit_cmar(train, **options),
                        score,
                        dict(metadata, options=options),
                    )
                )
            rows.append(
                _measure(
                    "vector-vecm-known-rank",
                    lambda: _fit_vector_vecm(train, rank=2),
                    score,
                    dict(
                        metadata,
                        options=dict(rank=2, difference_lags=1, intercept=True),
                    ),
                )
            )
            rows.append(
                _measure(
                    "unrestricted-var",
                    lambda: _VectorForecast(
                        fit_var(_flatten(train), order=2), X.shape[1:]
                    ),
                    score,
                    dict(metadata, options=dict(order=2, intercept=True)),
                )
            )
            dimension = int(np.prod(X.shape[1:]))
            transitions = np.array(
                [
                    data.companion[:dimension, :dimension],
                    data.companion[:dimension, dimension:],
                ]
            )
            for name, fit in (
                ("random-walk", _RandomWalk()),
                ("oracle-cmar", _OracleForecast(transitions, data.intercept)),
            ):
                rows.append(
                    _measure(name, lambda: fit, score, dict(metadata, options={}))
                )

        n = 120 if quick else 500
        for shape in ((3, 4), (2, 3, 2)):
            for regime in ("isotropic", "separable", "nonseparable"):
                data = tensor_ar_data(n + 50, shape=shape, regime=regime, seed=current)
                X, train = data.observations, data.observations[:n]
                metadata = dict(
                    task="multi-term-autoregression",
                    regime=regime,
                    replicate=replicate,
                    seed=current,
                    shape=list(shape),
                    n_train=n,
                    n_test=50,
                    true_terms=[2, 1],
                    separable_covariance_assumption=regime != "nonseparable",
                )

                def score(fit):
                    result = forecast_scores(fit, X, n)
                    if hasattr(fit, "transition_matrices"):
                        selected = fit.runs[fit.selected_start]
                        result.update(
                            transition_error=relative_frobenius_error(
                                data.transitions, np.asarray(fit.transition_matrices())
                            ),
                            spectral_radius=fit.spectral_radius,
                            successful_starts=sum(run.converged for run in fit.runs),
                            n_starts=len(fit.runs),
                            projection_converged=[
                                p.converged for p in selected.projection
                            ],
                            covariance_regularized=fit.covariance_regularized,
                            maximum_block_condition=selected.maximum_block_condition,
                            maximum_cancellation=selected.maximum_cancellation,
                            identification=[
                                d["identification"]
                                for d in fit.identification_diagnostics
                            ],
                        )
                        if fit.covariance_factors is not None:
                            result["covariance_error"] = relative_frobenius_error(
                                data.covariance, fit.innovation_covariance()
                            )
                    return result

                for method, terms in (
                    ("projection", (2, 1)),
                    ("ls", (2, 1)),
                    ("mle", (2, 1)),
                    ("ls", (1, 1)),
                ):
                    options = dict(
                        terms=terms,
                        method=method,
                        n_starts=1 if quick else 2,
                        max_iter=200,
                        tol=1e-7,
                        center=False,
                        random_state=current,
                    )
                    name = (
                        "tenar-"
                        + method
                        + ("-known-terms" if terms == (2, 1) else "-underfit-terms")
                    )
                    rows.append(
                        _measure(
                            name,
                            lambda: fit_tensor_ar(train, **options),
                            score,
                            dict(metadata, options=options),
                        )
                    )
                rows.append(
                    _measure(
                        "unrestricted-var",
                        lambda: _VectorForecast(
                            fit_var(_flatten(train), order=2, intercept=False), shape
                        ),
                        score,
                        dict(metadata, options=dict(order=2, intercept=False)),
                    )
                )
                oracle = _OracleForecast(data.transitions, np.zeros(shape))
                rows.append(
                    _measure(
                        "oracle-tenar",
                        lambda: oracle,
                        score,
                        dict(metadata, options={}),
                    )
                )
        print(
            f"Completed structured autoregression replicate {replicate+1}/{repeats}",
            flush=True,
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1193)
    parser.add_argument("--output", type=Path, default=Path("structured-ar.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    snapshot = source_snapshot(
        __file__, Path(__file__).with_name("structured_scenarios.py")
    )
    rows = run_suite(quick=args.quick, repeats=args.repeats, seed=args.seed)
    report = dict(
        schema_version=1,
        configuration=dict(quick=args.quick, repeats=args.repeats, seed=args.seed),
        environment=environment(),
        results=rows,
        **source_provenance(snapshot),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    errors = sum(r["status"] != "ok" for r in rows)
    unconverged = sum(r.get("converged") is False for r in rows)
    print(f"Saved {len(rows)} runs; {errors} errors; {unconverged} unconverged")
    if errors or report["source_changed_during_run"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
