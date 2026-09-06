"""Matrix ARMA forecasts against MAR, vector and known-parameter baselines.

Tsay (2024), Matrix-Variate Time Series Analysis: A Brief Review and Some New
Developments, https://doi.org/10.1111/insr.12558, supplies the rank-one MARMA
model. Forecast origins follow the fixed-fit rolling-origin design discussed
by Tashman (2000), https://doi.org/10.1016/S0169-2070(00)00065-0. These are
synthetic integration experiments, not paper replications or global-fit claims.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.advanced_matrix import _gaussian_score
from benchmarks.marma_scenarios import marma_data
from benchmarks.run import _measure, environment, source_provenance, source_snapshot
from mavats.autoregression import fit_mar
from mavats.baselines import fit_naive, fit_var
from mavats.metrics import mean_squared_error, relative_frobenius_error


class _Zero:
    converged = True
    n_iter = 0

    def __init__(self, train):
        self.fit = fit_naive(train, strategy="zero")

    def forecast(self, steps, history):
        return self.fit.forecast(steps)


class _TrueMARMA:
    """Known parameters and latent past innovations: extra oracle information.

    References
    ----------
    Tsay (2024), https://doi.org/10.1111/insr.12558. Conditional mean and
    covariance of the published model, not a fitted estimator.
    """

    converged = True
    n_iter = 0

    def __init__(self, data):
        self.data = data

    def forecast(self, steps, history):
        data = self.data
        origin = len(history)
        previous = history[-1].ravel(order="F")
        noise = data.innovations[origin - 1].ravel(order="F")
        predictions = []
        for _ in range(steps):
            previous = (
                data.intercept.ravel(order="F")
                + data.ar_transition @ previous
                - data.ma_transition @ noise
            )
            predictions.append(previous.reshape(data.intercept.shape, order="F"))
            noise = np.zeros_like(noise)
        return np.array(predictions)

    def forecast_covariance(self, steps):
        # Independent dense order-one recursion: Psi1=Phi-Theta, Psih=Phi Psih-1.
        impulse = np.eye(self.data.intercept.size)
        total = self.data.covariance.copy()
        covariances = [total.copy()]
        for h in range(1, steps):
            impulse = self.data.ar_transition @ impulse
            if h == 1:
                impulse -= self.data.ma_transition
            total = total + impulse @ self.data.covariance @ impulse.T
            covariances.append(total.copy())
        return np.array(covariances)


def _forecast_scores(fit, data, n, horizon=5):
    """Same held-out origins for both horizons, refiltering only past data."""
    predictions, observations = [], []
    for origin in range(n, len(data.observations) - horizon + 1):
        predictions.append(fit.forecast(horizon, history=data.observations[:origin]))
        observations.append(data.observations[origin : origin + horizon])
    predictions, observations = np.array(predictions), np.array(observations)
    result = dict(
        one_step_mse=mean_squared_error(observations[:, 0], predictions[:, 0]),
        five_step_mse=mean_squared_error(observations[:, -1], predictions[:, -1]),
        origins=len(observations),
        forecast_horizon=horizon,
    )
    if hasattr(fit, "forecast_covariance"):
        covariances = fit.forecast_covariance(horizon)
        for h, name in ((0, "one_step"), (horizon - 1, "five_step")):
            errors = observations[:, h] - predictions[:, h]
            result[name + "_negative_gaussian_log_score"] = _gaussian_score(
                errors, np.repeat(covariances[h][None], len(errors), axis=0)
            )
        result["innovation_covariance_error"] = relative_frobenius_error(
            data.covariance, covariances[0]
        )
    if hasattr(fit, "parameters"):
        parameters = fit.parameters
        result["ar_operator_error"] = relative_frobenius_error(
            data.ar_transition, np.kron(parameters.ar_right[0], parameters.ar_left[0])
        )
        result["ma_operator_error"] = relative_frobenius_error(
            data.ma_transition, np.kron(parameters.ma_right[0], parameters.ma_left[0])
        )
        diagnostic = fit.diagnostics
        result.update(
            ar_spectral_radius=diagnostic.ar_spectral_radius,
            ma_spectral_radius=diagnostic.ma_spectral_radius,
            is_stable=diagnostic.is_stable,
            is_invertible=diagnostic.is_invertible,
            structural_identification_certified=diagnostic.structural_identification_certified,
            n_starts=len(fit.runs),
            selected_start=fit.selected_start,
            successful_starts=sum(r.converged for r in fit.runs),
            failed_starts=sum(r.failed for r in fit.runs),
            selected_gradient_norm=fit.runs[fit.selected_start].gradient_norm,
            selected_stopping_message=fit.runs[fit.selected_start].message,
            selected_optimizer=fit.runs[fit.selected_start].optimizer,
            selected_feasible=fit.runs[fit.selected_start].feasible,
            selected_constraint_residuals=fit.runs[
                fit.selected_start
            ].constraint_residuals.tolist(),
            initialization_iterations=[r.initialization_iterations for r in fit.runs],
            initialization_converged=[r.initialization_converged for r in fit.runs],
            initialization_messages=[r.initialization_message for r in fit.runs],
            objective=fit.objective,
        )
    return result


def run_suite(*, quick=False, repeats=5, seed=1973):
    from mavats.marma import fit_marma

    rows = []
    n = 120 if quick else 500
    starts, max_iter = (1, 120) if quick else (2, 300)
    regimes = (
        ("isotropic", "near-cancellation")
        if quick
        else ("isotropic", "separable", "near-cancellation", "nonseparable")
    )
    for replicate in range(repeats):
        current = seed + replicate
        for regime in regimes:
            data = marma_data(n + 50, regime=regime, seed=current)
            train = data.observations[:n]
            metadata = dict(
                task="matrix-arma-forecast",
                regime=regime,
                replicate=replicate,
                seed=current,
                shape=[2, 3],
                n_train=n,
                n_test=50,
                true_orders=[1, 1],
                covariance_separable=regime != "nonseparable",
                forecasting_task="fixed-fit-common-origin-one-and-five-step",
            )
            options = dict(
                ar_order=1,
                ma_order=1,
                intercept=True,
                n_starts=starts,
                max_iter=max_iter,
                tol=1e-8,
                random_state=current,
                enforce_admissibility=True,
                stability_margin=1e-6,
            )
            methods = (
                (
                    "marma-ls-known-orders",
                    lambda: fit_marma(train, method="ls", **options),
                    dict(method="ls", **options),
                ),
                (
                    "marma-mle-known-orders",
                    lambda: fit_marma(train, method="mle", **options),
                    dict(method="mle", **options),
                ),
                (
                    "mar-als-order-one",
                    lambda: fit_mar(train, method="als", fit_intercept=True),
                    dict(method="als", order=1, fit_intercept=True),
                ),
                (
                    "mar-mle-order-one",
                    lambda: fit_mar(train, method="mle", fit_intercept=True),
                    dict(method="mle", order=1, fit_intercept=True),
                ),
                (
                    "var-order-one",
                    lambda: fit_var(train, order=1),
                    dict(order=1, intercept=True, ridge=0),
                ),
                (
                    "var-order-five",
                    lambda: fit_var(train, order=5),
                    dict(order=5, intercept=True, ridge=0),
                ),
                ("zero", lambda: _Zero(train), dict(strategy="zero")),
                (
                    "known-parameter-latent-innovation-oracle",
                    lambda: _TrueMARMA(data),
                    dict(extra_information="true-parameters-and-past-innovations"),
                ),
            )
            for name, function, configuration in methods:
                rows.append(
                    _measure(
                        name,
                        function,
                        lambda fit: _forecast_scores(fit, data, n),
                        dict(metadata, options=configuration),
                    )
                )
            print(
                f"Completed MARMA replicate {replicate + 1}/{repeats}: {regime}",
                flush=True,
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1973)
    parser.add_argument("--output", type=Path, default=Path("benchmark-marma.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    from benchmarks import advanced_matrix, marma_scenarios

    snapshot = source_snapshot(
        __file__, marma_scenarios.__file__, advanced_matrix.__file__
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
    errors = sum(row["status"] != "ok" for row in rows)
    unconverged = sum(
        not row.get("converged", False) for row in rows if row["status"] == "ok"
    )
    print(f"Saved {len(rows)} runs; {errors} errors; {unconverged} unconverged")
    if errors or report["source_changed_during_run"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
