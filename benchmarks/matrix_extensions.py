"""Paired sparse, threshold and decorrelation matrix-method benchmarks."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.run import _measure, environment
from mavats.autoregression import fit_mar
from mavats.baselines import fit_var
from mavats.decorrelation import fit_matrix_decorrelation
from mavats.factors import fit_lagged_factor
from mavats.metrics import (
    mean_squared_error,
    relative_frobenius_error,
    subspace_distance,
)
from mavats.simulation import simulate_mar
from mavats.sparse import fit_sparse_mar
from mavats.threshold import fit_threshold_factors


def _threshold_data(n, seed):
    rng = np.random.default_rng(seed)
    z = rng.uniform(-1, 1, n)
    row = np.linalg.qr(rng.normal(size=(6, 3)))[0]
    column = np.linalg.qr(rng.normal(size=(5, 3)))[0]
    loadings = ((row[:, :1], column[:, :2]), (row[:, 1:3], column[:, 2:3]))
    core = np.zeros((n, 2, 2))
    signal = np.empty((n, 6, 5))
    for t in range(n):
        if t:
            core[t] = 0.8 * core[t - 1] + rng.normal(size=(2, 2))
        a, b = loadings[int(z[t] >= 0)]
        signal[t] = a @ core[t, : a.shape[1], : b.shape[1]] @ b.T
    return signal + 0.04 * rng.normal(size=signal.shape), signal, z, loadings


def _decorrelation_data(n, seed):
    rng = np.random.default_rng(seed)
    coefficients = np.array([[0.92, 0.6, 0.2], [0.75, 0.4, 0.1], [0.55, 0.25, -0.1]])
    latent = np.zeros((n + 200, 3, 3))
    noise = rng.normal(size=latent.shape) * np.sqrt(1 - coefficients**2)
    for t in range(1, len(latent)):
        latent[t] = coefficients * latent[t - 1] + noise[t]
    row = np.array([[1, 0.3, -0.2], [0.1, 1, 0.4], [0.2, -0.2, 1]])
    column = np.array([[1, 0.4, 0.1], [-0.2, 1, 0.3], [0.1, 0.2, 1]])
    return row @ latent[200:] @ column.T + 3


def _block_forecast(train, predictors, grouping="threshold"):
    decomposition = fit_matrix_decorrelation(
        train, lags=1, correlation_lags=5, correlation_threshold=0.15, grouping=grouping
    )
    training = decomposition.blocks()
    histories = decomposition.blocks(predictors)
    prediction = []
    for blockrow, historyrow in zip(training, histories):
        prediction.append([])
        for block, history in zip(blockrow, historyrow):
            fit = fit_var(block)
            prediction[-1].append(
                np.concatenate([fit.forecast(1, history=one[None]) for one in history])
            )
    return SimpleNamespace(
        decomposition=decomposition, forecasts=decomposition.inverse_blocks(prediction)
    )


def _var_forecast(train, predictors):
    fit = fit_var(train)
    return SimpleNamespace(
        forecasts=np.concatenate(
            [fit.forecast(1, history=one[None]) for one in predictors]
        )
    )


def run_suite(*, quick=False, repeats=5, seed=48):
    records = []
    for replicate in range(repeats):
        current = seed + replicate
        n = 300 if quick else 1100
        A, B = np.diag([0.8, 0.7, 0.65]), np.diag([0.7, 0.8])
        x = simulate_mar(n + 50, A, B, random_state=current)
        train, test = x[:n], x[n:]

        def sparse_score(fit):
            predictions = np.concatenate(
                [fit.forecast(1, history=one[None]) for one in x[n - 1 : -1]]
            )
            record = dict(
                forecast_mse=mean_squared_error(test, predictions),
                operator_error=relative_frobenius_error(
                    np.kron(B, A), fit.coefficients[0]
                ),
            )
            if hasattr(fit, "row_support"):
                truth = np.r_[A.ravel() != 0, B.ravel() != 0]
                support = np.r_[fit.row_support.ravel(), fit.column_support.ravel()]
                record.update(
                    support_error=float(np.mean(support != truth)),
                    true_positive_rate=float(np.mean(support[truth])),
                    false_positive_rate=float(np.mean(support[~truth])),
                )
            return record

        options = {
            "spike_variance": 0.001,
            "slab_variance": 4.0,
            "inclusion_prior": [1.0, 1.0],
        }
        metadata = dict(
            task="sparse-forecast",
            regime="diagonal-sparse",
            replicate=replicate,
            seed=current,
            shape=[3, 2],
            n_train=n,
            options=options,
        )
        factories = {
            "sparse-emvs": lambda: fit_sparse_mar(train, spike_variance=0.001),
            "mar-als": lambda: fit_mar(train),
            "mar-mle": lambda: fit_mar(train, method="mle"),
            "var": lambda: fit_var(train, intercept=False),
        }
        for name, factory in factories.items():
            records.append(
                _measure(
                    name,
                    factory,
                    sparse_score,
                    dict(metadata, options=options if name == "sparse-emvs" else {}),
                )
            )

        n = 200 if quick else 400
        x, signal, z, loadings = _threshold_data(n + 50, current)

        def threshold_score(fit):
            if hasattr(fit, "threshold"):
                prediction = fit.reconstruct(x[n:], z[n:])
                spaces = float(
                    np.mean(
                        [
                            subspace_distance(a, b)
                            for truth, estimated in zip(loadings, fit.loadings)
                            for a, b in zip(truth, estimated)
                        ]
                    )
                )
                return dict(
                    signal_error=relative_frobenius_error(signal[:n], fit.signal),
                    held_out_signal_error=relative_frobenius_error(
                        signal[n:], prediction
                    ),
                    subspace_error=spaces,
                    threshold_error=abs(fit.threshold),
                    regime_error=float(
                        np.mean((z[n:] >= fit.threshold) != (z[n:] >= 0))
                    ),
                    ranks=[list(pair) for pair in fit.ranks],
                )
            prediction = fit.inverse_transform(fit.transform(x[n:]))
            return dict(
                signal_error=relative_frobenius_error(signal[:n], fit.signal),
                held_out_signal_error=relative_frobenius_error(signal[n:], prediction),
            )

        metadata = dict(
            task="threshold-factor",
            regime="unequal-ranks",
            replicate=replicate,
            seed=current,
            shape=[6, 5],
            n_train=n,
            options={"max_lag": 2, "trim": [0.1, 0.9]},
        )
        factories = {
            "threshold-estimated": lambda: fit_threshold_factors(
                x[:n], z[:n], max_lag=2
            ),
            "threshold-known-oracle": lambda: fit_threshold_factors(
                x[:n], z[:n], ((1, 2), (2, 1)), threshold=0, max_lag=2
            ),
            "global-factor-union-ranks": lambda: fit_lagged_factor(
                x[:n], (3, 3), lags=2
            ),
        }
        for name, factory in factories.items():
            records.append(_measure(name, factory, threshold_score, metadata))

        n = 2000 if quick else 10000
        x = _decorrelation_data(n + 100, current)
        train, test, predictors = x[:n], x[n:], x[n - 1 : -1]

        def block_score(fit):
            record = dict(forecast_mse=mean_squared_error(test, fit.forecasts))
            if hasattr(fit, "decomposition"):
                model = fit.decomposition
                record.update(
                    row_groups=[list(g) for g in model.row_groups],
                    column_groups=[list(g) for g in model.column_groups],
                    all_scalar_blocks=len(model.row_groups) == 3
                    and len(model.column_groups) == 3,
                    round_trip_error=relative_frobenius_error(
                        train, model.inverse_transform(model.series)
                    ),
                )
            return record

        metadata = dict(
            task="decorrelation-forecast",
            regime="scalar-blocks",
            replicate=replicate,
            seed=current,
            shape=[3, 3],
            n_train=n,
            timing_scope="fit transformation and predictors plus one-step predictions",
            options={"lags": 1, "correlation_lags": 5, "correlation_threshold": 0.15},
        )
        for name, factory in {
            "decorrelated-block-var": lambda: _block_forecast(train, predictors),
            "ratio-decorrelated-block-var": lambda: _block_forecast(
                train, predictors, grouping="ratio"
            ),
            "var": lambda: _var_forecast(train, predictors),
        }.items():
            options = (
                {}
                if name == "var"
                else dict(
                    metadata["options"],
                    grouping="ratio" if name.startswith("ratio-") else "threshold",
                )
            )
            records.append(
                _measure(name, factory, block_score, dict(metadata, options=options))
            )
        print(f"Completed extension replicate {replicate+1}/{repeats}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/matrix-extensions.json")
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    records = run_suite(quick=args.quick, repeats=args.repeats, seed=args.seed)
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "mavats").glob("*.py")) + [
        Path(__file__).resolve(),
        root / "benchmarks/run.py",
    ]
    report = dict(
        schema_version=1,
        configuration=dict(quick=args.quick, repeats=args.repeats, seed=args.seed),
        environment=environment(),
        source_sha256={
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        results=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    failures = [r for r in records if r["status"] != "ok"]
    print(
        f"Saved {len(records)} runs; {len(failures)} errors; {sum(not r.get('converged',True) for r in records)} unconverged"
    )
    for failure in failures:
        print(failure["method"], failure["error"])
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
