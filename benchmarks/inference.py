"""Repeated MAR interval coverage and specification-test size/power experiments.

Run ``python -m benchmarks.inference --repeats 200 --output ...``. Marginal
coverage uncertainty uses independent time-series replications, not entries
of the same fitted operator treated as independent observations.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.run import environment, source_provenance, source_snapshot
from mavats.autoregression import fit_mar
from mavats.inference import mar_inference, mar_specification_test


def _simulate(n, phi, covariance, seed):
    rng = np.random.default_rng(seed)
    f = np.zeros((n + 200, len(phi)))
    noise = rng.normal(size=f.shape) @ np.linalg.cholesky(covariance).T
    for t in range(1, len(f)):
        f[t] = phi @ f[t - 1] + noise[t]
    return f[200:].reshape(n, 2, 2).transpose(0, 2, 1)


def run_study(repeats=200, sample_sizes=(200, 800), seed=7123):
    A = np.array([[0.7, 0.2], [-0.1, 0.5]])
    B = np.array([[0.8, -0.15], [0.1, 0.4]])
    truth = np.kron(B, A)
    covariances = {
        "isotropic": np.eye(4),
        "separable": np.kron([[1.0, 0.4], [0.4, 1.2]], [[0.8, -0.2], [-0.2, 1.0]]),
        "nonseparable": np.array(
            [
                [1.0, 0.4, 0.1, 0.0],
                [0.4, 1.5, 0.0, -0.2],
                [0.1, 0.0, 0.7, 0.3],
                [0.0, -0.2, 0.3, 1.2],
            ]
        ),
    }
    rows = []
    for n in sample_sizes:
        for regime, covariance in covariances.items():
            for replicate in range(repeats):
                sample_seed = seed + replicate
                X = _simulate(n, truth, covariance, sample_seed)
                for method in ("projection", "als", "mle"):
                    row = dict(
                        task="coverage",
                        n=n,
                        regime=regime,
                        method=method,
                        replicate=replicate,
                        seed=sample_seed,
                        assumptions_satisfied=not (
                            method == "mle" and regime == "nonseparable"
                        ),
                    )
                    start = perf_counter()
                    try:
                        fit = fit_mar(X, method=method, tol=1e-11, max_iter=1000)
                        inference = mar_inference(X, fit)
                        lower, upper = inference.confidence_interval()
                        covered = (truth >= lower) & (truth <= upper)
                        row.update(
                            status="ok",
                            coverage=float(covered.mean()),
                            entry_coverage=covered.astype(int).tolist(),
                            mean_width=float(np.mean(upper - lower)),
                            iterations=fit.n_iter,
                        )
                    except (
                        ValueError,
                        FloatingPointError,
                        np.linalg.LinAlgError,
                    ) as exc:
                        row.update(status="error", error=str(exc))
                    row["seconds"] = perf_counter() - start
                    rows.append(row)
                if regime == "isotropic":
                    for null, phi in (
                        (True, truth),
                        (False, np.diag([0.8, 0.15, -0.7, 0.45])),
                    ):
                        series = (
                            X if null else _simulate(n, phi, covariance, sample_seed)
                        )
                        row = dict(
                            task="specification",
                            n=n,
                            regime="null" if null else "alternative",
                            method="kronecker-wald",
                            replicate=replicate,
                            seed=sample_seed,
                        )
                        try:
                            result = mar_specification_test(series)
                            row.update(
                                status="ok",
                                pvalue=result.pvalue,
                                rejected=result.pvalue < 0.05,
                                statistic=result.statistic,
                            )
                        except (
                            ValueError,
                            FloatingPointError,
                            np.linalg.LinAlgError,
                        ) as exc:
                            row.update(status="error", error=str(exc))
                        rows.append(row)
            print(f"Completed n={n}, {regime}, {repeats} replications", flush=True)
    return rows


def summary(rows):
    groups = {}
    for row in rows:
        key = (row["task"], row["n"], row["regime"], row["method"])
        groups.setdefault(key, []).append(row)
    output = []
    for (task, n, regime, method), group in groups.items():
        good = [r for r in group if r["status"] == "ok"]
        rates = np.array(
            [
                r["coverage"] if task == "coverage" else float(r["rejected"])
                for r in good
            ]
        )
        record = dict(
            task=task,
            n=n,
            regime=regime,
            method=method,
            successful=len(good),
            total=len(group),
            rates_condition_on_success=True,
            mean_rate=float(rates.mean()) if len(rates) else None,
            monte_carlo_se=(
                float(rates.std(ddof=1) / np.sqrt(len(rates)))
                if len(rates) > 1
                else None
            ),
        )
        if task == "coverage" and good:
            record["per_entry_coverage"] = np.mean(
                [r["entry_coverage"] for r in good], axis=0
            ).tolist()
            record["mean_interval_width"] = float(
                np.mean([r["mean_width"] for r in good])
            )
            record["assumptions_satisfied"] = good[0]["assumptions_satisfied"]
        elif good:
            # Wilson binomial bounds remain informative at zero/all rejections,
            # where the plug-in Monte Carlo standard error alone is misleading.
            p = float(rates.mean())
            count = len(rates)
            z = 1.959963984540054
            denominator = 1 + z * z / count
            center = (p + z * z / (2 * count)) / denominator
            half = (
                z
                * np.sqrt(p * (1 - p) / count + z * z / (4 * count * count))
                / denominator
            )
            record["binomial_wilson_95"] = [
                max(0.0, center - half),
                min(1.0, center + half),
            ]
        output.append(record)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--sample-sizes", type=int, nargs="+", default=[200, 800])
    parser.add_argument("--seed", type=int, default=7123)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/inference.json")
    )
    args = parser.parse_args()
    if args.repeats < 1 or min(args.sample_sizes) < 20:
        parser.error("need positive repeats and sample sizes >= 20")
    if args.output.suffix == ".jsonl":
        parser.error("summary output must differ from .jsonl raw output")
    snapshot = source_snapshot(__file__)
    results = run_study(args.repeats, tuple(args.sample_sizes), args.seed)
    report = dict(
        schema_version=1,
        configuration=dict(
            repeats=args.repeats, sample_sizes=args.sample_sizes, seed=args.seed
        ),
        environment=environment(),
        **source_provenance(snapshot),
        results_file=args.output.with_suffix(".jsonl").name,
        summary=summary(results),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in results)
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Saved {len(results)} experiments to {args.output}; {sum(r['status']!='ok' for r in results)} failures retained"
    )
    if report["source_changed_during_run"]:
        raise SystemExit("Source changed during the experiment; see provenance flags")


if __name__ == "__main__":
    main()
