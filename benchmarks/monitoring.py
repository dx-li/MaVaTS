"""Horizon-wide false-alarm and detection studies for sequential matrix monitors."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.run import environment, source_provenance, source_snapshot
from benchmarks.scenarios import monitoring_data
from mavats.calibration import calibrate_monitor
from mavats.monitoring import MatrixFactorMonitor


def _wilson(success, total):
    """Wilson 95% score interval for a binomial replication count.

    References
    ----------
    Wilson (1927), Probable Inference, the Law of Succession, and Statistical
    Inference, https://doi.org/10.1080/01621459.1927.10502953.
    The interval concerns Monte Carlo series counts, not dependent time points.
    """
    if not total:
        return None
    z = 1.959963984540054
    p = success / total
    d = 1 + z * z / total
    center = (p + z * z / (2 * total)) / d
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [max(0.0, center - half), min(1.0, center + half)]


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["regime"], row["method"]].append(row)
    output = []
    for (regime, method), group in sorted(groups.items()):
        good = [row for row in group if row["status"] == "ok"]
        record = dict(
            regime=regime,
            method=method,
            total=len(group),
            successful=len(good),
            rates_condition_on_success=True,
        )
        for key in ("alarm", "pre_event_alarm"):
            count = sum(row[key] for row in good)
            record[key + "_rate"] = count / len(good) if good else None
            record[key + "_wilson_95"] = _wilson(count, len(good))
        if regime in ("space-switch", "factor-increase"):
            at_risk = [row for row in good if not row["pre_event_alarm"]]
            detected = [row for row in at_risk if row["detected_after_event"]]
            record.update(
                at_risk_at_change=len(at_risk),
                post_event_detections=len(detected),
                right_censored=len(at_risk) - len(detected),
                detection_rate_conditional_on_no_pre_alarm=(
                    len(detected) / len(at_risk) if at_risk else None
                ),
                detection_wilson_95=_wilson(len(detected), len(at_risk)),
                mean_delay_conditional_on_detection=(
                    float(np.mean([row["post_event_delay"] for row in detected]))
                    if detected
                    else None
                ),
            )
        output.append(record)
    return output


def run_study(*, repeats=200, quick=False, seed=672, calibration_repeats=20000):
    training, horizon, shape = (25, 50, (10, 8)) if quick else (50, 100, (20, 15))
    configs = {
        "maximum-asymptotic": dict(horizon=horizon, statistic="maximum", alpha=0.05)
    }
    calibrations = []
    for name, statistic, eta in (
        ("maximum-gaussian", "maximum", 0.25),
        ("partial-sum-gaussian", "partial_sum", 0.25),
        ("darling-erdos-gaussian", "partial_sum", 0.5),
        ("renyi-gaussian", "partial_sum", 0.75),
    ):
        fit = calibrate_monitor(
            horizon,
            statistic=statistic,
            eta=eta,
            replications=calibration_repeats,
            random_state=seed + 100000,
        )
        configs[name] = fit.monitor_kwargs
        calibrations.append(
            dict(
                method=name,
                critical_value=fit.critical_value,
                replications=fit.replications,
                calibration_method=fit.method,
                quantile_interval=[
                    x if np.isfinite(x) else None for x in fit.quantile_interval
                ],
                calibration_seed=seed + 100000,
            )
        )
    rows = []
    for regime in ("null", "space-switch", "factor-increase", "same-space-volatility"):
        for replicate in range(repeats):
            data_seed = seed + replicate
            noise_seed = seed + 50000 + replicate
            X, change = monitoring_data(
                training, horizon, shape=shape, regime=regime, seed=data_seed
            )
            draws = np.random.default_rng(noise_seed).normal(size=horizon)
            for method, options in configs.items():
                row = dict(
                    task="monitoring",
                    regime=regime,
                    method=method,
                    replicate=replicate,
                    seed=data_seed,
                    noise_seed=noise_seed,
                    shape=list(shape),
                    training=training,
                    horizon=horizon,
                    change_step=change,
                    options=dict(
                        options, rank=1, projection_rank=2, power=8, epsilon=0.05
                    ),
                )
                start = perf_counter()
                try:
                    monitor = MatrixFactorMonitor(
                        X[:training], 1, 2, power=8, epsilon=0.05, **options
                    )
                    steps = monitor.update_many(X[training:], noise=draws)
                    alarm = monitor.alarm_step
                    row.update(
                        status="ok",
                        alarm=alarm is not None,
                        alarm_step=alarm,
                        pre_event_alarm=alarm is not None and alarm < change,
                        detected_after_event=alarm is not None and alarm >= change,
                        post_event_delay=(
                            alarm - change
                            if alarm is not None and alarm >= change
                            else None
                        ),
                        consumed=len(steps),
                        stop_reason=monitor.stop_reason,
                        max_drift=float(max(step.drift for step in steps)),
                        saturated=any(step.saturated for step in steps),
                    )
                except (
                    ValueError,
                    RuntimeError,
                    FloatingPointError,
                    np.linalg.LinAlgError,
                ) as exc:
                    row.update(status="error", error=str(exc))
                row["seconds"] = perf_counter() - start
                rows.append(row)
        print(f"Completed {regime}: {repeats} series per method", flush=True)
    return rows, calibrations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=672)
    parser.add_argument("--calibration-repeats", type=int, default=20000)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/results/monitoring.json")
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.calibration_repeats < 20:
        parser.error("need positive repeats and calibration-repeats >=20")
    if args.output.suffix == ".jsonl":
        parser.error("summary output must differ from .jsonl raw output")
    snapshot = source_snapshot(__file__, Path(__file__).with_name("scenarios.py"))
    rows, calibrations = run_study(
        repeats=args.repeats,
        quick=args.quick,
        seed=args.seed,
        calibration_repeats=args.calibration_repeats,
    )
    report = dict(
        schema_version=1,
        configuration=dict(
            repeats=args.repeats,
            quick=args.quick,
            seed=args.seed,
            calibration_repeats=args.calibration_repeats,
        ),
        environment=environment(),
        calibrations=calibrations,
        **source_provenance(snapshot),
        results_file=args.output.with_suffix(".jsonl").name,
        summary=summarize(rows),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
    )
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        f"Saved {len(rows)} runs; {sum(row['status']!='ok' for row in rows)} failures retained"
    )
    if report["source_changed_during_run"]:
        raise SystemExit("Source changed during the experiment; see provenance flags")


if __name__ == "__main__":
    main()
