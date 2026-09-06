"""Summarize retained replicates without dropping failures from the counts."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def summarize(report):
    groups = defaultdict(list)
    for row in report["results"]:
        groups[(row["task"], row["regime"], row["method"])].append(row)
    lines = [
        "# Benchmark integration results",
        "",
        "Scores are held-out one-step MSE for forecasts and relative in-sample signal",
        "error for factor/CP tasks. SE is the sample standard deviation divided by",
        "the square root of successful replications. This describes Monte Carlo",
        "variation for these fixed synthetic scenarios, not uncertainty about",
        "performance across arbitrary datasets. Successful unconverged runs remain",
        "in the summaries; failed runs remain in the counts. Oracle methods receive",
        "extra information. Fit times include full calls without dedicated warm-up.",
        "",
        "| Task | Scenario | Method | Successful / total | Unconverged | Mean score | SE | Median fit ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for (task, regime, method), rows in sorted(groups.items()):
        good = [row for row in rows if row["status"] == "ok"]
        unconverged = sum(not row["converged"] for row in good)
        scores = np.array(
            [row.get("forecast_mse", row.get("signal_error")) for row in good],
            dtype=float,
        )
        mean = f"{scores.mean():.5g}" if len(scores) else "—"
        se = (
            f"{scores.std(ddof=1) / np.sqrt(len(scores)):.3g}"
            if len(scores) > 1
            else "—"
        )
        timing = (
            f"{np.median([row['fit_seconds'] for row in good]) * 1000:.3g}"
            if good
            else "—"
        )
        lines.append(
            f"| {task} | {regime} | {method} | {len(good)} / {len(rows)} | {unconverged} | {mean} | {se} | {timing} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text()))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
