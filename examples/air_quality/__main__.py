"""Build an offline, auditable real-data teaching gallery.

python -m examples.air_quality --output docs/gallery
python -m examples.air_quality --methods mar-als projected-pca --output /tmp/gallery
Numerical source and paper attribution: docs/methods.md and docs/citations.md.
"""

import argparse
import hashlib
import json
import platform
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

import mavats

from .cases import CASES
from .data import DATA, load, preprocess

ROOT = Path(__file__).resolve().parents[2]
TRAIN = 273


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def source_hashes():
    paths = list((ROOT / "mavats").glob("*.py"))
    paths += list(Path(__file__).parent.glob("*.py"))
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def fit_case(case, x):
    """Fit only x[:TRAIN]; all custom restrictions are recorded in the code."""
    train = x[:TRAIN]
    options = dict(case.options)
    row = np.ones((4, 1))
    particulate = np.array([[1.0], [1.0], [0.0]])
    if case.id in ("constrained", "partial-constrained"):
        options.update(row_constraints=row, column_constraints=particulate)
    if case.id == "multiterm-constrained":
        options["constraints"] = [
            (np.eye(4), np.eye(3)[:, :2]),
            (np.eye(4), np.eye(3)[:, 2:]),
        ]
    if case.kind == "threshold":
        # Previous-day PM2.5 across stations, in transformed units. Drop day 1
        # rather than pretending its unavailable lag was observed.
        options["z"] = x[: TRAIN - 1, :, 0].mean(axis=1)
        train = x[1:TRAIN]
    if case.kind == "stability":
        options.update(
            c_grid=np.geomspace(1e-4, 100, 24),
            spatial_subsets=[([0, 1, 2], [0, 1]), ([0, 1, 2, 3], [0, 1, 2])],
            time_prefixes=[180, TRAIN],
            max_ranks=(1, 1),
        )
    if case.kind == "volatility":
        # First differences, training mean only. Missingness in either endpoint
        # is propagated when plotting observed squared changes below.
        train = np.diff(train, axis=0)
        train = train - train.mean(axis=0)
    if case.kind == "monitor":
        options.update(
            rank=1, projection_rank=1, horizon=len(x) - TRAIN, random_state=22
        )
        if case.id == "calibrated-monitor":
            calibration = mavats.calibrate_monitor(len(x) - TRAIN)
            options.update(calibration.monitor_kwargs)
        return mavats.MatrixFactorMonitor(train, **options)
    return getattr(mavats, case.api)(train, **options)


def mse(observed, prediction, mask):
    if observed.shape != prediction.shape or mask.shape != observed.shape:
        raise ValueError("Scoring shapes do not match")
    if not mask.any() or not np.isfinite(prediction).all():
        raise ValueError("No observed targets or nonfinite predictions")
    return float(np.mean((observed[mask] - prediction[mask]) ** 2))


def evaluate(case, result, x, mask, dates, axes, report):
    """Plot actual computed quantities; return arrays that allow score auditing."""
    a, b = axes
    test = x[TRAIN:]
    valid = mask[TRAIN:]
    test_dates = dates[TRAIN:]
    saved = {}
    for name in ("converged", "n_iter", "stop_reason", "ranks", "spectral_radius"):
        if hasattr(result, name):
            report[name] = jsonable(getattr(result, name))
    if hasattr(result, "pilot"):
        report["pilot_converged"] = bool(result.pilot.converged)
    if case.kind in ("forecast", "factor", "threshold"):
        fitted = result.result if case.api == "select_mar_rank" else result
        if case.kind == "forecast":
            if case.id == "naive-last":
                prediction = np.stack(
                    [
                        mavats.fit_naive(x[:t], strategy="last").forecast(1)[0]
                        for t in range(TRAIN, len(x))
                    ]
                )
            elif case.id == "naive-mean":
                prediction = fitted.forecast(len(test))
            else:
                prediction = np.stack(
                    [fitted.forecast(1, history=x[:t])[0] for t in range(TRAIN, len(x))]
                )
            label = "Rolling one-day forecast"
        elif case.kind == "threshold":
            z = x[TRAIN - 1 : -1, :, 0].mean(axis=1)
            prediction = result.inverse_transform(result.transform(test, z), z)
            report["threshold"] = float(result.threshold)
            label = "Same-day reconstruction"
        else:
            prediction = result.inverse_transform(result.transform(test))
            label = "Same-day reconstruction"
        error = mse(test, prediction, valid)
        report["metric"] = (
            "forecast_mse" if case.kind == "forecast" else "reconstruction_discrepancy"
        )
        report["metric_value"] = error
        report["observed_target_cells"] = int(valid.sum())
        observed_cell = test.reshape(len(test), -1)[:, 0].copy()
        observed_cell[~valid.reshape(len(test), -1)[:, 0]] = np.nan
        a.plot(test_dates, observed_cell, color="0.3", label="Observed", lw=1.3)
        a.plot(
            test_dates,
            prediction.reshape(len(test), -1)[:, 0],
            color="#0072B2",
            label=label,
            lw=1.4,
            linestyle="--",
        )
        a.set(
            ylabel="Standardized log(1 + concentration)",
            title="Aotizhongxin PM2.5" + (" · 00–11 h" if case.tensor else ""),
        )
        a.legend(fontsize=9)
        if case.kind == "forecast":
            last = x[TRAIN - 1 : -1]
            mean = np.broadcast_to(x[:TRAIN].mean(axis=0), test.shape)
            scores = [error, mse(test, last, valid), mse(test, mean, valid)]
            report["last_mse"], report["training_mean_mse"] = scores[1:]
            b.bar(
                ["Method", "Previous day", "Training mean"],
                scores,
                color=["#0072B2", "0.55", "0.75"],
            )
            b.set(
                ylabel="MSE (standardized log units squared)",
                xlabel="Frozen-parameter rolling one-day prediction",
                title="All observed held-out target cells",
            )
            saved.update(last=last, training_mean=mean)
        else:
            residual = np.where(valid, (test - prediction) ** 2, np.nan)
            by_day = np.nanmean(residual.reshape(len(test), -1), axis=1)
            b.plot(test_dates, by_day, color="#0072B2")
            b.set(
                ylabel="Mean squared reconstruction discrepancy",
                title="Observed entries only · clean signal unknown",
            )
        saved.update(observed=test, prediction=prediction, mask=valid)
    elif case.kind == "rank":
        ranks = result.ranks
        a.bar(["Station", "Pollutant"], ranks, color="#0072B2")
        a.set(
            ylabel="Selected rank",
            xlabel="Matrix mode",
            ylim=(0, 3),
            title="Training only · ranks are not validated truth",
        )
        a.set_yticks([0, 1, 2, 3])
        spectra = (
            result.pilot.eigenvalues if hasattr(result, "pilot") else result.eigenvalues
        )
        for i, spectrum in enumerate(spectra):
            b.plot(
                np.arange(1, len(spectrum) + 1),
                spectrum,
                "o-",
                label=["Station", "Pollutant"][i],
            )
        b.set(
            xlabel="Ordered component",
            ylabel="Training spectral value",
            title="Estimator-specific spectrum (scales differ)",
        )
        b.legend()
        saved.update(ranks=np.asarray(ranks))
    elif case.kind == "stability":
        for i, name in enumerate(("Station", "Pollutant")):
            a.step(result.c_grid, result.full_sample_ranks[i], where="mid", label=name)
            b.semilogx(result.c_grid, result.variances[i], "o-", label=name)
        a.set(xscale="log", xlabel="Penalty multiplier", ylabel="Full-training rank")
        b.set(xlabel="Penalty multiplier", ylabel="Subsample rank variance")
        a.legend()
        b.legend()
        choice = result.choose_plateaus(variance_tolerance=0, minimum_grid_points=3)
        report["plateau_complete"] = choice.complete
        report["selected_ranks"] = jsonable(choice.selected_ranks)
        report["plateau_reasons"] = jsonable(choice.reasons)
        saved.update(grid=result.c_grid, ranks=result.ranks, variances=result.variances)
    elif case.kind == "inference":
        lower, upper = result.confidence_interval()
        indices = np.arange(min(6, len(result.operator)))
        points = np.diag(result.operator)[indices]
        a.errorbar(
            indices,
            points,
            yerr=[points - np.diag(lower)[indices], np.diag(upper)[indices] - points],
            fmt="o",
            capsize=3,
            color="#0072B2",
        )
        a.axhline(0, color="0.5", lw=1)
        a.set(
            xlabel="First six column-major self-transition indices",
            ylabel="Transition coefficient",
            title="Nominal marginal 95% intervals",
        )
        im = b.imshow(
            result.operator,
            cmap="RdBu_r",
            vmin=-abs(result.operator).max(),
            vmax=abs(result.operator).max(),
        )
        b.figure.colorbar(im, ax=b, label="Coefficient (dimensionless)")
        b.set(
            xlabel="Lagged column-major cell",
            ylabel="Current column-major cell",
            title="Transition operator · assumptions unverified",
        )
        saved.update(operator=result.operator, lower=lower, upper=upper)
    elif case.kind == "specification":
        bound = float(abs(result.unrestricted_coefficient).max())
        im = a.imshow(
            result.unrestricted_coefficient, cmap="RdBu_r", vmin=-bound, vmax=bound
        )
        a.figure.colorbar(im, ax=a, label="Coefficient (dimensionless)")
        a.set(
            xlabel="Lagged column-major cell",
            ylabel="Current column-major cell",
            title="Unrestricted VAR transition",
        )
        b.bar(["Nominal p-value"], [result.pvalue], color="#0072B2")
        b.axhline(0.05, color="0.4", linestyle="--", label="0.05 reference")
        b.set(
            ylim=(0, 1),
            ylabel="Nominal asymptotic p-value",
            title="Not a test of every model assumption",
        )
        b.legend()
        report.update(
            statistic=result.statistic,
            pvalue=result.pvalue,
            degrees_of_freedom=result.degrees_of_freedom,
        )
    elif case.kind == "decorrelation":
        transformed = result.transform(test)
        before = np.corrcoef(test.reshape(len(test), -1), rowvar=False)
        after = np.corrcoef(transformed.reshape(len(test), -1), rowvar=False)
        for ax, matrix, title in zip(
            axes, (before, after), ("Observed coordinates", "Transformed coordinates")
        ):
            im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set(title=title, xlabel="Flattened cell", ylabel="Flattened cell")
            ax.figure.colorbar(im, ax=ax, label="Held-out contemporaneous correlation")
        report.update(
            row_groups=result.row_groups,
            column_groups=result.column_groups,
            round_trip_error=float(
                abs(result.inverse_transform(transformed) - test).max()
            ),
        )
        saved.update(before=before, after=after)
    elif case.kind == "volatility":
        changes = np.diff(x, axis=0)
        changes -= changes[: TRAIN - 1].mean(axis=0)
        heldout = changes[TRAIN - 1 :]
        filtered = result.filter(heldout)
        valid_changes = mask[TRAIN:] & mask[TRAIN - 1 : -1]
        energy = np.sum(heldout**2, axis=(1, 2))
        energy[~valid_changes.all(axis=(1, 2))] = np.nan
        a.plot(test_dates, energy, color="0.55", label="Observed squared-change norm")
        a.plot(
            test_dates,
            filtered.traces,
            color="#0072B2",
            label="Conditional covariance trace",
        )
        a.set(
            ylabel="Squared standardized log-change units",
            title="Covariance is computed before each observation",
        )
        a.legend(fontsize=8)
        b.plot(test_dates, filtered.negative_log_likelihoods, color="#0072B2")
        b.set(
            ylabel="Conditional Gaussian negative log score",
            title="Imputed inputs included · not a validated likelihood",
        )
        report["active_bounds"] = result.active_bounds
        saved.update(traces=filtered.traces, observed_energy=energy)
    elif case.kind == "monitor":
        steps = []
        for value in test:
            step = result.update(value)
            steps.append(step)
            if step.alarm:
                break
        times = test_dates[: len(steps)]
        a.plot(
            times,
            [s.statistic for s in steps],
            "o-",
            color="#0072B2",
            label="Randomized statistic",
        )
        a.axhline(result.critical_value, color="0.4", linestyle="--", label="Boundary")
        a.set(
            ylabel="Monitoring statistic",
            title="Stops at first alarm; no post-alarm continuation",
        )
        a.legend(fontsize=8)
        b.plot(
            test_dates,
            np.where(valid, test, np.nan)[:, :, 0].mean(axis=1),
            color="0.35",
        )
        if steps[-1].alarm:
            b.axvline(times[-1], color="#D55E00", label="First alarm", linestyle="--")
            b.legend()
        b.set(
            ylabel="Station-average transformed PM2.5",
            title="Context only · no known true change date",
        )
        report.update(
            first_alarm_date=str(times[-1]) if steps[-1].alarm else None,
            consumed=len(steps),
            calibration=result.calibration,
        )
        saved.update(
            statistic=np.array([s.statistic for s in steps]),
            critical_value=np.array(result.critical_value),
        )
    else:
        raise RuntimeError(f"Unhandled case kind: {case.kind}")
    return saved


def paper_links(case):
    function = getattr(mavats, case.api)
    canonical = f"{function.__module__}.{function.__name__}"
    index = (ROOT / "docs/citations.md").read_text(encoding="utf-8")
    row = next(
        line for line in index.splitlines() if line.startswith(f"| `{canonical}` |")
    )
    keys = re.findall(r"\[([\w]+)\]\(#", row)
    return ", ".join(f"[{key}](../citations.md#{key})" for key in keys)


def build_case(case, output, dates, values, counts):
    x, mask, preprocessing = preprocess(values, counts, TRAIN, case.tensor)
    report = {
        "id": case.id,
        "api": case.api,
        "kind": case.kind,
        "shape": list(x.shape),
        "preprocessing": preprocessing,
        "options": case.options,
        "status": "completed",
    }
    if case.kind == "decorrelation":
        figure, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    else:
        figure, axes = plt.subplots(2, 1, figsize=(9, 7), layout="constrained")
    figure.suptitle(case.title + " · Beijing 2014", fontsize=14)
    saved = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = fit_case(case, x)
            saved = evaluate(case, result, x, mask, dates, axes, report)
            flags = [report[k] for k in ("converged", "pilot_converged") if k in report]
            if any(not np.all(flag) for flag in flags):
                report["status"] = "not_converged"
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            report.update(status="rejected", error=f"{type(error).__name__}: {error}")
            for ax in axes:
                ax.clear()
            axes[0].plot(dates, x.reshape(len(x), -1)[:, 0], color="0.4")
            axes[0].axvline(dates[TRAIN], linestyle="--", color="#0072B2")
            axes[0].set(
                ylabel="Transformed first cell",
                title="Input context · no successful fit",
            )
            axes[1].axis("off")
            import textwrap

            axes[1].text(
                0,
                0.8,
                "Fit rejected — no estimate substituted\n\n"
                + textwrap.fill(str(error), 85),
                transform=axes[1].transAxes,
                va="top",
                fontsize=10,
            )
        report["warnings"] = list(dict.fromkeys(str(w.message) for w in caught))
    for ax in axes:
        if not ax.get_xlabel() and ax.axison:
            ax.set_xlabel("Recorded date (2014)")
        if ax.axison:
            ax.spines[["top", "right"]].set_visible(False)
    if report["status"] != "completed":
        figure.suptitle(
            case.title
            + " · Beijing 2014\n"
            + report["status"].replace("_", " ").upper()
            + " — diagnostic output only",
            fontsize=12,
        )
    figure.savefig(output / f"{case.id}.png", dpi=150)
    plt.close(figure)
    np.savez_compressed(output / f"{case.id}.npz", **saved)
    (output / f"{case.id}.json").write_text(
        json.dumps(jsonable(report), indent=2, allow_nan=False) + "\n"
    )
    options = ", ".join(f"{key}={value!r}" for key, value in case.options.items())
    special = (
        case.kind in ("threshold", "stability", "monitor", "volatility")
        or "constrained" in case.id
    )
    snippet = (
        "# See fit_case for the explicit constraints, lag alignment or stream setup.\n"
        "from examples.air_quality.__main__ import fit_case\n"
        "result = fit_case(case, x)"
        if special
        else f"from mavats import {case.api}\nresult = {case.api}(x[:273]"
        + (", " + options if options else "")
        + ")"
    )
    setup = (
        "from examples.air_quality.data import load, preprocess\n"
        "from examples.air_quality.cases import CASES\n"
        f'case = next(c for c in CASES if c.id == "{case.id}")\n'
        "dates, values, counts = load()\n"
        f"x, observed, preprocessing = preprocess(values, counts, train=273, tensor={case.tensor})\n"
    )
    snippet = setup + snippet
    result_text = (
        f"Training cells imputed: {preprocessing['imputed_training_cells']}; "
        f"held-out cells imputed: {preprocessing['imputed_test_cells']}. "
        "Gaps in the observed trace mark insufficient readings; predictions over "
        "those gaps are not scored.\n\n"
    )
    if "metric_value" in report:
        result_text += (
            f"Computed {report['metric']}: **{report['metric_value']:.4f}** "
            f"over {report['observed_target_cells']} observed target cells. "
        )
    if "last_mse" in report:
        result_text += (
            f"Previous-day MSE: {report['last_mse']:.4f}; "
            f"fixed-training-mean MSE: {report['training_mean_mse']:.4f}. "
            "Lower is better on this split only; no uncertainty interval is estimated."
        )
    page = f"""# {case.title}: an air-quality walkthrough

[Gallery and data protocol](../real-world-examples.md) · [All examples](index.md) · [API](https://dx-li.github.io/MaVaTS/mavats.html#{case.api})

## Question and applicability

{case.question}

{case.caution}

Paper attribution: {paper_links(case)}. This is a new teaching application,
not reproduction of a paper's empirical results or endorsement by its authors.

## Data and execution

Observed Beijing data: {tuple(x.shape)} = day × station × pollutant{(' × half-day' if case.tensor else '')}.
Train: January–September 2014 (273 days). Held out: October–December (92 days).
The [shared protocol](../real-world-examples.md#data-and-preprocessing) specifies
provenance, missingness, training-only transformations and observational limits.

From a checkout with the examples extra installed:

```bash
python -m examples.air_quality --methods {case.id} --output /tmp/mavats-gallery
```

The snippet includes shared preprocessing and the core call. Special setups
use `fit_case`, whose source explicitly defines constraints, lag alignment or
stream state. The executable [runner](../../examples/air_quality/__main__.py)
also contains the complete evaluation and plotting recipe.

```python
{snippet}
```

## Computed result

Run status: **{report['status']}**. This records numerical execution, not
scientific validity. Unconverged output is diagnostic only; rejected fits
have no substituted estimate. [Full diagnostics]({case.id}.json) include
warnings, convergence, preprocessing counts and method-specific quantities.

{result_text}

![{case.title}: computed output and diagnostics; see adjacent interpretation and JSON data]({case.id}.png)

The first-cell display is predeclared (Aotizhongxin PM2.5, morning for tensors),
not selected for visual appeal. Forecast/reconstruction metrics use all observed
held-out cells, excluding imputed targets. Reconstruction sees the target day;
it must not be read as a forecast or measured recovery of an unknown clean signal.
Diagnostic-only plots can instead use training data as explicitly labelled.

[Numerical plot/score arrays]({case.id}.npz) and [run provenance](provenance.json)
allow checking displayed results. No causal, health or regulatory conclusion follows.
"""
    (output / f"{case.id}.md").write_text(page, encoding="utf-8")
    print(
        case.id,
        report["status"],
        report.get("metric_value", report.get("error", "")),
        flush=True,
    )
    return jsonable(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/gallery"))
    parser.add_argument("--methods", nargs="+", choices=[c.id for c in CASES])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dates, values, counts = load()
    selected = [c for c in CASES if not args.methods or c.id in args.methods]
    reports = [build_case(c, args.output, dates, values, counts) for c in selected]
    provenance = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "mavats": mavats.__version__,
        "source_sha256": source_hashes(),
        "data": json.loads((DATA / "provenance.json").read_text()),
        "methods": [c.id for c in selected],
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    (args.output / "summary.json").write_text(
        json.dumps(reports, indent=2, allow_nan=False) + "\n"
    )
    rows = [
        f"| [{c.title}]({c.id}.md) | `{c.api}` | {c.kind} | {r['status']} |"
        for c, r in zip(selected, reports)
    ]
    (args.output / "index.md").write_text(
        "# Real-data method gallery\n\n"
        "Read the [data protocol and interpretation guide](../real-world-examples.md) first.\n"
        "All applications use the same fixed observational split; none establishes model validity.\n\n"
        "| Walkthrough | API | Output type | Execution status |\n"
        "| --- | --- | --- | --- |\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
