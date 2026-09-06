"""Bounded retained IC-constant stability experiments with explicit absences.

Han, Chen and Zhang (2022), Rank Determination in Tensor Factor Model,
https://doi.org/10.1214/22-EJS1991, Remark 8 and Section 5.4. These nested-grid
experiments are integration studies, not reproductions of the paper's tables.
Zero/cap plateaus are excluded under the explicit positive-rank convention;
absence of an admissible plateau is a result, not an execution failure.
"""

import argparse
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from benchmarks.factor_ranks import _scores
from benchmarks.rank_scenarios import rank_factor_data
from benchmarks.run import environment, source_provenance, source_snapshot
from mavats.rank_selection import select_tensor_rank
from mavats.rank_stability import tensor_rank_stability


def _json_safe(value):
    """Preserve unavailable diagnostics as null in strict JSON, never NaN."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, np.ndarray)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _settings(quick):
    sizes = ((4, 6), (5, 7), (6, 8)) if quick else ((6, 8), (8, 10), (10, 12))
    return dict(
        n=70 if quick else 300,
        shape=sizes[-1],
        spatial_subsets=tuple(tuple(tuple(range(d)) for d in dims) for dims in sizes),
        time_prefixes=(30, 50, 70) if quick else (100, 200, 300),
        max_ranks=(3, 3) if quick else (4, 4),
        c_grid=np.geomspace(0.01, 100.0, 9 if quick else 25),
    )


def _fit_score(train, data, n, options, multipliers):
    record = dict(penalty_multipliers=multipliers)
    phase = "fit"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = perf_counter()
        try:
            fit = select_tensor_rank(train, penalty_multiplier=multipliers, **options)
            record["fit_seconds"] = perf_counter() - start
            record.update(converged=bool(fit.converged), iterations=int(fit.n_iter))
            phase = "score"
            record.update(_scores(fit, data, n))
            record["status"] = "ok"
        except Exception as exc:
            if "fit_seconds" not in record:
                record["fit_seconds"] = perf_counter() - start
            record.update(
                status="error",
                failure_phase=phase,
                error=f"{type(exc).__name__}: {exc}",
            )
        record["warnings"] = [str(item.message) for item in caught]
    return record


def _run_case(data, settings, *, iterative, metadata):
    n = settings["n"]
    train = data.observations[:n]
    options = dict(
        method="tipup",
        criterion="ic",
        penalty=2,
        lags=1,
        max_ranks=settings["max_ranks"],
        nu=0.0,
        center=True,
        iterative=iterative,
        max_iter=100,
        tol=1e-8,
    )
    record = dict(
        metadata,
        method="ic2-stability-" + ("itipup" if iterative else "tipup"),
        options=options,
        c_grid=settings["c_grid"],
        spatial_subsets=settings["spatial_subsets"],
        time_prefixes=settings["time_prefixes"],
        plateau_options=dict(
            variance_tolerance=0.0, minimum_grid_points=2, allow_unconverged=False
        ),
    )
    record["fixed_c1"] = _fit_score(train, data, n, options, 1.0)
    record["final_refit"] = dict(status="not-run", reason="path_unavailable")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = perf_counter()
        phase = "path"
        try:
            path_options = {
                key: value for key, value in options.items() if key != "criterion"
            }
            path = tensor_rank_stability(
                train,
                settings["c_grid"],
                spatial_subsets=settings["spatial_subsets"],
                time_prefixes=settings["time_prefixes"],
                **path_options,
            )
            record["path_seconds"] = perf_counter() - start
            phase = "path_diagnostics"
            cells = [
                [
                    dict(
                        error=cell.error,
                        converged=cell.converged,
                        iterations=cell.n_iter,
                        ranks=None if cell.result is None else cell.result.ranks,
                        stopping_reason=(
                            None if cell.result is None else cell.result.stop_reason
                        ),
                    )
                    for cell in row
                ]
                for row in path.cells
            ]
            failed = sum(cell.error is not None for row in path.cells for cell in row)
            unfinished = sum(
                cell.result is not None and not cell.converged
                for row in path.cells
                for cell in row
            )
            choice = path.choose_plateaus(**record["plateau_options"])
            record.update(
                status="ok",
                ranks=path.ranks,
                full_sample_ranks=path.full_sample_ranks,
                variances=path.variances,
                variance_ddof=0,
                cells=cells,
                cell_converged=path.converged,
                cell_failures=failed,
                cell_unconverged=unfinished,
                all_cells_converged=bool(path.converged.all()),
                monotone=path.monotone,
                choice=asdict(choice),
                selection_complete=choice.complete,
                selected_penalty_multipliers=choice.penalty_multipliers,
                selected_path_ranks=choice.selected_ranks,
            )
            if choice.complete:
                record["final_refit"] = _fit_score(
                    train, data, n, options, choice.penalty_multipliers
                )
                if record["final_refit"]["status"] == "ok":
                    record["joint_refit_matches_selected_path_ranks"] = (
                        tuple(record["final_refit"]["ranks"]) == choice.selected_ranks
                    )
            else:
                record["final_refit"] = dict(
                    status="not-run", reason="incomplete_modewise_plateau_choice"
                )
        except Exception as exc:
            if "path_seconds" not in record:
                record["path_seconds"] = perf_counter() - start
            record.update(
                status="error",
                failure_phase=phase,
                error=f"{type(exc).__name__}: {exc}",
            )
        record["warnings"] = [str(item.message) for item in caught]
    return _json_safe(record)


def run_suite(*, quick=False, repeats=None, seed=2573):
    """Return eight path records per repetition; every fit is training-only."""
    repeats = (1 if quick else 10) if repeats is None else repeats
    if (
        isinstance(repeats, bool)
        or not isinstance(repeats, (int, np.integer))
        or repeats < 1
    ):
        raise ValueError("repeats must be a positive integer")
    settings = _settings(quick)
    n, shape = settings["n"], settings["shape"]
    records = []
    for replicate in range(repeats):
        current = seed + replicate
        for regime in ("strong", "weak", "cancellation", "no-factors"):
            data = rank_factor_data(n + 30, shape=shape, regime=regime, seed=current)
            metadata = dict(
                task="factor-rank-penalty-stability",
                regime=regime,
                replicate=replicate,
                seed=current,
                n_train=n,
                n_test=30,
                shape=shape,
                true_ranks=data.ranks,
                noise_temporally_white=data.noise_ar == 0,
                informative_tipup=regime in ("strong", "weak"),
                positive_rank_plateau_assumption=regime != "no-factors",
                weakest_loading_strength=0.5 if regime == "weak" else 0.0,
                assumed_nu=0.0,
                held_out_task="contemporaneous-denoising-not-forecasting",
                interpretation=(
                    "lag-one-inner-product-cancellation-stress"
                    if regime == "cancellation"
                    else (
                        "positive-rank-plateau-assumption-violation"
                        if regime == "no-factors"
                        else (
                            "unknown-strength-default-penalty"
                            if regime == "weak"
                            else "informative-strong-factors"
                        )
                    )
                ),
            )
            for iterative in (False, True):
                records.append(
                    _run_case(data, settings, iterative=iterative, metadata=metadata)
                )
            print(
                f"Completed stability replicate {replicate + 1}/{repeats}: {regime}",
                flush=True,
            )
    return records


def _execution_errors(rows):
    return sum(
        int(row["status"] != "ok")
        + row.get("cell_failures", 0)
        + int(row.get("fixed_c1", {}).get("status") == "error")
        + int(row.get("final_refit", {}).get("status") == "error")
        for row in rows
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2573)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark-rank-stability.json")
    )
    args = parser.parse_args()
    if args.repeats is not None and args.repeats < 1:
        parser.error("--repeats must be positive")
    from benchmarks import factor_ranks, rank_scenarios

    snapshot = source_snapshot(__file__, factor_ranks.__file__, rank_scenarios.__file__)
    repeats = args.repeats if args.repeats is not None else (1 if args.quick else 10)
    rows = run_suite(quick=args.quick, repeats=repeats, seed=args.seed)
    report = _json_safe(
        dict(
            schema_version=1,
            configuration=dict(quick=args.quick, repeats=repeats, seed=args.seed),
            environment=environment(),
            results=rows,
            **source_provenance(snapshot),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    errors = _execution_errors(rows)
    absent = sum(row.get("selection_complete") is False for row in rows)
    unfinished = sum(row.get("cell_unconverged", 0) for row in rows)
    print(
        f"Saved {len(rows)} paths; {errors} execution errors; "
        f"{absent} incomplete choices; {unfinished} unconverged path cells"
    )
    if errors or report["source_changed_during_run"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
