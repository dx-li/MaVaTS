"""Entrywise IHR versus matrixwise robust and nonrobust factor estimators."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from benchmarks.run import _measure, environment, source_provenance, source_snapshot
from benchmarks.structured_scenarios import contaminated_factor_data
from mavats.factors import fit_alpha_pca, fit_projected_pca
from mavats.huber import fit_huber_factor
from mavats.metrics import relative_frobenius_error, subspace_distance
from mavats.robust import fit_matrix_kendall


def run_suite(*, quick=False, repeats=10, seed=1693):
    from mavats.ihr import fit_ihr_factor, select_ihr_ranks

    rows = []
    for replicate in range(repeats):
        current = seed + replicate
        n, shape = (60, (5, 6)) if quick else (250, (8, 10))
        for regime in ("gaussian", "entry-outliers", "matrix-outliers"):
            data = contaminated_factor_data(
                n + 30, shape=shape, regime=regime, seed=current
            )
            train, test = data.observations[:n], data.observations[n:]
            metadata = dict(
                task="entrywise-robust-factors",
                regime=regime,
                replicate=replicate,
                seed=current,
                shape=list(shape),
                n_train=n,
                n_test=30,
                true_ranks=[2, 2],
                contaminated_training_entries=int(
                    np.count_nonzero(data.contamination_mask[:n])
                ),
                contaminated_test_entries=int(
                    np.count_nonzero(data.contamination_mask[n:])
                ),
                held_out_task="contemporaneous-denoising-not-forecasting",
            )

            def score(result):
                fit = result.fitted if hasattr(result, "fitted") else result
                extras = {}
                if hasattr(fit, "block_history"):
                    factors, diagnostics = fit.transform(test, return_diagnostics=True)
                    extras["test_score_solves_converged"] = all(
                        d.converged for d in diagnostics
                    )
                    extras["test_score_max_norm"] = max(
                        d.score_norm for d in diagnostics
                    )
                    extras["test_score_max_iterations"] = max(
                        d.n_iter for d in diagnostics
                    )
                else:
                    factors = fit.transform(test)
                if hasattr(result, "selected_ranks"):
                    extras.update(
                        selected_ranks=list(result.selected_ranks),
                        rank_pilot_converged=result.rank_pilot_converged,
                        rank_pilot_iterations=result.rank_pilot_iterations,
                        rank_pilot_stopping_reason=result.rank_pilot_stopping_reason,
                    )
                return dict(
                    signal_error=relative_frobenius_error(data.signal[:n], fit.signal),
                    held_out_signal_error=relative_frobenius_error(
                        data.signal[n:], fit.inverse_transform(factors)
                    ),
                    subspace_error=float(
                        np.mean(
                            [
                                subspace_distance(a, b)
                                for a, b in zip(data.loadings, fit.loadings)
                            ]
                        )
                    ),
                    ranks=list(fit.ranks),
                    threshold=getattr(fit, "threshold", None),
                    stopping_reason=getattr(fit, "stopping_reason", None),
                    **extras,
                )

            def automatic():
                selection = select_ihr_ranks(
                    train, (3, 3), max_iter=100, inner_max_iter=200
                )
                fit = fit_ihr_factor(
                    train, selection.ranks, max_iter=100, inner_max_iter=200
                )
                return SimpleNamespace(
                    fitted=fit,
                    selected_ranks=selection.ranks,
                    rank_pilot_converged=selection.pilot.converged,
                    rank_pilot_iterations=selection.pilot.n_iter,
                    rank_pilot_stopping_reason=selection.pilot.stopping_reason,
                    converged=fit.converged,
                    n_iter=fit.n_iter,
                )

            methods = (
                (
                    "ihr-known-ranks",
                    lambda: fit_ihr_factor(
                        train, (2, 2), max_iter=100, inner_max_iter=200
                    ),
                    dict(
                        ranks=[2, 2],
                        threshold="fixed-projected-pilot-MAD",
                        max_iter=100,
                        inner_max_iter=200,
                    ),
                ),
                (
                    "ihr-auto-ranks",
                    automatic,
                    dict(
                        max_ranks=[3, 3],
                        rank_method="ratio",
                        ridge=1e-4,
                        threshold="fixed-projected-pilot-MAD",
                        max_iter=100,
                        inner_max_iter=200,
                    ),
                ),
                (
                    "matrixwise-huber-known-ranks",
                    lambda: fit_huber_factor(train, (2, 2)),
                    dict(ranks=[2, 2], threshold="median-initial-matrix-residual-norm"),
                ),
                (
                    "matrix-kendall-known-ranks",
                    lambda: fit_matrix_kendall(train, (2, 2)),
                    dict(ranks=[2, 2]),
                ),
                (
                    "alpha-pca-known-ranks",
                    lambda: fit_alpha_pca(train, (2, 2), alpha=0),
                    dict(ranks=[2, 2], alpha=0),
                ),
                (
                    "projected-pca-known-ranks",
                    lambda: fit_projected_pca(train, (2, 2)),
                    dict(ranks=[2, 2]),
                ),
            )
            for name, factory, options in methods:
                rows.append(
                    _measure(name, factory, score, dict(metadata, options=options))
                )
        print(
            f"Completed entrywise robust replicate {replicate+1}/{repeats}", flush=True
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1693)
    parser.add_argument("--output", type=Path, default=Path("entrywise-robust.json"))
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
