"""Paired partial/multi-term constrained factor integration experiments.

Reference model/estimators: Chen, Tsay and Chen (2020), Constrained Factor
Models for High-Dimensional Matrix-Variate Time Series,
https://doi.org/10.1080/01621459.2019.1584899. Known-loading projections and
independent single-term sums below are explicitly labeled comparison devices,
not additional published multi-term estimators.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.marma_scenarios import multiterm_factor_data, partial_factor_data
from benchmarks.run import _measure, environment, source_provenance, source_snapshot
from mavats.constrained import fit_constrained_factor
from mavats.factors import fit_lagged_factor, fit_projected_pca
from mavats.metrics import relative_frobenius_error, subspace_distance


class _KnownLoadingProjection:
    """Observed-data LS projection into known signal spans; extra information.

    References
    ----------
    Chen, Tsay and Chen (2020), https://doi.org/10.1080/01621459.2019.1584899.
    The model supplies the signal spans; known-loading joint LS is a benchmark
    completion, not the paper's loading estimator or a universal lower bound.
    """

    converged = True
    n_iter = 0

    def __init__(self, X, terms):
        self.shape = X.shape[1:]
        self.terms = terms
        self.design = np.concatenate([np.kron(c, r) for r, c in terms], axis=1)
        u, s, vh = np.linalg.svd(self.design, full_matrices=False)
        if s[-1] <= s[0] * np.finfo(float).eps * max(self.design.shape):
            raise ValueError("known signal design is rank deficient")
        self.inverse = (vh.T / s) @ u.T
        self.signal = self.inverse_transform(self.transform(X))

    def transform(self, X):
        return np.stack([x.ravel(order="F") for x in X]) @ self.inverse.T

    def inverse_transform(self, factors):
        flat = factors @ self.design.T
        return np.stack([x.reshape(self.shape, order="F") for x in flat])


class _IndependentSum:
    """Deliberately unadjusted single-term fits; overlap is double-counted.

    References
    ----------
    Chen, Tsay and Chen (2020), https://doi.org/10.1080/01621459.2019.1584899.
    Uses the single-term estimator without the paper's competing-span
    adjustment; this comparison is not the published multi-term method.
    """

    converged = True
    n_iter = 0

    def __init__(self, X, constraints):
        self.fits = [
            fit_constrained_factor(X, (1, 1), row_constraints=r, column_constraints=c)
            for r, c in constraints
        ]
        self.loadings = tuple(f.loadings for f in self.fits)
        self.signal = sum(f.signal for f in self.fits)

    def transform(self, X):
        return tuple(f.transform(X) for f in self.fits)

    def inverse_transform(self, cores):
        return sum(f.inverse_transform(core) for f, core in zip(self.fits, cores))


def _score_signal(fit, data, n):
    reconstructed = fit.inverse_transform(fit.transform(data.observations[n:]))
    return dict(
        signal_error=relative_frobenius_error(data.signal[:n], fit.signal),
        held_out_signal_error=relative_frobenius_error(data.signal[n:], reconstructed),
    )


def _projection_ranks(loadings, constraints):
    """Ranks relative to supplied spans, with tolerance on the original scale."""
    projection = constraints @ constraints.T @ loadings
    tolerance = (
        np.finfo(float).eps * max(loadings.shape) * np.linalg.norm(loadings, ord=2)
    )
    return [
        int(np.count_nonzero(np.linalg.svd(part, compute_uv=False) > tolerance))
        for part in (projection, loadings - projection)
    ]


def run_suite(*, quick=False, repeats=10, seed=2173):
    from mavats.constrained import (
        fit_multiterm_constrained_factor,
        fit_partial_constrained_factor,
    )

    rows = []
    n, shape = (80, (6, 8)) if quick else (400, (8, 10))
    for replicate in range(repeats):
        current = seed + replicate
        for regime in (
            "full-interactions",
            "diagonal-only",
            "weak-complement",
            "misspecified-constraints",
        ):
            data = partial_factor_data(n + 30, shape=shape, regime=regime, seed=current)
            train = data.observations[:n]
            constraints = dict(
                row_constraints=data.row_constraints,
                column_constraints=data.column_constraints,
            )
            fixed = dict(row_ranks=data.row_ranks, column_ranks=data.column_ranks)
            metadata = dict(
                task="partial-constrained-factors",
                regime=regime,
                replicate=replicate,
                seed=current,
                shape=list(shape),
                n_train=n,
                n_test=30,
                generating_row_ranks=list(data.row_ranks),
                generating_column_ranks=list(data.column_ranks),
                true_supplied_projection_row_ranks=_projection_ranks(
                    data.loadings[0], data.row_constraints
                ),
                true_supplied_projection_column_ranks=_projection_ranks(
                    data.loadings[1], data.column_constraints
                ),
                held_out_task="contemporaneous-denoising-not-forecasting",
                supplied_constraints_correct=regime != "misspecified-constraints",
            )

            def score(fit):
                result = _score_signal(fit, data, n)
                if hasattr(fit, "loadings"):
                    result["subspace_error"] = float(
                        np.mean(
                            [
                                subspace_distance(a, b)
                                for a, b in zip(data.loadings, fit.loadings)
                            ]
                        )
                    )
                    result["ranks"] = list(fit.ranks)
                if hasattr(fit, "row_ranks"):
                    result["selected_row_ranks"] = list(fit.row_ranks)
                    result["selected_column_ranks"] = list(fit.column_ranks)
                    result["interactions"] = fit.interactions
                    result["group_rank_diagnostics"] = fit.rank_diagnostics
                return result

            methods = (
                (
                    "partial-fixed-group-ranks",
                    lambda: fit_partial_constrained_factor(
                        train, **constraints, **fixed
                    ),
                    dict(**fixed, interactions=True),
                ),
                (
                    "partial-auto-ranks",
                    lambda: fit_partial_constrained_factor(train, **constraints),
                    dict(
                        row_ranks=[None, None],
                        column_ranks=[None, None],
                        interactions=True,
                    ),
                ),
                (
                    "partial-diagonal-fixed-group-ranks",
                    lambda: fit_partial_constrained_factor(
                        train, **constraints, **fixed, interactions=False
                    ),
                    dict(**fixed, interactions=False),
                ),
                (
                    "fully-constrained-known-primary-ranks",
                    lambda: fit_constrained_factor(train, (1, 2), **constraints),
                    dict(ranks=[1, 2]),
                ),
                (
                    "unconstrained-lagged-known-total-ranks",
                    lambda: fit_lagged_factor(train, (3, 3)),
                    dict(ranks=[3, 3]),
                ),
                (
                    "projected-pca-known-total-ranks",
                    lambda: fit_projected_pca(train, (3, 3)),
                    dict(ranks=[3, 3]),
                ),
                (
                    "known-loading-projection-oracle",
                    lambda: _KnownLoadingProjection(train, [data.loadings]),
                    dict(extra_information="true-loading-spaces"),
                ),
            )
            for name, function, options in methods:
                rows.append(
                    _measure(
                        name,
                        function,
                        score,
                        dict(metadata, options=options),
                    )
                )

        for regime in ("orthogonal", "overlapping", "near-overlap"):
            data = multiterm_factor_data(
                n + 30, shape=shape, regime=regime, seed=current + 200
            )
            train = data.observations[:n]
            metadata = dict(
                task="multi-term-constrained-factors",
                regime=regime,
                replicate=replicate,
                seed=current + 200,
                shape=list(shape),
                n_train=n,
                n_test=30,
                true_ranks=[[1, 1], [1, 1]],
                held_out_task="contemporaneous-denoising-not-forecasting",
            )

            def score_multi(fit):
                result = _score_signal(fit, data, n)
                if hasattr(fit, "component_signals"):
                    result["component_signal_error"] = [
                        relative_frobenius_error(target[:n], estimate)
                        for target, estimate in zip(
                            data.component_signals, fit.component_signals
                        )
                    ]
                if hasattr(fit, "identification_diagnostics"):
                    result["identification_diagnostics"] = (
                        fit.identification_diagnostics
                    )
                    result["rank_diagnostics"] = fit.rank_diagnostics
                    result["ranks"] = fit.ranks
                if hasattr(fit, "score_condition"):
                    result["score_condition"] = fit.score_condition
                    result["score_design_rank"] = fit.score_design_rank
                return result

            methods = (
                (
                    "multi-term-known-ranks",
                    lambda: fit_multiterm_constrained_factor(
                        train, data.constraints, [(1, 1), (1, 1)]
                    ),
                    dict(ranks=[[1, 1], [1, 1]]),
                ),
                (
                    "independent-constrained-sum",
                    lambda: _IndependentSum(train, data.constraints),
                    dict(ranks=[[1, 1], [1, 1]], overlap_adjusted=False),
                ),
                (
                    "unconstrained-lagged-known-total-ranks",
                    lambda: fit_lagged_factor(train, (2, 2)),
                    dict(ranks=[2, 2]),
                ),
                (
                    "projected-pca-known-total-ranks",
                    lambda: fit_projected_pca(train, (2, 2)),
                    dict(ranks=[2, 2]),
                ),
                (
                    "known-loading-joint-projection-oracle",
                    lambda: _KnownLoadingProjection(train, data.loadings),
                    dict(extra_information="true-term-loading-spaces"),
                ),
            )
            for name, function, options in methods:
                rows.append(
                    _measure(
                        name,
                        function,
                        score_multi,
                        dict(metadata, options=options),
                    )
                )
        print(
            f"Completed constrained-factor replicate {replicate + 1}/{repeats}",
            flush=True,
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2173)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark-constrained-extensions.json")
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    from benchmarks import marma_scenarios

    snapshot = source_snapshot(__file__, marma_scenarios.__file__)
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
