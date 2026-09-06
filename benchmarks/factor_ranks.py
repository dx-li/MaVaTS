"""Published IC/ER factor ranks, weak signal and exact lag cancellation.

Han, Chen and Zhang (2022), https://doi.org/10.1214/22-EJS1991, supplies the
forty criterion/penalty/projection combinations. Adjacent ratios are the
conventional comparison of Wang, Liu and Chen (2019),
https://doi.org/10.1016/j.jeconom.2018.09.013. Known-loading and training-mean
comparators receive explicitly labeled information or impose a null model.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks.rank_scenarios import rank_factor_data
from benchmarks.run import _measure, environment, source_provenance, source_snapshot
from mavats.factors import eigenvalue_ratio
from mavats.metrics import mean_squared_error, relative_frobenius_error
from mavats.tensor import fit_tensor_factor


class _Projection:
    """Fixed known-loading or empty-loading projection with a training mean.

    References
    ----------
    Han, Chen and Zhang (2022), https://doi.org/10.1214/22-EJS1991, supplies
    the Tucker signal model. True loading spaces are extra information;
    the empty-loading fit is a historical-mean comparator, not a rank test.
    """

    converged = True
    n_iter = 0

    def __init__(self, X, loadings):
        self.mean = X.mean(axis=0)
        self.loadings = tuple(
            np.linalg.qr(a, mode="reduced")[0] if a.shape[1] else a.copy()
            for a in loadings
        )
        self.ranks = tuple(a.shape[1] for a in self.loadings)
        self.signal = self.inverse_transform(self.transform(X))

    def transform(self, X):
        row, col = self.loadings
        return np.array([row.T @ (x - self.mean) @ col for x in X])

    def inverse_transform(self, cores):
        row, col = self.loadings
        return np.array([row @ f @ col.T + self.mean for f in cores])


def _adjacent_fit(X, method, iterative, max_ranks, lags):
    """Existing adjacent-ratio initialization with a matched search cap.

    References
    ----------
    Wang, Liu and Chen (2019), https://doi.org/10.1016/j.jeconom.2018.09.013.
    The initialized ranks stay fixed during tensor projection, unlike the
    rank-updating Han et al. (2022) estimator. Pilot time is included.
    """
    pilot = fit_tensor_factor(X, ranks=(1, 1), method=method, lags=lags, center=True)
    ranks = tuple(
        eigenvalue_ratio(values, max_rank=maximum)
        for values, maximum in zip(pilot.eigenvalues, max_ranks)
    )
    return fit_tensor_factor(
        X, ranks, method=method, lags=lags, iterative=iterative, center=True
    )


def _scores(fit, data, n):
    held_out = fit.inverse_transform(fit.transform(data.observations[n:]))
    ranks = tuple(int(r) for r in fit.ranks)
    difference = np.array(ranks) - np.array(data.ranks)
    result = dict(
        ranks=list(ranks),
        rank_correct=bool(ranks == data.ranks),
        rank_absolute_error=float(np.abs(difference).mean()),
        underestimated_modes=int(np.count_nonzero(difference < 0)),
        overestimated_modes=int(np.count_nonzero(difference > 0)),
        training_signal_mse=mean_squared_error(data.signal[:n], fit.signal),
        held_out_signal_mse=mean_squared_error(data.signal[n:], held_out),
    )
    if any(data.ranks):
        result["held_out_relative_signal_error"] = relative_frobenius_error(
            data.signal[n:], held_out
        )
    if hasattr(fit, "rank_history"):
        result.update(
            initial_ranks=list(fit.initial_ranks),
            starting_ranks=list(fit.starting_ranks),
            rank_history=[list(r) for r in fit.rank_history],
            signal_ranks=list(fit.signal_ranks),
            projector_changes=list(fit.projector_changes),
            stopping_reason=fit.stop_reason,
            terminal_diagnostics=[
                dict(
                    rank=step.rank,
                    eigenvalues=step.eigenvalues.tolist(),
                    log_eigenvalue_scale=step.log_eigenvalue_scale,
                    log_penalty=step.log_penalty,
                    candidate_ranks=step.candidate_ranks.tolist(),
                    scores=step.scores.tolist(),
                    log_score_scale=step.log_score_scale,
                    projected_shape=list(step.projected_shape),
                )
                for step in fit.diagnostics
            ],
        )
    return result


def run_suite(*, quick=False, repeats=10, seed=2573):
    from mavats.rank_selection import select_tensor_rank

    n, shape = (70, (6, 8)) if quick else (300, (10, 12))
    max_ranks = (3, 3) if quick else (4, 4)
    configurations = [
        (regime, 1)
        for regime in (
            "strong",
            "weak",
            "cancellation",
            "white-factors",
            "no-factors",
            "serial-noise",
        )
    ] + [("cancellation", 2)]
    penalties = (1, 2) if quick else range(1, 6)
    rows = []
    for replicate in range(repeats):
        current = seed + replicate
        for regime, lags in configurations:
            data = rank_factor_data(n + 30, shape=shape, regime=regime, seed=current)
            train = data.observations[:n]
            metadata = dict(
                task="factor-rank-selection",
                regime=regime,
                lags=lags,
                replicate=replicate,
                seed=current,
                shape=list(shape),
                n_train=n,
                n_test=30,
                true_ranks=list(data.ranks),
                noise_temporally_white=data.noise_ar == 0,
                informative_topup=regime not in ("white-factors", "no-factors"),
                informative_tipup=regime not in ("white-factors", "no-factors")
                and (regime != "cancellation" or lags > 1),
                weakest_loading_strength=0.5 if regime == "weak" else 0.0,
                held_out_task="contemporaneous-denoising-not-forecasting",
            )
            methods = []
            for method in ("topup", "tipup"):
                for iterative in (False, True):
                    name = ("i" if iterative else "") + method
                    for criterion in ("ic", "er"):
                        for penalty in penalties:
                            options = dict(
                                method=method,
                                iterative=iterative,
                                criterion=criterion,
                                penalty=penalty,
                                max_ranks=max_ranks,
                                lags=lags,
                                center=True,
                                nu=0,
                                penalty_multiplier=1,
                                c0=0.1,
                                max_iter=100,
                                tol=1e-8,
                            )
                            methods.append(
                                (
                                    f"{criterion}{penalty}-{name}",
                                    lambda options=options: select_tensor_rank(
                                        train, **options
                                    ),
                                    options,
                                )
                            )
                    baseline = dict(
                        method=method,
                        iterative=iterative,
                        max_ranks=max_ranks,
                        lags=lags,
                    )
                    methods.append(
                        (
                            "adjacent-" + name,
                            lambda baseline=baseline: _adjacent_fit(train, **baseline),
                            dict(baseline, center=True, rank_updates=False),
                        )
                    )
                    if regime == "weak":
                        options = dict(
                            method=method,
                            iterative=iterative,
                            criterion="ic",
                            penalty=2,
                            max_ranks=max_ranks,
                            lags=lags,
                            center=True,
                            nu=0.5,
                            penalty_multiplier=1,
                            c0=0.1,
                            max_iter=100,
                            tol=1e-8,
                        )
                        methods.append(
                            (
                                "oracle-strength-ic2-" + name,
                                lambda options=options: select_tensor_rank(
                                    train, **options
                                ),
                                dict(
                                    options,
                                    extra_information="true-weakest-loading-strength",
                                ),
                            )
                        )
            methods.extend(
                [
                    (
                        "known-loading-projection-oracle",
                        lambda: _Projection(train, data.loadings),
                        dict(
                            extra_information="true-loading-spaces-not-true-mean",
                            center=True,
                        ),
                    ),
                    (
                        "training-mean",
                        lambda: _Projection(
                            train, [np.empty((dim, 0)) for dim in shape]
                        ),
                        dict(imposed_ranks=[0, 0], center=True),
                    ),
                ]
            )
            for name, function, options in methods:
                rows.append(
                    _measure(
                        name,
                        function,
                        lambda fit: _scores(fit, data, n),
                        dict(metadata, options=options),
                    )
                )
            print(
                f"Completed rank replicate {replicate+1}/{repeats}: {regime}, lags={lags}",
                flush=True,
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2573)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark-factor-ranks.json")
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    from benchmarks import rank_scenarios

    snapshot = source_snapshot(__file__, rank_scenarios.__file__)
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
    unconverged = sum(
        not r.get("converged", False) for r in rows if r["status"] == "ok"
    )
    print(f"Saved {len(rows)} runs; {errors} errors; {unconverged} unconverged")
    if errors or report["source_changed_during_run"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
