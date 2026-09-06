"""Benchmark shared seeded data: ``python -m benchmarks.run --quick``.

Timings include fitting only, with simulation and scoring outside the timer.
Failed methods remain visible. Results are stress tests, not replications of
the published papers' complete simulation experiments.
"""

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import warnings
from contextlib import redirect_stdout
from itertools import permutations
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

import mavats
from mavats import (
    fit_alpha_pca,
    fit_constrained_factor,
    fit_cp_factor,
    fit_huber_factor,
    fit_lagged_factor,
    fit_mar,
    fit_matrix_kendall,
    fit_naive,
    fit_projected_pca,
    fit_tensor_factor,
    fit_var,
)
from mavats.metrics import (
    mean_squared_error,
    relative_frobenius_error,
    subspace_distance,
)
from mavats.simulation import simulate_factor, simulate_mar


def _measure(method, function, score, metadata):
    record = dict(metadata, method=method)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start = perf_counter()
        phase = "fit"
        try:
            fit = function()
            record["fit_seconds"] = perf_counter() - start
            phase = "score"
            record.update(score(fit))
            phase = "diagnostics"
            record["converged"] = bool(getattr(fit, "converged", True))
            record["iterations"] = int(getattr(fit, "n_iter", 0))
            record["status"] = "ok"
        except Exception as exc:
            if "fit_seconds" not in record:
                record["fit_seconds"] = perf_counter() - start
            record["failure_phase"] = phase
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
        record["warnings"] = [str(w.message) for w in caught]
    return record


def _factor_score(fit, signal, loadings):
    return {
        "signal_error": relative_frobenius_error(signal, fit.signal),
        "subspace_error": float(
            np.mean([subspace_distance(u, v) for u, v in zip(loadings, fit.loadings)])
        ),
        "ranks": list(fit.ranks),
    }


def run_suite(*, quick=False, repeats=5, seed=2026):
    """Return per-replicate results on common data for every competing method."""
    rows = []
    for replicate in range(repeats):
        current_seed = seed + replicate
        rng = np.random.default_rng(current_seed)
        shape = (3, 4) if quick else (8, 10)
        n = 120 if quick else 500
        a = np.diag(np.linspace(0.5, 0.8, shape[0]))
        b = np.diag(np.linspace(0.6, 0.9, shape[1]))
        a[0, -1], b[-1, 0] = 0.15, -0.1
        for regime in ("isotropic", "correlated", "reduced-rank"):
            aa, bb = a.copy(), b.copy()
            if regime == "reduced-rank":
                aa[-1], bb[-1] = 0, 0
            row_cov = (
                np.eye(shape[0])
                if regime != "correlated"
                else 0.6
                ** abs(np.subtract.outer(np.arange(shape[0]), np.arange(shape[0])))
            )
            col_cov = (
                np.eye(shape[1])
                if regime != "correlated"
                else 0.5
                ** abs(np.subtract.outer(np.arange(shape[1]), np.arange(shape[1])))
            )
            x = simulate_mar(
                n + 20,
                aa,
                bb,
                row_cov=row_cov,
                column_cov=col_cov,
                random_state=current_seed,
            )
            training, test = x[:n], x[n:]

            def score_forecast(fit):
                if hasattr(fit, "history") or hasattr(fit, "left"):
                    predicted = np.stack(
                        [
                            fit.forecast(1, history=x[: n + i])[0]
                            for i in range(len(test))
                        ]
                    )
                else:
                    predicted = fit.forecast(len(test))
                result = {"forecast_mse": mean_squared_error(test, predicted)}
                if hasattr(fit, "left"):
                    result["operator_error"] = relative_frobenius_error(
                        np.kron(bb, aa), np.kron(fit.B, fit.A)
                    )
                    result["spectral_radius"] = fit.spectral_radius
                return result

            factories = {
                "mar-projection": lambda: fit_mar(training, method="projection"),
                "mar-als": lambda: fit_mar(training, method="als"),
                "mar-mle": lambda: fit_mar(training, method="mle"),
                "var": lambda: fit_var(training, intercept=False),
                "var-ridge": lambda: fit_var(training, intercept=False, ridge=1.0),
                "zero": lambda: fit_naive(training, strategy="zero"),
                "training-mean": lambda: fit_naive(training, strategy="mean"),
            }
            if regime == "reduced-rank":
                factories["rr-mar-oracle-ranks"] = lambda: fit_mar(
                    training, ranks=(shape[0] - 1, shape[1] - 1)
                )
            metadata = dict(
                task="forecast",
                regime=regime,
                seed=current_seed,
                replicate=replicate,
                shape=list(shape),
                n_train=n,
            )
            for name, factory in factories.items():
                rows.append(_measure(name, factory, score_forecast, metadata))
            # Last-value and oracle conditional-mean baselines share the same origins.
            rows.append(
                dict(
                    metadata,
                    method="last-value",
                    status="ok",
                    converged=True,
                    iterations=0,
                    fit_seconds=0.0,
                    warnings=[],
                    forecast_mse=mean_squared_error(test, x[n - 1 : -1]),
                )
            )
            rows.append(
                dict(
                    metadata,
                    method="oracle-mar",
                    status="ok",
                    converged=True,
                    iterations=0,
                    fit_seconds=0.0,
                    warnings=[],
                    forecast_mse=mean_squared_error(test, aa @ x[n - 1 : -1] @ bb.T),
                )
            )

        factor_shape, ranks = ((5, 6) if quick else (20, 25)), (2, 2)
        base = simulate_factor(n, factor_shape, ranks, random_state=current_seed)
        for regime in ("gaussian", "elliptical-t3", "outliers"):
            observations, signal = base.observations.copy(), base.signal.copy()
            if regime == "elliptical-t3":
                radial = np.sqrt(3 / rng.chisquare(3, size=n))[:, None, None]
                observations *= radial
                signal *= radial
            elif regime == "outliers":
                indices = rng.choice(n, size=max(1, n // 20), replace=False)
                observations[indices] += rng.normal(
                    scale=30, size=observations[indices].shape
                )
            factories = {
                "alpha-pca": lambda: fit_alpha_pca(observations, ranks),
                "projected-pca": lambda: fit_projected_pca(observations, ranks),
                "lagged-factor": lambda: fit_lagged_factor(observations, ranks, lags=2),
                "matrix-kendall": lambda: fit_matrix_kendall(observations, ranks),
                "huber-factor": lambda: fit_huber_factor(observations, ranks),
                "tipup": lambda: fit_tensor_factor(
                    observations, ranks, method="tipup", lags=2
                ),
                "itipup": lambda: fit_tensor_factor(
                    observations, ranks, method="tipup", lags=2, iterative=True
                ),
                "itopup": lambda: fit_tensor_factor(
                    observations, ranks, method="topup", lags=2, iterative=True
                ),
                # Oracle-known constraint spans: reported separately, not a fair
                # competitor that has the same information as unconstrained fits.
                "constrained-oracle-spans": lambda: fit_constrained_factor(
                    observations,
                    ranks,
                    row_constraints=base.loadings[0],
                    column_constraints=base.loadings[1],
                    lags=2,
                ),
            }
            metadata = dict(
                task="matrix-factor",
                regime=regime,
                seed=current_seed,
                replicate=replicate,
                shape=list(factor_shape),
                n_train=n,
            )
            for name, factory in factories.items():
                rows.append(
                    _measure(
                        name,
                        factory,
                        lambda fit: _factor_score(fit, signal, base.loadings),
                        metadata,
                    )
                )

        # Distinct scalar dynamics identify individual nonorthogonal CP
        # components. The default proxies use only observed data.
        cp_rng = np.random.default_rng(current_seed)
        cp_a, cp_b = cp_rng.normal(size=(6, 3)), cp_rng.normal(size=(5, 3))
        cp_a /= np.linalg.norm(cp_a, axis=0)
        cp_b /= np.linalg.norm(cp_b, axis=0)
        cp_scores = cp_rng.normal(size=(max(n, 200), 3))
        for t in range(1, len(cp_scores)):
            cp_scores[t] += np.array([-0.8, 0.4, 0.9]) * cp_scores[t - 1]
        cp_signal = np.einsum("tr,ir,jr->tij", cp_scores, cp_a, cp_b)
        cp_observations = cp_signal + 0.02 * cp_rng.normal(size=cp_signal.shape)

        def score_cp(fit):
            similarity = abs(cp_a.T @ fit.A) * abs(cp_b.T @ fit.B)
            match = max(
                sum(similarity[i, p[i]] for i in range(3)) / 3
                for p in permutations(range(3))
            )
            return {
                "signal_error": relative_frobenius_error(cp_signal, fit.signal),
                "component_error": float(np.clip(1 - match, 0, 1)),
                "condition_number": fit.condition_number,
                "rank": fit.rank,
            }

        rows.append(
            _measure(
                "cp-refined",
                lambda: fit_cp_factor(cp_observations, 3),
                score_cp,
                dict(
                    task="matrix-cp",
                    regime="distinct-ar",
                    seed=current_seed,
                    replicate=replicate,
                    shape=[6, 5],
                    n_train=len(cp_scores),
                ),
            )
        )

        data = simulate_factor(
            n, (4, 5, 3) if quick else (10, 12, 8), (2, 2, 2), random_state=current_seed
        )
        for method in ("tipup", "topup"):
            for iterative in (False, True):
                rows.append(
                    _measure(
                        ("i" if iterative else "") + method,
                        lambda: fit_tensor_factor(
                            data.observations,
                            (2, 2, 2),
                            method=method,
                            iterative=iterative,
                            lags=2,
                        ),
                        lambda fit: _factor_score(fit, data.signal, data.loadings),
                        dict(
                            task="tensor-factor",
                            regime="gaussian",
                            seed=current_seed,
                            replicate=replicate,
                            shape=list(data.signal.shape[1:]),
                            n_train=n,
                        ),
                    )
                )
    return rows


def environment():
    """Capture software, BLAS and requested thread context for every runner."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        np.show_config()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "mavats": mavats.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas": buffer.getvalue(),
        "thread_environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
    }


def source_snapshot(*drivers):
    """Fingerprint source files before any long experiment, not after it."""
    root = Path(__file__).resolve().parents[1]
    paths = set((root / "mavats").glob("*.py"))
    paths.update(Path(p).resolve() for p in (*drivers, __file__))
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(paths)
    }


def source_provenance(snapshot):
    """Expose source edits during a run instead of attributing results to them."""
    root = Path(__file__).resolve().parents[1]
    changed = [
        name
        for name, digest in snapshot.items()
        if not (root / name).is_file()
        or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
    ]
    return dict(source_sha256=snapshot, source_changed_during_run=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.json"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    snapshot = source_snapshot(__file__)
    rows = run_suite(quick=args.quick, repeats=args.repeats, seed=args.seed)
    report = {
        "schema_version": 1,
        "configuration": {
            "quick": args.quick,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "environment": environment(),
        "results": rows,
        **source_provenance(snapshot),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    errors = sum(row["status"] != "ok" for row in rows)
    unconverged = sum(
        not row.get("converged", False) for row in rows if row["status"] == "ok"
    )
    print(
        f"{len(rows)} runs; {errors} errors; {unconverged} unconverged. Results: {args.output}"
    )
    if errors or report["source_changed_during_run"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
