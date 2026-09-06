"""Entrywise iterative Huber regression, explicitly arXiv:2306.03317v1 (2023).

This is not matrixwise residual-norm reweighting. The accepted work is listed
under a different title; equivalence with that unavailable text is unverified.

References
----------
He, Kong, Liu and Zhao (2023), Robust Statistical Inference for
Large-Dimensional Matrix-Valued Time Series via Iterative Huber Regression.
https://arxiv.org/abs/2306.03317v1
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import lstsq

from ._validation import as_series, finite_scalar, positive_int, ranks_tuple
from .factors import FactorResult, _canonical_signs, _prepare, fit_projected_pca
from .huber import _initial_spaces


def _positive(value, name):
    value = finite_scalar(value, name, minimum=0)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _loss(residual, threshold):
    absolute = np.abs(residual)
    clipped = np.minimum(absolute, threshold)
    # Avoid squaring large residuals in the linear branch.
    return float(np.mean(clipped * (absolute - 0.5 * clipped)))


def _loss_change(residual, increment, threshold):
    """Huber loss difference without subtracting nearly equal total losses."""
    updated = residual + increment
    before, after = np.abs(residual), np.abs(updated)
    change = np.empty_like(residual)
    quadratic = (before <= threshold) & (after <= threshold)
    change[quadratic] = increment[quadratic] * (
        residual[quadratic] + 0.5 * increment[quadratic]
    )
    same_sign = (residual >= 0) == (updated >= 0)
    linear = ~quadratic & same_sign & (before >= threshold) & (after >= threshold)
    change[linear] = threshold * np.sign(residual[linear]) * increment[linear]
    crossing = ~(quadratic | linear)
    old, new = before[crossing], after[crossing]
    absolute_increment = np.where(
        same_sign[crossing],
        np.sign(residual[crossing]) * increment[crossing],
        new - old,
    )
    clipped_old, clipped_new = np.minimum(old, threshold), np.minimum(new, threshold)
    change[crossing] = clipped_old * absolute_increment + (
        clipped_new - clipped_old
    ) * (new - 0.5 * (clipped_new + clipped_old))
    return float(np.mean(change))


@dataclass
class IHRRegressionDiagnostics:
    """Inner convex solve diagnostics; score uses column-scaled design units."""

    converged: bool
    n_iter: int
    score_norm: float
    objective_history: np.ndarray
    stopping_reason: str


def _regression(design, response, threshold, *, initial=None, max_iter=100, tol=1e-9):
    """Solve a fixed-threshold convex Huber regression by SVD-based IRLS.

    Eq. (2.6) of the 2023 preprint. No intercept or ridge is added. The score
    criterion uses mean absolute design entries and min(threshold, max|y|).
    """
    design, response = np.asarray(design), np.asarray(response)
    scales = np.max(np.abs(design), axis=0)
    if np.any(scales == 0) or not np.isfinite(scales).all():
        raise ValueError("Huber regression design is rank deficient or nonfinite")
    z = design / scales
    cutoff = np.finfo(float).eps * max(z.shape)
    least, _, rank, _ = lstsq(z, response, cond=cutoff, lapack_driver="gelsd")
    if rank != z.shape[1]:
        raise ValueError("Huber regression design is numerically rank deficient")
    beta = least if initial is None else np.asarray(initial) * scales
    residual = response - z @ beta
    value = _loss(residual, threshold)
    history = [value]
    denominator = np.maximum(np.mean(np.abs(z), axis=0), np.finfo(float).tiny)
    score_scale = min(
        threshold, max(float(np.max(np.abs(response))), np.finfo(float).tiny)
    )

    def score(e):
        return float(
            np.max(
                np.abs(z.T @ (np.clip(e, -threshold, threshold) / score_scale))
                / len(e)
                / denominator
            )
        )

    norm = score(residual)
    converged, reason, iteration = norm <= tol, "max_iter", 0
    if converged:
        reason = "score_tolerance"
    for iteration in range(1, max_iter + 1):
        if converged:
            iteration -= 1
            break
        weight = np.ones(len(response))
        large = np.abs(residual) > threshold
        weight[large] = threshold / np.abs(residual[large])
        root = np.sqrt(weight)
        candidate, _, rank, _ = lstsq(
            z * root[:, None], response * root, cond=cutoff, lapack_driver="gelsd"
        )
        if rank != z.shape[1]:
            reason = "weighted_rank_deficient"
            iteration -= 1
            break
        candidate_residual = response - z @ candidate
        next_value = _loss(candidate_residual, threshold)
        if not np.isfinite(next_value):
            raise FloatingPointError("Huber regression objective is not representable")
        increment = -(z @ (candidate - beta))
        # Direct totals can differ by one ULP while a near-optimal step really
        # decreases loss. Piecewise differences retain its small quadratic term.
        if _loss_change(residual, increment, threshold) > 0:
            reason = "descent_stalled"
            iteration -= 1
            break
        beta, residual, value = candidate, candidate_residual, next_value
        history.append(value)
        norm = score(residual)
        converged = norm <= tol
        if converged:
            reason = "score_tolerance"
    return beta / scales, IHRRegressionDiagnostics(
        converged, iteration, norm, np.asarray(history), reason
    )


def _normalize(row, column, factors):
    """Signal-preserving Eqs. (2.3)--(2.5), returned in orthonormal units.

    Public factors are sqrt(p1*p2) times paper factors; spectra below use the
    paper units. Nonzero distinct spectra identify axes only up to signs.
    """
    bases, maps = [], []
    for matrix in (row, column):
        u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
        if singular[-1] <= np.finfo(float).eps * max(matrix.shape) * singular[0]:
            raise ValueError("updated loading matrix is numerically rank deficient")
        bases.append(u)
        maps.append(singular[:, None] * vt)
    core = maps[0] @ factors @ maps[1].T
    moments = (
        np.einsum("tab,tcb->ac", core, core),
        np.einsum("tab,tac->bc", core, core),
    )
    rotations, spectra = [], []
    for basis, moment in zip(bases, moments):
        values, vectors = np.linalg.eigh(moment / len(core))
        vectors = vectors[:, ::-1]
        loading = basis @ vectors
        signed = _canonical_signs(loading)
        vectors *= np.where(np.sum(loading * signed, axis=0) < 0, -1.0, 1.0)
        rotations.append(vectors)
        spectra.append(np.maximum(values[::-1], 0) / (len(row) * len(column)))
    return (
        bases[0] @ rotations[0],
        bases[1] @ rotations[1],
        rotations[0].T @ core @ rotations[1],
        tuple(spectra),
    )


def _factor_regressions(X, row, column, threshold, max_iter, tol, initial=None):
    design = np.kron(column, row)
    factors, diagnostics = [], []
    for t, x in enumerate(X):
        start = None if initial is None else initial[t].ravel(order="F")
        beta, info = _regression(
            design,
            x.ravel(order="F"),
            threshold,
            initial=start,
            max_iter=max_iter,
            tol=tol,
        )
        factors.append(beta.reshape((row.shape[1], column.shape[1]), order="F"))
        diagnostics.append(info)
    return np.asarray(factors), tuple(diagnostics)


@dataclass
class IHRFactorResult(FactorResult):
    """Entrywise Huber decomposition with robust (not projected) factor scores.

    Threshold is in original entry units. Objective history and eigenvalues
    use X/data_scale units; eigenvalues additionally use the paper's loading
    normalization R'R=p1*I, C'C=p2*I. No inference is supplied. `block_history`
    records (row, column, factor) diagnostic tuples for each attempted sweep.
    """

    threshold: float = 0.0
    objective_history: np.ndarray | None = None
    block_history: tuple = ()
    stopping_reason: str = "max_iter"
    inner_max_iter: int = 100
    inner_tol: float = 1e-9

    def transform(self, X, *, return_diagnostics=False):
        """Fit each new core by entrywise Huber regression at the training threshold.

        Uses only that observation and fitted loadings/mean. By default a warning
        reports unfinished inner solves; return_diagnostics=True instead returns
        (factors, per-observation diagnostics). No refit or future observations.

        References
        ----------
        He, Kong, Liu and Zhao (2023), Robust Statistical Inference for
        Large-Dimensional Matrix-Valued Time Series via Iterative Huber
        Regression, Algorithm 1 factor-regression block.
        https://arxiv.org/abs/2306.03317v1
        Holding training loadings and threshold fixed is the out-of-sample
        scoring convention used here, not a forecasting or inference method.
        """
        X = as_series(X, min_samples=1)
        if X.shape[1:] != self.mean.shape:
            raise ValueError("observation dimensions must match the fitted model")
        if not isinstance(return_diagnostics, (bool, np.bool_)):
            raise ValueError("return_diagnostics must be boolean")
        cores, diagnostics = [], []
        for x in X:
            # Per-observation scaling keeps numerical stopping decisions
            # independent of other (possibly future) transform observations.
            scale = max(
                float(np.max(np.abs(x))),
                float(np.max(np.abs(self.mean))),
                np.finfo(float).tiny,
            )
            with np.errstate(over="ignore"):
                threshold = self.threshold / scale
            if threshold == 0:
                raise ValueError(
                    "threshold is not representable relative to transform data"
                )
            core, info = _factor_regressions(
                (x / scale - self.mean / scale)[None],
                *self.loadings,
                threshold,
                self.inner_max_iter,
                self.inner_tol,
            )
            with np.errstate(over="ignore", invalid="ignore"):
                core *= scale
            if not np.isfinite(core).all():
                raise FloatingPointError(
                    "robust factor scores exceed original data units"
                )
            cores.append(core[0])
            diagnostics.extend(info)
        core, diagnostics = np.asarray(cores), tuple(diagnostics)
        if return_diagnostics:
            return core, diagnostics
        if not all(info.converged for info in diagnostics):
            warnings.warn(
                "some robust factor regressions did not converge; request diagnostics",
                RuntimeWarning,
                stacklevel=2,
            )
        return core

    def inverse_transform(self, factors):
        """Reconstruct robust cores, rejecting unrepresentable original-unit output.

        References
        ----------
        He, Kong, Liu and Zhao (2023), Robust Statistical Inference for
        Large-Dimensional Matrix-Valued Time Series via Iterative Huber
        Regression. https://arxiv.org/abs/2306.03317v1
        The optional training-mean restoration is an implementation convention.
        """
        factors = as_series(factors, min_samples=1)
        if factors.shape[1:] != self.ranks:
            raise ValueError("factor dimensions must match fitted ranks")
        scale = max(
            float(np.max(np.abs(factors))),
            float(np.max(np.abs(self.mean))),
            np.finfo(float).tiny,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            signal = (
                self.loadings[0] @ (factors / scale) @ self.loadings[1].T
                + self.mean / scale
            ) * scale
        if not np.isfinite(signal).all():
            raise FloatingPointError("reconstruction exceeds original data units")
        return signal


def fit_ihr_factor(
    X,
    ranks,
    *,
    threshold=None,
    center=False,
    initial=None,
    max_iter=100,
    inner_max_iter=100,
    tol=1e-7,
    inner_tol=1e-9,
):
    """Fit fixed-threshold entrywise IHR, arXiv:2306.03317v1, Eq. (2.2)/Alg. 1.

    Row, column and factor blocks are convex Huber regressions, not matrixwise
    weighted projections. Each accepted sweep includes signal-preserving SVD
    normalization. A local convergence flag requires small relative signal/loss
    changes and successful inner scores; it does not establish a global optimum.

    `threshold=None` freezes 1.345*1.483*median(abs(residual)) from a one-sweep
    projected-PCA pilot, following Section 4.4's fixed-threshold calibration.
    A numerically zero pilot scale requires an explicit positive threshold.
    This is NOT Section 2's repeatedly updated regression-specific MAD rule.
    Explicit thresholds have original ENTRY units, not matrix Frobenius units.
    `initial` is a pair of full-rank loading bases; otherwise alpha=0 PCA.
    `center=True` removes an ordinary, nonrobust training mean.

    The accepted work's new title is Winsorized Mean Matrix Factor Model; its
    full text was unavailable for reconciliation. No equivalence, inferential
    standard errors or guarantees for the finite algorithm path are claimed.
    References
    ----------
    He, Kong, Liu and Zhao (2023), Robust Statistical Inference for
    Large-Dimensional Matrix-Valued Time Series via Iterative Huber Regression,
    equation (2.2), Algorithm 1 and Section 4.4.
    https://arxiv.org/html/2306.03317v1
    """
    X, work, mean, scale = _prepare(X, center)
    ranks = ranks_tuple(ranks, X.shape[1:])
    if ranks is None:
        raise ValueError("provide two explicit ranks, or use select_ihr_ranks")
    max_iter = positive_int(max_iter, "max_iter")
    inner_max_iter = positive_int(inner_max_iter, "inner_max_iter")
    tol, inner_tol = _positive(tol, "tol"), _positive(inner_tol, "inner_tol")
    row, column = _initial_spaces(initial, work, ranks)
    core = row.T @ work @ column
    if threshold is None:
        pilot = fit_projected_pca(work, ranks, max_iter=1)
        tau = 1.345 * 1.483 * float(np.median(np.abs(pilot.residuals)))
        if tau <= 64 * np.finfo(float).eps * max(float(np.max(np.abs(work))), 1):
            raise ValueError(
                "pilot residual scale is numerically zero; supply threshold"
            )
        threshold = tau * scale
        if not np.isfinite(threshold) or threshold == 0:
            raise ValueError(
                "automatic threshold is not representable in original units"
            )
    else:
        threshold = _positive(threshold, "threshold")
        with np.errstate(over="ignore"):
            tau = threshold / scale
        # +infinity means exactly quadratic loss at all finite residuals.
        if tau == 0:
            raise ValueError("threshold is not representable relative to data scale")
    row, column, core, spectra = _normalize(row, column, core)
    signal = row @ core @ column.T
    history = [_loss(work - signal, tau)]
    blocks, converged, reason, iteration = [], False, "max_iter", 0
    for iteration in range(1, max_iter + 1):
        row_design = (core @ column.T).transpose(0, 2, 1).reshape(-1, ranks[0])
        updated_rows, row_info = [], []
        for i in range(X.shape[1]):
            beta, info = _regression(
                row_design,
                work[:, i, :].ravel(),
                tau,
                initial=row[i],
                max_iter=inner_max_iter,
                tol=inner_tol,
            )
            updated_rows.append(beta)
            row_info.append(info)
        new_row = np.asarray(updated_rows)
        column_design = (new_row @ core).reshape(-1, ranks[1])
        updated_columns, column_info = [], []
        for j in range(X.shape[2]):
            beta, info = _regression(
                column_design,
                work[:, :, j].ravel(),
                tau,
                initial=column[j],
                max_iter=inner_max_iter,
                tol=inner_tol,
            )
            updated_columns.append(beta)
            column_info.append(info)
        new_column = np.asarray(updated_columns)
        new_core, factor_info = _factor_regressions(
            work, new_row, new_column, tau, inner_max_iter, inner_tol, core
        )
        blocks.append((tuple(row_info), tuple(column_info), factor_info))
        try:
            new_row, new_column, new_core, new_spectra = _normalize(
                new_row, new_column, new_core
            )
        except ValueError:
            reason = "rank_deficient_update"
            iteration -= 1
            break
        candidate = new_row @ new_core @ new_column.T
        value = _loss(work - candidate, tau)
        change = np.linalg.norm(candidate - signal) / max(
            np.linalg.norm(signal), np.finfo(float).tiny
        )
        inner_ok = all(info.converged for group in blocks[-1] for info in group)
        if not np.isfinite(value) or value > history[-1]:
            roundoff = 64 * np.finfo(float).eps * max(history[-1], np.finfo(float).tiny)
            converged = bool(
                np.isfinite(value)
                and value - history[-1] <= roundoff
                and change <= np.sqrt(tol)
                and inner_ok
            )
            reason = "numerical_tolerance" if converged else "descent_stalled"
            iteration -= 1
            break
        improvement = (history[-1] - value) / max(history[-1], np.finfo(float).tiny)
        row, column, core, spectra, signal = (
            new_row,
            new_column,
            new_core,
            new_spectra,
            candidate,
        )
        history.append(value)
        if improvement <= tol and change <= np.sqrt(tol) and inner_ok:
            converged, reason = True, "converged"
            break
    with np.errstate(over="ignore", invalid="ignore"):
        factors, signal = core * scale, signal * scale + mean
        residuals = X - signal
    if not all(np.isfinite(a).all() for a in (factors, signal, residuals)):
        raise FloatingPointError("IHR decomposition exceeds original data units")
    return IHRFactorResult(
        factors,
        signal,
        (row, column),
        residuals,
        mean,
        spectra,
        "entrywise-ihr-2023",
        scale,
        iteration,
        converged,
        threshold,
        np.asarray(history),
        tuple(blocks),
        reason,
        inner_max_iter,
        inner_tol,
    )


@dataclass
class IHRRankResult:
    """Preprint Section 3.2 rank criteria from an overfitted IHR model.

    `log_ratios` avoids overflow; `thresholds` uses pilot normalized variance
    units. The pilot need not converge: inspect pilot.converged. Zero selected
    ranks under the threshold rule are not silently changed to one.
    """

    ranks: tuple
    method: str
    pilot: IHRFactorResult
    log_ratios: tuple | None
    thresholds: tuple | None


def select_ihr_ranks(X, max_ranks, *, method="ratio", ridge=1e-4, **fit_options):
    """Fit oversized ranks and apply the 2023 preprint's two-way rank rules.

    `max_ranks` are fitted dimensions m1,m2, strictly above the true ranks for
    the theory. Ratio searches 1..m_l-1 and uses sigma_i/(sigma_(i+1)+ridge/D²)
    where D²=min(T*p1,T*p2,p1*p2). Ridge has ORIGINAL variance units; scale it
    by a² when scaling X by a to retain the same ratio criterion.
    Threshold selection implements Section 4.3's P_l=sigma_(l,1)*D^(-2/3).
    It can select zero and is scale invariant. Neither rule tests a no-factor
    null with calibrated size. No post-selection inference is provided.

    References
    ----------
    He, Kong, Liu and Zhao (2023), Robust Statistical Inference for
    Large-Dimensional Matrix-Valued Time Series via Iterative Huber Regression,
    Section 4.3. https://arxiv.org/abs/2306.03317v1
    The pilot uses the fixed-threshold implementation in ``fit_ihr_factor``;
    equivalence to the renamed accepted successor remains unverified.
    """
    if method not in ("ratio", "threshold"):
        raise ValueError("method must be 'ratio' or 'threshold'")
    ridge = _positive(ridge, "ridge")
    pilot = fit_ihr_factor(X, max_ranks, **fit_options)
    t, p, q = pilot.signal.shape
    d2 = min(t * p, t * q, p * q)
    if method == "threshold":
        cutoffs = tuple(values[0] * d2 ** (-1 / 3) for values in pilot.eigenvalues)
        ranks = tuple(
            int(np.count_nonzero(values > cutoff))
            for values, cutoff in zip(pilot.eigenvalues, cutoffs)
        )
        return IHRRankResult(ranks, method, pilot, None, cutoffs)
    if any(rank < 2 for rank in pilot.ranks):
        raise ValueError("ratio selection needs max_ranks >=2 in both modes")
    log_ridge = np.log(ridge) - np.log(d2) - 2 * np.log(pilot.data_scale)
    logs = []
    for values in pilot.eigenvalues:
        with np.errstate(divide="ignore"):
            logarithms = np.log(values)
        logs.append(logarithms[:-1] - np.logaddexp(logarithms[1:], log_ridge))
    return IHRRankResult(
        tuple(int(np.argmax(v)) + 1 for v in logs), method, pilot, tuple(logs), None
    )
