"""Published IC/ER rank determination for dynamic Tucker factor models.

References
----------
Han, Chen and Zhang (2022), Rank Determination in Tensor Factor Model,
Electronic Journal of Statistics 16, 1726--1803.
https://doi.org/10.1214/22-EJS1991
Primary equations and implementation conventions: https://arxiv.org/html/2011.07131v3
Sections 2--3, equations (1), (2), (6)--(8), and Remarks 5--7.
"""

import math
from dataclasses import dataclass

import numpy as np

from ._validation import as_series, finite_scalar, positive_int
from .factors import _canonical_signs, _lagged_moment, _prepare, _project
from .tensor import _unfold_series


def _logs(values):
    result = np.full(np.shape(values), -np.inf, dtype=float)
    np.log(values, out=result, where=np.asarray(values) > 0)
    return result


def _log_penalty(shape, n, h0, mode, criterion, penalty, nu, c0):
    """Equations (7)--(8), using ORIGINAL dimensions in every sweep."""
    ld = sum(math.log(d) for d in shape)
    lk, lt, lh = math.log(shape[mode]), math.log(n), math.log(h0)
    if criterion == "ic":
        common = lh + (2 - 2 * nu) * ld
        harmonic_log = ld + lt - np.logaddexp(ld, lt)
        logarithm = (
            harmonic_log if penalty in (1, 2) else min(lk if penalty == 5 else ld, lt)
        )
        if logarithm <= 0:
            raise ValueError("this sample size and shape give a nonpositive IC penalty")
        rate = -lt if penalty in (1, 3) else np.logaddexp(-lt, -ld)
        return float(common + rate + math.log(logarithm))
    base = lh + 2 * ld - 2 * lt
    return float(
        (
            lh + math.log(c0),
            base,
            base - 2 * lk,
            np.logaddexp(base - 2 * lk, lh + 2 * lk - 2 * lt),
            np.logaddexp(base - lk, lh + ld + lk - 2 * lt),
        )[penalty - 1]
    )


def _criterion(eigenvalues, log_eigenvalue_scale, log_penalty, criterion, max_rank):
    """Evaluate exact criteria without subtracting large common IC tails."""
    values = np.asarray(eigenvalues)
    lv = _logs(values)
    relative_penalty = log_penalty - log_eigenvalue_scale
    if criterion == "ic":
        candidates = np.arange(max_rank + 1)
        # IC(m)-IC(m-1)=G-lambda_m. Sorted eigenvalues make this convex.
        rank = min(int(np.count_nonzero(lv > relative_penalty)), max_rank)
        tails = np.logaddexp.accumulate(lv[::-1])[::-1]
        log_scores = np.logaddexp(
            tails[candidates], _logs(candidates) + relative_penalty
        )
        common = float(max(log_scores.max(), relative_penalty))
        scores = np.exp(log_scores - common)
        return dict(
            rank=rank,
            candidate_ranks=candidates,
            scores=scores,
            log_scores=log_scores + log_eigenvalue_scale,
            log_score_scale=common + log_eigenvalue_scale,
        )
    candidates = np.arange(1, max_rank + 1)
    # Ratio = 1 - gap/(lambda_m+H). Compare the improvement rather than
    # rounded ratios, which can all equal one when H dominates the spectrum.
    denominator = np.logaddexp(lv[:max_rank], relative_penalty)
    offset = max(0.0, relative_penalty)
    merit = _logs(values[:max_rank] - values[1 : max_rank + 1]) - (denominator - offset)
    rank = int(np.argmax(merit)) + 1
    log_scores = np.logaddexp(lv[1 : max_rank + 1], relative_penalty) - denominator
    return dict(
        rank=rank,
        candidate_ranks=candidates,
        scores=np.exp(log_scores),
        log_scores=log_scores,
        log_score_scale=0.0,
    )


@dataclass
class RankSelectionStep:
    """One mode's criterion evaluated at its sequential update.

    ``eigenvalues * exp(log_eigenvalue_scale)`` are the physical fourth-power
    spectrum; neither exponentiation nor physical spectra need be representable.
    ``log_penalty`` is in those same physical units. IC ``scores`` share the
    positive factor ``exp(log_score_scale)``; ER scores are ordinary ratios.
    ``log_scores`` retain physical IC values or dimensionless ER log ratios.
    Rounding may tie displayed scores even when stable criterion comparisons
    distinguish them. The exact-tie convention is the smallest candidate rank.
    ``projected_shape`` records the data dimensions used for this moment, NOT
    the original dimensions retained in the theoretical penalty.

    References
    ----------
    Han, Chen and Zhang (2022), Rank Determination in Tensor Factor Model,
    equations (1)--(2), (7)--(8). https://doi.org/10.1214/22-EJS1991
    """

    rank: int
    eigenvalues: np.ndarray
    log_eigenvalue_scale: float
    log_penalty: float
    candidate_ranks: np.ndarray
    scores: np.ndarray
    log_scores: np.ndarray
    log_score_scale: float
    projected_shape: tuple


def _restore(array, scale, name):
    with np.errstate(over="ignore", invalid="ignore"):
        result = array * scale
    if not np.isfinite(result).all():
        raise ValueError(f"{name} cannot be represented in physical data units")
    return result


@dataclass
class TensorRankSelectionResult:
    """Selected dynamic ranks, fitted Tucker spaces, and complete sweep history.

    ``history[0]`` is the unprojected selection; subsequent entries contain one
    sequential sweep. ``initial_ranks`` are those unprojected choices, while
    ``starting_ranks`` are the conservative projection initialization. The final
    ``ranks`` are per-mode criterion choices, not necessarily minimal multilinear
    ranks: any zero mode implies an identically zero centered signal, regardless
    of other modes' reported ranks. ``signal_ranks`` makes this distinction.
    Convergence means ranks AND loading projectors stopped changing, not correct
    rank recovery. Iterative spectra refer to the corresponding sequential
    update, not a new simultaneous recomputation using all final loadings.

    References
    ----------
    Han, Chen and Zhang (2022), Rank Determination in Tensor Factor Model,
    Section 3.2 and Remarks 5--7. https://doi.org/10.1214/22-EJS1991
    """

    ranks: tuple
    loadings: tuple
    factors: np.ndarray
    signal: np.ndarray
    residuals: np.ndarray
    mean: np.ndarray
    history: tuple
    initial_ranks: tuple
    starting_ranks: tuple
    rank_history: tuple
    projector_changes: tuple
    method: str
    criterion: str
    penalty: int
    nu: float
    c0: float
    log_penalty_multipliers: tuple
    max_ranks: tuple
    lags: tuple
    data_scale: float
    n_iter: int
    converged: bool
    stop_reason: str

    @property
    def diagnostics(self):
        return self.history[-1]

    @property
    def eigenvalues(self):
        return tuple(step.eigenvalues for step in self.diagnostics)

    @property
    def signal_ranks(self):
        return (0,) * len(self.ranks) if 0 in self.ranks else self.ranks

    def transform(self, X):
        """Project new data using only fitted spaces and the training mean.

        References
        ----------
        Han, Chen and Zhang (2022), Rank Determination in Tensor Factor Model,
        equations (4)--(5). https://doi.org/10.1214/22-EJS1991
        Fixed-loading Tucker projection is reconstruction, not forecasting.
        """
        X = as_series(X, min_samples=1, ndim=len(self.ranks) + 1)
        if X.shape[1:] != self.mean.shape:
            raise ValueError("observation dimensions must match the fitted model")
        axes = tuple(range(1, X.ndim))
        scale = np.maximum(
            np.max(np.abs(X), axis=axes, keepdims=True),
            float(np.max(np.abs(self.mean))),
        )
        scale = np.where(scale > 0, scale, 1.0)
        return _restore(
            _project(X / scale - self.mean / scale, self.loadings),
            scale,
            "transformed factors",
        )

    def inverse_transform(self, factors):
        """Reconstruct with the training mean, including empty zero-rank cores.

        References
        ----------
        Han, Chen and Zhang (2022), Rank Determination in Tensor Factor Model,
        equation (4). https://doi.org/10.1214/22-EJS1991
        Restoring a fitted deterministic mean is the centering convention.
        """
        raw = np.asarray(factors)
        if (
            np.iscomplexobj(raw)
            or raw.dtype.kind not in "biuf"
            or raw.ndim != len(self.ranks) + 1
            or raw.shape[0] < 1
            or raw.shape[1:] != self.ranks
        ):
            raise ValueError("factors must be finite real cores with the fitted ranks")
        factors = np.asarray(raw, dtype=float)
        if not np.isfinite(factors).all():
            raise ValueError("factors must be finite real cores with the fitted ranks")
        axes = tuple(range(1, factors.ndim))
        scale = np.maximum(
            np.max(np.abs(factors), axis=axes, keepdims=True, initial=0),
            float(np.max(np.abs(self.mean))),
        )
        scale = np.where(scale > 0, scale, 1.0)
        return _restore(
            _project(factors / scale, self.loadings, inverse=True) + self.mean / scale,
            scale,
            "reconstructed signal",
        )


def _per_mode(value, modes, name, *, logarithmic=False):
    if np.ndim(value) == 0:
        values = (value,) * modes
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be scalar or one value per mode") from exc
        if len(values) != modes:
            raise ValueError(f"{name} must be scalar or one value per mode")
    answer = []
    for entry in values:
        number = finite_scalar(entry, name)
        if not logarithmic and number <= 0:
            raise ValueError(f"{name} must be strictly positive")
        answer.append(number if logarithmic else math.log(number))
    return tuple(answer)


def select_tensor_rank(
    X,
    *,
    method="tipup",
    criterion="er",
    penalty=1,
    max_ranks=None,
    lags=1,
    iterative=False,
    nu=0.0,
    penalty_multiplier=1.0,
    log_penalty_multiplier=None,
    c0=0.1,
    initial_ranks=None,
    max_iter=100,
    tol=1e-8,
    center=True,
):
    """Select Tucker ranks by the published TOPUP/TIPUP IC or ER criteria.

    Parameters
    ----------
    X : array_like, shape (T, d1, ..., dK)
        Finite real training observations, K >= 1 and every d_k >= 2.
    method : {'topup', 'tipup'}
        Past-first lag outer/inner product moments, Section 3.2.
    criterion : {'ic', 'er'}
        Equation (1) permits 0,...,m*; equation (2) permits 1,...,m*.
    penalty : int, 1,...,5
        Exact g1,...,g5 (IC) or h1,...,h5 (ER), equations (7)--(8).
    max_ranks : sequence of int, optional
        Search bounds m*_k < d_k; defaults to floor(d_k/2). Zero bounds are
        allowed only for IC. Full-dimension search extensions are not provided.
    lags : int, default 1
        h0: accumulate lags 1,...,h0, without averaging the moment over lags.
    iterative : bool, default False
        Sequential equation (6) updates, reselecting every rank in each sweep.
    nu : float in [0, 1], default 0
        IC strength tuning, ideally weakest-factor exponent delta1. Zero is the
        paper's default strong-factor setting, not an estimate of strength.
    penalty_multiplier, log_penalty_multiplier : scalar or sequence, optional
        Positive multiplicative constants, one per mode if desired. The log
        alternative supports constants outside float range and requires the
        ordinary multiplier to remain its scalar default 1. Constants apply
        to FIXED physical fourth-power penalties, not relative eigenvalue
        heuristics. Under X -> a X, add 4 log(abs(a)) to log multipliers to
        preserve the criterion. Keeping constants fixed need NOT preserve ranks.
    c0 : positive float, default 0.1
        Constant in h1; ignored by other displayed penalty families.
    initial_ranks : sequence of int, optional
        Iterative projection initialization only, up to d_k. Default is
        min(2*r_initial, r_initial+3, d_k), following Remark 5 with dimension cap.
    max_iter : positive int, default 100
        Maximum sequential sweeps, excluding the unprojected initialization.
    tol : positive float, default 1e-8
        Stop when ranks agree and the maximum Frobenius projector change < tol.
    center : bool, default True
        Remove and retain the training mean, as recommended in Remark 6 when
        demeaning does not remove a factor dimension (constant factors violate
        that premise). Held-out transformations use this training mean only.

    Notes
    -----
    Original d and d_k enter penalties even after projection. Only moments use
    projected dimensions. No penalty calibration, confidence intervals, ER
    mock-zero eigenvalue, or full-dimension rank extension is implied. Serially
    white noise, informative lag moments, factor strengths and the rate
    conditions of Section 4 are substantive assumptions; numerical convergence
    does not check them. TIPUP can lose factors by lag-inner-product cancellation.
    IC zero-mode selections have exact empty-core reconstruction. With an empty
    projected mode subsequent moments are zero, an absorbing loss of information
    which the positive-rank asymptotic theory does not remedy.

    References
    ----------
    Han, Chen and Zhang (2022), Rank Determination in Tensor Factor Model,
    equations (1), (2), (6)--(8), Remarks 5--7.
    https://doi.org/10.1214/22-EJS1991
    https://arxiv.org/html/2011.07131v3
    """
    method, criterion = str(method).lower(), str(criterion).lower()
    if method not in {"topup", "tipup"}:
        raise ValueError("method must be 'topup' or 'tipup'")
    if criterion not in {"ic", "er"}:
        raise ValueError("criterion must be 'ic' or 'er'")
    penalty = positive_int(penalty, "penalty")
    if penalty > 5:
        raise ValueError("penalty must be in 1,...,5")
    if not isinstance(iterative, (bool, np.bool_)):
        raise ValueError("iterative must be boolean")
    h0 = positive_int(lags, "lags")
    max_iter = positive_int(max_iter, "max_iter")
    tol = finite_scalar(tol, "tol", minimum=0)
    nu = finite_scalar(nu, "nu", minimum=0)
    c0 = finite_scalar(c0, "c0", minimum=0)
    if tol == 0 or c0 == 0 or nu > 1:
        raise ValueError("tol and c0 must be positive, and nu must be <= 1")
    ndim = np.ndim(X)
    if ndim < 2:
        raise ValueError("X needs time and at least one spatial mode")
    X, work, mean, scale = _prepare(X, center, ndim=ndim)
    shape, n = X.shape[1:], len(X)
    if min(shape) < 2 or h0 >= n:
        raise ValueError("all mode dimensions must be >= 2 and lags < sample size")
    if max_ranks is None:
        max_ranks = tuple(d // 2 for d in shape)
    else:
        try:
            max_ranks = tuple(max_ranks)
        except TypeError as exc:
            raise ValueError("max_ranks must contain one bound per mode") from exc
        if len(max_ranks) != len(shape):
            raise ValueError("max_ranks must contain one bound per mode")
        max_ranks = tuple(
            positive_int(r, "max_rank", minimum=0 if criterion == "ic" else 1)
            for r in max_ranks
        )
        if any(r >= d for r, d in zip(max_ranks, shape)):
            raise ValueError("max_ranks must be strictly below mode dimensions")
    if log_penalty_multiplier is None:
        log_multipliers = _per_mode(
            penalty_multiplier, len(shape), "penalty_multiplier"
        )
    else:
        if (
            np.ndim(penalty_multiplier) != 0
            or finite_scalar(penalty_multiplier, "penalty_multiplier") != 1
        ):
            raise ValueError(
                "supply either penalty_multiplier or log_penalty_multiplier"
            )
        log_multipliers = _per_mode(
            log_penalty_multiplier,
            len(shape),
            "log_penalty_multiplier",
            logarithmic=True,
        )
    log_penalties = tuple(
        _log_penalty(shape, n, h0, k, criterion, penalty, nu, c0) + c
        for k, c in enumerate(log_multipliers)
    )
    lag_values = tuple(range(1, h0 + 1))
    log_scale = 4 * math.log(scale)

    def update(projected, mode):
        moment = (
            np.zeros((shape[mode], shape[mode]))
            if projected.size == 0
            else _lagged_moment(_unfold_series(projected, mode), lag_values, method)
        )
        if not np.isfinite(moment).all():
            raise ValueError("lag moment cannot be represented after normalization")
        values, vectors = np.linalg.eigh((moment + moment.T) * 0.5)
        values = np.maximum(values[::-1], 0)
        vectors = _canonical_signs(vectors[:, ::-1])
        chosen = _criterion(
            values, log_scale, log_penalties[mode], criterion, max_ranks[mode]
        )
        step = RankSelectionStep(
            eigenvalues=values,
            log_eigenvalue_scale=log_scale,
            log_penalty=log_penalties[mode],
            projected_shape=projected.shape[1:],
            **chosen,
        )
        return vectors, step

    first = [update(work, k) for k in range(len(shape))]
    initial = tuple(pair[1].rank for pair in first)
    if initial_ranks is not None:
        if not iterative:
            raise ValueError("initial_ranks apply only to iterative selection")
        try:
            starting = tuple(initial_ranks)
        except TypeError as exc:
            raise ValueError("initial_ranks must contain one rank per mode") from exc
        if len(starting) != len(shape):
            raise ValueError("initial_ranks must contain one rank per mode")
        starting = tuple(
            positive_int(r, "initial_rank", minimum=0 if criterion == "ic" else 1)
            for r in starting
        )
        if any(r > d for r, d in zip(starting, shape)):
            raise ValueError("initial_ranks cannot exceed mode dimensions")
    else:
        starting = (
            tuple(min(2 * r, r + 3, d) for r, d in zip(initial, shape))
            if iterative
            else initial
        )
    loadings = [pair[0][:, :r] for pair, r in zip(first, starting)]
    history = [tuple(pair[1] for pair in first)]
    rank_history, changes = [starting], []
    converged, iteration = not iterative, 0
    if iterative:
        for iteration in range(1, max_iter + 1):
            old = list(loadings)
            steps = []
            for mode in range(len(shape)):
                vectors, step = update(_project(work, loadings, skip=mode), mode)
                loadings[mode] = vectors[:, : step.rank]
                steps.append(step)
            current = tuple(u.shape[1] for u in loadings)
            change = max(
                float(np.linalg.norm(u @ u.T - v @ v.T)) for u, v in zip(loadings, old)
            )
            history.append(tuple(steps))
            changes.append(change)
            rank_history.append(current)
            if current == tuple(u.shape[1] for u in old) and change < tol:
                converged = True
                break
    ranks = tuple(u.shape[1] for u in loadings)
    core = _project(work, loadings)
    reconstructed = _project(core, loadings, inverse=True) + mean / scale
    signal = _restore(reconstructed, scale, "fitted signal")
    factors = _restore(core, scale, "fitted factors")
    residuals = _restore(X / scale - reconstructed, scale, "fitted residuals")
    return TensorRankSelectionResult(
        ranks,
        tuple(loadings),
        factors,
        signal,
        residuals,
        mean,
        tuple(history),
        initial,
        starting,
        tuple(rank_history),
        tuple(changes),
        ("i" if iterative else "") + method,
        criterion,
        penalty,
        nu,
        c0,
        log_multipliers,
        max_ranks,
        lag_values,
        scale,
        iteration,
        converged,
        (
            "rank_and_projector_tolerance"
            if iterative and converged
            else ("max_iter" if iterative else "noniterative")
        ),
    )
