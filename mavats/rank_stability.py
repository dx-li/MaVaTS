"""Subsample stability of tensor-factor IC penalty constants.

References
----------
Han, Y., Chen, R., and Zhang, C.-H. (2022), "Rank Determination in Tensor
Factor Model", Electronic Journal of Statistics 16, 1726--1803,
https://arxiv.org/html/2011.07131v3, Remark 8 and Section 5.4.
The path implements the published empirical variance. Plateau selection is
an explicit finite-grid interpretation, not a consistency or optimality claim.
"""

from dataclasses import dataclass

import numpy as np

from ._validation import as_series, finite_scalar, positive_int


@dataclass(frozen=True)
class RankStabilityCell:
    """One IC fit, or its recorded numerical failure; inherits the module reference."""

    result: object | None
    error: str | None = None

    @property
    def converged(self):
        return self.result is not None and bool(self.result.converged)

    @property
    def n_iter(self):
        return None if self.result is None else self.result.n_iter


@dataclass(frozen=True)
class RankStabilityInterval:
    """A maximal constant-full-rank, low-variance run on the supplied c grid.

    Index endpoints are inclusive. Reasons retain rejected runs, including the
    maximum-rank plateau and zero-rank tail discussed in the module reference.
    """

    start_index: int
    end_index: int
    lower_multiplier: float
    upper_multiplier: float
    rank: int
    admissible: bool
    reasons: tuple
    touches_grid_boundary: bool


@dataclass(frozen=True)
class RankStabilityChoice:
    """Modewise finite-grid choices, not a jointly refitted rank estimate.

    Missing choices are None, never silently replaced by a boundary value.
    With iterative IC, refitting the chosen modewise multipliers jointly may
    change ranks because the mode updates are coupled. See the module reference.
    """

    selected_indices: tuple
    penalty_multipliers: tuple
    selected_ranks: tuple
    intervals: tuple
    reasons: tuple
    variance_tolerance: float
    minimum_grid_points: int
    allow_unconverged: bool
    point_reasons: tuple

    @property
    def complete(self):
        return all(index is not None for index in self.selected_indices)


@dataclass(frozen=True)
class TensorRankStabilityResult:
    """Full IC stability path; no observations or subsamples are selected secretly.

    ``ranks[k,c,j]`` uses -1 only for failed cells. ``variances[k,c]`` uses
    divisor J and is NaN if any cell at that c failed (no case deletion).
    ``cells[c][j]`` retains the complete rank-selector result and convergence.
    The final subsample is full data, hence ``full_sample_ranks=ranks[:,:,-1]``.
    All indices are zero-based; time_prefixes are exclusive stopping positions.
    References: Han, Chen and Zhang (2022), Remark 8 and Section 5.4,
    https://arxiv.org/html/2011.07131v3.
    """

    c_grid: np.ndarray
    spatial_subsets: tuple
    time_prefixes: tuple
    max_ranks: tuple
    options: dict
    ranks: np.ndarray
    variances: np.ndarray
    cells: tuple

    @property
    def full_sample_ranks(self):
        return self.ranks[:, :, -1].copy()

    @property
    def converged(self):
        """Boolean (c, subsample) array, not a path-level success certificate."""
        return np.array([[cell.converged for cell in row] for row in self.cells])

    @property
    def monotone(self):
        """Whether all successful ranks are nonincreasing in c, by mode.

        A mode with failed cells returns False. Iterative projection/rank
        coupling need not preserve the fixed-moment IC monotonicity.
        """
        return np.array(
            [
                bool(np.all(mode >= 0) and np.all(np.diff(mode, axis=0) <= 0))
                for mode in self.ranks
            ]
        )

    def choose_plateaus(
        self, *, variance_tolerance, minimum_grid_points, allow_unconverged=False
    ):
        """Choose the first admissible stability interval separately for each mode.

        Both numerical tolerances are required: points must be consecutive in
        the supplied grid, have variance <= variance_tolerance and constant
        full-sample rank. Rank zero and the fixed maximum rank are excluded.
        By default every subsample fit at every selected point must converge.
        The representative is the lower-middle gridpoint, with no interpolation.
        Rejected intervals and reasons are retained; missing choices are None.

        References
        ----------
        Han, Chen and Zhang (2022), "Rank Determination in Tensor Factor Model",
        https://arxiv.org/html/2011.07131v3, Remark 8 and Section 5.4.
        Their second-interval discussion motivates rejecting the initial
        maximum-rank plateau. The explicit finite-grid width, variance tolerance,
        convergence filter and representative are package discretization choices.
        """
        variance_tolerance = finite_scalar(
            variance_tolerance, "variance_tolerance", minimum=0
        )
        minimum_grid_points = positive_int(
            minimum_grid_points, "minimum_grid_points", minimum=2
        )
        if not isinstance(allow_unconverged, (bool, np.bool_)):
            raise ValueError("allow_unconverged must be boolean")
        complete_c = np.all(self.ranks >= 0, axis=(0, 2))
        converged_c = self.converged.all(axis=1)
        selected, multipliers, chosen_ranks, all_intervals, reasons = [], [], [], [], []
        all_point_reasons = []
        for mode, maximum in enumerate(self.max_ranks):
            full = self.ranks[mode, :, -1]
            low = complete_c & np.isfinite(self.variances[mode])
            low &= self.variances[mode] <= variance_tolerance
            if not allow_unconverged:
                low &= converged_c
            point_reasons = []
            for c_index in range(len(self.c_grid)):
                rejected = []
                if not complete_c[c_index]:
                    rejected.append("failed_subsample_fit")
                if not np.isfinite(self.variances[mode, c_index]):
                    rejected.append("variance_unavailable")
                elif self.variances[mode, c_index] > variance_tolerance:
                    rejected.append("variance_above_tolerance")
                if not allow_unconverged and not converged_c[c_index]:
                    rejected.append("unconverged_subsample_fit")
                point_reasons.append(tuple(rejected))
            all_point_reasons.append(tuple(point_reasons))
            intervals = []
            index = 0
            while index < len(self.c_grid):
                if not low[index]:
                    index += 1
                    continue
                end = index
                while (
                    end + 1 < len(self.c_grid)
                    and low[end + 1]
                    and full[end + 1] == full[index]
                ):
                    end += 1
                rejected = []
                if full[index] == maximum:
                    rejected.append("maximum_rank_plateau")
                if full[index] == 0:
                    rejected.append("zero_rank_under_positive_rank_assumption")
                if end - index + 1 < minimum_grid_points:
                    rejected.append("too_few_grid_points")
                intervals.append(
                    RankStabilityInterval(
                        index,
                        end,
                        float(self.c_grid[index]),
                        float(self.c_grid[end]),
                        int(full[index]),
                        not rejected,
                        tuple(rejected),
                        index == 0 or end == len(self.c_grid) - 1,
                    )
                )
                index = end + 1
            first = next((region for region in intervals if region.admissible), None)
            if first is None:
                selected.append(None)
                multipliers.append(None)
                chosen_ranks.append(None)
                reasons.append("no_admissible_region_on_supplied_grid")
            else:
                representative = (first.start_index + first.end_index) // 2
                selected.append(representative)
                multipliers.append(float(self.c_grid[representative]))
                chosen_ranks.append(first.rank)
                reasons.append("first_admissible_region_lower_middle_gridpoint")
            all_intervals.append(tuple(intervals))
        return RankStabilityChoice(
            tuple(selected),
            tuple(multipliers),
            tuple(chosen_ranks),
            tuple(all_intervals),
            tuple(reasons),
            variance_tolerance,
            minimum_grid_points,
            bool(allow_unconverged),
            tuple(all_point_reasons),
        )


def _grids(X, c_grid, spatial_subsets, time_prefixes, max_ranks, lags):
    raw = np.asarray(c_grid)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf" or len(raw) < 2:
        raise ValueError("c_grid must be a real vector with at least two points")
    grid = raw.astype(float, copy=True)
    if not np.isfinite(grid).all() or np.any(grid <= 0) or np.any(np.diff(grid) <= 0):
        raise ValueError("c_grid must be finite, positive and strictly increasing")
    try:
        subsets, prefixes = tuple(spatial_subsets), tuple(time_prefixes)
    except TypeError as exc:
        raise ValueError(
            "supply explicit spatial_subsets and time_prefixes sequences"
        ) from exc
    if len(subsets) < 2 or len(subsets) != len(prefixes):
        raise ValueError("need at least two paired spatial subsets/time prefixes")
    prefixes = tuple(positive_int(t, "time prefix", minimum=lags + 1) for t in prefixes)
    if prefixes[-1] != len(X) or any(a > b for a, b in zip(prefixes, prefixes[1:])):
        raise ValueError("time prefixes must be nondecreasing and end at len(X)")
    shape = X.shape[1:]
    if np.ndim(max_ranks) == 0:
        caps = (positive_int(max_ranks, "max_ranks"),) * len(shape)
    else:
        try:
            caps = tuple(positive_int(r, "max rank") for r in max_ranks)
        except TypeError as exc:
            raise ValueError(
                "max_ranks must be integer or one integer per mode"
            ) from exc
        if len(caps) != len(shape):
            raise ValueError("max_ranks must have one entry per spatial mode")
    clean, previous = [], None
    for subset in subsets:
        try:
            subset = tuple(subset)
        except TypeError as exc:
            raise ValueError(
                "each spatial subset needs one index vector per mode"
            ) from exc
        if len(subset) != len(shape):
            raise ValueError("each spatial subset needs one index vector per mode")
        entry = []
        for mode, (indices, dimension) in enumerate(zip(subset, shape)):
            raw = np.asarray(indices)
            if raw.ndim != 1 or raw.dtype.kind not in "iu" or not raw.size:
                raise ValueError("spatial indices must be nonempty integer vectors")
            if (
                np.any(raw < 0)
                or np.any(raw >= dimension)
                or len(np.unique(raw)) != len(raw)
            ):
                raise ValueError("spatial indices must be unique and in bounds")
            current = tuple(int(i) for i in raw)
            if caps[mode] >= len(current):
                raise ValueError(
                    "fixed max_ranks must be smaller than every subsample dimension"
                )
            if previous is not None and not set(previous[mode]) < set(current):
                raise ValueError(
                    "spatial subsets must be strictly nested in every mode"
                )
            entry.append(current)
        previous = tuple(entry)
        clean.append(previous)
    if any(set(indices) != set(range(d)) for indices, d in zip(clean[-1], shape)):
        raise ValueError("last spatial subset must contain the full data in every mode")
    return grid, tuple(clean), prefixes, caps


def tensor_rank_stability(
    X,
    c_grid,
    *,
    spatial_subsets,
    time_prefixes,
    max_ranks,
    method="tipup",
    penalty=2,
    lags=1,
    iterative=False,
    nu=0.0,
    center=True,
    max_iter=100,
    tol=1e-8,
):
    """Compute the published IC penalty-constant subsample-stability path.

    c_grid is a supplied positive increasing vector. At each point the same c
    multiplies all mode penalties; modewise choices may subsequently differ.
    spatial_subsets[j][k] gives actual indices in mode k, time_prefixes[j]
    gives the number of initial observations. Sets must grow strictly in every
    spatial mode and finish with full data; prefixes may repeat but must end at
    len(X). At least two paired subsamples are required. No random subsampling,
    shuffling, out-of-sample tuning, rank-cap clipping or interpolation occurs.

    Every cell reruns IC with that subsample's dimensions and T, including its
    own centering and iterative initialization. Numerical fit failures are
    recorded, not omitted from the variance. Their rank is -1 and variance NaN.
    Finite unconverged results remain available but are ineligible for default
    plateau selection. options records the controls passed to the IC selector.

    References
    ----------
    Han, Y., Chen, R., and Zhang, C.-H. (2022), "Rank Determination in Tensor
    Factor Model", https://arxiv.org/html/2011.07131v3, Remark 8 and Section 5.4.
    S[k,c] is the population variance over all supplied subsamples (ddof=0).
    The explicit index grid and numerical plateau interpretation are user/package
    choices, not an automatically consistent selector for arbitrary data.
    """
    # Imported here to keep the independent path records lightweight.
    from .rank_selection import select_tensor_rank

    lags = positive_int(lags, "lags")
    X = as_series(X, min_samples=lags + 1, ndim=None)
    grid, subsets, prefixes, caps = _grids(
        X, c_grid, spatial_subsets, time_prefixes, max_ranks, lags
    )
    if method not in ("tipup", "topup"):
        raise ValueError("method must be tipup or topup")
    penalty = positive_int(penalty, "penalty")
    if penalty > 5:
        raise ValueError("penalty must be an integer from 1 through 5")
    nu = finite_scalar(nu, "nu", minimum=0)
    if nu > 1:
        raise ValueError("nu must lie between zero and one")
    if not isinstance(iterative, (bool, np.bool_)) or not isinstance(
        center, (bool, np.bool_)
    ):
        raise ValueError("iterative and center must be boolean")
    max_iter = positive_int(max_iter, "max_iter")
    tol = finite_scalar(tol, "tol", minimum=0)
    if tol == 0:
        raise ValueError("tol must be positive")
    options = dict(
        method=method,
        criterion="ic",
        penalty=penalty,
        max_ranks=caps,
        lags=lags,
        iterative=bool(iterative),
        nu=nu,
        center=bool(center),
        max_iter=max_iter,
        tol=tol,
    )
    ranks = np.full((X.ndim - 1, len(grid), len(subsets)), -1, dtype=int)
    cells = [[None] * len(subsets) for _ in grid]
    # Materialize only one subsample at a time. Preserve the exact supplied order.
    for j, (indices, stop) in enumerate(zip(subsets, prefixes)):
        sample = X[:stop]
        for axis, index in enumerate(indices, 1):
            sample = np.take(sample, index, axis=axis)
        for c_index, multiplier in enumerate(grid):
            try:
                fitted = select_tensor_rank(
                    sample, penalty_multiplier=float(multiplier), **options
                )
                ranks[:, c_index, j] = fitted.ranks
                cells[c_index][j] = RankStabilityCell(fitted)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                cells[c_index][j] = RankStabilityCell(
                    None, f"{type(exc).__name__}: {exc}"
                )
    variances = ranks.astype(float).var(axis=2, ddof=0)
    variances[np.any(ranks < 0, axis=2)] = np.nan
    return TensorRankStabilityResult(
        grid,
        subsets,
        prefixes,
        caps,
        options,
        ranks,
        variances,
        tuple(tuple(row) for row in cells),
    )
