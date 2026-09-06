"""Independent scalar-factor and finite-grid oracles for IC stability.

Han, Chen and Zhang (2022), Remark 8 and Section 5.4:
https://arxiv.org/html/2011.07131v3. Plateau discretization is a package
convention; these tests do not claim that every stable interval is consistent.
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from mavats.rank_stability import (
    RankStabilityCell,
    TensorRankStabilityResult,
    tensor_rank_stability,
)


def _design():
    return dict(
        spatial_subsets=(
            ((4, 0, 2), (5, 1, 3), (6, 2, 0)),
            ((2, 4, 1, 0), (1, 3, 5, 0, 2), (0, 6, 4, 2)),
            (tuple(range(4, -1, -1)), tuple(range(5, -1, -1)), tuple(range(6, -1, -1))),
        ),
        time_prefixes=(7, 10, 12),
        max_ranks=(2, 2, 2),
    )


@pytest.mark.parametrize("method", ["topup", "tipup"])
@pytest.mark.parametrize("center", [False, True])
def test_scalar_factor_analytic_spectrum_uses_each_subsample_dimensions_and_mean(
    method, center
):
    t = np.arange(12, dtype=float)
    factor = (t * t - 3 * t + 2) / 10
    loading = tuple(np.arange(1, d + 1, dtype=float) / d for d in (5, 6, 7))
    X = np.einsum("t,i,j,k->tijk", factor, *loading)
    grid = np.array([0.001, 0.1, 10, 1000, 100000])
    design = _design()
    path = tensor_rank_stability(
        X, grid, method=method, center=center, lags=2, penalty=2, nu=0.4, **design
    )
    expected = np.empty((3, len(grid), 3), dtype=int)
    for j, (indices, stop) in enumerate(
        zip(design["spatial_subsets"], design["time_prefixes"])
    ):
        f = factor[:stop].copy()
        if center:
            f -= math.fsum(f) / stop
        temporal = math.fsum(
            (math.fsum(f[s - h] * f[s] for s in range(h, stop)) / (stop - h)) ** 2
            for h in (1, 2)
        )
        spatial = math.prod(
            math.fsum(loading[k][i] ** 2 for i in index)
            for k, index in enumerate(indices)
        )
        eigenvalue = temporal * spatial**2
        d = math.prod(len(index) for index in indices)
        base = 2 * d**1.2 * (1 / stop + 1 / d) * math.log(d * stop / (d + stop))
        for c_index, c in enumerate(grid):
            expected[:, c_index, j] = int(eigenvalue > c * base)
            cell = path.cells[c_index][j]
            assert cell.error is None and cell.converged
            for diagnostic in cell.result.diagnostics:
                assert diagnostic.log_penalty == pytest.approx(math.log(c * base))
                actual = diagnostic.eigenvalues[0] * math.exp(
                    diagnostic.log_eigenvalue_scale
                )
                assert actual == pytest.approx(eigenvalue, rel=3e-12)
    np.testing.assert_array_equal(path.ranks, expected)
    means = expected.sum(axis=2) / 3
    variance = sum((expected[:, :, j] - means) ** 2 for j in range(3)) / 3
    np.testing.assert_allclose(path.variances, variance)
    np.testing.assert_array_equal(path.full_sample_ranks, expected[:, :, -1])


def _path(ranks, converged):
    modes, count, samples = ranks.shape
    cells = tuple(
        tuple(
            (
                RankStabilityCell(None, "failed")
                if np.any(ranks[:, c, j] < 0)
                else RankStabilityCell(
                    SimpleNamespace(converged=converged[c, j], n_iter=1)
                )
            )
            for j in range(samples)
        )
        for c in range(count)
    )
    variance = ranks.astype(float).var(axis=2)
    variance[np.any(ranks < 0, axis=2)] = np.nan
    return TensorRankStabilityResult(
        np.geomspace(0.001, 1e6, count),
        (),
        tuple(range(samples)),
        (4,) * modes,
        {},
        ranks,
        variance,
        cells,
    )


def _plateau_choice_oracle(path, tolerance, width, relaxed):
    choices = []
    for mode in range(len(path.max_ranks)):
        blocks = []
        current = []
        for c in range(len(path.c_grid)):
            valid = all(
                path.cells[c][j].result is not None
                and (relaxed or path.cells[c][j].result.converged)
                for j in range(len(path.cells[c]))
            )
            valid &= math.isfinite(path.variances[mode, c])
            valid &= path.variances[mode, c] <= tolerance
            rank = path.ranks[mode, c, -1]
            if current and (not valid or rank != path.ranks[mode, current[-1], -1]):
                blocks.append(current)
                current = []
            if valid:
                current.append(c)
        if current:
            blocks.append(current)
        admissible = [
            block
            for block in blocks
            if len(block) >= width
            and 0 < path.ranks[mode, block[0], -1] < path.max_ranks[mode]
        ]
        choices.append(
            None if not admissible else admissible[0][(len(admissible[0]) - 1) // 2]
        )
    return tuple(choices)


@pytest.mark.parametrize("relaxed", [False, True])
@pytest.mark.parametrize("tolerance", [0, 0.25, 3])
def test_plateau_engine_matches_independent_partition_oracle(relaxed, tolerance):
    rng = np.random.default_rng(710)
    for _ in range(30):
        # Repeated blocks yield genuine candidates; scattered failures and
        # unfinished fits must split them rather than shrink the denominator.
        ranks = np.repeat(rng.integers(0, 5, size=(2, 5, 3)), 3, axis=1)
        converged = rng.random((15, 3)) > 0.08
        for c, j in zip(rng.integers(0, 15, 2), rng.integers(0, 3, 2)):
            ranks[:, c, j] = -1
        path = _path(ranks, converged)
        expected = _plateau_choice_oracle(path, tolerance, 2, relaxed)
        result = path.choose_plateaus(
            variance_tolerance=tolerance,
            minimum_grid_points=2,
            allow_unconverged=relaxed,
        )
        assert result.selected_indices == expected
        assert result.complete == all(index is not None for index in expected)
        for mode, index in enumerate(expected):
            if index is None:
                assert result.selected_ranks[mode] is None
                assert result.penalty_multipliers[mode] is None
            else:
                assert result.selected_ranks[mode] == path.ranks[mode, index, -1]
                assert result.penalty_multipliers[mode] == path.c_grid[index]


def test_failed_cell_never_becomes_admissible_when_unconverged_is_allowed(monkeypatch):
    from mavats import rank_selection

    def incomplete(sample, **kwargs):
        if sample.shape[0] == 10:
            raise np.linalg.LinAlgError("deliberate failed middle subsample")
        return SimpleNamespace(ranks=(1, 1, 1), converged=False, n_iter=1)

    monkeypatch.setattr(rank_selection, "select_tensor_rank", incomplete)
    path = tensor_rank_stability(np.ones((12, 5, 6, 7)), [0.01, 0.1, 1], **_design())
    assert np.isnan(path.variances).all()
    assert np.all(path.ranks[:, :, 1] == -1)
    assert all(path.cells[c][1].result is None for c in range(3))
    relaxed = path.choose_plateaus(
        variance_tolerance=100, minimum_grid_points=2, allow_unconverged=True
    )
    assert not relaxed.complete and relaxed.selected_indices == (None, None, None)
    assert all(
        "failed_subsample_fit" in reasons for reasons in relaxed.point_reasons[0]
    )


def test_programming_errors_are_not_misreported_as_numerical_failures(monkeypatch):
    from mavats import rank_selection

    def broken(sample, **kwargs):
        raise RuntimeError("implementation bug is not a statistical failure")

    monkeypatch.setattr(rank_selection, "select_tensor_rank", broken)
    with pytest.raises(RuntimeError, match="implementation bug"):
        tensor_rank_stability(np.ones((12, 5, 6, 7)), [0.01, 0.1], **_design())
