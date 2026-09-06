"""Independent dense-spectrum and finite-grid IC stability checks.

Reference: Han, Chen and Zhang (2022), Rank Determination in Tensor Factor
Model, https://arxiv.org/html/2011.07131v3, Remark 8 and Section 5.4.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from mavats.rank_stability import (
    RankStabilityCell,
    TensorRankStabilityResult,
    tensor_rank_stability,
)


def _design(shape=(5, 6), n=32):
    return dict(
        spatial_subsets=[
            (tuple(range(3)), tuple(range(4))),
            (tuple(range(4)), tuple(range(5))),
            (tuple(range(shape[0])), tuple(range(shape[1]))),
        ],
        time_prefixes=[n - 9, n - 4, n],
        max_ranks=(2, 2),
    )


def _direct_ranks(X, c, caps, method, lags, penalty, nu, center):
    # Explicit physical-unit dense covariance tensors, never package kernels.
    data = X - X.mean(axis=0) if center else X
    n = len(data)
    d = np.prod(data.shape[1:])
    ranks = []
    for mode, cap in enumerate(caps):
        dimension = data.shape[mode + 1]
        unfolded = np.moveaxis(data, mode + 1, 1).reshape(n, dimension, -1)
        moment = np.zeros((dimension, dimension))
        for lag in range(1, lags + 1):
            if method == "tipup":
                cross = sum(
                    a @ b.T for a, b in zip(unfolded[:-lag], unfolded[lag:])
                ) / (n - lag)
            else:
                cross = sum(
                    np.einsum("ia,jb->iajb", a, b)
                    for a, b in zip(unfolded[:-lag], unfolded[lag:])
                )
                cross = cross.reshape(dimension, -1) / (n - lag)
            moment += cross @ cross.T
        eigenvalues = np.linalg.eigvalsh(moment)[::-1]
        logarithm = (
            np.log(d * n / (d + n)) if penalty == 2 else np.log(min(dimension, n))
        )
        threshold = c * lags * d ** (2 - 2 * nu) * (1 / n + 1 / d) * logarithm
        objectives = [sum(eigenvalues[r:]) + r * threshold for r in range(cap + 1)]
        ranks.append(np.argmin(objectives))
    return ranks


@pytest.mark.parametrize("method", ["tipup", "topup"])
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("penalty,nu", [(2, 0.0), (5, 0.3)])
def test_dense_physical_spectrum_oracle(method, center, penalty, nu):
    data = np.random.default_rng(532).normal(size=(32, 5, 6)) + 0.7
    grid = np.array([1e-5, 1e-3, 0.01, 0.05, 0.3, 1.0, 10.0])
    design = _design()
    path = tensor_rank_stability(
        data,
        grid,
        method=method,
        center=center,
        penalty=penalty,
        nu=nu,
        lags=2,
        **design,
    )
    expected = np.empty_like(path.ranks)
    for j, (subset, stop) in enumerate(
        zip(design["spatial_subsets"], design["time_prefixes"])
    ):
        sample = data[:stop][:, subset[0]][:, :, subset[1]]
        for i, c in enumerate(grid):
            expected[:, i, j] = _direct_ranks(
                sample, c, (2, 2), method, 2, penalty, nu, center
            )
    np.testing.assert_array_equal(path.ranks, expected)
    # Independent arithmetic uses the population, not sample, denominator.
    means = expected.sum(axis=2) / expected.shape[2]
    variance = sum((expected[:, :, j] - means) ** 2 for j in range(3)) / 3
    np.testing.assert_allclose(path.variances, variance)
    assert path.converged.all() and path.monotone.all()
    assert all(
        cell.n_iter == 0 and cell.error is None for row in path.cells for cell in row
    )
    np.testing.assert_array_equal(path.full_sample_ranks, expected[:, :, -1])


def _hand_path(ranks, converged=None):
    ranks = np.asarray(ranks)
    modes, count, samples = ranks.shape
    converged = (
        np.ones((count, samples), dtype=bool) if converged is None else converged
    )
    cells = tuple(
        tuple(
            RankStabilityCell(
                SimpleNamespace(
                    converged=bool(converged[c, j]),
                    n_iter=3,
                    ranks=tuple(ranks[:, c, j]),
                )
            )
            for j in range(samples)
        )
        for c in range(count)
    )
    return TensorRankStabilityResult(
        np.arange(1, count + 1, dtype=float),
        (),
        tuple(range(samples)),
        (4,) * modes,
        {},
        ranks,
        ranks.var(axis=2, ddof=0),
        cells,
    )


def test_second_interval_not_global_variance_minimum_and_modewise_choices():
    ranks = np.repeat(
        np.array(
            [
                [4, 4, 3, 2, 2, 2, 1, 1, 0, 0],
                [4, 4, 4, 3, 2, 1, 1, 1, 0, 0],
            ]
        )[:, :, None],
        3,
        axis=2,
    )
    ranks[0, 2, 0] = 4
    ranks[1, 3:5, 0] += 1
    path = _hand_path(ranks)
    choice = path.choose_plateaus(variance_tolerance=0, minimum_grid_points=2)
    assert choice.complete
    assert choice.selected_indices == (4, 6)
    assert choice.penalty_multipliers == (5.0, 7.0)
    assert choice.selected_ranks == (2, 1)
    assert np.argmin(path.variances[0]) == 0  # Incorrect global argmin picks cap.
    assert choice.intervals[0][0].reasons == ("maximum_rank_plateau",)
    assert choice.intervals[0][-1].reasons == (
        "zero_rank_under_positive_rank_assumption",
    )
    assert choice.intervals[0][0].touches_grid_boundary
    assert choice.point_reasons[0][2] == ("variance_above_tolerance",)


def test_convergence_filter_breaks_intervals_unless_explicitly_disabled():
    ranks = np.repeat(np.array([[4, 4, 2, 2, 2, 2, 0, 0]])[:, :, None], 3, axis=2)
    converged = np.ones((8, 3), dtype=bool)
    converged[3, 1] = False
    path = _hand_path(ranks, converged)
    strict = path.choose_plateaus(variance_tolerance=0, minimum_grid_points=3)
    assert not strict.complete and strict.selected_indices == (None,)
    assert "unconverged_subsample_fit" in strict.point_reasons[0][3]
    relaxed = path.choose_plateaus(
        variance_tolerance=0, minimum_grid_points=3, allow_unconverged=True
    )
    assert relaxed.selected_indices == (3,)
    assert not path.converged.all()


def test_tolerance_constant_rank_and_no_admissible_region():
    ranks = np.array([[[2, 2, 3], [2, 2, 2], [1, 1, 1], [0, 0, 0]]])
    path = _hand_path(ranks)
    # Equal low variance alone does not merge different full-sample ranks.
    choice = path.choose_plateaus(variance_tolerance=0.3, minimum_grid_points=2)
    assert not choice.complete
    assert choice.penalty_multipliers == (None,)
    assert choice.reasons == ("no_admissible_region_on_supplied_grid",)
    ranks[0, 0] = [2, 3, 2]
    choice = _hand_path(ranks).choose_plateaus(
        variance_tolerance=0.3, minimum_grid_points=2
    )
    assert choice.selected_indices == (0,)
    assert choice.intervals[0][0].touches_grid_boundary  # Explicitly truncated grid.


def test_record_numerical_failure_without_variance_case_deletion(monkeypatch):
    from mavats import rank_selection

    real = rank_selection.select_tensor_rank

    def failing(data, **kwargs):
        if kwargs["penalty_multiplier"] == 0.05 and data.shape[1] == 4:
            raise FloatingPointError("injected numerical failure")
        return real(data, **kwargs)

    monkeypatch.setattr(rank_selection, "select_tensor_rank", failing)
    path = tensor_rank_stability(
        np.random.default_rng(3).normal(size=(32, 5, 6)), [0.01, 0.05, 0.1], **_design()
    )
    assert path.cells[1][1].result is None
    assert "injected numerical failure" in path.cells[1][1].error
    assert path.cells[1][1].n_iter is None
    assert not path.converged[1, 1]
    assert (path.ranks[:, 1, 1] == -1).all() and np.isnan(path.variances[:, 1]).all()
    assert not path.monotone.any()
    choice = path.choose_plateaus(variance_tolerance=1, minimum_grid_points=2)
    assert "failed_subsample_fit" in choice.point_reasons[0][1]


def test_planted_rank_one_path_and_actual_nonrandom_indices():
    rng = np.random.default_rng(97)
    factors = np.zeros(150)
    for t in range(1, len(factors)):
        factors[t] = 0.8 * factors[t - 1] + rng.normal()
    data = factors[:, None, None] * np.ones((1, 6, 7)) + 0.1 * rng.normal(
        size=(150, 6, 7)
    )
    subsets = [
        ([4, 0, 3], [6, 2, 0]),
        ([4, 0, 3, 1], [6, 2, 0, 4]),
        (list(range(6)), list(range(7))),
    ]
    path = tensor_rank_stability(
        data,
        np.geomspace(1e-6, 100, 40),
        spatial_subsets=subsets,
        time_prefixes=[100, 125, 150],
        max_ranks=2,
    )
    choice = path.choose_plateaus(variance_tolerance=0, minimum_grid_points=3)
    assert choice.selected_ranks == (1, 1)
    assert path.spatial_subsets[0] == ((4, 0, 3), (6, 2, 0))
    subsets[0][0][0] = 2
    assert path.spatial_subsets[0][0][0] == 4


def test_own_centering_causality_and_physical_fourth_power_scaling():
    rng = np.random.default_rng(21)
    data = rng.normal(size=(32, 5, 6))
    grid = np.geomspace(1e-5, 1, 12)
    design = _design()
    path = tensor_rank_stability(data, grid, **design)
    for scale in (1e-75, 1e75):
        changed = tensor_rank_stability(data * scale, grid * scale**4, **design)
        np.testing.assert_array_equal(changed.ranks, path.ranks)
    shifted = tensor_rank_stability(data + 1e6, grid, **design)
    np.testing.assert_array_equal(shifted.ranks, path.ranks)
    data[23:] += 100
    changed = tensor_rank_stability(data, grid, **design)
    np.testing.assert_array_equal(changed.ranks[:, :, 0], path.ranks[:, :, 0])


def test_iterative_cells_keep_real_stopping_status():
    data = np.random.default_rng(539).normal(size=(32, 5, 6))
    path = tensor_rank_stability(
        data, [1e-7, 1e-4, 0.01], iterative=True, max_iter=1, tol=1e-14, **_design()
    )
    assert not path.converged.all()
    assert all(cell.result is not None for row in path.cells for cell in row)
    assert all(cell.n_iter == 1 for row in path.cells for cell in row)
    choice = path.choose_plateaus(variance_tolerance=100, minimum_grid_points=2)
    assert not choice.complete


@pytest.mark.parametrize("order", [1, 3])
def test_vector_and_order_three_tensor_paths(order):
    data = np.random.default_rng(35).normal(size=(24,) + (5,) * order)
    subsets = [tuple(tuple(range(size)) for _ in range(order)) for size in (3, 4, 5)]
    prefixes = (16, 20, 24)
    grid = [1e-5, 0.01, 1]
    path = tensor_rank_stability(
        data, grid, spatial_subsets=subsets, time_prefixes=prefixes, max_ranks=2
    )
    assert path.ranks.shape == (order, 3, 3)
    for j, stop in enumerate(prefixes):
        sample = data[(slice(stop),) + (slice(j + 3),) * order]
        for c_index, c in enumerate(grid):
            expected = _direct_ranks(sample, c, (2,) * order, "tipup", 1, 2, 0, True)
            np.testing.assert_array_equal(path.ranks[:, c_index, j], expected)


def test_equal_time_prefixes_and_reordered_full_sample_are_explicit():
    data = np.random.default_rng(905).normal(size=(32, 5, 6))
    options = _design()
    options["time_prefixes"] = [32, 32, 32]
    options["spatial_subsets"][-1] = (list(range(4, -1, -1)), list(range(5, -1, -1)))
    path = tensor_rank_stability(data, [0.001, 0.01, 0.1], **options)
    assert path.time_prefixes == (32, 32, 32)
    assert path.spatial_subsets[-1][0] == (4, 3, 2, 1, 0)
    for i, c in enumerate(path.c_grid):
        np.testing.assert_array_equal(
            path.full_sample_ranks[:, i],
            _direct_ranks(data, c, (2, 2), "tipup", 1, 2, 0, True),
        )


@pytest.mark.parametrize(
    "grid",
    [
        [1],
        [1, 1],
        [2, 1],
        [0, 1],
        [-1, 1],
        [1, np.nan],
        [1, np.inf],
        [True, False],
        [[1, 2]],
    ],
)
def test_invalid_c_grid(grid):
    with pytest.raises(ValueError, match="c_grid"):
        tensor_rank_stability(np.ones((32, 5, 6)), grid, **_design())


@pytest.mark.parametrize(
    "override",
    [
        {"time_prefixes": [20, 19, 32]},
        {"time_prefixes": [20, 25, 31]},
        {"time_prefixes": [True, 25, 32]},
        {"time_prefixes": [1, 25, 32]},
        {"max_ranks": (3, 2)},
        {"max_ranks": (2,)},
        {"max_ranks": True},
        {"max_ranks": 0},
        {"lags": 32},
        {"method": "pca"},
        {"penalty": 6},
        {"penalty": True},
        {"nu": 2},
        {"center": 1},
        {"iterative": 1},
        {"tol": 0},
        {"max_iter": 0},
    ],
)
def test_invalid_controls(override):
    options = _design()
    options.update(override)
    with pytest.raises(ValueError):
        tensor_rank_stability(np.ones((32, 5, 6)), [0.1, 1], **options)


@pytest.mark.parametrize(
    "subsets",
    [
        [([0, 1, 2], [0, 1, 2, 3])],
        [([0, 1, 2], [0, 1, 2, 3])] * 3,
        [
            ([0, 1, 2], [0, 1, 2, 3]),
            ([0, 1, 3, 4], [0, 1, 2, 3, 4]),
            (list(range(5)), list(range(6))),
        ],
        [
            ([0, 1, 1], [0, 1, 2, 3]),
            ([0, 1, 2, 3], [0, 1, 2, 3, 4]),
            (list(range(5)), list(range(6))),
        ],
        [
            ([0.0, 1, 2], [0, 1, 2, 3]),
            ([0, 1, 2, 3], [0, 1, 2, 3, 4]),
            (list(range(5)), list(range(6))),
        ],
        [
            ([-1, 1, 2], [0, 1, 2, 3]),
            ([0, 1, 2, 3], [0, 1, 2, 3, 4]),
            (list(range(5)), list(range(6))),
        ],
    ],
)
def test_invalid_subsets(subsets):
    options = _design()
    options["spatial_subsets"] = subsets
    with pytest.raises(ValueError):
        tensor_rank_stability(np.ones((32, 5, 6)), [0.1, 1], **options)


@pytest.mark.parametrize(
    "options",
    [
        {"variance_tolerance": -1, "minimum_grid_points": 2},
        {"variance_tolerance": np.nan, "minimum_grid_points": 2},
        {"variance_tolerance": 0, "minimum_grid_points": 1},
        {"variance_tolerance": 0, "minimum_grid_points": 2, "allow_unconverged": 1},
    ],
)
def test_invalid_plateau_controls(options):
    with pytest.raises(ValueError):
        _hand_path(np.ones((1, 3, 2), dtype=int)).choose_plateaus(**options)
