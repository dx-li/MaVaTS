import numpy as np
from numpy.testing import assert_allclose

from benchmarks.factor_ranks import _adjacent_fit, _Projection, _scores
from benchmarks.rank_scenarios import rank_factor_data
from mavats.rank_selection import select_tensor_rank


def test_known_loading_projection_matches_dense_held_out_least_squares():
    data = rank_factor_data(40, shape=(6, 8), regime="weak")
    fit = _Projection(data.observations[:25], data.loadings)
    design = np.kron(*data.loadings[::-1])
    flat = np.array([(x - fit.mean).ravel(order="F") for x in data.observations[25:]])
    core = np.linalg.lstsq(design, flat.T, rcond=None)[0]
    expected = (
        np.array([x.reshape((6, 8), order="F") for x in (design @ core).T]) + fit.mean
    )
    actual = fit.inverse_transform(fit.transform(data.observations[25:]))
    assert_allclose(actual, expected, atol=3e-15)
    assert_allclose(
        fit.transform(data.observations[25:28]),
        fit.transform(data.observations[25:])[:3],
    )


def test_zero_signal_scoring_uses_defined_absolute_error():
    data = rank_factor_data(40, shape=(6, 8), regime="no-factors")
    fit = _Projection(data.observations[:25], data.loadings)
    score = _scores(fit, data, 25)
    assert score["rank_correct"]
    assert "held_out_relative_signal_error" not in score
    assert_allclose(
        score["held_out_signal_mse"], np.mean(data.observations[:25].mean(axis=0) ** 2)
    )
    assert_allclose(fit.signal, np.repeat(fit.mean[None], 25, axis=0))


def test_adjacent_comparator_holds_pilot_ranks_fixed_with_matched_cap():
    data = rank_factor_data(70, shape=(6, 8))
    first = _adjacent_fit(data.observations, "tipup", False, (3, 3), 1)
    final = _adjacent_fit(data.observations, "tipup", True, (3, 3), 1)
    assert first.ranks == final.ranks
    assert all(r <= 3 for r in final.ranks)


def test_published_criterion_benchmark_retains_history_and_physical_penalties():
    data = rank_factor_data(50, shape=(6, 8))
    fit = select_tensor_rank(
        data.observations[:35], criterion="ic", penalty=2, iterative=True
    )
    score = _scores(fit, data, 35)
    assert score["rank_history"] == [list(r) for r in fit.rank_history]
    assert score["starting_ranks"] == list(fit.starting_ranks)
    assert score["stopping_reason"] == fit.stop_reason
    for value, step in zip(score["terminal_diagnostics"], fit.diagnostics):
        assert value["log_penalty"] == step.log_penalty
        assert value["log_eigenvalue_scale"] == step.log_eigenvalue_scale
