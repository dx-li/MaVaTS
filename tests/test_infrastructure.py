import numpy as np
import pytest

from mavats.baselines import fit_naive, fit_var
from mavats.metrics import (
    mean_squared_error,
    relative_frobenius_error,
    subspace_distance,
)
from mavats.model_selection import rolling_forecast
from mavats.simulation import (
    mar_spectral_radius,
    matrix_normal,
    simulate_factor,
    simulate_mar,
)


def test_matrix_normal_covariance_and_singular_support():
    row = np.array([[2.0, 0.4], [0.4, 1.0]])
    col = np.array([[1.0, 0.3, 0.0], [0.3, 2.0, 0.1], [0.0, 0.1, 0.5]])
    x = matrix_normal(40000, row, col, random_state=1)
    flat = x.transpose(0, 2, 1).reshape(len(x), -1)
    np.testing.assert_allclose(np.cov(flat.T), np.kron(col, row), atol=0.055)
    x = matrix_normal(10, np.zeros((2, 2)), col, random_state=1)
    assert np.count_nonzero(x) == 0
    with pytest.raises(ValueError):
        matrix_normal(1, np.diag([1.0, -1.0]), col)


def test_simulation_deterministic_recurrence():
    a = np.array([[0.4, 0.2], [0.0, 0.3]])
    b = np.array([[0.7]])
    initial = np.ones((1, 2, 1))
    x = simulate_mar(
        5, a, b, burnin=0, initial=initial, row_cov=np.zeros((2, 2)), random_state=1
    )
    previous = initial[0]
    for observation in x:
        previous = a @ previous @ b.T
        np.testing.assert_allclose(observation, previous)
    assert mar_spectral_radius(a, b) == pytest.approx(0.28)
    with pytest.raises(ValueError, match="stationary"):
        simulate_mar(5, np.eye(2), np.eye(2))


def test_simulation_reproducibility_without_global_rng():
    np.random.seed(102)
    state = np.random.get_state()
    x = simulate_mar(10, [[0.5]], [[0.5]], random_state=91)
    np.testing.assert_array_equal(
        x, simulate_mar(10, [[0.5]], [[0.5]], random_state=91)
    )
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])
    np.testing.assert_array_equal(
        simulate_factor(5, (3, 4, 2), (2, 2, 1), random_state=2).observations,
        simulate_factor(5, (3, 4, 2), (2, 2, 1), random_state=2).observations,
    )


def test_tensor_simulation_reconstructs_truth():
    data = simulate_factor(30, (4, 5, 3), (2, 2, 1), noise_std=0, random_state=5)
    expected = np.einsum("tabc,ia,jb,kc->tijk", data.factors, *data.loadings)
    np.testing.assert_allclose(data.signal, expected)
    np.testing.assert_array_equal(data.signal, data.observations)


def test_metrics_identifiability():
    q = np.linalg.qr(np.random.default_rng(9).normal(size=(8, 2)))[0]
    assert subspace_distance(q, q @ np.array([[2.0, 1.0], [0.0, 3.0]])) < 3e-8
    assert subspace_distance(np.eye(4)[:, :2], np.eye(4)[:, 2:]) == pytest.approx(1)
    assert relative_frobenius_error([0], [0]) == 0
    assert np.isinf(relative_frobenius_error([0], [1]))
    assert mean_squared_error([1, 2], [2, 4]) == 2.5
    with pytest.raises(ValueError):
        subspace_distance(np.zeros((3, 1)), np.ones((3, 1)))


def test_var_matches_independent_column_vector_regression():
    x = simulate_mar(
        500, [[0.8, 0.15], [0.0, 0.7]], [[0.9, 0.1], [0.0, 0.8]], random_state=23
    )
    model = fit_var(x, order=2)
    flat = np.stack([observation.flatten(order="F") for observation in x])
    design = np.c_[flat[1:-1], flat[:-2], np.ones(len(x) - 2)]
    beta = np.linalg.lstsq(design, flat[2:], rcond=None)[0]
    np.testing.assert_allclose(model.coefficients[0], beta[:4].T, atol=1e-12)
    forecast = model.forecast(2)
    first = np.r_[flat[-1], flat[-2], 1.0] @ beta
    second = np.r_[first, flat[-1], 1.0] @ beta
    np.testing.assert_allclose(forecast[1].flatten(order="F"), second)


def test_ridge_rank_deficient_constant():
    x = np.full((20, 2, 2), 7.0)
    fit = fit_var(x, order=2, ridge=1.0)
    np.testing.assert_allclose(fit.forecast(2), 7.0)
    assert np.count_nonzero(fit.coefficients) == 0


def test_rolling_forecast_never_exposes_future():
    x = np.arange(20.0).reshape(20, 1, 1)
    seen = []

    def factory(training):
        seen.append(training.copy())
        return fit_naive(training)

    result = rolling_forecast(
        x, factory, initial_train_size=8, horizon=3, step=4, window=5
    )
    np.testing.assert_array_equal(result.origins, [8, 12, 16])
    np.testing.assert_array_equal(result.errors, np.tile([1.0, 4.0, 9.0], (3, 1)))
    for origin, training in zip(result.origins, seen):
        np.testing.assert_array_equal(training, x[origin - 5 : origin])


@pytest.mark.parametrize("bad", [True, 1.2, 0, -1])
def test_simulation_rejects_bad_counts(bad):
    with pytest.raises(ValueError):
        simulate_mar(bad, [[0.2]], [[0.5]])


@pytest.mark.parametrize("magnitude", [1e-200, 1e200])
def test_relative_error_is_invariant_to_extreme_scales(magnitude):
    actual = magnitude * np.array([[1.0, -2.0], [3.0, -0.5]])
    assert relative_frobenius_error(actual, 2 * actual) == pytest.approx(1.0)


def test_metrics_avoid_intermediate_overflow_and_underflow():
    assert mean_squared_error(
        np.full((20, 10), 1e154), np.zeros((20, 10))
    ) == pytest.approx(1e308)
    tiny_mse = mean_squared_error([1e-160], [0])
    assert tiny_mse > 0
    assert tiny_mse / 1e-320 == pytest.approx(1.0)
    assert np.isinf(mean_squared_error([1e308], [-1e308]))
    assert relative_frobenius_error([1e308], [-1e308]) == pytest.approx(2.0)
    assert relative_frobenius_error([1e-200], [1.0]) == pytest.approx(1e200)
    actual = np.array([1e308])
    predicted = np.nextafter(actual, 0.0)
    expected = ((actual - predicted) / actual).item()
    assert relative_frobenius_error(actual, predicted) == pytest.approx(expected, abs=0)


def test_subspace_distance_resolves_small_angles_and_column_scaling():
    theta = 1e-10
    a = np.array([[1.0], [0.0]])
    b = np.array([[np.cos(theta)], [np.sin(theta)]])
    assert subspace_distance(a, b) == pytest.approx(theta, rel=1e-8, abs=0)
    q = np.linalg.qr(np.random.default_rng(9).normal(size=(10, 2)))[0]
    assert subspace_distance(q, q * [1e-200, 1e200]) < 1e-14
    assert subspace_distance(np.eye(3)[:, :1], np.eye(3)[:, :2]) == pytest.approx(
        np.sqrt(1 / 3)
    )


def test_multilag_simulation_matches_dense_companion_with_nonsymmetric_factors():
    left = np.array([[[0.3, 0.2], [-0.1, 0.4]], [[-0.1, 0.1], [0.0, -0.15]]])
    right = np.array([[[0.5, -0.2], [0.1, 0.3]], [[0.3, 0.1], [-0.1, 0.2]]])
    initial = np.arange(8.0).reshape(2, 2, 2)
    intercept = np.array([[0.1, -0.2], [0.3, 0.4]])
    phi = [np.kron(b, a) for a, b in zip(left, right)]
    companion = np.block([[phi[0], phi[1]], [np.eye(4), np.zeros((4, 4))]])
    assert mar_spectral_radius(left, right) == pytest.approx(
        max(abs(np.linalg.eigvals(companion)))
    )
    state = np.r_[initial[-1].ravel(order="F"), initial[-2].ravel(order="F")]
    offset = np.r_[intercept.ravel(order="F"), np.zeros(4)]
    expected = []
    for _ in range(10):
        state = companion @ state + offset
        expected.append(state[:4].reshape(2, 2, order="F"))
    simulated = simulate_mar(
        7,
        left,
        right,
        initial=initial,
        intercept=intercept,
        row_cov=np.zeros((2, 2)),
        burnin=3,
        random_state=0,
    )
    np.testing.assert_allclose(simulated, expected[3:], atol=1e-14)


def test_simulation_handles_extreme_equivalent_factor_scalings():
    args = dict(
        initial=np.full((1, 1, 1), 1e200),
        burnin=0,
        row_cov=np.zeros((1, 1)),
        random_state=3,
    )
    ordinary = simulate_mar(8, [[0.4]], [[0.5]], **args)
    extreme = simulate_mar(8, [[0.4e250]], [[0.5e-250]], **args)
    np.testing.assert_allclose(extreme, ordinary, rtol=1e-14)


def test_matrix_normal_covariance_root_handles_extreme_finite_covariances():
    rng_seed = 333
    ordinary = matrix_normal(20, np.ones((2, 2)), np.eye(2), random_state=rng_seed)
    extreme = matrix_normal(
        20, np.full((2, 2), 1e308), np.eye(2) * 1e-308, random_state=rng_seed
    )
    np.testing.assert_allclose(extreme, ordinary, rtol=1e-14)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"check_stationarity": 1},
        {"check_stationarity": "false"},
        {"intercept": np.ones((1, 1), dtype=complex)},
        {"initial": np.ones((1, 1, 1), dtype=complex)},
    ],
)
def test_simulation_rejects_invalid_boolean_and_complex_data(kwargs):
    with pytest.raises(ValueError):
        simulate_mar(3, [[0.5]], [[0.5]], **kwargs)


def test_matrix_normal_rejects_complex_mean():
    with pytest.raises(ValueError, match="real"):
        matrix_normal(3, np.eye(2), np.eye(2), mean=np.ones((2, 2), dtype=complex))


def test_explosive_simulation_reports_overflow():
    with pytest.raises(FloatingPointError, match="overflowed"):
        simulate_mar(
            1200,
            [[2.0]],
            [[2.0]],
            check_stationarity=False,
            burnin=0,
            initial=np.ones((1, 1, 1)),
            row_cov=np.zeros((1, 1)),
        )


def test_var_and_naive_center_without_overflow():
    x = np.full((10, 2, 3), 1e308)
    fitted = fit_var(x, order=2)
    assert fitted.design_scale == 1e308
    assert np.isfinite(fitted.singular_values).all()
    np.testing.assert_array_equal(fitted.forecast(3), np.full((3, 2, 3), 1e308))
    np.testing.assert_array_equal(fit_naive(x, strategy="mean").forecast(1), x[:1])


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_var_coefficients_and_forecasts_are_scale_equivariant(scale):
    X = simulate_mar(120, [[0.5]], [[0.6]], random_state=19)
    reference = fit_var(X, order=2)
    scaled = fit_var(X * scale, order=2)
    np.testing.assert_allclose(scaled.coefficients, reference.coefficients, atol=1e-14)
    np.testing.assert_allclose(
        scaled.forecast(5) / scale, reference.forecast(5), atol=1e-14
    )


def test_var_scaled_ridge_matches_independent_augmented_regression():
    X = np.random.default_rng(7).normal(size=(80, 2, 3)) * 5
    ridge = 17.0
    fitted = fit_var(X, order=2, ridge=ridge)
    flat = np.stack([x.ravel(order="F") for x in X])
    D = np.concatenate((flat[1:-1], flat[:-2]), axis=1)
    Y = flat[2:]
    Dmean, Ymean = D.mean(axis=0), Y.mean(axis=0)
    oracle = np.linalg.lstsq(
        np.r_[D - Dmean, np.sqrt(ridge) * np.eye(12)],
        np.r_[Y - Ymean, np.zeros((12, 6))],
        rcond=None,
    )[0]
    np.testing.assert_allclose(
        fitted.coefficients, oracle.reshape(2, 6, 6).transpose(0, 2, 1), atol=2e-15
    )
    np.testing.assert_allclose(fitted.intercept, Ymean - Dmean @ oracle, atol=2e-15)


@pytest.mark.parametrize("intercept", [1, "false", None])
def test_var_requires_boolean_intercept(intercept):
    with pytest.raises(ValueError, match="boolean"):
        fit_var(np.ones((10, 2, 2)), intercept=intercept)


def test_rolling_aggregate_avoids_overflow_and_rejects_complex_forecasts():
    class Constant:
        def __init__(self, value):
            self.value = value

        def forecast(self, horizon):
            return np.full((horizon, 1, 1), self.value)

    X = np.zeros((10, 1, 1))
    result = rolling_forecast(
        X, lambda training: Constant(1e154), initial_train_size=4, horizon=2
    )
    assert result.mse == pytest.approx(1e308)
    with pytest.raises(ValueError, match="real"):
        rolling_forecast(X, lambda training: Constant(1 + 1j), initial_train_size=4)
