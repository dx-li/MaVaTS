"""Independent dense oracles for Tsay's minus-MA conditional model.

References: Tsay, R. S. (2024), Matrix-Variate Time Series Analysis: A Brief
Review and Some New Developments, https://doi.org/10.1111/insr.12558,
equations (4), (5), (30). These tests derive dense Gaussian equations rather
than importing filtering, gradient, or forecasting kernels from the package.
"""

import numpy as np
import pytest


def _vectorize(series):
    return np.asarray([matrix.ravel(order="F") for matrix in series])


def _operators(pairs):
    return [np.kron(right, left) for left, right in pairs]


def _lag_matrix(coefficients, length, dimension):
    result = np.eye(length * dimension)
    for lag, coefficient in enumerate(coefficients, 1):
        for time in range(lag, length):
            result[
                time * dimension : (time + 1) * dimension,
                (time - lag) * dimension : (time - lag + 1) * dimension,
            ] -= coefficient
    return result


def _dense_conditional(data, ar, ma, intercept, covariance):
    observations = _vectorize(data)
    prefix = max(len(ar), len(ma))
    length = len(data) - prefix
    dimension = observations.shape[1]
    response = observations[prefix:].copy() - intercept.ravel(order="F")
    for time in range(prefix, len(data)):
        for lag, coefficient in enumerate(ar, 1):
            response[time - prefix] -= coefficient @ observations[time - lag]
    innovation_map = _lag_matrix(ma, length, dimension)
    innovation_vector = np.linalg.solve(innovation_map, response.ravel())
    innovations = innovation_vector.reshape(length, dimension)
    # Since the map is unit block triangular its determinant is exactly one.
    # A second Gaussian density uses the dense correlated response covariance.
    response_covariance = (
        innovation_map @ np.kron(np.eye(length), covariance) @ innovation_map.T
    )
    sign, logdet = np.linalg.slogdet(response_covariance)
    assert sign == 1
    objective = 0.5 * (
        length * dimension * np.log(2 * np.pi)
        + logdet
        + response.ravel() @ np.linalg.solve(response_covariance, response.ravel())
    )
    return innovations, objective


def _dense_future(history, innovations, ar, ma, intercept, covariance, steps):
    history, innovations = _vectorize(history), _vectorize(innovations)
    dimension = history.shape[1]
    constants = np.tile(intercept.ravel(order="F"), (steps, 1))
    for time in range(steps):
        for lag, coefficient in enumerate(ar, 1):
            if time < lag:
                constants[time] += coefficient @ history[time - lag]
        for lag, coefficient in enumerate(ma, 1):
            if time < lag:
                constants[time] -= coefficient @ innovations[time - lag]
    autoregression = _lag_matrix(ar, steps, dimension)
    moving_average = _lag_matrix(ma, steps, dimension)
    means = np.linalg.solve(autoregression, constants.ravel()).reshape(steps, dimension)
    shock_map = np.linalg.solve(autoregression, moving_average)
    joint_covariance = shock_map @ np.kron(np.eye(steps), covariance) @ shock_map.T
    marginals = np.asarray(
        [
            joint_covariance[
                t * dimension : (t + 1) * dimension,
                t * dimension : (t + 1) * dimension,
            ]
            for t in range(steps)
        ]
    )
    return means, marginals


def _design(seed=842):
    rng = np.random.default_rng(seed)
    ar = [
        (rng.normal(size=(2, 2)) * 0.16, rng.normal(size=(3, 3)) * 0.2),
        (rng.normal(size=(2, 2)) * 0.12, rng.normal(size=(3, 3)) * 0.14),
    ]
    ma = [
        (rng.normal(size=(2, 2)) * 0.25, rng.normal(size=(3, 3)) * 0.24),
        (rng.normal(size=(2, 2)) * 0.18, rng.normal(size=(3, 3)) * 0.2),
    ]
    left = np.array([[1.2, 0.35], [0.35, 0.8]])
    right = np.array([[0.9, 0.2, -0.1], [0.2, 1.3, 0.25], [-0.1, 0.25, 1.1]])
    intercept = np.array([[0.4, -0.2, 0.1], [0.6, 0.15, -0.3]])
    data = rng.normal(size=(17, 2, 3)) + intercept
    return data, ar, ma, intercept, (left, right)


def _parameters(ar, ma, intercept, covariance):
    from mavats.marma import MARMAParameters

    return MARMAParameters(
        ar_left=[a for a, _ in ar],
        ar_right=[b for _, b in ar],
        ma_left=[a for a, _ in ma],
        ma_right=[b for _, b in ma],
        intercept=intercept,
        row_covariance=covariance[0],
        column_covariance=covariance[1],
    )


def _central_gradient(function, point):
    derivative = np.empty_like(point)
    for index in range(len(point)):
        step = 2e-5 * max(1.0, abs(point[index]))
        upper, lower = point.copy(), point.copy()
        upper[index] += step
        lower[index] -= step
        derivative[index] = (function(upper) - function(lower)) / (2 * step)
    return derivative


@pytest.mark.parametrize(
    "ar_order,ma_order", [(1, 2), (2, 1), (2, 2), (1, 0), (0, 2), (0, 0)]
)
def test_filter_against_dense_conditional_gaussian(ar_order, ma_order):
    from mavats.marma import filter_marma

    data, ar, ma, intercept, covariance = _design()
    ar, ma = ar[:ar_order], ma[:ma_order]
    parameters = _parameters(ar, ma, intercept, covariance)
    result = filter_marma(data, parameters)
    prefix = max(ar_order, ma_order)
    innovations, objective = _dense_conditional(
        data, _operators(ar), _operators(ma), intercept, np.kron(*covariance[::-1])
    )
    np.testing.assert_array_equal(
        result.conditioning_mask, np.arange(len(data)) < prefix
    )
    np.testing.assert_array_equal(result.residuals[:prefix], 0)
    np.testing.assert_allclose(
        _vectorize(result.residuals[prefix:]), innovations, rtol=2e-13, atol=2e-13
    )
    assert np.isnan(result.fitted_values[:prefix]).all()
    np.testing.assert_allclose(
        result.fitted_values[prefix:],
        data[prefix:] - result.residuals[prefix:],
        rtol=2e-13,
        atol=2e-13,
    )
    assert result.log_likelihood == pytest.approx(-objective, rel=2e-13)


def test_prefix_is_conditioned_not_filtered_and_ma_sign_is_negative():
    from mavats.marma import filter_marma

    ar = []
    ma = [(np.ones((1, 1)), np.array([[0.7]]))]
    data = np.array([100.0, 2.0, 3.0, -1.0]).reshape(-1, 1, 1)
    covariance = (np.ones((1, 1)), np.ones((1, 1)))
    parameters = _parameters(ar, ma, np.zeros((1, 1)), covariance)
    result = filter_marma(data, parameters)
    # An implementation that filtered the conditioning prefix would give
    # second innovation 72 rather than 2; the third exposes the MA sign.
    np.testing.assert_allclose(result.residuals.ravel(), [0.0, 2.0, 4.4, 2.08])


@pytest.mark.parametrize("steps", [1, 2, 6])
def test_fixed_parameter_forecasts_against_dense_noncommuting_system(steps):
    from mavats.marma import filter_marma

    data, ar, ma, intercept, covariance = _design()
    parameters = _parameters(ar, ma, intercept, covariance)
    result = filter_marma(data, parameters)
    ar, ma = _operators(ar), _operators(ma)
    assert np.linalg.norm(ar[0] @ ma[0] - ma[0] @ ar[0]) > 1e-3
    means, variances = _dense_future(
        data,
        result.residuals,
        ar,
        ma,
        intercept,
        np.kron(*covariance[::-1]),
        steps,
    )
    np.testing.assert_allclose(
        _vectorize(result.forecast(steps)), means, rtol=3e-13, atol=3e-13
    )
    np.testing.assert_allclose(
        result.forecast_covariance(steps), variances, rtol=3e-13, atol=3e-13
    )


@pytest.mark.parametrize("gauge", [1e-200, 1e200, -1e-200, -1e200])
def test_compensating_coefficient_gauges_preserve_the_filter(gauge):
    from mavats.marma import filter_marma

    data, ar, ma, intercept, covariance = _design()
    expected = filter_marma(data, _parameters(ar, ma, intercept, covariance))
    ar_scaled = [(a * gauge, b / gauge) for a, b in ar]
    ma_scaled = [(a / gauge, b * gauge) for a, b in ma]
    result = filter_marma(
        data, _parameters(ar_scaled, ma_scaled, intercept, covariance)
    )
    np.testing.assert_allclose(result.residuals, expected.residuals, atol=3e-13)
    np.testing.assert_allclose(result.forecast(4), expected.forecast(4), atol=3e-13)
    assert result.log_likelihood == pytest.approx(expected.log_likelihood, rel=2e-13)


@pytest.mark.parametrize("scale", [1e-100, 1e100])
def test_physical_data_units_preserve_gaussian_filter_and_covariance(scale):
    from mavats.marma import filter_marma

    data, ar, ma, intercept, covariance = _design()
    expected = filter_marma(data, _parameters(ar, ma, intercept, covariance))
    scaled_covariance = ((covariance[0] * scale) * scale, covariance[1])
    result = filter_marma(
        data * scale,
        _parameters(ar, ma, intercept * scale, scaled_covariance),
    )
    np.testing.assert_allclose(result.residuals / scale, expected.residuals, atol=3e-12)
    np.testing.assert_allclose(
        result.forecast(4) / scale, expected.forecast(4), atol=3e-12
    )
    np.testing.assert_allclose(
        (result.forecast_covariance(4) / scale) / scale,
        expected.forecast_covariance(4),
        rtol=3e-12,
        atol=3e-12,
    )
    sample_size = len(data) - max(len(ar), len(ma))
    expected_likelihood = expected.log_likelihood - sample_size * 6 * np.log(scale)
    assert result.log_likelihood == pytest.approx(expected_likelihood, rel=3e-13)


@pytest.mark.parametrize("method", ["ls", "mle"])
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_recursive_score_against_independent_dense_coordinate_differences(
    method, fit_intercept
):
    from mavats.marma import _objective_gradient, _ParameterChart

    data, ar, ma, intercept, covariance = _design()
    if not fit_intercept:
        intercept = np.zeros_like(intercept)
    initial = _parameters(ar, ma, intercept, covariance)
    chart = _ParameterChart(
        initial, fit_intercept=fit_intercept, covariance=method == "mle"
    )
    point = chart.pack(initial)
    count = len(data) - max(len(ar), len(ma))

    def dense_loss(vector):
        decoded = chart.decode(vector)
        a, b, l, r, constant = decoded[:5]
        covariance = (
            np.kron(decoded[6] @ decoded[6].T, decoded[5] @ decoded[5].T)
            if method == "mle"
            else np.eye(data.shape[1] * data.shape[2])
        )
        errors, objective = _dense_conditional(
            data,
            _operators(zip(a, b)),
            _operators(zip(l, r)),
            constant,
            covariance,
        )
        return objective / count if method == "mle" else np.sum(errors**2) / count

    loss, gradient = _objective_gradient(point, data, chart, method=method)
    assert loss == pytest.approx(dense_loss(point), rel=3e-13)
    expected_gradient = _central_gradient(dense_loss, point)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=2e-6, atol=2e-7)


def test_second_order_root_signs_and_structural_nonidentification():
    from mavats.marma import marma_diagnostics

    ar = [(np.ones((1, 1)), np.array([[value]])) for value in [1.2, -0.4]]
    ma = [(np.ones((1, 1)), np.array([[value]])) for value in [1.1, -0.35]]
    parameters = _parameters(
        ar, ma, np.zeros((1, 1)), (np.ones((1, 1)), np.ones((1, 1)))
    )
    diagnostics = marma_diagnostics(parameters)
    assert diagnostics.is_stable and diagnostics.is_invertible
    expected_ar = max(abs(np.roots([1.0, -1.2, 0.4])))
    expected_ma = max(abs(np.roots([1.0, -1.1, 0.35])))
    assert diagnostics.ar_spectral_radius == pytest.approx(expected_ar)
    assert diagnostics.ma_spectral_radius == pytest.approx(expected_ma)
    # Cancelling stable AR/MA polynomials do not give an identified model.
    cancelled = _parameters(
        ar, ar, np.zeros((1, 1)), (np.ones((1, 1)), np.ones((1, 1)))
    )
    assert not marma_diagnostics(cancelled).structural_identification_certified


def test_tiny_representable_conditional_mean_is_not_residual_subtraction():
    from mavats.marma import MARMAParameters, filter_marma

    parameters = MARMAParameters([[[1.0]]], [[[1e-200]]], [], [], [[0.0]])
    result = filter_marma(np.array([1.0, 2.0, 3.0]).reshape(-1, 1, 1), parameters)
    np.testing.assert_allclose(
        result.fitted_values[1:].ravel() / 1e-200, [1.0, 2.0], rtol=1e-14
    )


def test_white_noise_forecast_ignores_irrelevant_history_scale():
    from mavats.marma import MARMAParameters, filter_marma

    parameters = MARMAParameters([], [], [], [], [[1e-200]])
    result = filter_marma(np.full((2, 1, 1), 1e308), parameters)
    np.testing.assert_array_equal(result.forecast(2), np.full((2, 1, 1), 1e-200))
    np.testing.assert_array_equal(result.fitted_values, np.full((2, 1, 1), 1e-200))


def test_positive_factored_variance_cannot_silently_materialize_as_zero():
    from mavats.marma import MARMAParameters, filter_marma

    parameters = MARMAParameters([], [], [], [], [[0.0]], [[1e-200]], [[1e-200]])
    result = filter_marma(np.array([1e-200, -1e-200]).reshape(-1, 1, 1), parameters)
    assert np.isfinite(result.log_likelihood)
    with pytest.raises((FloatingPointError, ValueError), match="covariance|variance"):
        result.forecast_covariance(1)


@pytest.mark.parametrize("method", ["ls", "mle"])
def test_overflowing_supplied_start_is_rejected_not_reported_converged(method):
    from mavats.marma import MARMAParameters, fit_marma

    parameters = MARMAParameters([], [], [[[1.0]]], [[[1e308]]], [[0.0]])
    data = np.arange(1.0, 9.0).reshape(-1, 1, 1)
    with pytest.raises(ValueError, match="all MARMA starts failed"):
        fit_marma(
            data,
            ar_order=0,
            ma_order=1,
            method=method,
            intercept=False,
            initial=parameters,
            n_starts=1,
            max_iter=3,
            enforce_admissibility=False,
        )


def test_near_unit_ma_filter_preserves_unclipped_recursion():
    from mavats.marma import MARMAParameters, filter_marma, marma_diagnostics

    theta = 1.0 - 1e-10
    parameters = MARMAParameters([], [], [[[1.0]]], [[[theta]]], [[0.0]])
    data = np.ones((500, 1, 1))
    result = filter_marma(data, parameters)
    # Stable evaluation of sum(theta**j), independently of innovation recursion.
    powers = np.arange(1.0, len(data))
    expected = -np.expm1(powers * np.log(theta)) / (1 - theta)
    np.testing.assert_allclose(result.residuals[1:].ravel(), expected, rtol=3e-13)
    assert not marma_diagnostics(parameters).is_invertible
    assert marma_diagnostics(parameters, tolerance=0).is_invertible


def test_wrong_shape_initial_is_rejected_before_dense_root_allocation(monkeypatch):
    import mavats.marma as marma

    data, ar, ma, intercept, covariance = _design()
    initial = _parameters(ar, ma, intercept, covariance)

    def forbidden_roots(*args, **kwargs):
        raise AssertionError("a mismatched initial model must not allocate companions")

    monkeypatch.setattr(marma, "_radii", forbidden_roots)
    with pytest.raises(ValueError, match="initial parameters.*dimensions"):
        marma.fit_marma(data[:, :, :2], 2, 2, initial=initial, n_starts=1)


def test_unresolved_stability_margin_cannot_admit_a_unit_root():
    from mavats.marma import MARMAParameters, fit_marma

    initial = MARMAParameters([[[1.0]]], [[[1.0]]], [], [], [[0.0]])
    with pytest.raises(ValueError, match="margin|stationarity"):
        fit_marma(
            np.arange(1.0, 10.0).reshape(-1, 1, 1),
            1,
            0,
            initial=initial,
            intercept=False,
            n_starts=1,
            max_iter=1,
            stability_margin=1e-20,
        )


@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_supplied_gaussian_start_preserves_its_first_conditional_objective(scale):
    from mavats.marma import fit_marma

    data, ar, ma, intercept, covariance = _design()
    data = data * scale
    intercept = intercept * scale
    covariance = ((covariance[0] * scale) * scale, covariance[1])
    initial = _parameters(ar, ma, intercept, covariance)
    _, expected = _dense_conditional(
        data,
        _operators(ar),
        _operators(ma),
        intercept,
        np.kron(*covariance[::-1]),
    )
    result = fit_marma(
        data, 2, 2, method="mle", initial=initial, n_starts=1, max_iter=1
    )
    count = len(data) - 2
    expected_work = expected / count - data.shape[1] * data.shape[2] * np.log(
        result.data_scale
    )
    run = result.runs[0]
    assert run.objective_history[0] == pytest.approx(expected_work, rel=2e-12)
    assert run.initialization_converged is None
    assert run.feasible
    assert not result.diagnostics.structural_identification_certified
