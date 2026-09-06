"""Independent algebraic and behavioral checks for Samadi--De Alwis EMAR."""

import numpy as np
import pytest

from mavats import fit_envelope_mar, fit_mar
from mavats._envelope_optimization import _fit_envelope, _profile_value_gradient, _spd
from mavats.envelope import _block_update, _moments, _nll, _right_whiten


def _positive(rng, dim):
    a = rng.normal(size=(dim, dim))
    return a @ a.T + np.eye(dim)


def _dense_nll(residual, row, column):
    covariance = np.kron(column, row)
    vectors = residual.transpose(0, 2, 1).reshape(len(residual), -1)
    return 0.5 * (
        len(residual)
        * (np.linalg.slogdet(covariance)[1] + covariance.shape[0] * np.log(2 * np.pi))
        + np.sum(vectors * np.linalg.solve(covariance, vectors.T).T)
    )


def _series(seed=42, count=180, order=1):
    rng = np.random.default_rng(seed)
    m, n = 3, 4
    r = np.linalg.qr(rng.normal(size=(m, m)))[0]
    c = np.linalg.qr(rng.normal(size=(n, n)))[0]
    left = np.stack([r[:, :2] @ rng.normal(size=(2, m)) for _ in range(order)])
    right = np.stack([c[:, :2] @ rng.normal(size=(2, n)) for _ in range(order)])
    contraction = sum(
        np.linalg.norm(a, 2) * np.linalg.norm(b, 2) for a, b in zip(left, right)
    )
    right *= 0.6 / contraction
    lr = r @ np.diag(np.sqrt([0.5, 1.0, 3.0]))
    lc = c @ np.diag(np.sqrt([0.4, 0.8, 2.0, 3.0]))
    values = np.zeros((count + 150, m, n))
    for t in range(order, len(values)):
        values[t] = 0.1 + lr @ rng.normal(size=(m, n)) @ lc.T
        for lag, (a, b) in enumerate(zip(left, right), 1):
            values[t] += a @ values[t - lag] @ b.T
    return values[-count:]


@pytest.mark.parametrize("m,n", [(2, 3), (4, 2)])
def test_cholesky_whitening_and_dense_likelihood(m, n):
    rng = np.random.default_rng(2)
    residual = rng.normal(size=(21, m, n))
    row, column = _positive(rng, m), _positive(rng, n)
    white = _right_whiten(residual, column)
    expected = sum(e @ np.linalg.solve(column, e.T) for e in residual)
    np.testing.assert_allclose(
        np.einsum("tij,tkj->ik", white, white), expected, atol=1e-12
    )
    np.testing.assert_allclose(
        _nll(residual, row, column), _dense_nll(residual, row, column), rtol=1e-13
    )


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_profile_tangent_gradient_and_rotation_invariance(rank):
    rng = np.random.default_rng(12)
    m = 4
    residual, response = _positive(rng, m), _positive(rng, m)
    basis = np.linalg.qr(rng.normal(size=(m, rank)))[0]
    factor = _spd(response, "response")[1]
    value, gradient = _profile_value_gradient(basis, residual, factor)
    inv = np.linalg.inv(response)  # independent small dense oracle
    expected = (
        np.linalg.slogdet(basis.T @ residual @ basis)[1]
        + np.linalg.slogdet(basis.T @ inv @ basis)[1]
    )
    np.testing.assert_allclose(value, expected, atol=1e-13)
    np.testing.assert_allclose(basis.T @ gradient, 0, atol=1e-13)
    direction = rng.normal(size=basis.shape)
    direction -= basis @ (basis.T @ direction)

    def objective(step):
        u = np.linalg.qr(basis + step * direction)[0]
        return (
            np.linalg.slogdet(u.T @ residual @ u)[1]
            + np.linalg.slogdet(u.T @ inv @ u)[1]
        )

    derivative = (objective(1e-5) - objective(-1e-5)) / 2e-5
    np.testing.assert_allclose(
        derivative, np.sum(gradient * direction), rtol=2e-7, atol=2e-8
    )
    rotation = np.linalg.qr(rng.normal(size=(rank, rank)))[0]
    other_value, other_gradient = _profile_value_gradient(
        basis @ rotation, residual, factor
    )
    np.testing.assert_allclose(other_value, value, atol=1e-13)
    np.testing.assert_allclose(other_gradient, gradient @ rotation, atol=1e-12)


def test_grassmann_against_exhaustive_angle_oracle_and_scaling():
    residual = np.array([[0.5, 0.2], [0.2, 1.3]])
    response = np.array([[2.0, -0.6], [-0.6, 1.5]])
    angles = np.linspace(0, np.pi, 20001)
    u = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    objectives = np.log(np.einsum("ti,ij,tj->t", u, residual, u)) + np.log(
        np.einsum("ti,ij,tj->t", u, np.linalg.inv(response), u)
    )
    results = []
    for scale1, scale2 in [(1, 1), (1e-180, 1e180)]:
        basis, diagnostic = _fit_envelope(
            residual * scale1,
            response * scale2,
            1,
            initial=None,
            starts=4,
            rng=np.random.default_rng(2),
            max_iter=300,
            tol=1e-6,
        )
        value, _ = _profile_value_gradient(
            basis, residual, _spd(response, "response")[1]
        )
        assert value <= min(objectives) + 1e-8
        assert diagnostic["converged"]
        for run in diagnostic["runs"]:
            assert np.max(np.diff(run["objective_history"]), initial=0) <= 1e-12
        results.append(basis @ basis.T)
    np.testing.assert_allclose(*results, atol=1e-6)


@pytest.mark.parametrize("order", [1, 2])
def test_joint_lag_profile_matches_independent_dense_weighted_regression(order):
    rng = np.random.default_rng(22)
    count, m, n, rank = 31, 3, 4, 2
    y = rng.normal(size=(count, m, n))
    lags = [rng.normal(size=y.shape) for _ in range(order)]
    b = rng.normal(size=(order, n, n))
    col = _positive(rng, n)
    a, residual, response = _moments(y, lags, b, col)
    f = np.concatenate([z @ coef.T for z, coef in zip(lags, b)], axis=1)
    weight = np.linalg.inv(col)
    gram = sum(v @ weight @ v.T for v in f)
    cross = sum(target @ weight @ v.T for target, v in zip(y, f))
    expected = np.linalg.solve(gram, cross.T).T
    np.testing.assert_allclose(a, expected, atol=2e-13)
    error = y - a @ f
    np.testing.assert_allclose(
        residual, sum(e @ weight @ e.T for e in error) / (count * n), atol=1e-13
    )
    np.testing.assert_allclose(
        response, sum(e @ weight @ e.T for e in y) / (count * n), atol=1e-13
    )
    fitted_a, row, basis, diagnostic = _block_update(
        y, lags, b, col, rank, initial=None, starts=3, rng=rng, max_iter=200, tol=1e-6
    )
    # Vectorized weighted regression with coefficients eta, holding the fitted
    # envelope fixed. Independent Kronecker design/Cholesky likelihood oracle.
    designs = np.concatenate([np.kron(v.T, basis) for v in f], axis=0)
    values = y.transpose(0, 2, 1).reshape(-1)
    chol = np.linalg.cholesky(np.kron(col, row))
    wd = np.concatenate([np.linalg.solve(chol, d) for d in np.split(designs, count)])
    wy = np.concatenate([np.linalg.solve(chol, v) for v in np.split(values, count)])
    eta = np.linalg.lstsq(wd, wy, rcond=None)[0].reshape(rank, order * m, order="F")
    np.testing.assert_allclose(fitted_a, basis @ eta, atol=2e-12)
    complement = np.linalg.qr(basis, mode="complete")[0][:, rank:]
    profile = (
        np.linalg.slogdet(basis.T @ residual @ basis)[1]
        + np.linalg.slogdet(complement.T @ response @ complement)[1]
    )
    expected_nll = (
        0.5
        * count
        * (
            m * np.linalg.slogdet(col)[1]
            + n * profile
            + m * n * (1 + np.log(2 * np.pi))
        )
    )
    np.testing.assert_allclose(
        _dense_nll(y - fitted_a @ f, row, col), expected_nll, rtol=1e-13
    )


@pytest.mark.parametrize("order", [1, 2])
def test_fit_reducing_spaces_intercept_forecast_and_diagnostics(order):
    data = _series(order=order)
    before = data.copy()
    result = fit_envelope_mar(data, (2, 2), order=order, max_iter=40)
    np.testing.assert_array_equal(data, before)
    assert result.converged, (result.stop_reason, result.optimization_history[-1])
    assert result.inner_converged and result.likelihood_converged
    assert len(result.objective_history) == result.n_iter
    assert np.max(np.diff(result.objective_history)) < 1e-7
    for coefficients, covariance, basis in [
        (result.left, result.row_covariance, result.row_envelope),
        (result.right, result.column_covariance, result.column_envelope),
    ]:
        p = basis @ basis.T
        np.testing.assert_allclose(p @ coefficients, coefficients, atol=1e-12)
        np.testing.assert_allclose(p @ covariance @ (np.eye(len(p)) - p), 0, atol=1e-12)
        assert np.linalg.eigvalsh(covariance)[0] > 0
    fitted = np.broadcast_to(result.intercept, result.fitted_values.shape).copy()
    for lag, (a, b) in enumerate(zip(result.left, result.right), 1):
        fitted += a @ data[order - lag : len(data) - lag] @ b.T
    np.testing.assert_allclose(result.fitted_values, fitted, atol=1e-12)
    np.testing.assert_allclose(result.residuals.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(
        result.log_likelihood,
        -_dense_nll(result.residuals, result.row_covariance, result.column_covariance),
        rtol=1e-13,
    )
    path = list(data[-order:].copy())
    expected = []
    for _ in range(3):
        next_value = result.intercept.copy()
        for lag, (a, b) in enumerate(zip(result.left, result.right), 1):
            next_value += a @ path[-lag] @ b.T
        path.append(next_value)
        expected.append(next_value)
    np.testing.assert_allclose(result.forecast(3), expected, atol=1e-12)
    np.testing.assert_allclose(
        result.forecast(3, history=data[-order:]), expected, atol=1e-12
    )
    # Passing an alternative history must use only its most recent p entries.
    alternate = data.copy()
    alternate[:-order] = 10000
    np.testing.assert_allclose(
        result.forecast(3, history=alternate), expected, atol=1e-12
    )


def test_full_envelope_boundary_matches_unrestricted_mle():
    data = _series(count=300)
    unrestricted = fit_mar(
        data,
        method="mle",
        fit_intercept=True,
        covariance_floor=0,
        max_iter=400,
        tol=1e-11,
    )
    result = fit_envelope_mar(
        data,
        (3, 4),
        initial=(unrestricted.left, unrestricted.right),
        initial_covariance=(
            unrestricted.row_covariance,
            unrestricted.column_covariance,
        ),
        tol=1e-11,
    )
    assert result.converged
    np.testing.assert_allclose(
        result.coefficients, unrestricted.coefficients, rtol=1e-4, atol=2e-6
    )
    np.testing.assert_allclose(
        result.log_likelihood, unrestricted.log_likelihood, atol=1e-6
    )


@pytest.mark.parametrize("scale", [1e-250, 1e250])
def test_scale_equivariance_and_finite_likelihood(scale):
    data = _series(count=120)
    base = fit_envelope_mar(data, (2, 2), max_iter=15, inner_tol=2e-5)
    scaled = fit_envelope_mar(data * scale, (2, 2), max_iter=15, inner_tol=2e-5)
    np.testing.assert_allclose(scaled.coefficients, base.coefficients, atol=3e-5)
    np.testing.assert_allclose(scaled.forecast(2) / scale, base.forecast(2), atol=1e-4)
    np.testing.assert_allclose(
        scaled.row_covariance / scale, base.row_covariance, atol=1e-4
    )
    np.testing.assert_allclose(
        scaled.column_covariance / scale, base.column_covariance, atol=1e-4
    )
    np.testing.assert_allclose(
        scaled.log_likelihood,
        base.log_likelihood - base.residuals.size * np.log(scale),
        atol=1e-6,
    )


def test_nonconvergence_is_explicit_and_rng_is_local():
    np.random.seed(65)
    state = np.random.get_state()
    result = fit_envelope_mar(_series(count=80), (2, 2), max_iter=1, inner_max_iter=1)
    after = np.random.get_state()
    np.testing.assert_array_equal(after[1], state[1])
    assert after[2:] == state[2:]
    assert not result.converged
    assert not result.likelihood_converged
    assert result.stop_reason == "max_iter"
    assert len(result.optimization_history) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"envelope_dims": None},
        {"envelope_dims": (0, 2)},
        {"envelope_dims": (4, 2)},
        {"envelope_dims": (True, 2)},
        {"order": 0},
        {"order": 1.0},
        {"tol": -1},
        {"tol": np.nan},
        {"inner_tol": True},
        {"inner_max_iter": 0},
        {"max_iter": 0},
        {"inner_starts": 0},
        {"warmup_max_iter": 0},
        {"fit_intercept": 1},
        {"max_dense_elements": 0},
        {"initial": (np.eye(2), np.eye(4))},
        {"initial": (np.eye(3) * 1j, np.eye(4))},
        {"initial_covariance": (np.zeros((3, 3)), np.eye(4))},
        {"initial_covariance": (np.eye(3) * 1j, np.eye(4))},
        {"initial_covariance": (np.triu(np.ones((3, 3))), np.eye(4))},
        {"initial_envelopes": (np.ones((3, 2)), np.eye(4)[:, :2])},
    ],
)
def test_input_validation(kwargs):
    options = dict(envelope_dims=(2, 2), max_iter=1)
    options.update(kwargs)
    with pytest.raises(ValueError):
        fit_envelope_mar(_series(count=40), **options)


@pytest.mark.parametrize(
    "data",
    [
        np.zeros((20, 3, 4)),
        np.ones((20, 3, 4)),
        np.full((20, 3, 4), np.nan),
        np.ones((20, 3, 4), dtype=complex),
    ],
)
def test_degenerate_and_invalid_data_fail(data):
    with pytest.raises(ValueError):
        fit_envelope_mar(data, (2, 2))


def test_saturated_design_and_zero_initial_fail_without_hidden_flooring():
    data = np.random.default_rng(12).normal(size=(4, 3, 2))
    with pytest.raises(ValueError, match="singular|rank deficient"):
        fit_envelope_mar(
            data,
            (2, 1),
            order=2,
            initial=(
                np.repeat(np.eye(3)[None], 2, axis=0),
                np.repeat(np.eye(2)[None], 2, axis=0),
            ),
        )
    with pytest.raises(ValueError, match="rank deficient"):
        fit_envelope_mar(_series(), (2, 2), initial=(np.eye(3), np.zeros((4, 4))))


def test_dense_guard_does_not_prevent_matrix_forecast():
    result = fit_envelope_mar(
        _series(order=2), (2, 2), order=2, max_iter=2, max_dense_elements=1
    )
    for name in ("coefficients", "spectral_radius", "is_stable"):
        with pytest.raises(ValueError, match="max_dense_elements"):
            getattr(result, name)
    with pytest.raises(ValueError, match="max_dense_elements"):
        result.residual_covariance()
    assert result.forecast(2).shape == (2, 3, 4)


@pytest.mark.parametrize("dims", [(1, 4), (3, 1), (1, 1)])
def test_one_full_space_is_still_free_coefficient_model_and_no_intercept(dims):
    data = _series(count=150)
    bases = (np.eye(3)[:, : dims[0]], np.eye(4)[:, : dims[1]])
    result = fit_envelope_mar(
        data,
        dims,
        fit_intercept=False,
        initial_envelopes=bases,
        initial=(np.eye(3), np.eye(4)),
        max_iter=15,
    )
    np.testing.assert_array_equal(result.intercept, np.zeros((3, 4)))
    assert result.warmup_converged is None
    assert len(result.optimization_history[0]["row"]["runs"]) == (
        1 if dims[0] == 3 else 4
    )
    assert len(result.optimization_history[0]["column"]["runs"]) == (
        1 if dims[1] == 4 else 4
    )
    assert not np.allclose(result.B, np.eye(4))
    assert not np.allclose(result.A, np.eye(3))
    np.testing.assert_allclose(np.linalg.norm(result.A), 1, atol=1e-13)
    assert result.A.flat[np.argmax(np.abs(result.A))] > 0


def test_numerically_increasing_sweep_rolls_back_and_is_not_convergence(monkeypatch):
    import mavats.envelope as module

    data = _series(count=80)
    first = fit_envelope_mar(data, (2, 2), max_iter=1)
    original_nll = module._nll
    calls = []

    def increasing(*args):
        calls.append(1)
        return original_nll(*args) + (10000 if len(calls) > 1 else 0)

    monkeypatch.setattr(module, "_nll", increasing)
    fit = fit_envelope_mar(data, (2, 2), max_iter=5)
    assert fit.stop_reason == "likelihood_increase"
    assert not fit.converged
    assert fit.n_iter == 1
    assert len(fit.optimization_history) == 2
    assert not fit.optimization_history[-1]["accepted"]
    np.testing.assert_allclose(fit.coefficients, first.coefficients)
    np.testing.assert_allclose(fit.objective_history, first.objective_history)
    np.testing.assert_allclose(fit.fitted_values, first.fitted_values)


def test_warmup_output_spaces_are_extra_starts_not_fixed_estimates():
    data = _series(count=130)
    warm = fit_mar(
        data, method="mle", fit_intercept=True, max_iter=100, covariance_floor=0
    )
    bases = tuple(
        np.linalg.svd(np.concatenate(coef, axis=1), full_matrices=False)[0][:, :2]
        for coef in (warm.left, warm.right)
    )
    automatic = fit_envelope_mar(data, (2, 2), max_iter=10)
    explicit = fit_envelope_mar(data, (2, 2), max_iter=10, initial_envelopes=bases)
    for axis in ("row", "column"):
        assert len(automatic.optimization_history[0][axis]["runs"]) == 4
    np.testing.assert_allclose(
        automatic.log_likelihood, explicit.log_likelihood, atol=1e-8
    )
    np.testing.assert_allclose(automatic.coefficients, explicit.coefficients, atol=1e-7)
    assert (
        np.linalg.norm(
            automatic.row_envelope @ automatic.row_envelope.T - bases[0] @ bases[0].T
        )
        > 1e-4
    )
