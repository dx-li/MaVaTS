"""CMAR checks against independently assembled dense regressions and likelihoods."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mavats.cointegration import _block_update, cmar_i1_diagnostics, fit_cmar


def planted(n=400, *, p=3, q=2, ranks=(1, 1), seed=32, lags=1):
    """Direct matrix recurrence, not any fitting/forecast/simulation kernel."""
    rng = np.random.default_rng(seed)
    u = np.linalg.qr(rng.normal(size=(p, ranks[0])))[0]
    v = np.linalg.qr(rng.normal(size=(q, ranks[1])))[0]
    a1, a2 = -0.65 * u @ u.T, v @ v.T
    b1 = np.repeat((0.2 * np.eye(p))[None], lags, axis=0)
    b2 = np.repeat((0.4 / max(lags, 1) * np.eye(q))[None], lags, axis=0)
    constant = rng.normal(size=(p, q)) * 0.015
    row = 0.35 ** np.abs(np.arange(p)[:, None] - np.arange(p))
    column = 0.25 ** np.abs(np.arange(q)[:, None] - np.arange(q))
    x = np.zeros((n + 50, p, q))
    for t in range(lags + 1, len(x)):
        delta = a1 @ x[t - 1] @ a2.T + constant
        for j in range(1, lags + 1):
            delta += b1[j - 1] @ (x[t - j] - x[t - j - 1]) @ b2[j - 1].T
        error = (
            np.linalg.cholesky(row)
            @ rng.normal(size=(p, q))
            @ np.linalg.cholesky(column).T
        )
        x[t] = x[t - 1] + delta + 0.3 * error
    return (
        x[50:],
        dict(A1=a1, A2=a2, short_run_left=b1, short_run_right=b2, intercept=constant),
        (u, v),
    )


def block_oracle(Y, X, differences, opposite, short, rank, intercept, covariance):
    """Moment-based RRR with symmetric whitening, unlike implementation SVD/Cholesky."""
    n, p, q = Y.shape
    if covariance is None:
        white = np.eye(q)
    else:
        val, vec = np.linalg.eigh(covariance)
        white = (vec / np.sqrt(val)) @ vec.T
    ys, xs, zs = [], [], []
    for t in range(n):
        for j in range(q):
            ys.append(Y[t] @ white[:, j])
            xs.append(X[t] @ opposite.T @ white[:, j])
            z = [d[t] @ b.T @ white[:, j] for d, b in zip(differences, short)]
            if intercept:
                z.append(white[:, j])
            zs.append(np.concatenate(z) if z else np.empty(0))
    y, x, z = np.array(ys).T, np.array(xs).T, np.array(zs).T
    if z.shape[0]:
        projection = z.T @ np.linalg.solve(z @ z.T, z)
        yr, xr = y - y @ projection, x - x @ projection
    else:
        yr, xr = y, x
    xx, yx = xr @ xr.T, yr @ xr.T
    unrestricted = yx @ np.linalg.inv(xx)
    if covariance is None:
        root, invroot = np.eye(p), np.eye(p)
    else:
        errors = yr - unrestricted @ xr
        val, vec = np.linalg.eigh(errors @ errors.T / (n * q))
        root = (vec * np.sqrt(val)) @ vec.T
        invroot = (vec / np.sqrt(val)) @ vec.T
    explained = invroot @ yx @ np.linalg.solve(xx, yx.T) @ invroot
    _, vec = np.linalg.eigh(explained)
    u = vec[:, -rank:]
    a = root @ u @ u.T @ invroot @ unrestricted
    psi = (y - a @ x) @ z.T @ np.linalg.inv(z @ z.T) if z.shape[0] else np.empty((p, 0))
    residual = y - a @ x - psi @ z
    sigma = residual @ residual.T / (n * q)
    return a, psi, sigma


@pytest.mark.parametrize("method", ["ls", "mle"])
@pytest.mark.parametrize(
    "intercept,lags", [(False, 0), (True, 0), (False, 2), (True, 2)]
)
def test_each_block_matches_independent_profiled_rrr(method, intercept, lags):
    rng = np.random.default_rng(44)
    n, p, q = 45, 3, 4
    y, x = rng.normal(size=(2, n, p, q))
    differences = [rng.normal(size=(n, p, q)) for _ in range(lags)]
    opposite = np.diag([1.0, 0.8, 0, 0])
    short = np.array([rng.normal(size=(q, q)) + np.eye(q) for _ in range(lags)])
    covariance = (
        None if method == "ls" else 0.4 ** np.abs(np.arange(q)[:, None] - np.arange(q))
    )
    a, b, d, sigma, _ = _block_update(
        y, x, differences, opposite, short, 2, intercept, covariance, 1e-12
    )
    expected_a, expected_psi, expected_sigma = block_oracle(
        y, x, differences, opposite, short, 2, intercept, covariance
    )
    assert_allclose(a, expected_a, atol=2e-12, rtol=2e-11)
    pieces = [matrix for matrix in b] + ([d] if intercept else [])
    if pieces:
        assert_allclose(
            np.concatenate(pieces, axis=1), expected_psi, atol=2e-12, rtol=2e-11
        )
    if method == "mle":
        assert_allclose(sigma, expected_sigma, atol=2e-12)


@pytest.mark.parametrize("method", ["ls", "mle"])
@pytest.mark.parametrize("lags", [0, 1, 2])
def test_dense_error_correction_equations_likelihood_and_forecast(method, lags):
    x, _, _ = planted(n=220, lags=lags)
    fit = fit_cmar(x[:200], (1, 1), method=method, difference_lags=lags, n_starts=1)
    flat = x[:200].transpose(0, 2, 1).reshape(200, -1)
    delta = np.diff(flat, axis=0)
    pi = np.kron(fit.A2, fit.A1)
    gammas = [
        np.kron(b2, b1) for b1, b2 in zip(fit.short_run_left, fit.short_run_right)
    ]
    expected = []
    for t in range(lags + 1, 200):
        yhat = pi @ flat[t - 1] + fit.intercept.ravel(order="F")
        for j, gamma in enumerate(gammas, start=1):
            yhat += gamma @ delta[t - j - 1]
        expected.append(yhat)
    expected = np.array(expected)
    assert_allclose(
        fit.fitted_differences.transpose(0, 2, 1).reshape(len(expected), -1),
        expected,
        atol=2e-12,
    )
    residual = delta[lags:] - expected
    assert_allclose(
        fit.residuals.transpose(0, 2, 1).reshape(len(expected), -1),
        residual,
        atol=2e-12,
    )
    assert_allclose(fit.long_run_operator, pi)
    assert fit.cointegration_rank == 1
    assert_allclose(fit.A1, fit.alpha1 @ fit.beta1.T, atol=1e-14)
    assert_allclose(fit.A2, fit.alpha2 @ fit.beta2.T, atol=1e-14)
    assert_allclose(np.linalg.norm(fit.A1), 1)
    for b in fit.short_run_left:
        assert_allclose(np.linalg.norm(b), 1)
    assert_allclose(
        fit.cointegrating_scores(x[:3]).ravel(),
        (
            fit.cointegrating_vectors.T @ x[:3].transpose(0, 2, 1).reshape(3, -1).T
        ).ravel(),
    )
    if method == "mle":
        covariance = np.kron(fit.column_covariance, fit.row_covariance)
        sign, logdet = np.linalg.slogdet(covariance)
        assert sign == 1
        nll = 0.5 * (
            len(pi) * np.log(2 * np.pi)
            + logdet
            + np.einsum("ti,it->t", residual, np.linalg.solve(covariance, residual.T))
        )
        assert_allclose(fit.log_likelihood, -nll.sum(), atol=1e-9)
        assert_allclose(np.trace(fit.row_covariance), 3)
    else:
        assert fit.log_likelihood is None
        assert_allclose(
            fit.objective, np.mean(np.sum((residual / fit.data_scale) ** 2, axis=1))
        )
    # Independent dense level-VAR recursion; no future test values enter.
    history = list(flat[-lags - 1 :].copy())
    future = []
    for _ in range(5):
        yhat = sum(
            a @ history[-j] for j, a in enumerate(fit.level_coefficients, start=1)
        ) + fit.intercept.ravel(order="F")
        history.append(yhat)
        future.append(yhat)
    forecast = fit.forecast(5).transpose(0, 2, 1).reshape(5, -1)
    assert_allclose(forecast, future, atol=5e-12)
    assert_allclose(fit.forecast(5, history=x[:200]), fit.forecast(5))
    assert all(np.all(np.diff(run.objective_history) <= 1e-8) for run in fit.runs)


@pytest.mark.parametrize("method", ["ls", "mle"])
def test_vector_limit_is_rank_constrained_vecm_not_stationary_mar(method):
    x, _, _ = planted(n=350, p=3, q=1, lags=1)
    fit = fit_cmar(x, (1, 1), difference_lags=1, method=method, n_starts=1)
    differences = np.diff(x, axis=0)
    # q=1 removes Kronecker restrictions on vector coefficients: one RRR oracle.
    a, psi, _ = block_oracle(
        differences[1:],
        x[1:-1],
        [differences[:-1]],
        np.ones((1, 1)),
        np.ones((1, 1, 1)),
        1,
        True,
        np.ones((1, 1)) if method == "mle" else None,
    )
    assert_allclose(fit.long_run_operator, a, atol=1e-10)
    assert_allclose(fit.short_run_operators[0], psi[:, :3], atol=1e-10)
    assert_allclose(fit.intercept, psi[:, 3:], atol=1e-10)
    assert fit.i1_diagnostics().unit_roots_expected == 2


@pytest.mark.parametrize("method", ["ls", "mle"])
def test_planted_bilinear_cointegration_spaces_recover(method):
    x, true, spaces = planted(n=1000, q=3, ranks=(1, 2), seed=23)
    fit = fit_cmar(x, (1, 2), method=method, difference_lags=1, n_starts=2)
    assert fit.converged
    assert fit.i1_diagnostics().compatible
    assert fit.i1_diagnostics().unit_roots_expected == 7
    for estimated, beta in zip(fit.cointegration_projectors, spaces):
        assert np.linalg.norm(estimated - beta @ beta.T) < 0.12
    expected_pi = np.kron(true["A2"], true["A1"])
    assert (
        np.linalg.norm(fit.long_run_operator - expected_pi)
        / np.linalg.norm(expected_pi)
        < 0.3
    )


def test_i1_diagnostics_detect_i2_explosion_and_nonunit_circle_boundaries():
    regular = cmar_i1_diagnostics(np.diag([-0.5, 0]), np.ones((1, 1)))
    assert regular.compatible and regular.unit_roots_expected == 1
    assert_allclose(regular.long_run_impact, np.diag([0, 1]))
    i2 = cmar_i1_diagnostics(np.array([[0, 1], [0, 0]]), np.ones((1, 1)))
    assert not i2.compatible
    assert i2.unit_roots_observed == 2 and i2.unit_roots_expected == 1
    assert i2.long_run_impact is None
    explosive = cmar_i1_diagnostics(np.diag([0.1, 0]), np.ones((1, 1)))
    assert not explosive.compatible and explosive.unstable_roots_count == 1
    negative_unit = cmar_i1_diagnostics(np.diag([-2.0, 0]), np.ones((1, 1)))
    assert not negative_unit.compatible and negative_unit.boundary_roots_count == 1
    stationary = cmar_i1_diagnostics(-0.5 * np.eye(2), np.ones((1, 1)))
    assert not stationary.compatible and stationary.unit_roots_expected == 0
    random_walk = cmar_i1_diagnostics(np.zeros((2, 2)), np.ones((1, 1)))
    assert random_walk.compatible and random_walk.long_run_rank == 0


def test_full_complements_and_long_run_impact_not_mode_complement_products():
    x, true, _ = planted(n=20, q=3, ranks=(1, 2))
    result = cmar_i1_diagnostics(
        true["A1"], true["A2"], true["short_run_left"], true["short_run_right"]
    )
    assert result.compatible
    assert len(result.complement_singular_values) == 9 - 2
    pi = np.kron(true["A2"], true["A1"])
    gamma = np.eye(9) - np.kron(true["short_run_right"][0], true["short_run_left"][0])
    impact = result.long_run_impact
    assert_allclose(pi @ impact, 0, atol=1e-14)
    assert_allclose(impact @ pi, 0, atol=1e-14)
    assert_allclose(impact @ gamma @ impact, impact, atol=2e-14)


@pytest.mark.parametrize("method", ["ls", "mle"])
@pytest.mark.parametrize("scale", [1e-150, -1e-150, 1e150, -1e150])
def test_global_scale_equivariance(method, scale):
    x, _, _ = planted(n=150)
    kwargs = dict(ranks=(1, 1), difference_lags=1, method=method, n_starts=1)
    fit, scaled = fit_cmar(x, **kwargs), fit_cmar(x * scale, **kwargs)
    assert_allclose(
        fit.long_run_operator, scaled.long_run_operator, rtol=1e-7, atol=1e-9
    )
    assert_allclose(
        fit.short_run_operators, scaled.short_run_operators, rtol=1e-7, atol=1e-9
    )
    assert_allclose(scaled.residuals / scale, fit.residuals, atol=1e-7)
    assert_allclose(scaled.forecast(3) / scale, fit.forecast(3), atol=1e-7)
    assert_allclose(
        scaled.cointegrating_scores(x * scale) / scale,
        fit.cointegrating_scores(x),
        atol=1e-7,
    )
    if method == "mle":
        assert_allclose(
            scaled.log_likelihood - fit.log_likelihood,
            -len(fit.residuals) * 6 * np.log(abs(scale)),
            rtol=1e-12,
        )


def test_multistart_seed_locality_best_objective_and_input_copying():
    x, true, _ = planted(n=170)
    original = x.copy()
    rng = np.random.default_rng(84)
    before = deepcopy(rng.bit_generator.state)
    fit = fit_cmar(
        x, (1, 1), difference_lags=1, initial=true, n_starts=3, random_state=rng
    )
    repeat = fit_cmar(
        x, (1, 1), difference_lags=1, initial=true, n_starts=3, random_state=84
    )
    assert rng.bit_generator.state == before
    assert_allclose(fit.long_run_operator, repeat.long_run_operator)
    assert fit.objective == min(run.objective for run in fit.runs if run.n_iter)
    assert_array_equal(x, original)
    x[:] = 0
    true["A1"][:] = 0
    assert np.linalg.norm(fit.A1) > 0
    assert_array_equal(fit.history, original[-2:])


def test_limited_and_failed_starts_are_retained(monkeypatch):
    x, true, _ = planted(n=150)
    limited = fit_cmar(x, (1, 1), difference_lags=1, max_iter=1, n_starts=2)
    assert not limited.converged
    assert all(run.n_iter == 1 for run in limited.runs)
    # Inject a numerical failure on one start to audit reporting/continuation.
    import mavats.cointegration as module

    block = module._block_update
    calls = [0]

    def fail_first(*args, **kwargs):
        calls[0] += 1
        if calls[0] == 1:
            raise np.linalg.LinAlgError("injected singular block")
        return block(*args, **kwargs)

    monkeypatch.setattr(module, "_block_update", fail_first)
    fit = fit_cmar(
        x, (1, 1), difference_lags=1, initial=true, n_starts=2, random_state=7
    )
    assert fit.runs[0].failure and fit.runs[0].n_iter == 0
    assert fit.selected_start == 1


def test_dense_allocation_guards_and_random_initialization_path():
    x, _, _ = planted(n=120)
    with pytest.raises(ValueError, match="dense initialization"):
        fit_cmar(x, (1, 1), max_dense_dimension=5)
    fit = fit_cmar(x, (1, 1), init="random", max_dense_dimension=5, n_starts=1)
    for name in (
        "long_run_operator",
        "short_run_operators",
        "level_coefficients",
        "cointegrating_vectors",
    ):
        with pytest.raises(ValueError, match="dense representation"):
            getattr(fit, name)
    with pytest.raises(ValueError, match="dense companion"):
        fit.i1_diagnostics()
    assert fit.i1_diagnostics(max_dense_dimension=6).unit_roots_expected == 5
    assert fit.forecast(1).shape == (1, 3, 2)


@pytest.mark.parametrize(
    "option",
    [
        {"ranks": None},
        {"ranks": (0, 1)},
        {"ranks": (3, 2)},
        {"ranks": (4, 1)},
        {"ranks": (1,)},
        {"difference_lags": -1},
        {"difference_lags": True},
        {"intercept": "yes"},
        {"method": "johansen"},
        {"n_starts": 0},
        {"max_iter": 0},
        {"tol": 0},
        {"rcond": 0},
        {"rcond": 1},
        {"init": "pca"},
        {"max_dense_dimension": 0},
        {"initial": []},
        {"initial": {"A1": np.eye(3)}},
        {"initial": {"unknown": 1}},
    ],
)
def test_invalid_configuration(option):
    x, _, _ = planted(n=40)
    kwargs = dict(ranks=(1, 1), n_starts=1)
    kwargs.update(option)
    with pytest.raises(ValueError):
        fit_cmar(x, **kwargs)


@pytest.mark.parametrize(
    "x",
    [
        np.zeros((10, 3, 2)),
        np.ones((10, 3, 2)),
        np.ones((10, 3, 2), dtype=complex),
        np.full((10, 3, 2), np.nan),
        np.ones((2, 3, 2)),
    ],
)
def test_unidentified_or_invalid_observations_rejected(x):
    with pytest.raises(ValueError):
        fit_cmar(x, (1, 1), n_starts=1)


def test_forecast_and_score_input_guards_and_represented_overflow():
    x, _, _ = planted(n=120)
    fit = fit_cmar(x, (1, 1), difference_lags=1, n_starts=1)
    with pytest.raises(ValueError):
        fit.forecast(1, history=x[:1])
    with pytest.raises(ValueError):
        fit.forecast(0)
    with pytest.raises(ValueError):
        fit.cointegrating_scores(np.ones((3, 2, 3)))
    # Force a representable signed cancellation through scaled contractions.
    fit.beta1 = np.array([[1], [1], [-1]]) / np.sqrt(3)
    fit.beta2 = np.array([[1], [1]]) / np.sqrt(2)
    large = np.full((1, 3, 2), 1.3e308)
    assert np.isfinite(fit.cointegrating_scores(large)).all()
    fit.beta1 = np.ones((3, 1)) / np.sqrt(3)
    with pytest.raises(FloatingPointError, match="scores"):
        fit.cointegrating_scores(large)
    fit.A1[:] = 1e200
    with pytest.raises(FloatingPointError, match="forecast"):
        fit.forecast(2)


def test_finite_near_maximum_covariance_is_not_lost_to_symmetrization():
    rng = np.random.default_rng(8)
    x = np.zeros((90, 2, 1))
    for t in range(1, len(x)):
        x[t] = np.diag([0.6, 1]) @ x[t - 1] + rng.normal(size=(2, 1))
    fit = fit_cmar(x, (1, 1), method="mle", intercept=False, n_starts=1)
    scaled = fit_cmar(x * 1e154, (1, 1), method="mle", intercept=False, n_starts=1)
    assert np.isfinite(scaled.column_covariance).all()
    assert_allclose(scaled.column_covariance / 1e308, fit.column_covariance, rtol=1e-12)


@pytest.mark.parametrize("gauge", [1e-200, -1e-200, 1e200, -1e200])
def test_equivalent_extreme_reciprocal_initial_coefficient_gauges(gauge):
    rng = np.random.default_rng(8)
    x = np.zeros((90, 2, 1))
    for t in range(1, len(x)):
        x[t] = np.diag([0.6, 1]) @ x[t - 1] + rng.normal(size=(2, 1))
    initial = dict(A1=np.diag([-0.4, 0]), A2=np.ones((1, 1)))
    reference = fit_cmar(x, (1, 1), initial=initial, intercept=False, n_starts=1)
    shifted = dict(A1=initial["A1"] * gauge, A2=initial["A2"] / gauge)
    fit = fit_cmar(x, (1, 1), initial=shifted, intercept=False, n_starts=1)
    assert_allclose(fit.long_run_operator, reference.long_run_operator, atol=1e-14)
    assert_allclose(fit.objective, reference.objective, rtol=1e-14)


@pytest.mark.parametrize("method", ["ls", "mle"])
def test_unrestricted_intercept_translation_preserves_stopping_and_coefficients(method):
    x, _, _ = planted(n=220, q=3, ranks=(1, 2), seed=58)
    offset = np.full(x.shape[1:], 1e6)
    kwargs = dict(ranks=(1, 2), difference_lags=1, method=method, n_starts=1)
    fit = fit_cmar(x, **kwargs)
    shifted = fit_cmar(x + offset, **kwargs)
    assert fit.converged and shifted.converged
    assert fit.n_iter == shifted.n_iter
    assert_allclose(fit.long_run_operator, shifted.long_run_operator, atol=2e-8)
    assert_allclose(fit.short_run_operators, shifted.short_run_operators, atol=2e-8)
    assert_allclose(shifted.forecast(3) - offset, fit.forecast(3), atol=2e-7)
    assert_allclose(
        shifted.intercept + shifted.A1 @ offset @ shifted.A2.T, fit.intercept, atol=2e-7
    )
    if method == "mle":
        assert_allclose(shifted.log_likelihood, fit.log_likelihood, atol=2e-6)
