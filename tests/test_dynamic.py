"""Independent dense Gaussian, constrained optimization and dynamic oracles."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from mavats.dynamic import (
    _em_step,
    _inverse_action,
    _loading_moments,
    _loading_update,
    _objective,
    _pooled_ar,
    _rotate,
    _scores,
    _select_ranks,
    _spectrum,
    _stepwise,
    fit_two_way_dynamic,
)


def _parameters(seed=8):
    rng = np.random.default_rng(seed)
    n, m, c, r = 4, 5, 2, 3
    Lam = np.sqrt(n) * np.linalg.qr(rng.normal(size=(n, c)))[0]
    L = np.sqrt(m) * np.linalg.qr(rng.normal(size=(m, r)))[0]
    # Nondiagonal matrices exercise covariance EM before identifying rotation.
    B = rng.normal(size=(r, r))
    psi_f = B @ B.T + 0.4 * np.eye(r)
    B = rng.normal(size=(c, c))
    psi_g = B @ B.T + 0.6 * np.eye(c)
    Y = rng.normal(size=(7, n, m))
    return Y, L, Lam, psi_f, psi_g, 0.7


def _latent_oracle(Y, L, Lam, psi_f, psi_g, noise):
    T, n, m = Y.shape
    r, c = L.shape[1], Lam.shape[1]
    H = np.zeros((n * m, n * r + m * c))
    for k in range(n * r):
        basis = np.zeros((n, r))
        basis.flat[k] = 1
        H[:, k] = (basis @ L.T).ravel()
    for k in range(m * c):
        basis = np.zeros((m, c))
        basis.flat[k] = 1
        H[:, n * r + k] = (Lam @ basis.T).ravel()
    P = np.zeros((n * r + m * c, n * r + m * c))
    P[: n * r, : n * r] = np.kron(np.eye(n), psi_f)
    P[n * r :, n * r :] = np.kron(np.eye(m), psi_g)
    K = H @ P @ H.T + noise * np.eye(n * m)
    means = (P @ H.T @ np.linalg.solve(K, Y.reshape(T, -1).T)).T
    conditional = P - P @ H.T @ np.linalg.solve(K, H @ P)
    return H, K, means, conditional


def test_spectral_inverse_objective_and_scores_match_dense_latent_gaussian():
    args = _parameters()
    Y, L, Lam, psi_f, psi_g, noise = args
    H, K, means, _ = _latent_oracle(*args)
    n, m = Y.shape[1:]
    r = L.shape[1]
    # Independently check row-major Kronecker orientation.
    expected_K = (
        np.kron(np.eye(n), L @ psi_f @ L.T)
        + np.kron(Lam @ psi_g @ Lam.T, np.eye(m))
        + noise * np.eye(n * m)
    )
    assert_allclose(K, expected_K, atol=2e-14)
    action = _inverse_action(Y, _spectrum(L, Lam, psi_f, psi_g, noise))
    assert_allclose(
        action.reshape(len(Y), -1).T,
        np.linalg.solve(K, Y.reshape(len(Y), -1).T),
        atol=1e-13,
    )
    F, G = _scores(*args)
    assert_allclose(F.reshape(len(Y), -1), means[:, : n * r], atol=2e-13)
    assert_allclose(G.reshape(len(Y), -1), means[:, n * r :], atol=2e-13)
    assert_allclose(
        (F @ L.T + Lam @ G.transpose(0, 2, 1)).reshape(len(Y), -1),
        means @ H.T,
        atol=2e-13,
    )
    objective = -np.linalg.slogdet(K)[1] - np.mean(np.sum(Y * action, axis=(1, 2)))
    assert_allclose(_objective(*args), objective, atol=1e-12)


def test_full_em_moments_match_dense_oracle_including_cross_effects():
    args = _parameters()
    Y, L, Lam, psi_f, psi_g, noise = args
    H, K, means, conditional = _latent_oracle(*args)
    T, n, m = Y.shape
    r, c = L.shape[1], Lam.shape[1]
    f, g, s, active = _em_step(*args)
    expected_f = np.zeros((r, r))
    expected_g = np.zeros((c, c))
    for i in range(n):
        indices = slice(i * r, (i + 1) * r)
        unit = means[:, indices]
        expected_f += unit.T @ unit / T + conditional[indices, indices]
    for j in range(m):
        indices = slice(n * r + j * c, n * r + (j + 1) * c)
        unit = means[:, indices]
        expected_g += unit.T @ unit / T + conditional[indices, indices]
    residual = Y.reshape(T, -1) - means @ H.T
    expected_s = (np.sum(residual**2) / T + np.trace(H @ conditional @ H.T)) / (n * m)
    assert_allclose(f, expected_f / n, atol=1e-13)
    assert_allclose(g, expected_g / m, atol=1e-13)
    assert_allclose(s, expected_s, atol=1e-13)
    assert not active
    assert _objective(Y, L, Lam, f, g, s) > _objective(*args)
    # Keeping just independent marginal uncertainty double-counts the overlap.
    independent = conditional.copy()
    independent[: n * r, n * r :] = 0
    independent[n * r :, : n * r] = 0
    incorrect = (np.sum(residual**2) / T + np.trace(H @ independent @ H.T)) / (n * m)
    assert incorrect > expected_s + 0.1
    assert np.linalg.norm(f - np.diag(np.diag(f))) > 0.01


def test_rotation_preserves_covariance_scores_signal_and_normalization():
    Y, L, Lam, psi_f, psi_g, noise = _parameters()
    LL, ff = _rotate(L, psi_f)
    RR, gg = _rotate(Lam, psi_g)
    assert_allclose(LL @ ff @ LL.T, L @ psi_f @ L.T, atol=2e-13)
    assert_allclose(RR @ gg @ RR.T, Lam @ psi_g @ Lam.T, atol=2e-13)
    assert_allclose(LL.T @ LL, len(L) * np.eye(L.shape[1]), atol=2e-14)
    assert np.all(LL[0] >= 0) and np.all(RR[0] >= 0)
    assert np.all(np.diff(np.diag(ff)) < 0)
    F, G = _scores(Y, L, Lam, psi_f, psi_g, noise)
    rotated_F, rotated_G = _scores(Y, LL, RR, ff, gg, noise)
    assert_allclose(
        F @ L.T + Lam @ G.transpose(0, 2, 1),
        rotated_F @ LL.T + RR @ rotated_G.transpose(0, 2, 1),
        atol=2e-13,
    )


def test_loading_moment_matches_paper_expansion_and_conditional_objective():
    Y, L, Lam, psi_f, psi_g, noise = _parameters()
    L, psi_f = _rotate(L, psi_f)
    Lam, psi_g = _rotate(Lam, psi_g)
    T, n, m = Y.shape
    moments = _loading_moments(Y, Lam, np.diag(psi_f), np.diag(psi_g), noise)
    direct = []
    for f in np.diag(psi_f):
        W = f / (noise * (noise + m * f)) * np.eye(n)
        for k, g in enumerate(np.diag(psi_g)):
            coefficient = (
                1 / noise
                - 1 / (noise + m * f)
                - 1 / (noise + n * g)
                + 1 / (noise + m * f + n * g)
            ) / (n * m)
            W -= coefficient * np.outer(Lam[:, k], Lam[:, k])
        direct.append(sum(y.T @ W @ y for y in Y) / T)
    assert_allclose(moments, direct, atol=1e-13)
    updated, _, _ = _loading_update(moments, L, 1000, 1e-11)
    before = np.einsum("is,sij,js->", L, moments, L)
    after = np.einsum("is,sij,js->", updated, moments, updated)
    assert_allclose(
        _objective(Y, updated, Lam, psi_f, psi_g, noise)
        - _objective(Y, L, Lam, psi_f, psi_g, noise),
        after - before,
        atol=1e-12,
    )


def test_polar_updates_indefinite_forms_and_stationary_tangent_gradient():
    rng = np.random.default_rng(19)
    raw = rng.normal(size=(2, 4, 4))
    moments = (raw + raw.transpose(0, 2, 1)) / 2 - 3 * np.eye(4)[None]
    initial = 2 * np.linalg.qr(rng.normal(size=(4, 2)))[0]
    loading, _, converged = _loading_update(moments, initial, 2000, 1e-13)
    assert converged
    assert_allclose(loading.T @ loading, 4 * np.eye(2), atol=1e-13)
    product = np.einsum("sij,js->is", moments, loading)
    multiplier = loading.T @ product / 4
    tangent = product - loading @ (multiplier + multiplier.T) / 2
    assert np.linalg.norm(tangent) < 2e-5
    rankone, _, done = _loading_update(moments[:1], initial[:, :1], 1, 1e-7)
    assert done
    assert_allclose(
        (rankone.T @ moments[0] @ rankone).item(),
        4 * np.linalg.eigvalsh(moments[0])[-1],
    )


def _sample(T=400, seed=10, mean=0):
    rng = np.random.default_rng(seed)
    n, m, c, r = 6, 5, 1, 2
    Lam = np.sqrt(n) * np.linalg.qr(rng.normal(size=(n, c)))[0]
    L = np.sqrt(m) * np.linalg.qr(rng.normal(size=(m, r)))[0]
    F = np.zeros((T + 100, n, r))
    G = np.zeros((T + 100, m, c))
    for t in range(1, len(F)):
        F[t] = F[t - 1] * [0.8, 0.3] + rng.normal(size=(n, r)) * [0.7, 0.5]
        G[t] = G[t - 1] * 0.6 + rng.normal(size=(m, c)) * 0.5
    F, G = F[100:], G[100:]
    signal = F @ L.T + Lam @ G.transpose(0, 2, 1)
    Y = mean + signal + rng.normal(size=signal.shape) * 0.3
    return Y, signal, L, Lam


def test_end_to_end_recovery_dynamics_forecasts_and_no_mutation():
    Y, signal, L, Lam = _sample()
    before = Y.copy()
    fit = fit_two_way_dynamic(Y, (1, 2), tol=1e-8)
    assert fit.converged
    assert fit.ranks == (1, 2) and fit.orders == (1, 1)
    assert fit.F.shape == (len(Y), 6, 2) and fit.G.shape == (len(Y), 5, 1)
    assert_allclose(fit.column_ar, [[0.8, 0.3]], atol=0.07)
    assert_allclose(fit.row_ar, [[0.6]], atol=0.07)
    assert_allclose(fit.noise_variance, 0.09, atol=0.015)
    assert np.mean((fit.signal - signal) ** 2) < 0.06
    assert_allclose(
        fit.column_loadings @ fit.column_loadings.T / 5, L @ L.T / 5, atol=0.04
    )
    assert_allclose(
        fit.row_loadings @ fit.row_loadings.T / 6, Lam @ Lam.T / 6, atol=0.04
    )
    assert_allclose(
        fit.column_loadings.T @ fit.column_loadings, 5 * np.eye(2), atol=1e-13
    )
    assert_array_equal(Y, before)
    assert_allclose(fit.signal + fit.residuals, Y)
    assert_allclose(fit.reconstruct(Y), fit.signal, atol=2e-13)
    assert_allclose(fit.transform(Y)[0], fit.F, atol=2e-13)
    expected = (
        fit.F[-1] * fit.column_ar[0]
    ) @ fit.column_loadings.T + fit.row_loadings @ (fit.G[-1] * fit.row_ar[0]).T
    assert_allclose(fit.forecast(2)[0], expected, atol=1e-13)
    assert_allclose(fit.forecast(2, history=Y[-1:]), fit.forecast(2), atol=1e-13)
    assert fit.is_stable
    assert len(fit.objective_history) == fit.n_iter + 1
    assert fit.inner_converged[-1].all()
    assert np.diff(fit.objective_history).min() >= -1e-10
    assert all(np.diff(h).min() >= -1e-10 for h in fit.em_objective_histories)


def test_pooled_ar2_matches_direct_scalar_design_and_paper_variance_denominator():
    rng = np.random.default_rng(2)
    scores = rng.normal(size=(14, 3, 2))
    coefficients, variance, residuals = _pooled_ar(scores, 2)
    for j in range(2):
        design = []
        response = []
        for t in range(2, len(scores)):
            for i in range(3):
                design.append([scores[t - 1, i, j], scores[t - 2, i, j]])
                response.append(scores[t, i, j])
        design, response = np.asarray(design), np.asarray(response)
        expected = np.linalg.lstsq(design, response, rcond=None)[0]
        error = response - design @ expected
        assert_allclose(coefficients[:, j], expected)
        assert_allclose(variance[j], np.sum(error**2) / (3 * 14))
        assert_allclose(residuals[:, :, j].ravel(), error)


def test_stepwise_initialization_and_rank_selection_projected_residual_oracle():
    Y, _, _, _ = _sample(T=200)
    L, Lam, history = _stepwise(Y, (1, 2), 20, 1e-10)
    assert_allclose(L.T @ L, 5 * np.eye(2), atol=1e-13)
    assert_allclose(Lam.T @ Lam, 6 * np.eye(1), atol=1e-13)
    assert np.diff(history).max() <= 1e-12
    ranks, rank_history, spectra, converged, cycle = _select_ranks(Y, (3, 2), 0.01, 20)
    assert ranks == (1, 2) and converged and not cycle
    # Independently reproduce the first rank iteration, including T scaling.
    T, n, m = Y.shape
    Q = np.linalg.eigh(sum(y @ y.T for y in Y))[1][:, -3:]
    residual = [(np.eye(n) - Q @ Q.T) @ y for y in Y]
    vals = np.linalg.eigvalsh(sum(e.T @ e for e in residual) / T)[::-1]
    delta = max(n**-0.5, m**-0.5, T**-0.5)
    r = 1 + np.argmax(vals[:2] / (vals[1:3] + 0.01 * delta))
    Q = np.linalg.eigh(sum(y.T @ y for y in Y))[1][:, -r:]
    residual = [y @ (np.eye(m) - Q @ Q.T) for y in Y]
    vals = np.linalg.eigvalsh(sum(e @ e.T for e in residual) / T)[::-1]
    c = 1 + np.argmax(vals[:3] / (vals[1:4] + 0.01 * delta))
    assert_array_equal(rank_history[1], [c, r])
    assert spectra[0].shape == (6,) and spectra[1].shape == (5,)
    fit = fit_two_way_dynamic(Y, rank_bounds=(3, 2))
    assert fit.ranks == (1, 2) and fit.rank_converged


def test_centering_scaled_input_and_initialization_contract():
    Y, _, L, Lam = _sample(T=130)
    fit = fit_two_way_dynamic(Y, (1, 2), center=True, initial=(Lam, L), tol=1e-8)
    shifted = fit_two_way_dynamic(
        7 * Y + 25, (1, 2), center=True, initial=(Lam, L), tol=1e-8
    )
    assert_allclose(shifted.signal, 7 * fit.signal + 25, atol=2e-7)
    assert_allclose(shifted.forecast(2), 7 * fit.forecast(2) + 25, atol=2e-7)
    assert_allclose(shifted.noise_variance, 49 * fit.noise_variance, rtol=1e-7)
    assert_allclose(fit.mean, Y.mean(axis=0), atol=1e-14)
    assert fit.initialization_history.size == 0


def test_iteration_limits_remain_visible_and_floors_are_reported():
    Y, _, _, _ = _sample(T=50)
    fit = fit_two_way_dynamic(
        Y,
        (1, 2),
        max_iter=1,
        em_max_iter=1,
        loading_max_iter=1,
        tol=1e-14,
        variance_floor=0.2,
    )
    assert fit.n_iter == 1 and not fit.converged
    assert not fit.inner_converged.all()
    assert fit.variance_regularized
    assert fit.forecast(1).shape == (1, 6, 5)


def test_rank_iteration_limit_and_data_units_for_ratio_ridge():
    Y, _, _, _ = _sample(T=120)
    limited = fit_two_way_dynamic(Y, rank_bounds=(3, 2), rank_max_iter=1)
    assert limited.rank_history.shape == (2, 2)
    assert not limited.rank_converged
    assert not limited.rank_cycle
    base = fit_two_way_dynamic(Y, rank_bounds=(3, 2), ratio_ridge=0.03)
    scaled = fit_two_way_dynamic(11 * Y, rank_bounds=(3, 2), ratio_ridge=0.03 * 121)
    assert_array_equal(base.rank_history, scaled.rank_history)
    assert_allclose(base.signal, scaled.signal / 11, atol=1e-10)


@pytest.mark.parametrize("scale", [1e-140, 1e140])
def test_extreme_representable_data_scaling(scale):
    Y, _, L, Lam = _sample(T=50)
    base = fit_two_way_dynamic(Y, (1, 2), initial=(Lam, L))
    scaled = fit_two_way_dynamic(Y * scale, (1, 2), initial=(Lam, L))
    assert_allclose(scaled.signal / scale, base.signal, atol=2e-12)
    assert_allclose(
        (scaled.noise_variance / scale) / scale, base.noise_variance, rtol=1e-11
    )
    assert_allclose(scaled.forecast(2) / scale, base.forecast(2), atol=2e-12)


def test_unrepresentable_physical_variance_raises():
    Y, _, L, Lam = _sample(T=40)
    with pytest.raises(FloatingPointError, match="underflow"):
        fit_two_way_dynamic(Y * 1e-250, (1, 2), initial=(Lam, L))


@pytest.mark.parametrize(
    "options",
    [
        {"ranks": (6, 2)},
        {"ranks": (0, 1)},
        {"ranks": (1, 2.0)},
        {"orders": (1, 100)},
        {"orders": True},
        {"rank_bounds": (6, 2)},
        {"ratio_ridge": 0},
        {"ratio_ridge": np.nan},
        {"center": 1},
        {"max_iter": 0},
        {"loading_max_iter": False},
        {"em_max_iter": 1.2},
        {"tol": 0},
        {"variance_floor": -1},
        {"initial": 3},
        {"initial": (np.eye(6)[:, :1], np.eye(5)[:, :2])},
    ],
)
def test_invalid_parameters(options):
    Y, _, _, _ = _sample(T=30)
    kwargs = {"ranks": (1, 2)}
    kwargs.update(options)
    with pytest.raises(ValueError):
        fit_two_way_dynamic(Y, **kwargs)


def test_bad_inputs_and_forecast_shapes():
    for Y in (np.zeros((10, 3, 4)), np.ones((10, 1, 4)), np.full((10, 3, 4), np.nan)):
        with pytest.raises(ValueError):
            fit_two_way_dynamic(Y)
    with pytest.raises(ValueError):
        fit_two_way_dynamic(np.ones((10, 3, 4)), center=True)
    fit = fit_two_way_dynamic(_sample(T=40)[0], (1, 2), orders=(1, 2))
    with pytest.raises(ValueError):
        fit.forecast(True)
    with pytest.raises(ValueError):
        fit.forecast(1, history=np.ones((1, 6, 5)))
    with pytest.raises(ValueError):
        fit.transform(np.ones((3, 5, 6)))
    with pytest.raises(ValueError):
        fit.inverse_transform(np.ones((3, 6, 2)), np.ones((2, 5, 1)))
