"""Independent convex-regression and matrix-normalization oracles for IHR."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import brentq, least_squares

from mavats.factors import fit_projected_pca
from mavats.ihr import (
    _factor_regressions,
    _loss,
    _loss_change,
    _normalize,
    _regression,
    fit_ihr_factor,
    select_ihr_ranks,
)


def data(seed=2, shape=(35, 7, 6), ranks=(2, 2), noise=0.1):
    rng = np.random.default_rng(seed)
    row = np.linalg.qr(rng.normal(size=(shape[1], ranks[0])))[0] * np.sqrt(shape[1])
    col = np.linalg.qr(rng.normal(size=(shape[2], ranks[1])))[0] * np.sqrt(shape[2])
    core = rng.normal(size=(shape[0], *ranks))
    signal = row @ core @ col.T
    return signal + noise * rng.normal(size=shape), signal, row, col


def convex_oracle(z, y, tau):
    # Trust-region robust least squares is algorithmically independent of IRLS.
    result = least_squares(
        lambda b: z @ b - y,
        np.zeros(z.shape[1]),
        jac=lambda b: z.copy(),
        loss="huber",
        f_scale=tau,
        gtol=1e-12,
        ftol=1e-12,
        xtol=1e-12,
        max_nfev=2000,
    )
    return result.x


def test_scalar_regression_matches_clipped_score_root():
    y = np.array([-3.0, -0.3, 0.1, 0.2, 0.5, 19.0])
    tau = 0.7
    expected = brentq(lambda b: np.clip(y - b, -tau, tau).sum(), -3, 19)
    beta, info = _regression(np.ones((len(y), 1)), y, tau, max_iter=1000, tol=1e-11)
    assert_allclose(beta, [expected], atol=1e-10)
    assert info.converged
    assert np.all(
        np.diff(info.objective_history)
        <= 4 * np.finfo(float).eps * info.objective_history[0]
    )


def test_loss_difference_retains_sub_ulp_improvement_and_threshold_crossings():
    from decimal import Decimal, localcontext

    old = np.array([-9.0, -0.70000000001, -0.2, 0.3, 0.69999999999, 80.0])
    change = np.array([1e-10, 2e-11, -1e-10, 2e-10, 2e-11, -1e-10])
    with localcontext() as context:
        context.prec = 70
        tau = Decimal.from_float(0.7)

        def exact(e):
            return e * e / 2 if abs(e) <= tau else tau * abs(e) - tau * tau / 2

        expected = sum(
            exact(Decimal.from_float(e) + Decimal.from_float(d))
            - exact(Decimal.from_float(e))
            for e, d in zip(old, change)
        ) / len(old)
    assert_allclose(
        _loss_change(old, change, 0.7), float(expected), rtol=1e-7, atol=1e-25
    )


def test_dense_regression_and_kronecker_factor_design_match_convex_oracle():
    rng = np.random.default_rng(3)
    row, col = rng.normal(size=(4, 2)), rng.normal(size=(3, 2))
    X = rng.normal(size=(2, 4, 3))
    X[0, 1, 2] = 20
    actual, diagnostics = _factor_regressions(X, row, col, 0.8, 1000, 1e-9)
    design = np.array([np.kron(col[j], row[i]) for j in range(3) for i in range(4)])
    for t in range(2):
        expected = convex_oracle(design, X[t].ravel(order="F"), 0.8)
        assert_allclose(actual[t].ravel(order="F"), expected, atol=2e-6)
        assert diagnostics[t].score_norm < 1e-7


def test_normalization_preserves_signal_and_paper_covariance_units():
    rng = np.random.default_rng(15)
    row, col = rng.normal(size=(5, 2)), rng.normal(size=(4, 3))
    core = rng.normal(size=(10, 2, 3))
    r, c, f, spectra = _normalize(row, col, core)
    assert_allclose(r @ f @ c.T, row @ core @ col.T, atol=1e-13)
    assert_allclose(r.T @ r, np.eye(2), atol=1e-14)
    assert_allclose(c.T @ c, np.eye(3), atol=1e-14)
    paper_f = f / np.sqrt(20)
    assert_allclose(
        np.mean(paper_f @ paper_f.transpose(0, 2, 1), axis=0),
        np.diag(spectra[0]),
        atol=1e-13,
    )
    assert_allclose(
        np.mean(paper_f.transpose(0, 2, 1) @ paper_f, axis=0),
        np.diag(spectra[1]),
        atol=1e-13,
    )


def test_first_sweep_matches_explicit_row_column_factor_oracles():
    X, _, row, col = data(shape=(5, 4, 3), ranks=(1, 1), noise=0.3)
    X[1, 0, 2] += 9
    row, col = row / np.sqrt(4), col / np.sqrt(3)
    f = row.T @ X @ col
    tau = 0.4
    r = np.array(
        [
            convex_oracle(
                np.array([f[t] @ col[j] for t in range(5) for j in range(3)]),
                X[:, i, :].ravel(),
                tau,
            )
            for i in range(4)
        ]
    )
    c = np.array(
        [
            convex_oracle(
                np.array([f[t].T @ r[i] for t in range(5) for i in range(4)]),
                X[:, :, j].ravel(),
                tau,
            )
            for j in range(3)
        ]
    )
    fs = np.array(
        [convex_oracle(np.kron(c, r), x.ravel(order="F"), tau).reshape(1, 1) for x in X]
    )
    fit = fit_ihr_factor(
        X,
        (1, 1),
        initial=(row, col),
        threshold=tau,
        max_iter=1,
        inner_max_iter=1000,
        inner_tol=1e-10,
    )
    assert_allclose(fit.signal, r @ fs @ c.T, atol=2e-6)
    assert not fit.converged


def test_quadratic_limit_matches_least_squares_and_projected_signal():
    X, _, _, _ = data()
    fit = fit_ihr_factor(X, (2, 2), threshold=1e10, max_iter=100)
    pe = fit_projected_pca(X, (2, 2), max_iter=100, tol=1e-10)
    assert_allclose(fit.signal, pe.signal, atol=3e-5)
    beta, info = _regression(
        np.array([[1.0, 0], [1, 1], [1, 2]]),
        np.array([1.0, 3, 4]),
        np.inf,
        initial=np.zeros(2),
    )
    assert_allclose(
        beta, np.linalg.lstsq([[1, 0], [1, 1], [1, 2]], [1, 3, 4], rcond=None)[0]
    )
    assert info.converged


def test_fixed_threshold_descent_recovery_and_robust_transform():
    X, signal, row, col = data()
    X[0, 1, 2] += 80
    fit = fit_ihr_factor(
        X, (2, 2), threshold=0.3, initial=(row, col), inner_max_iter=300
    )
    assert np.all(np.diff(fit.objective_history) <= 0)
    assert np.linalg.norm(fit.signal - signal) / np.linalg.norm(signal) < 0.06
    robust, diagnostics = fit.transform(X, return_diagnostics=True)
    ordinary = fit.loadings[0].T @ X @ fit.loadings[1]
    assert np.linalg.norm(robust[0] - ordinary[0]) > 1
    assert_allclose(fit.inverse_transform(robust)[1:], fit.signal[1:], atol=2e-3)
    assert max(info.score_norm for info in diagnostics) < 1e-7
    future = X[:3].copy()
    one = fit.transform(future, return_diagnostics=True)[0]
    future[2] *= 3
    two = fit.transform(future, return_diagnostics=True)[0]
    assert_allclose(one[:2], two[:2], atol=1e-8)


@pytest.mark.parametrize("scale", [1e-250, 1e-150, 1e150, 1e250, -3.0])
def test_extreme_scale_equivariance(scale):
    X, _, row, col = data(shape=(12, 4, 3), ranks=(1, 1))
    opts = dict(initial=(row, col), max_iter=8, inner_max_iter=300)
    first = fit_ihr_factor(X, (1, 1), threshold=0.3, **opts)
    second = fit_ihr_factor(X * scale, (1, 1), threshold=0.3 * abs(scale), **opts)
    assert_allclose(second.signal / scale, first.signal, atol=2e-7)
    assert_allclose(
        second.objective_history[-1], first.objective_history[-1], rtol=1e-7
    )
    assert_allclose(
        second.transform(X[:2] * scale, return_diagnostics=True)[0] / scale,
        first.transform(X[:2], return_diagnostics=True)[0],
        atol=2e-7,
    )


def test_pilot_threshold_is_frozen_and_input_preserved():
    X, _, _, _ = data()
    original = X.copy()
    result = fit_ihr_factor(X, (2, 2), max_iter=2)
    pilot = fit_projected_pca(X, (2, 2), max_iter=1)
    assert_allclose(
        result.threshold, 1.345 * 1.483 * np.median(np.abs(pilot.residuals))
    )
    assert_array_equal(X, original)


def test_huge_quadratic_threshold_does_not_erase_tiny_transform_data():
    X, _, _, _ = data(shape=(10, 4, 3), ranks=(1, 1))
    X *= 1e-250
    result = fit_ihr_factor(X, (1, 1), threshold=1e250)
    actual, diagnostics = result.transform(X[:2], return_diagnostics=True)
    expected = result.loadings[0].T @ X[:2] @ result.loadings[1]
    assert_allclose(actual / 1e-250, expected / 1e-250, atol=1e-12)
    assert all(d.converged for d in diagnostics)


def test_transform_batches_cannot_change_earlier_numeric_stopping():
    X, _, _, _ = data(shape=(12, 4, 3), ranks=(1, 1))
    result = fit_ihr_factor(X, (1, 1), threshold=0.3)
    first, first_diagnostics = result.transform(X[:1], return_diagnostics=True)
    altered = np.concatenate((X[:1], X[1:2] * 1e100))
    together, diagnostics = result.transform(altered, return_diagnostics=True)
    assert_array_equal(first[0], together[0])
    assert_array_equal(
        first_diagnostics[0].objective_history, diagnostics[0].objective_history
    )


def test_centered_decomposition_and_inverse_use_fixed_training_mean():
    X, _, _, _ = data(shape=(12, 4, 3), ranks=(1, 1))
    X += 8
    result = fit_ihr_factor(X, (1, 1), center=True, threshold=0.3)
    assert_allclose(result.mean, X.mean(axis=0))
    assert_allclose(result.inverse_transform(result.factors), result.signal)
    assert_allclose(result.residuals + result.signal, X)


def test_rank_rules_use_paper_core_units_and_log_ridge():
    X, _, _, _ = data(shape=(25, 6, 5), ranks=(1, 1), noise=0.2)
    fit = select_ihr_ranks(X, (2, 2), threshold=0.4, max_iter=5)
    d2 = min(25 * 6, 25 * 5, 6 * 5)
    for spectrum, ratios in zip(fit.pilot.eigenvalues, fit.log_ratios):
        original = spectrum * fit.pilot.data_scale**2
        assert_allclose(np.exp(ratios), original[:-1] / (original[1:] + 1e-4 / d2))
    threshold = select_ihr_ranks(
        X, (2, 2), threshold=0.4, max_iter=5, method="threshold"
    )
    for spectrum, cutoff, rank in zip(
        threshold.pilot.eigenvalues, threshold.thresholds, threshold.ranks
    ):
        assert_allclose(cutoff, spectrum[0] * d2 ** (-1 / 3))
        assert rank == np.count_nonzero(spectrum > cutoff)


def test_ratio_rank_rule_preserves_scaling_when_absolute_ridge_scales():
    X, _, _, _ = data(shape=(12, 4, 3), ranks=(1, 1))
    first = select_ihr_ranks(X, (2, 2), threshold=0.3, max_iter=3)
    second = select_ihr_ranks(
        X * 1e100, (2, 2), threshold=0.3e100, ridge=1e196, max_iter=3
    )
    assert first.ranks == second.ranks
    for a, b in zip(first.log_ratios, second.log_ratios):
        assert_allclose(a, b, atol=1e-9)


def test_threshold_selection_does_not_silently_force_rank_one():
    result = select_ihr_ranks(
        np.ones((5, 1, 1)), (1, 1), method="threshold", threshold=1
    )
    assert result.ranks == (0, 0)


@pytest.mark.parametrize(
    "options",
    [
        {"threshold": 0},
        {"threshold": True},
        {"threshold": 1j},
        {"max_iter": 0},
        {"inner_max_iter": True},
        {"tol": 0},
        {"inner_tol": -1},
        {"center": "yes"},
    ],
)
def test_invalid_options(options):
    X, _, _, _ = data()
    with pytest.raises(ValueError):
        fit_ihr_factor(X, (2, 2), **options)


def test_rank_and_scale_boundaries():
    with pytest.raises(ValueError):
        fit_ihr_factor(np.ones((4, 2, 2)), None)
    with pytest.raises(ValueError):
        fit_ihr_factor(np.ones((4, 2, 2)), (1, 1))
    with pytest.raises(ValueError):
        _regression(np.ones((3, 2)), np.arange(3), 1)
    X, _, _, _ = data()
    fit = fit_ihr_factor(X, (2, 2), max_iter=1, inner_max_iter=1)
    assert not fit.converged
    with pytest.raises(ValueError):
        fit.transform(X[:, :3])
    with pytest.warns(RuntimeWarning):
        fit.transform(X)
