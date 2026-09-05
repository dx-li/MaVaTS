import numpy as np
import pytest

from mavats.alphaPCA import estimate_alpha_PCA, estimate_cov_Cj, estimate_cov_Ri
from mavats.factors import (
    eigenvalue_ratio,
    fit_alpha_pca,
    fit_lagged_factor,
    fit_projected_pca,
)


def projector(Q):
    return Q @ Q.T


def fixture(seed=42, noise=0):
    rng = np.random.default_rng(seed)
    Q = np.linalg.qr(rng.normal(size=(7, 2)))[0]
    V = np.linalg.qr(rng.normal(size=(6, 2)))[0]
    F = rng.normal(size=(120, 2, 2))
    for t in range(1, len(F)):
        F[t] += 0.7 * F[t - 1]
    signal = Q @ F @ V.T
    return signal + noise * rng.normal(size=signal.shape), signal, (Q, V)


def reference_loading(moment, rank):
    return np.linalg.eigh(moment)[1][:, -rank:]


@pytest.mark.parametrize("fit", [fit_alpha_pca, fit_projected_pca, fit_lagged_factor])
def test_noiseless_subspace_and_reconstruction(fit):
    X, _, truth = fixture()
    result = fit(X, (2, 2))
    for actual, expected in zip(result.loadings, truth):
        np.testing.assert_allclose(projector(actual), projector(expected), atol=1e-11)
    np.testing.assert_allclose(result.signal, X, atol=1e-11)
    np.testing.assert_allclose(
        result.inverse_transform(result.transform(X)), X, atol=1e-11
    )
    np.testing.assert_allclose(result.residuals, 0, atol=1e-11)


@pytest.mark.parametrize("alpha", [-1, 0, 2.5])
def test_alpha_matches_independent_mean_covariance_formula(alpha):
    X, _, _ = fixture(noise=0.3)
    X = X + np.arange(42).reshape(7, 6) / 10
    mean = X.mean(0)
    centered = X - mean
    row = sum(x @ x.T for x in centered) / len(X) + (1 + alpha) * mean @ mean.T
    col = sum(x.T @ x for x in centered) / len(X) + (1 + alpha) * mean.T @ mean
    result = fit_alpha_pca(X, (2, 3), alpha=alpha)
    for estimated, moment, rank in zip(result.loadings, (row, col), (2, 3)):
        np.testing.assert_allclose(
            projector(estimated), projector(reference_loading(moment, rank)), atol=1e-11
        )


def test_projected_pca_uses_both_initial_spaces_in_first_step():
    X, _, _ = fixture(noise=0.6)
    initial = fit_alpha_pca(X, (2, 2))
    row_projected = X @ initial.loadings[1]
    col_projected = X.transpose(0, 2, 1) @ initial.loadings[0]
    expected = [
        reference_loading(sum(x @ x.T for x in series), 2)
        for series in (row_projected, col_projected)
    ]
    actual = fit_projected_pca(X, (2, 2))
    for Q, V in zip(actual.loadings, expected):
        np.testing.assert_allclose(projector(Q), projector(V), atol=1e-11)


def test_lagged_factor_matches_explicit_column_pair_covariances():
    X, _, _ = fixture(noise=0.4)
    result = fit_lagged_factor(X, (2, 2), lags=(1, 3))
    for data, Q in zip((X, X.transpose(0, 2, 1)), result.loadings):
        moment = np.zeros((data.shape[1], data.shape[1]))
        for h in (1, 3):
            for i in range(data.shape[2]):
                for j in range(data.shape[2]):
                    cov = data[:-h, :, i].T @ data[h:, :, j] / (len(X) - h)
                    moment += cov @ cov.T
        np.testing.assert_allclose(
            projector(Q), projector(reference_loading(moment, 2)), atol=1e-11
        )


def test_centering_uses_training_mean_for_new_observations():
    X, _, _ = fixture()
    X += np.arange(42).reshape(7, 6)
    result = fit_alpha_pca(X, (2, 2), center=True)
    np.testing.assert_allclose(result.signal, X, atol=1e-11)
    future = X[:3] + 2
    expected = result.loadings[0].T @ (future - X.mean(0)) @ result.loadings[1]
    np.testing.assert_allclose(result.transform(future), expected, atol=1e-11)
    with pytest.raises(ValueError):
        result.transform(np.ones((3, 6, 7)))
    with pytest.raises(ValueError):
        result.inverse_transform(np.ones((3, 2, 3)))


@pytest.mark.parametrize("fit", [fit_alpha_pca, fit_projected_pca, fit_lagged_factor])
def test_rank_defaults_degenerate_modes_and_scale_invariance(fit):
    X, _, _ = fixture()
    result = fit(X)
    assert result.ranks == (2, 2)
    large = fit(X * 1e100, (2, 2))
    tiny = fit(X * 1e-100, (2, 2))
    np.testing.assert_allclose(large.signal / 1e100, X, atol=1e-11)
    np.testing.assert_allclose(tiny.signal / 1e-100, X, atol=1e-11)
    zeros = fit(np.zeros((5, 1, 1)))
    assert zeros.ranks == (1, 1)
    np.testing.assert_array_equal(zeros.signal, 0)


@pytest.mark.parametrize(
    "ranks", [(0, 1), (1, 0), (8, 1), (1, 7), (1.2, 1), (True, 1), (1,), 2]
)
def test_rank_validation(ranks):
    with pytest.raises(ValueError):
        fit_alpha_pca(fixture()[0], ranks)


@pytest.mark.parametrize("alpha", [-2, np.nan, np.inf, True, 2j, [0]])
def test_alpha_validation(alpha):
    with pytest.raises(ValueError):
        fit_alpha_pca(fixture()[0], (2, 2), alpha=alpha)


@pytest.mark.parametrize("lags", [0, -1, 120, [], [1, 1], [0], True, 1.2])
def test_lag_validation(lags):
    with pytest.raises(ValueError):
        fit_lagged_factor(fixture()[0], (2, 2), lags=lags)


def test_eigenvalue_ratio_order_scale_and_zero():
    values = np.array([100, 80, 0.1, 0.09, 0.08, 0.07])
    for spectrum in (values, values[::-1], values * 1e-200):
        assert eigenvalue_ratio(spectrum) == 2
    assert eigenvalue_ratio(np.zeros(6)) == 1
    assert eigenvalue_ratio([2]) == 1


def test_legacy_alpha_signal_is_not_divided_by_dimensions_twice():
    X, _, _ = fixture()
    F, signal, R, C = estimate_alpha_PCA(X, 0, 2, 2)
    np.testing.assert_allclose(signal, X, atol=1e-11)
    np.testing.assert_allclose(R @ F @ C.T, signal, atol=1e-11)
    np.testing.assert_allclose(R.T @ R, 7 * np.eye(2), atol=1e-11)
    np.testing.assert_allclose(C.T @ C, 6 * np.eye(2), atol=1e-11)


def test_legacy_covariance_matches_score_hac_and_rotation():
    X, _, _ = fixture(noise=0.3)
    alpha, m, index = 0.7, 3, 1
    F, _, R, C = estimate_alpha_PCA(X, alpha, 2, 2)
    residual = X - R @ F @ C.T
    scores = np.array(
        [
            (factor + alpha * F.mean(0)) @ C.T @ error[index]
            for factor, error in zip(F, residual)
        ]
    ) / np.sqrt(X.shape[2])
    middle = scores.T @ scores / len(X)
    for h in range(1, m + 1):
        cross = scores[h:].T @ scores[:-h] / len(X)
        middle += (1 - h / (m + 1)) * (cross + cross.T)
    adjusted = X + (np.sqrt(alpha + 1) - 1) * X.mean(0)
    moment = sum(x @ x.T for x in adjusted) / np.prod(X.shape)
    information = R.T @ moment @ R / X.shape[1]
    inverse = np.linalg.inv(information)  # Independent small reference only.
    expected = inverse @ middle @ inverse
    actual = estimate_cov_Ri(X, F, R, C, index, alpha, m)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert np.linalg.eigvalsh(actual).min() >= -1e-12
    O = np.array([[0.6, -0.8], [0.8, 0.6]])
    rotated = estimate_cov_Ri(X, O.T @ F, R @ O, C, index, alpha, m)
    np.testing.assert_allclose(rotated, O.T @ actual @ O, atol=1e-12)
    transpose = estimate_cov_Ri(
        X.transpose(0, 2, 1), F.transpose(0, 2, 1), C, R, 2, alpha, m
    )
    np.testing.assert_allclose(estimate_cov_Cj(X, F, R, C, 2, alpha, m), transpose)


@pytest.mark.parametrize("index,m", [(-1, 0), (7, 0), (0, -1), (0, 120), (0, 1.2)])
def test_covariance_rejects_invalid_index_or_bandwidth(index, m):
    X, _, _ = fixture(noise=0.3)
    F, _, R, C = estimate_alpha_PCA(X, 0, 2, 2)
    with pytest.raises(ValueError):
        estimate_cov_Ri(X, F, R, C, index, 0, m)
