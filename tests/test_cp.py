import numpy as np
import pytest

from mavats.cp import (
    CPIdentificationError,
    _pca_proxy,
    _refined_loadings,
    fit_cp_factor,
)


def cp_fixture(seed=14, samples=200, noise=0):
    rng = np.random.default_rng(seed)
    A, B = rng.normal(size=(6, 3)), rng.normal(size=(5, 3))
    A /= np.linalg.norm(A, axis=0)
    B /= np.linalg.norm(B, axis=0)
    scores = rng.normal(size=(samples, 3))
    for t in range(1, samples):
        scores[t] += np.array([-0.8, 0.4, 0.9]) * scores[t - 1]
    signal = np.einsum("tr,ir,jr->tij", scores, A, B)
    return signal + noise * rng.normal(size=signal.shape), signal, scores, A, B


def test_exact_cp_recovers_individual_nonorthogonal_components_and_scores():
    X, _, scores, A, B = cp_fixture()
    result = fit_cp_factor(X, 3)
    # Individual components, not just row and column subspaces, must match.
    similarity = abs(A.T @ result.A) * abs(B.T @ result.B)
    permutation = similarity.argmax(axis=1)
    assert len(set(permutation)) == 3
    np.testing.assert_allclose(similarity[np.arange(3), permutation], 1, atol=1e-11)
    signs = np.sign(
        np.sum(A * result.A[:, permutation], axis=0)
        * np.sum(B * result.B[:, permutation], axis=0)
    )
    np.testing.assert_allclose(
        result.scores[:, permutation] * signs, scores, atol=1e-10
    )
    np.testing.assert_allclose(result.signal, X, atol=1e-11)
    np.testing.assert_allclose(result.residuals, 0, atol=1e-11)
    np.testing.assert_allclose(np.linalg.norm(result.A, axis=0), 1, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(result.B, axis=0), 1, atol=1e-12)
    assert not np.allclose(result.A.T @ result.A, np.eye(3))


def test_cp_scores_use_joint_least_squares_and_reconstruct_unseen_data():
    X, _, _, A, B = cp_fixture()
    result = fit_cp_factor(X, 3)
    future_scores = np.array([[1, -2, 3], [-4, 5, 6]])
    future = np.einsum("tr,ir,jr->tij", future_scores, A, B)
    projected = result.transform(future)
    np.testing.assert_allclose(result.inverse_transform(projected), future, atol=1e-11)
    # This is deliberately a nonorthogonal basis: separate inner products
    # are not the scores.
    inner_products = np.einsum("tij,ir,jr->tr", X, result.A, result.B)
    assert np.linalg.norm(inner_products - result.scores) > 1


def test_noisy_cp_recovers_components_and_denoises():
    X, truth, _, A, B = cp_fixture(samples=1200, noise=0.02)
    result = fit_cp_factor(X, 3)
    similarity = abs(A.T @ result.A) * abs(B.T @ result.B)
    assert np.min(similarity.max(axis=1)) > 0.995
    assert np.linalg.norm(result.signal - truth) < np.linalg.norm(X - truth)


@pytest.mark.parametrize("scale", [1e-100, 1e100, -3])
def test_scale_invariance(scale):
    X, _, _, _, _ = cp_fixture()
    result = fit_cp_factor(X * scale, 3)
    np.testing.assert_allclose(result.signal / scale, X, atol=1e-10)


def test_center_and_proxy_affine_invariance():
    X, _, factors, _, _ = cp_fixture()
    mean = np.arange(30).reshape(6, 5) / 5
    xi, eta = factors.sum(axis=1), factors @ np.array([0.2, 0.5, 0.3])
    result = fit_cp_factor(X + mean, 3, center=True, xi=xi, eta=eta)
    shifted = fit_cp_factor(X + mean, 3, center=True, xi=3 * xi + 7, eta=-2 * eta + 4)
    np.testing.assert_allclose(result.signal, X + mean, atol=1e-10)
    np.testing.assert_allclose(shifted.signal, result.signal, atol=1e-10)
    np.testing.assert_allclose(shifted.eigenvalues, result.eigenvalues, atol=1e-10)
    np.testing.assert_allclose(result.transform(result.mean[None]), 0, atol=1e-12)


def test_reference_refined_equations_recover_loading_columns():
    rng = np.random.default_rng(51)
    U = rng.normal(size=(3, 3))
    V = rng.normal(size=(3, 3))
    U /= np.linalg.norm(U, axis=0)
    V /= np.linalg.norm(V, axis=0)
    H1 = U @ np.diag([0.4, -0.7, 1.1]) @ V.T
    H2 = U @ np.diag([0.4, -0.7, 1.1]) @ np.diag([-0.5, 0.3, 0.8]) @ V.T
    actual_U, actual_V, values, _ = _refined_loadings(H1, H2, 1e10, 1e-8)
    np.testing.assert_allclose(values, [-0.5, 0.3, 0.8], atol=1e-12)
    for actual, truth in ((actual_U, U), (actual_V, V)):
        np.testing.assert_allclose(abs(np.sum(actual * truth, axis=0)), 1, atol=1e-12)
    # The published normal-equation eigenproblem gives the same eigenvalues.
    J = np.linalg.inv(H1.T @ H1) @ H1.T @ H2
    np.testing.assert_allclose(np.sort(np.linalg.eigvals(J).real), values, atol=1e-11)


def test_reference_full_sample_mean_lag_covariance_pipeline():
    X, _, factors, _, _ = cp_fixture()
    xi, eta = factors @ [0.4, 0.2, 0.3], factors @ [0.1, 0.6, 0.5]
    centered = X - X.mean(0)
    xi_centered, eta_centered = xi - xi.mean(), eta - eta.mean()
    moments = [
        sum(y * x for y, x in zip(centered[k:], xi_centered[:-k])) / (len(X) - k)
        for k in (1, 3)
    ]
    P = np.linalg.eigh(sum(s @ s.T for s in moments))[1][:, -3:]
    Q = np.linalg.eigh(sum(s.T @ s for s in moments))[1][:, -3:]
    Z = P.T @ centered @ Q
    H = [
        sum(z * x for z, x in zip(Z[k:], eta_centered[:-k])) / (len(X) - k)
        for k in (1, 2)
    ]
    eigenvalues = np.sort(
        np.linalg.eigvals(np.linalg.inv(H[0].T @ H[0]) @ H[0].T @ H[1]).real
    )
    result = fit_cp_factor(X, 3, lags=(1, 3), xi=xi, eta=eta)
    np.testing.assert_allclose(result.eigenvalues, eigenvalues, atol=1e-11)
    np.testing.assert_allclose(result.signal, X, atol=1e-11)


@pytest.mark.parametrize(
    "H1,H2,message",
    [
        (np.diag([1, 0]), np.eye(2), "singular"),
        (np.eye(2), np.diag([1, 0]), "singular"),
        (np.eye(2), 0.7 * np.eye(2), "not separated"),
        (np.eye(2), np.array([[0, -1], [1, 0]]), "non-real"),
        (np.diag([1, 1e-12]), np.eye(2), "ill-conditioned"),
    ],
)
def test_unidentified_moment_pencils_fail_explicitly(H1, H2, message):
    with pytest.raises(CPIdentificationError, match=message):
        _refined_loadings(H1, H2, 1e10, 1e-8)


def test_rank_one_and_degenerate_data():
    rng = np.random.default_rng(71)
    scores = np.sin(np.arange(40) / 3)
    X = np.einsum("t,i,j->tij", scores, rng.normal(size=4), rng.normal(size=3))
    result = fit_cp_factor(X, 1)
    np.testing.assert_allclose(result.signal, X, atol=1e-11)
    with pytest.raises(CPIdentificationError):
        fit_cp_factor(np.ones((40, 4, 3)), 1)
    with pytest.raises(CPIdentificationError):
        fit_cp_factor(X, 2)
    with pytest.raises(CPIdentificationError):
        fit_cp_factor(X, 1, xi=np.ones(40))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rank": 0},
        {"rank": 6},
        {"rank": True},
        {"rank": 1.5},
        {"lags": 200},
        {"lags": []},
        {"proxy_variance": 0},
        {"proxy_variance": 1.1},
        {"condition_limit": 1},
        {"eigenvalue_tol": 0},
        {"center": "yes"},
        {"xi": np.zeros(5)},
        {"eta": np.full(200, np.nan)},
        {"xi": np.ones(200) * 1j},
    ],
)
def test_validation(kwargs):
    X, _, _, _, _ = cp_fixture()
    with pytest.raises(ValueError):
        fit_cp_factor(X, **({"rank": 3} | kwargs))


def test_transform_validation():
    result = fit_cp_factor(cp_fixture()[0], 3)
    with pytest.raises(ValueError):
        result.transform(np.ones((2, 5, 6)))
    with pytest.raises(ValueError):
        result.inverse_transform(np.ones((2, 2)))


def test_default_proxy_is_average_of_pca_scores_at_variance_cutoff():
    X = np.diag([10.0, 5.0, 0.01, 0.001])[:, None, :]
    matrix = X.reshape(4, 4)
    matrix -= matrix.mean(axis=0)
    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)
    count = np.searchsorted(np.cumsum(s**2) / np.sum(s**2), 0.99) + 1
    for i in range(count):
        if Vt[i, abs(Vt[i]).argmax()] < 0:
            U[:, i] *= -1
    expected = (U[:, :count] * s[:count]).mean(axis=1)
    np.testing.assert_allclose(_pca_proxy(X, 0.99), expected, atol=1e-12)
