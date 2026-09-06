import numpy as np
import pytest

from mavats.factors import _lagged_moment, fit_lagged_factor
from mavats.tensor import fit_tensor_factor


def data_fixture(noise=0):
    rng = np.random.default_rng(197)
    loadings = [np.linalg.qr(rng.normal(size=(d, 2)))[0] for d in (5, 4, 3)]
    F = rng.normal(size=(150, 2, 2, 2))
    for t in range(1, len(F)):
        F[t] += 0.8 * F[t - 1]
    signal = np.einsum("tabc,ia,jb,kc->tijk", F, *loadings)
    return signal + noise * rng.normal(size=signal.shape), signal, loadings


@pytest.mark.parametrize("method", ["topup", "tipup"])
@pytest.mark.parametrize("iterative", [False, True])
def test_tensor_recovers_noiseless_spaces_and_signal(method, iterative):
    X, signal, truth = data_fixture()
    result = fit_tensor_factor(X, (2, 2, 2), method=method, iterative=iterative, lags=2)
    for Q, V in zip(result.loadings, truth):
        np.testing.assert_allclose(Q @ Q.T, V @ V.T, atol=1e-11)
    np.testing.assert_allclose(result.signal, signal, atol=1e-11)
    np.testing.assert_allclose(
        result.inverse_transform(result.transform(X)), signal, atol=1e-11
    )
    assert result.converged


@pytest.mark.parametrize("method", ["topup", "tipup"])
def test_tensor_moments_match_explicit_outer_or_inner_products(method):
    rng = np.random.default_rng(6)
    unfolded = rng.normal(size=(9, 3, 4))
    moment = np.zeros((3, 3))
    for h in (1, 3):
        if method == "topup":
            outer = sum(
                np.multiply.outer(a, b) for a, b in zip(unfolded[:-h], unfolded[h:])
            ) / (9 - h)
            matrix = outer.reshape(3, -1)
        else:
            matrix = sum(a @ b.T for a, b in zip(unfolded[:-h], unfolded[h:])) / (9 - h)
        moment += matrix @ matrix.T
    np.testing.assert_allclose(
        _lagged_moment(unfolded, (1, 3), method), moment, atol=1e-12
    )


def test_matrix_topup_agrees_with_wang_liu_chen_estimator():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(40, 5, 4))
    tensor = fit_tensor_factor(X, (2, 2), method="topup", lags=2)
    matrix = fit_lagged_factor(X, (2, 2), lags=2)
    np.testing.assert_allclose(tensor.signal, matrix.signal, atol=1e-12)


@pytest.mark.parametrize("method", ["topup", "tipup"])
def test_iteration_matches_independent_sequential_mode_updates(method):
    X, _, _ = data_fixture(noise=0.4)
    initial = fit_tensor_factor(X, (2, 2, 2), method=method)
    Q = list(initial.loadings)
    for k in range(3):
        projected = X.copy()
        for j in range(3):
            if j != k:
                projected = np.moveaxis(
                    np.tensordot(projected, Q[j], axes=(j + 1, 0)), -1, j + 1
                )
        unfolded = np.moveaxis(projected, k + 1, 1).reshape(len(X), X.shape[k + 1], -1)
        if method == "topup":
            cross = sum(
                np.multiply.outer(a, b) for a, b in zip(unfolded[:-1], unfolded[1:])
            )
            matrix = cross.reshape(X.shape[k + 1], -1)
        else:
            matrix = sum(a @ b.T for a, b in zip(unfolded[:-1], unfolded[1:]))
        Q[k] = np.linalg.svd(matrix, full_matrices=False)[0][:, :2]
    result = fit_tensor_factor(X, (2, 2, 2), method=method, iterative=True, max_iter=1)
    for actual, expected in zip(result.loadings, Q):
        np.testing.assert_allclose(actual @ actual.T, expected @ expected.T, atol=1e-11)
    assert result.n_iter == 1
    assert not result.converged


def test_tensor_auto_rank_and_centered_full_rank_roundtrip():
    X, _, _ = data_fixture()
    result = fit_tensor_factor(X)
    assert result.ranks == (2, 2, 1)  # Default ER searches at most half each dimension.
    X += np.arange(60).reshape(5, 4, 3)
    result = fit_tensor_factor(X, (5, 4, 3), center=True)
    np.testing.assert_allclose(result.signal, X, atol=1e-11)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "unknown"},
        {"iterative": 1},
        {"lags": 150},
        {"max_iter": 0},
        {"tol": 0},
        {"ranks": (2, 2)},
    ],
)
def test_tensor_validation(kwargs):
    with pytest.raises(ValueError):
        fit_tensor_factor(data_fixture()[0], **kwargs)
