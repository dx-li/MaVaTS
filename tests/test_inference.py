"""Independent derivative/normal-equation oracles for published MAR inference."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi2

from mavats.autoregression import fit_mar
from mavats.inference import (
    _jacobian,
    _jacobian_batch,
    _operator_jacobian,
    mar_inference,
    mar_specification_test,
)
from mavats.simulation import simulate_mar


def _data(size=600, seed=47):
    return simulate_mar(
        size,
        [[0.7, 0.2], [-0.1, 0.5]],
        [[0.8, -0.15], [0.1, 0.4]],
        row_cov=[[1.0, 0.2], [0.2, 1.4]],
        column_cov=[[0.8, 0.1], [0.1, 1.0]],
        random_state=seed,
    )


def test_prediction_and_operator_jacobians_finite_difference_nonsquare():
    rng = np.random.default_rng(4)
    A, B, X = rng.normal(size=(2, 2)), rng.normal(size=(3, 3)), rng.normal(size=(2, 3))
    parameters = np.r_[A.ravel(order="F"), B.ravel(order="F")]
    observed, operator = [], []
    for e in np.eye(len(parameters)):
        plus, minus = parameters + 1e-5 * e, parameters - 1e-5 * e
        ap, bp = plus[:4].reshape(2, 2, order="F"), plus[4:].reshape(3, 3, order="F")
        am, bm = minus[:4].reshape(2, 2, order="F"), minus[4:].reshape(3, 3, order="F")
        observed.append(((ap @ X @ bp.T - am @ X @ bm.T) / 2e-5).ravel(order="F"))
        operator.append(((np.kron(bp, ap) - np.kron(bm, am)) / 2e-5).ravel(order="F"))
    assert_allclose(_jacobian(X, A, B), np.array(observed).T, atol=1e-9)
    assert_allclose(_operator_jacobian(A, B), np.array(operator).T, atol=1e-9)
    batch = np.stack([X, 2 * X, -3 * X])
    assert_allclose(
        _jacobian_batch(batch, A, B),
        np.stack([_jacobian(z, A, B) for z in batch]),
        atol=1e-13,
    )


@pytest.mark.parametrize("method", ["als", "mle"])
def test_covariance_matches_paper_vec_transpose_formula(method):
    x = _data()
    fit = fit_mar(x, method=method, tol=1e-11)
    answer = mar_inference(x, fit)
    a, b = fit.A, fit.B
    residuals = fit.residuals.transpose(0, 2, 1).reshape(len(x) - 1, -1)
    sigma = (
        residuals.T @ residuals / (len(x) - 1)
        if method == "als"
        else np.kron(fit.column_covariance, fit.row_covariance)
    )
    gamma = np.r_[a.ravel(order="F"), np.zeros(4)]
    gram, meat = np.zeros((8, 8)), np.zeros((8, 8))
    for z in x[:-1]:
        # Theorem 3 uses vec(B.T), deliberately unlike the public API.
        W = np.concatenate(
            [np.kron(b @ z.T, np.eye(2)), np.kron(np.eye(2), a @ z)], axis=1
        )
        if method == "als":
            gram += W.T @ W / (len(x) - 1)
            meat += W.T @ sigma @ W / (len(x) - 1)
        else:
            gram += W.T @ np.linalg.inv(sigma) @ W / (len(x) - 1)
    if method == "mle":
        meat = gram.copy()
    inv = np.linalg.inv(gram + np.outer(gamma, gamma))
    expected = inv @ meat @ inv / (len(x) - 1)
    permutation = [0, 1, 2, 3, 4, 6, 5, 7]
    assert_allclose(
        answer.parameter_covariance,
        expected[np.ix_(permutation, permutation)],
        rtol=1e-9,
        atol=1e-12,
    )
    assert_allclose(answer.parameter_covariance @ gamma, 0, atol=1e-12)


def test_projection_covariance_matches_independent_delta_derivative():
    x = _data()
    inference = mar_inference(x, method="projection")
    f = x.transpose(0, 2, 1).reshape(len(x), 4)
    phi = np.linalg.lstsq(f[:-1], f[1:], rcond=None)[0].T
    e = f[1:] - f[:-1] @ phi.T
    varcov = np.kron(np.linalg.inv(f[:-1].T @ f[:-1]), e.T @ e / (len(x) - 1))
    # Theorem 2 differentiates at the rank-one null, estimated by the
    # projected operator, not at the unrestricted noisy sample coefficient.
    phi = np.kron(inference.result.B, inference.result.A)

    def projected_parameters(p):
        # Independent block rearrangement of Phi into vec(A) vec(B).T.
        blocks = np.column_stack(
            [
                p[2 * j : 2 * j + 2, 2 * k : 2 * k + 2].ravel(order="F")
                for k in range(2)
                for j in range(2)
            ]
        )
        u, s, vh = np.linalg.svd(blocks, full_matrices=False)
        aa = u[:, 0]
        bb = s[0] * vh[0]
        sign = np.sign(aa @ inference.result.A.ravel(order="F"))
        return np.r_[aa, bb] * sign

    derivative = []
    for unit in np.eye(16):
        direction = unit.reshape(4, 4, order="F") * 1e-5
        derivative.append(
            (
                projected_parameters(phi + direction)
                - projected_parameters(phi - direction)
            )
            / 2e-5
        )
    derivative = np.column_stack(derivative)
    assert_allclose(
        inference.parameter_covariance,
        derivative @ varcov @ derivative.T,
        atol=1e-10,
        rtol=1e-7,
    )


@pytest.mark.parametrize("method", ["projection", "als", "mle"])
@pytest.mark.parametrize("scale", [1e-150, 1e150])
def test_inference_scale_invariance(method, scale):
    x = _data()
    original = mar_inference(x, method=method)
    scaled = mar_inference(x * scale, method=method)
    assert_allclose(
        scaled.parameter_covariance,
        original.parameter_covariance,
        rtol=1e-7,
        atol=1e-12,
    )
    assert_allclose(
        scaled.operator_standard_errors, original.operator_standard_errors, rtol=1e-7
    )


def test_operator_standard_errors_and_intervals_are_identification_consistent():
    result = mar_inference(_data())
    covariance = result.operator_covariance()
    assert_allclose(
        result.operator_standard_errors.ravel(order="F") ** 2,
        np.diag(covariance),
        atol=1e-15,
    )
    for target in ("operator", "left", "right"):
        lower, upper = result.confidence_interval(target=target)
        assert np.all(upper >= lower)
    lower, upper = result.confidence_interval(level=0.95)
    assert_allclose((lower + upper) / 2, result.operator)
    with pytest.raises(ValueError):
        result.confidence_interval(level=1.0)


def test_fixed_sign_anchor_preserves_operator_and_does_not_mutate_fit():
    x = simulate_mar(
        1000, np.diag([1.0, -1.0]) / np.sqrt(2), np.eye(2) * 0.5, random_state=8
    )
    fit = fit_mar(x)
    original = fit.A.copy(), fit.B.copy()
    first = mar_inference(x, fit, sign_anchor=(0, 0))
    second = mar_inference(x, fit, sign_anchor=(1, 1))
    assert first.result.A[0, 0] > 0 and second.result.A[1, 1] > 0
    assert_allclose(first.result.A, -second.result.A)
    assert_allclose(first.operator, second.operator)
    assert_allclose(
        first.operator_standard_errors, second.operator_standard_errors, atol=1e-12
    )
    assert_allclose(fit.A, original[0])
    assert_allclose(fit.B, original[1])
    with pytest.raises(ValueError, match="sign_anchor"):
        mar_inference(x, sign_anchor=(2, 2))


def test_specification_matches_literal_pseudoinverse_equation():
    x = _data()
    result = mar_specification_test(x)
    T = len(x)
    f = x.transpose(0, 2, 1).reshape(T, 4)
    p = np.linalg.lstsq(f[:-1], f[1:], rcond=None)[0].T
    e = f[1:] - f[:-1] @ p.T
    gamma = f[:-1].T @ f[:-1] / (T - 1)
    xi = np.kron(np.linalg.inv(gamma), e.T @ e / (T - 1))
    indices = np.arange(16).reshape(4, 4, order="F")

    def rearrange(matrix):
        return np.column_stack(
            [
                matrix[2 * j : 2 * j + 2, 2 * k : 2 * k + 2].ravel(order="F")
                for k in range(2)
                for j in range(2)
            ]
        )

    idx = rearrange(indices).ravel(order="F")
    xi = xi[np.ix_(idx, idx)]
    a, b = result.projection_left.ravel(order="F"), result.projection_right.ravel(
        order="F"
    )
    P = np.kron(np.eye(4) - np.outer(b, b) / np.dot(b, b), np.eye(4) - np.outer(a, a))
    residual = (rearrange(p) - np.outer(a, b)).ravel(order="F")
    statistic = (T - 1) * residual @ np.linalg.pinv(P @ xi @ P) @ residual
    assert_allclose(result.statistic, statistic, rtol=1e-10)
    assert_allclose(result.pvalue, chi2.sf(statistic, 9), rtol=1e-10)
    assert_allclose(
        np.linalg.norm(p - np.kron(result.projection_right, result.projection_left)),
        np.linalg.norm(rearrange(p) - np.outer(a, b)),
    )


def test_specification_is_scale_invariant_and_rejects_nonkronecker_alternative():
    x = _data()
    assert_allclose(
        mar_specification_test(x * 1e150).statistic,
        mar_specification_test(x).statistic,
        rtol=1e-10,
    )
    rng = np.random.default_rng(33)
    phi = np.diag([0.8, 0.15, -0.7, 0.45])
    f = np.zeros((2200, 4))
    for i in range(1, len(f)):
        f[i] = phi @ f[i - 1] + rng.normal(size=4)
    alternative = f[200:].reshape(-1, 2, 2).transpose(0, 2, 1)
    assert mar_specification_test(alternative).pvalue < 1e-8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"order": 2},
        {"ridge": 1.0},
        {"ranks": (1, 1)},
        {"fit_intercept": True},
        {"max_iter": 1, "tol": 0},
    ],
)
def test_incompatible_fits_are_rejected(kwargs):
    x = _data()
    with pytest.raises(ValueError):
        mar_inference(x, fit_mar(x, **kwargs))


def test_invalid_data_guards_and_fit_mismatch():
    x = _data()
    with pytest.raises(ValueError, match="observations"):
        mar_inference(x[::-1], fit_mar(x))
    with pytest.raises(ValueError, match="max_dimension"):
        mar_inference(x, max_dimension=3)
    with pytest.raises(ValueError):
        mar_specification_test(np.ones((5, 1, 2)))
    with pytest.raises(ValueError):
        mar_specification_test(np.zeros((40, 2, 2)))
