"""TenAR blocks checked against independent dense and entrywise algebra."""

from dataclasses import replace
from itertools import product
from math import prod

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import mavats.tensor_autoregression as tenar
from mavats.autoregression import fit_mar
from mavats.tensor_autoregression import fit_tensor_ar


def kron(matrices):
    value = np.ones((1, 1))
    for matrix in reversed(matrices):
        value = np.kron(value, matrix)
    return value


def flat(X):
    return np.stack([x.ravel(order="F") for x in X])


def spatial(rows, shape):
    return np.stack([row.reshape(shape, order="F") for row in rows])


def parameters(shape=(2, 3, 2), terms=(2, 1), seed=79):
    rng = np.random.default_rng(seed)
    result = []
    for count in terms:
        lag = []
        for _ in range(count):
            term = [rng.normal(size=(d, d)) + np.eye(d) for d in shape]
            term = [a / np.linalg.norm(a, ord=2) for a in term]
            term[-1] *= 0.65 / sum(terms)
            lag.append(term)
        result.append(lag)
    return result


def covariance_factors(shape=(2, 3, 2)):
    return [
        0.45 ** np.abs(np.arange(d)[:, None] - np.arange(d)[None, :]) for d in shape
    ]


def data(n=180, shape=(2, 3, 2), terms=(2, 1), lags=(1, 2), seed=72):
    matrices = parameters(shape, terms)
    transitions = [sum(kron(term) for term in lag) for lag in matrices]
    covariances = covariance_factors(shape)
    rng = np.random.default_rng(seed)
    order = max(lags)
    values = np.zeros((n + 100 + order, prod(shape)))
    noise = (
        rng.normal(size=(n + 100, prod(shape)))
        @ np.linalg.cholesky(kron(covariances)).T
    )
    for t in range(order, len(values)):
        values[t] = (
            sum(phi @ values[t - lag] for lag, phi in zip(lags, transitions))
            + noise[t - order]
        )
    return spatial(values[-n:], shape), matrices, covariances


def entrywise_apply(X, matrices):
    shape = X.shape[1:]
    result = np.zeros_like(X)
    for output in product(*(range(d) for d in shape)):
        for source in product(*(range(d) for d in shape)):
            weight = prod(a[i, j] for a, i, j in zip(matrices, output, source))
            result[(slice(None), *output)] += X[(slice(None), *source)] * weight
    return result


def dense_block_design(predictor, mode):
    """Each column varies one mode coefficient, using explicit tensor entries."""
    shape = predictor.shape[1:]
    d, dimension = shape[mode], prod(shape)
    design = np.zeros((len(predictor) * dimension, d * d))
    for output in product(*(range(s) for s in shape)):
        row = np.ravel_multi_index(output, shape, order="F")
        for source in range(d):
            index = list(output)
            index[mode] = source
            column = output[mode] + d * source
            design[row::dimension, column] = predictor[(slice(None), *index)]
    return design


def test_nonsquare_three_mode_contraction_and_rearrangement_entrywise_oracle():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(5, 2, 3, 2))
    matrices = parameters()[0][0]
    expected = entrywise_apply(X, matrices)
    assert_allclose(tenar._apply(X, matrices), expected, atol=2e-16)
    assert_allclose(flat(expected), flat(X) @ kron(matrices).T, atol=2e-16)
    rearranged = tenar._rearrange(kron(matrices), X.shape[1:])
    expected_tensor = np.einsum("i,j,k->ijk", *(a.ravel(order="F") for a in matrices))
    assert_allclose(rearranged, expected_tensor, rtol=3e-16, atol=0)
    for mode in range(3):
        indices = [k for k in range(3) if k != mode]
        expected_unfold = []
        for x in X:
            one = np.empty((X.shape[mode + 1], prod(X.shape[i + 1] for i in indices)))
            for index in product(*(range(d) for d in X.shape[1:])):
                column = np.ravel_multi_index(
                    tuple(index[i] for i in indices),
                    tuple(X.shape[i + 1] for i in indices),
                    order="F",
                )
                one[index[mode], column] = x[index]
            expected_unfold.append(one)
        assert_array_equal(tenar._unfold(X, mode), expected_unfold)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("weighted", [False, True])
def test_conditional_coefficient_blocks_match_independent_dense_ls_gls(mode, weighted):
    rng = np.random.default_rng(401)
    response = rng.normal(size=(9, 2, 3, 2))
    predictor = rng.normal(size=response.shape)
    covariances = covariance_factors()
    design = dense_block_design(predictor, mode)
    y = flat(response).ravel()
    if weighted:
        precision = np.kron(np.eye(len(response)), np.linalg.inv(kron(covariances)))
        expected = np.linalg.solve(
            design.T @ precision @ design, design.T @ precision @ y
        )
    else:
        expected = np.linalg.lstsq(design, y, rcond=None)[0]
    actual, condition = tenar._coefficient_block(
        response, predictor, mode, covariances if weighted else None
    )
    assert condition >= 1
    assert_allclose(actual.ravel(order="F"), expected, rtol=4e-13, atol=2e-15)


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_covariance_block_matches_independent_kronecker_precision_and_score(mode):
    residuals = np.random.default_rng(13).normal(size=(11, 2, 3, 2))
    covariances = covariance_factors()
    other = [c for k, c in enumerate(covariances) if k != mode]
    precision = np.linalg.inv(kron(other))
    expected = np.zeros_like(covariances[mode])
    for x in residuals:
        matrix = np.moveaxis(x, mode, 0).reshape(len(expected), -1, order="F")
        expected += matrix @ precision @ matrix.T
    expected /= len(residuals) * prod(residuals.shape[1:]) / len(expected)
    actual, regularized = tenar._covariance_block(residuals, covariances, mode)
    assert not regularized
    assert_allclose(actual, expected, rtol=1e-13, atol=2e-15)
    covariances[mode] = actual
    baseline = dense_nll(residuals, covariances)
    for i in range(len(actual)):
        varied = [c.copy() for c in covariances]
        varied[mode][i, i] += 1e-5
        assert dense_nll(residuals, varied) > baseline


def dense_nll(residuals, covariances):
    covariance = kron(covariances)
    vectors = flat(residuals)
    return 0.5 * (
        len(vectors)
        * (vectors.shape[1] * np.log(2 * np.pi) + np.linalg.slogdet(covariance)[1])
        + np.sum(vectors * np.linalg.solve(covariance, vectors.T).T)
    )


def test_likelihood_and_normalization_match_dense_gaussian_covariance():
    residuals = np.random.default_rng(33).normal(size=(12, 2, 3, 2))
    covariances = covariance_factors()
    assert_allclose(
        tenar._negative_loglike(residuals, covariances),
        dense_nll(residuals, covariances),
        rtol=2e-15,
    )
    normalized = tenar._normalize_covariances(covariances)
    assert_allclose(kron(normalized), kron(covariances), atol=2e-16)
    assert_allclose([np.linalg.norm(c) for c in normalized[:-1]], 1)


@pytest.mark.parametrize("method", ["ls", "mle"])
def test_complete_multiterm_multilag_sweep_matches_dense_conditional_oracle(method):
    X = np.random.default_rng(801).normal(size=(14, 2, 3, 2))
    shape, lags = X.shape[1:], (1, 3)
    expected = parameters()
    initial = tenar._copy_coefficients(expected)
    # The public canonical term order is decreasing product Frobenius norm.
    for lag in expected:
        lag.sort(key=lambda term: prod(np.linalg.norm(a) for a in term), reverse=True)
    covariance = covariance_factors()
    target, predictors = flat(X[3:]), [X[3 - lag : len(X) - lag] for lag in lags]
    precision = np.kron(np.eye(len(target)), np.linalg.inv(kron(covariance)))
    for i, lag in enumerate(expected):
        for r, term in enumerate(lag):
            for mode, d in enumerate(shape):
                partial = target.copy()
                for j, otherlag in enumerate(expected):
                    for s, otherterm in enumerate(otherlag):
                        if (i, r) != (j, s):
                            partial -= flat(predictors[j]) @ kron(otherterm).T
                other = [np.eye(d) if k == mode else a for k, a in enumerate(term)]
                predictor = spatial(flat(predictors[i]) @ kron(other).T, shape)
                design = dense_block_design(predictor, mode)
                if method == "mle":
                    coefficient = np.linalg.solve(
                        design.T @ precision @ design,
                        design.T @ precision @ partial.ravel(),
                    )
                else:
                    coefficient = np.linalg.lstsq(design, partial.ravel(), rcond=None)[
                        0
                    ]
                term[mode] = coefficient.reshape((d, d), order="F")
    transitions = np.array([sum(kron(term) for term in lag) for lag in expected])
    residual_vectors = target - sum(
        flat(z) @ p.T for z, p in zip(predictors, transitions)
    )
    residuals = spatial(residual_vectors, shape)
    options = dict(covariance_initial=covariance) if method == "mle" else {}
    actual = fit_tensor_ar(
        X,
        terms=(2, 1),
        lags=lags,
        initial=initial,
        method=method,
        max_iter=1,
        **options,
    )
    assert actual.n_iter == 1
    assert_allclose(actual.transition_matrices(), transitions, rtol=4e-12, atol=3e-14)
    assert_allclose(actual.residuals, residuals, rtol=4e-12, atol=2e-14)
    if method == "mle":
        for mode in range(len(shape) - 1, -1, -1):
            inverse = np.linalg.inv(
                kron([c for k, c in enumerate(covariance) if k != mode])
            )
            scatter = np.zeros((shape[mode], shape[mode]))
            for e in residuals:
                matrix = np.moveaxis(e, mode, 0).reshape(shape[mode], -1, order="F")
                scatter += matrix @ inverse @ matrix.T
            covariance[mode] = scatter / (len(residuals) * prod(shape) / shape[mode])
        assert_allclose(
            actual.innovation_covariance(), kron(covariance), rtol=4e-12, atol=3e-14
        )


def test_hierarchical_covariance_projection_recovers_separable_population_moment():
    shape = (2, 3, 2)
    covariance = kron(covariance_factors(shape))
    # These twelve vectors have exactly the prescribed uncentered second moment.
    residuals = np.sqrt(len(covariance)) * np.linalg.cholesky(covariance).T
    factors, active = tenar._covariance_projection(residuals, shape, 0)
    assert not active
    assert_allclose(kron(factors), covariance, atol=2e-15)


@pytest.mark.parametrize("method", ["ls", "mle"])
def test_matrix_single_term_one_sweep_matches_existing_mar(method):
    X, coefficients, covariance = data(n=70, shape=(2, 3), terms=(1,), lags=(1,))
    initial = coefficients[0][0]
    kwargs = {"covariance_initial": covariance} if method == "mle" else {}
    tensor = fit_tensor_ar(X, initial=coefficients, method=method, max_iter=1, **kwargs)
    kwargs = {"initial_covariance": covariance} if method == "mle" else {}
    matrix = fit_mar(
        X,
        initial=initial,
        method="als" if method == "ls" else method,
        max_iter=1,
        covariance_floor=0,
        **kwargs,
    )
    assert_allclose(tensor.transition_matrices(), matrix.coefficients, atol=2e-15)
    assert_allclose(tensor.residuals, matrix.residuals, atol=3e-15)
    assert_allclose(tensor.forecast(3), matrix.forecast(3), atol=3e-15)
    if method == "mle":
        assert_allclose(
            tensor.innovation_covariance(),
            np.kron(matrix.column_covariance, matrix.row_covariance),
            atol=3e-15,
        )
        assert_allclose(tensor.log_likelihood, matrix.log_likelihood, atol=2e-13)


def test_matrix_projection_matches_exact_mar_nearest_kronecker():
    X, _, _ = data(n=70, shape=(2, 3), terms=(1,), lags=(1,))
    tensor = fit_tensor_ar(X, method="projection")
    matrix = fit_mar(X, method="projection")
    assert_allclose(tensor.transition_matrices(), matrix.coefficients, atol=2e-15)
    assert tensor.runs[0].projection[0].method == "matrix_svd"
    assert tensor.converged


def test_matrix_multiterm_canonicalization_is_exact_and_orthogonal():
    original = parameters(shape=(3, 4), terms=(3,), seed=139)
    canonical = tenar._canonicalize(original)
    assert_allclose(
        sum(kron(t) for t in canonical[0]),
        sum(kron(t) for t in original[0]),
        atol=2e-16,
    )
    for mode in range(2):
        vectors = np.column_stack([t[mode].ravel() for t in canonical[0]])
        gram = vectors.T @ vectors
        assert_allclose(gram, np.diag(np.diag(gram)), atol=2e-15)
    assert_allclose([np.linalg.norm(t[0]) for t in canonical[0]], 1)
    norms = [np.linalg.norm(t[1]) for t in canonical[0]]
    assert norms == sorted(norms, reverse=True)


def test_repeated_matrix_singular_values_are_not_claimed_identified():
    e0, e1 = np.diag([1.0, 0]), np.diag([0.0, 1])
    canonical = tenar._canonicalize([[[e0, e0], [e1, e1]]])
    diagnostic = tenar._coefficient_diagnostics(canonical)[0]
    assert diagnostic["identification"] == "unseparated_matrix_singular_values"


def test_local_cp_projection_recovers_a_rank_one_tensor_with_diagnostics():
    matrices = parameters(terms=(1,))[0][0]
    tensor = tenar._rearrange(kron(matrices), (2, 3, 2))
    factors, diagnostic = tenar._cp_project(
        tensor, 1, np.random.default_rng(0), 30, 1e-12
    )
    assert_allclose(tenar._cp_tensor(factors), tensor, atol=1e-16)
    assert diagnostic.converged
    assert diagnostic.method == "local_cp_als"
    assert np.all(np.diff(diagnostic.objective_history) <= 1e-14)


@pytest.mark.parametrize("method", ["ls", "mle"])
def test_nonsquare_multilag_multiterm_fit_is_dense_prediction_and_likelihood_consistent(
    method,
):
    X, _, _ = data(n=180)
    fitted = fit_tensor_ar(
        X, terms=(2, 1), method=method, n_starts=2, max_iter=25, projection_max_iter=40
    )
    transitions = fitted.transition_matrices()
    expected = flat(X[1:-1]) @ transitions[0].T + flat(X[:-2]) @ transitions[1].T
    assert_allclose(flat(fitted.fitted_values), expected, rtol=3e-13, atol=3e-15)
    assert_allclose(fitted.fitted_values + fitted.residuals, X[2:], atol=1e-15)
    assert len(fitted.runs) == 2
    assert fitted.objective <= fitted.runs[0].objective + 1e-12
    for run in fitted.runs:
        assert np.all(np.diff(run.objective_history) <= 1e-8)
    assert fitted.order == 2
    if method == "mle":
        assert_allclose(
            fitted.log_likelihood,
            -dense_nll(fitted.residuals, fitted.covariance_factors),
            atol=2e-12,
        )
        assert not fitted.covariance_regularized


def test_gapped_lags_forecast_and_companion_use_every_complete_transition():
    X, initial, _ = data(n=90, terms=(2, 1), lags=(1, 3))
    fitted = fit_tensor_ar(X, terms=(2, 1), lags=(1, 3), initial=initial, max_iter=3)
    transitions = fitted.transition_matrices()
    d = prod(X.shape[1:])
    expected_companion = np.zeros((3 * d, 3 * d))
    expected_companion[:d, :d] = transitions[0]
    expected_companion[:d, 2 * d :] = transitions[1]
    expected_companion[d:, :-d] = np.eye(2 * d)
    assert_array_equal(fitted.companion_matrix(), expected_companion)
    assert_allclose(
        fitted.spectral_radius, np.max(abs(np.linalg.eigvals(expected_companion)))
    )
    state = np.concatenate([X[-i].ravel(order="F") for i in range(1, 4)])
    expected = []
    for _ in range(4):
        state = expected_companion @ state
        expected.append(state[:d].reshape(X.shape[1:], order="F"))
    assert_allclose(fitted.forecast(4), expected, atol=2e-15)
    assert_allclose(fitted.forecast(4, history=X[-3:]), expected, atol=2e-15)
    with pytest.raises(ValueError):
        fitted.forecast(1, history=X[-2:])


def test_stability_uses_sum_not_individual_term_radii_and_has_dense_guard():
    X, _, _ = data(n=40, shape=(2, 2), terms=(1,), lags=(1,))
    fit = fit_tensor_ar(X, max_iter=1)
    term = (np.eye(2), 0.6 * np.eye(2))
    altered = replace(fit, coefficients=((term, term),), terms=(2,))
    assert_allclose(altered.spectral_radius, 1.2)
    assert not altered.is_stable
    with pytest.raises(ValueError, match="companion"):
        replace(altered, max_dense_dimension=3).spectral_radius
    with pytest.raises(ValueError, match="transition"):
        altered.transition_matrices(max_dimension=3)


def test_random_initialization_avoids_dense_projection_for_large_modes():
    X = np.random.default_rng(51).normal(size=(12, 3, 4, 3))
    with pytest.raises(ValueError, match="dense VAR projection"):
        fit_tensor_ar(X, max_dense_dimension=8, max_iter=1)
    fitted = fit_tensor_ar(X, init="random", max_dense_dimension=8, max_iter=2)
    assert fitted.forecast(1).shape == (1, 3, 4, 3)
    assert np.isfinite(fitted.spectral_radius)  # one-term, one-lag exact shortcut
    with pytest.raises(ValueError):
        fitted.transition_matrices()


@pytest.mark.parametrize("factor", [1e-100, 1e100, -3.0])
@pytest.mark.parametrize("method", ["ls", "mle"])
def test_data_unit_changes_preserve_coefficients_and_map_covariance_and_forecasts(
    factor, method
):
    X, initial, covariances = data(n=70, terms=(1,), lags=(1,))
    options = dict(method=method, initial=initial, max_iter=4)
    base = fit_tensor_ar(X, **options)
    scaled = fit_tensor_ar(X * factor, **options)
    assert_allclose(
        scaled.transition_matrices(), base.transition_matrices(), atol=2e-13
    )
    assert_allclose(scaled.forecast(2) / factor, base.forecast(2), atol=2e-13)
    assert_allclose(
        scaled.objective_history, base.objective_history, rtol=2e-13, atol=2e-12
    )
    if method == "mle":
        assert_allclose(
            scaled.innovation_covariance() / factor / factor,
            base.innovation_covariance(),
            rtol=2e-13,
            atol=2e-14,
        )
        assert_allclose(
            scaled.log_likelihood,
            base.log_likelihood - base.residuals.size * np.log(abs(factor)),
            atol=1e-10,
        )


def test_centering_is_fixed_mean_extension_and_has_no_future_information():
    X, initial, _ = data(n=90, terms=(1,), lags=(1,))
    offset = np.arange(12).reshape(2, 3, 2) * 2.0
    a = fit_tensor_ar(X, initial=initial, center=True, max_iter=3)
    b = fit_tensor_ar(X + offset, initial=initial, center=True, max_iter=3)
    assert_allclose(b.mean, X.mean(axis=0) + offset, atol=3e-15)
    assert_allclose(b.transition_matrices(), a.transition_matrices(), atol=2e-14)
    assert_allclose(b.forecast(2), a.forecast(2) + offset, atol=8e-15)
    assert_allclose(b.residuals, a.residuals, atol=2e-14)


@pytest.mark.parametrize("method", ["ls", "mle"])
def test_centered_large_offset_does_not_cause_premature_objective_convergence(method):
    X, initial, _ = data(n=90, terms=(1,), lags=(1,))
    base = fit_tensor_ar(X, initial=initial, center=True, method=method)
    offset = fit_tensor_ar(X + 1e8, initial=initial, center=True, method=method)
    assert base.converged and offset.converged
    assert offset.n_iter == base.n_iter
    assert_allclose(offset.transition_matrices(), base.transition_matrices(), atol=2e-8)


def test_known_nonsquare_multiterm_transition_recovery_without_true_initialization():
    X, truth, _ = data(n=2000)
    fitted = fit_tensor_ar(X, terms=(2, 1), method="mle", n_starts=2, max_iter=100)
    expected = np.array([sum(kron(term) for term in lag) for lag in truth])
    assert fitted.converged
    assert np.linalg.norm(fitted.transition_matrices() - expected) < 0.35
    assert fitted.is_stable


def test_failed_sweep_retains_last_complete_objective_and_reports_failure(monkeypatch):
    X, initial, _ = data(n=40, terms=(1,), lags=(1,))

    def fail(*args, **kwargs):
        raise ValueError("independent injected block failure")

    monkeypatch.setattr(tenar, "_coefficient_block", fail)
    fitted = fit_tensor_ar(X, initial=initial, n_starts=1)
    assert not fitted.converged
    assert fitted.n_iter == 0
    assert all(
        run.status == "failed" and "injected" in run.message for run in fitted.runs
    )
    expected = sum(kron(term) for term in initial[0])
    assert_allclose(fitted.transition_matrices()[0], expected, atol=2e-16)


def test_tiny_retained_predictions_are_not_lost_by_response_minus_residual(monkeypatch):
    X = np.random.default_rng(3).normal(size=(10, 2, 2))

    def fail(*args, **kwargs):
        raise ValueError("retain initial")

    monkeypatch.setattr(tenar, "_coefficient_block", fail)
    fit = fit_tensor_ar(X, initial=[[[np.eye(2), np.eye(2) * 1e-200]]])
    assert_allclose(fit.fitted_values, X[:-1] * 1e-200, rtol=1e-15, atol=0)


def test_forecast_finite_nearmax_mean_uses_maxabs_not_frobenius_norm():
    X, _, _ = data(n=40, shape=(2, 2), terms=(1,), lags=(1,))
    fit = fit_tensor_ar(X, max_iter=1)
    mean = np.full((2, 2), 1e308)
    constant = replace(fit, mean=mean, _history=mean[None])
    assert_array_equal(constant.forecast(2), np.stack([mean, mean]))


def test_single_term_stability_avoids_intermediate_product_underflow():
    X, _, _ = data(n=40, terms=(1,), lags=(1,))
    fit = fit_tensor_ar(X, max_iter=1)
    # Unit-Frobenius nearly nilpotent modes can have arbitrarily tiny radii.
    first = np.array([[1e-200, 1.0], [0.0, 1e-200]])
    second = np.diag(np.full(3, 1e-200))
    second[0, 1] = 1.0
    third = np.eye(2) * 1e300
    unusual = replace(fit, coefficients=(((first, second, third),),))
    assert_allclose(unusual.spectral_radius, 1e-100, rtol=2e-13, atol=0)


@pytest.mark.parametrize(
    "scales", [(1e-200, 1e200, 1e200, 1e-200), (1e200, 1e-200, 1e-200, 1e200)]
)
def test_compensating_four_mode_gauges_preserve_initial_model_and_covariance(scales):
    X = np.random.default_rng(72).normal(size=(40, 2, 2, 2, 2))
    base_term = [np.eye(2) for _ in scales]
    base_term[-1] *= 0.2
    term = [matrix * scale for matrix, scale in zip(base_term, scales)]
    normalized = tenar._normalize_term(term)
    assert_allclose(kron(normalized), 0.2 * np.eye(16), atol=2e-16)
    normalized_cov = tenar._normalize_covariances(
        [np.eye(2) * scale for scale in scales]
    )
    assert_allclose(kron(normalized_cov), np.eye(16), atol=1e-15)
    base = fit_tensor_ar(
        X,
        initial=[[base_term]],
        method="mle",
        covariance_initial=[np.eye(2)] * 4,
        max_iter=1,
    )
    scaled = fit_tensor_ar(
        X,
        initial=[[term]],
        method="mle",
        covariance_initial=[np.eye(2) * scale for scale in scales],
        max_iter=1,
    )
    assert scaled.n_iter == 1
    assert_allclose(
        scaled.transition_matrices(), base.transition_matrices(), atol=2e-14
    )
    assert_allclose(
        scaled.innovation_covariance(), base.innovation_covariance(), atol=2e-14
    )


@pytest.mark.parametrize("scale", [1.0, 1e-100, 1e-200, 1e200])
def test_component_cancellation_diagnostics_are_amplitude_invariant(scale):
    one = tenar._canonicalize([[[np.eye(2), np.eye(2) * scale]]])
    diagnostic = tenar._coefficient_diagnostics(one)[0]
    assert_allclose(diagnostic["cancellation_ratio"], 1, atol=4e-16)
    assert diagnostic["identification"] == "single_nonzero_term"
    assert diagnostic["mode_column_ranks"] == (1, 1)
    cancelling = tenar._canonicalize(
        [
            [
                [np.eye(2), np.eye(2), np.eye(2) * scale],
                [np.eye(2), np.eye(2), -0.999 * np.eye(2) * scale],
            ]
        ]
    )
    diagnostic = tenar._coefficient_diagnostics(cancelling)[0]
    assert_allclose(diagnostic["cancellation_ratio"], 1999, rtol=2e-9)


def test_singular_covariance_blocks_require_explicit_regularization():
    residuals = np.ones((4, 2, 3, 2))
    covariances = [np.eye(d) for d in residuals.shape[1:]]
    with pytest.raises(ValueError, match="rank deficient"):
        tenar._covariance_block(residuals, covariances, 1)
    covariance, active = tenar._covariance_block(residuals, covariances, 1, floor=1e-6)
    assert active
    assert np.linalg.eigvalsh(covariance)[0] > 0
    with pytest.raises(ValueError, match="rank deficient"):
        tenar._coefficient_block(residuals, residuals, 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"terms": ()},
        {"terms": (0,)},
        {"terms": (True,)},
        {"terms": 1},
        {"terms": (1, 1), "lags": (2, 1)},
        {"terms": (1, 1), "lags": (1, 1)},
        {"lags": (1, 2)},
        {"lags": (100,)},
        {"method": "tucker"},
        {"init": "auto"},
        {"n_starts": 0},
        {"max_iter": 0},
        {"max_dense_dimension": 0},
        {"tol": 0},
        {"projection_tol": 0},
        {"center": 1},
        {"covariance_floor": 1e-6},
        {"method": "mle", "covariance_floor": 1},
        {"initial": np.eye(2)},
        {"method": "projection", "init": "random"},
    ],
)
def test_invalid_fit_options(kwargs):
    X = np.random.default_rng(9).normal(size=(30, 2, 3, 2))
    with pytest.raises(ValueError):
        fit_tensor_ar(X, **kwargs)


def test_invalid_data_initial_covariance_and_seeded_input_preservation():
    with pytest.raises(ValueError):
        fit_tensor_ar(np.ones((20, 2)))
    with pytest.raises(ValueError):
        fit_tensor_ar(np.zeros((20, 2, 3)))
    with pytest.raises(ValueError):
        fit_tensor_ar(np.ones((20, 2, 3)), center=True)
    with pytest.raises(ValueError):
        fit_tensor_ar(np.ones((20, 2, 3)), terms=(5,))
    X, initial, _ = data(n=40, terms=(1,), lags=(1,))
    original = X.copy()
    saved = tenar._copy_coefficients(initial)
    with pytest.raises(ValueError):
        fit_tensor_ar(
            X,
            initial=initial,
            method="mle",
            covariance_initial=[np.eye(2), -np.eye(3), np.eye(2)],
        )
    a = fit_tensor_ar(X, init="random", n_starts=2, max_iter=2, random_state=22)
    b = fit_tensor_ar(X, init="random", n_starts=2, max_iter=2, random_state=22)
    assert_array_equal(a.transition_matrices(), b.transition_matrices())
    fit_tensor_ar(X, initial=initial, max_iter=2)
    assert_array_equal(X, original)
    for original_term, after in zip(saved[0], initial[0]):
        for original_matrix, after_matrix in zip(original_term, after):
            assert_array_equal(original_matrix, after_matrix)
