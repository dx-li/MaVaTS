"""Asymptotic MAR(1) inference from Chen, Xiao and Yang (2021).

Theorems 2--4 and the specification test in Section 4.2:
https://doi.org/10.1016/j.jeconom.2020.07.015.
These are fixed-dimension, large-sample results for stationary zero-mean
MAR(1) with iid innovations, nonsingular coefficients and innovation covariance.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.stats import chi2, norm

from ._validation import as_series, finite_scalar, positive_int
from .autoregression import (
    MARResult,
    _rearrange_phi,
    fit_mar,
    nearest_kronecker_product,
)


def _condition(matrix, limit, name):
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[0] == 0 or singular[-1] <= singular[0] / limit:
        raise ValueError(
            f"{name} is singular or ill-conditioned for asymptotic inference"
        )
    return float(singular[0] / singular[-1])


def _positive_definite(matrix, limit, name):
    matrix = (matrix + matrix.T) / 2
    values = np.linalg.eigvalsh(matrix)
    if values[-1] <= 0 or values[0] <= values[-1] / limit:
        raise ValueError(
            f"{name} is not positive definite at the requested condition limit"
        )
    return matrix


def _jacobian(X, A, B):
    """Derivative of vec(A X B.T) wrt [vec(A), vec(B)], all column-major."""
    m, n = X.shape
    transpose = np.arange(n * n).reshape(n, n, order="F").T.ravel(order="F")
    return np.concatenate(
        (np.kron(B @ X.T, np.eye(m)), np.kron(np.eye(n), A @ X)[:, transpose]), axis=1
    )


def _operator_jacobian(A, B):
    m, n = len(A), len(B)
    directions = []
    for unit in np.eye(m * m):
        directions.append(np.kron(B, unit.reshape(m, m, order="F")).ravel(order="F"))
    for unit in np.eye(n * n):
        directions.append(np.kron(unit.reshape(n, n, order="F"), A).ravel(order="F"))
    return np.column_stack(directions)


def _jacobian_batch(X, A, B):
    """Bounded time batch of the same column-major derivatives."""
    T, m, n = X.shape
    left = np.einsum("tjk,il->tjikl", B @ X.transpose(0, 2, 1), np.eye(m)).reshape(
        T, m * n, m * m
    )
    right = np.einsum("tik,jl->tjikl", A @ X, np.eye(n)).reshape(T, m * n, n * n)
    return np.concatenate((left, right), axis=2)


def _var_moments(X, condition_limit):
    T, m, n = X.shape
    flat = X.transpose(0, 2, 1).reshape(T, m * n)
    past, future = flat[:-1], flat[1:]
    _condition(past, condition_limit**0.5, "VAR design")
    coefficient = np.linalg.lstsq(past, future, rcond=None)[0].T
    residuals = future - past @ coefficient.T
    gamma = _positive_definite(
        past.T @ past / len(past), condition_limit, "lag covariance"
    )
    sigma = _positive_definite(
        residuals.T @ residuals / len(past),
        condition_limit,
        "VAR innovation covariance",
    )
    # Cov(vec(Phi)) = Gamma^{-1} kron Sigma / N, with column-major vec(Phi).
    covariance = np.kron(np.linalg.solve(gamma, np.eye(m * n)), sigma)
    return coefficient, covariance


def _rearrangement_indices(m, n):
    d = m * n
    return _rearrange_phi(np.arange(d * d).reshape(d, d, order="F"), m, n).ravel(
        order="F"
    )


@dataclass
class MARInferenceResult:
    """Plug-in covariance of normalized A/B estimates, not their sqrt(N) limit.

    ``parameter_covariance`` uses [vec(A), vec(B)] in column-major order;
    ||A||_F=1 fixes scale. Signs are aligned so A[sign_anchor] is positive;
    that true entry must be nonzero for coefficient intervals to have a locally
    fixed identification convention. Operator intervals have no sign ambiguity.
    ``operator`` and ``operator_standard_errors`` have shape (m*n,m*n).
    The dense covariance of the operator is available only by explicit call.
    Intervals are marginal Wald intervals, not simultaneous confidence bands.
    """

    result: MARResult
    parameter_covariance: np.ndarray
    operator_standard_errors: np.ndarray
    condition_number: float
    n_observations: int
    sign_anchor: tuple
    _operator_derivative: np.ndarray

    @property
    def operator(self):
        return np.kron(self.result.B, self.result.A)

    @property
    def left_standard_errors(self):
        m = len(self.result.A)
        return np.sqrt(
            np.maximum(np.diag(self.parameter_covariance)[: m * m], 0)
        ).reshape(m, m, order="F")

    @property
    def right_standard_errors(self):
        m, n = len(self.result.A), len(self.result.B)
        return np.sqrt(
            np.maximum(np.diag(self.parameter_covariance)[m * m :], 0)
        ).reshape(n, n, order="F")

    def operator_covariance(self):
        """Materialize covariance of column-vectorized operator: (mn)^4 entries."""
        return (
            self._operator_derivative
            @ self.parameter_covariance
            @ self._operator_derivative.T
        )

    def confidence_interval(self, *, level=0.95, target="operator"):
        """Return lower/upper arrays for marginal asymptotic normal intervals.

        At products whose two underlying A/B entries both vanish, the first-order
        variance degenerates and ordinary Wald coverage is not justified.
        """
        level = finite_scalar(level, "level")
        if not 0 < level < 1:
            raise ValueError("level must lie strictly between zero and one")
        if target == "operator":
            estimate, se = self.operator, self.operator_standard_errors
        elif target == "left":
            estimate, se = self.result.A, self.left_standard_errors
        elif target == "right":
            estimate, se = self.result.B, self.right_standard_errors
        else:
            raise ValueError("target must be 'operator', 'left', or 'right'")
        width = norm.isf((1 - level) / 2) * se
        return estimate - width, estimate + width


def mar_inference(
    X,
    result=None,
    *,
    method="als",
    sign_anchor=(0, 0),
    condition_limit=1e10,
    max_dimension=32,
):
    """Plug-in MAR(1) standard errors for projection, ALS or separable MLE.

    Implements Theorems 2--4 of Chen, Xiao and Yang (2021), with sample means
    replacing expectations and N=T-1 effective observations. Public parameter
    ordering consistently uses vec(B), applying the permutation needed by the
    paper's vec(B.T) ALS/MLE formula. A common data scale is removed before
    computing moments; coefficient standard errors are invariant to data units.

    If ``result`` is omitted, fit the named method at a tight optimization
    tolerance. A supplied result must match X and be converged, stable, without
    intercept, ridge or reduced-rank constraints. MLE covariance flooring must
    be inactive. Innovations must be iid with finite second moments and, for
    the ALS/MLE theorems, absolutely continuous (the paper's Condition R).
    This is not a HAC covariance or
    post-selection inference. Correct separability is required for MLE inference.

    ``max_dimension`` bounds rows*columns before dense allocations, protecting
    against accidental (mn)^4 memory use in projection inference. Raise it
    deliberately for larger problems. Numerically ill-conditioned sample
    problems are rejected; sample guards cannot establish population identification.
    An asymptotic interval is not a finite-sample coverage guarantee.

    ``sign_anchor`` is a fixed (row,column) index of A whose true value must be
    nonzero. A/B are jointly sign-flipped in a copied result to make this entry
    positive. This avoids the discontinuity of the estimator's largest-entry
    sign rule at tied opposite-signed entries. The caller's result is unchanged.
    """
    X = as_series(X, min_samples=3)
    m, n = X.shape[1:]
    try:
        anchor = tuple(
            positive_int(v, "sign_anchor index", minimum=0) for v in sign_anchor
        )
    except TypeError as exc:
        raise ValueError("sign_anchor must be a pair of matrix indices") from exc
    if len(anchor) != 2 or max(anchor) >= m:
        raise ValueError("sign_anchor must index the left coefficient matrix")
    maximum = positive_int(max_dimension, "max_dimension")
    limit = finite_scalar(condition_limit, "condition_limit")
    if limit <= 1:
        raise ValueError("condition_limit must exceed one")
    if m * n > maximum:
        raise ValueError(
            "matrix vector dimension exceeds max_dimension for dense inference"
        )
    if result is None:
        result = fit_mar(X, method=method, max_iter=1000, tol=1e-11)
    if not isinstance(result, MARResult) or result.order != 1:
        raise ValueError("result must be a MAR(1) fit")
    if (
        result.method not in ("als", "projection", "mle")
        or result.ridge
        or result.ranks is not None
    ):
        raise ValueError("inference requires an unconstrained unpenalized MAR fit")
    if (
        result.A.shape != (m, m)
        or result.B.shape != (n, n)
        or result.residuals.shape != X[1:].shape
    ):
        raise ValueError("result dimensions do not match X")
    if np.any(result.intercept != 0):
        raise ValueError("the published inference here requires a zero-intercept fit")
    if not result.converged or not result.is_stable:
        raise ValueError("inference requires a converged stationary fit")
    if result.covariance_regularized:
        raise ValueError(
            "the published MLE inference requires inactive covariance flooring"
        )
    if abs(result.A[anchor]) <= np.finfo(float).eps * m:
        raise ValueError(
            "sign anchor is numerically zero; choose a nonzero population coefficient"
        )
    if result.A[anchor] < 0:
        result = replace(result, left=-result.left, right=-result.right)
    A, B = result.A, result.B
    _condition(A, limit, "left coefficient")
    _condition(B, limit, "right coefficient")
    if not np.isclose(np.linalg.norm(A), 1.0, rtol=1e-10):
        raise ValueError("left coefficient must have unit Frobenius norm")
    scale = float(np.max(np.abs(X))) or 1.0
    work = X / scale
    predicted = A @ work[:-1] @ B.T
    residuals = work[1:] - predicted
    if not np.allclose(result.residuals / scale, residuals, atol=1e-10, rtol=1e-8):
        raise ValueError("result was not fitted to the supplied observations")
    N, d, q = len(X) - 1, m * n, m * m + n * n
    alpha, beta = A.ravel(order="F"), B.ravel(order="F")
    if result.method == "projection":
        _, xi = _var_moments(work, limit)
        indices = _rearrangement_indices(m, n)
        xi = xi[np.ix_(indices, indices)]
        beta_norm = np.linalg.norm(beta)
        v0 = np.concatenate(
            (
                np.kron(
                    (beta / beta_norm)[None], np.eye(m * m) - np.outer(alpha, alpha)
                )
                / beta_norm,
                np.kron(np.eye(n * n), alpha[None]),
            ),
            axis=0,
        )
        covariance = v0 @ xi @ v0.T / N
        condition = _condition(work[:-1].reshape(N, d), limit, "VAR design")
    else:
        flat = residuals.transpose(0, 2, 1).reshape(N, d)
        if result.method == "mle":
            sigma = np.kron(
                result.column_covariance / scale, result.row_covariance / scale
            )
        else:
            sigma = flat.T @ flat / N
        sigma = _positive_definite(sigma, limit, "innovation covariance")
        gram, meat = np.zeros((q, q)), np.zeros((q, q))
        for start in range(0, N, 128):
            J = _jacobian_batch(work[start : min(start + 128, N)], A, B)
            if result.method == "mle":
                weighted = np.linalg.solve(sigma, J.transpose(1, 0, 2).reshape(d, -1))
                weighted = weighted.reshape(d, len(J), q).transpose(1, 0, 2)
                gram += np.einsum("tdi,tdj->ij", J, weighted, optimize=True) / N
            else:
                gram += np.einsum("tdi,tdj->ij", J, J, optimize=True) / N
                meat += np.einsum("tdi,de,tej->ij", J, sigma, J, optimize=True) / N
        if result.method == "mle":
            meat = gram.copy()
        gamma = np.r_[alpha, np.zeros(n * n)]
        H = _positive_definite(
            gram + np.outer(gamma, gamma), limit, "constrained information"
        )
        condition = _condition(H, limit, "constrained information")
        covariance = np.linalg.solve(H, np.linalg.solve(H, meat).T).T / N
    covariance = (covariance + covariance.T) / 2
    derivative = _operator_jacobian(A, B)
    variances = np.einsum(
        "ij,jk,ik->i", derivative, covariance, derivative, optimize=True
    )
    errors = np.sqrt(np.maximum(variances, 0)).reshape(d, d, order="F")
    return MARInferenceResult(
        result, covariance, errors, condition, N, anchor, derivative
    )


@dataclass
class MARSpecificationResult:
    """Wald specification test of one Kronecker VAR transition operator."""

    statistic: float
    pvalue: float
    degrees_of_freedom: int
    projection_left: np.ndarray
    projection_right: np.ndarray
    unrestricted_coefficient: np.ndarray
    condition_number: float
    n_observations: int


def mar_specification_test(X, *, condition_limit=1e10, max_dimension=16):
    """Test VAR(1) coefficient Phi = B ⊗ A using the paper's Section 4.2.

    Requires a stationary zero-mean process with iid nonsingular innovations.
    Under the null, both Kronecker factors must be nonsingular. The statistic
    has asymptotic chi-square df=(m²-1)(n²-1). A small p-value rejects that
    coefficient structure; a large p-value does not establish it.

    Uses N=T-1 and sample second moments. Instead of a numerically singular
    projected covariance and a pseudoinverse, an orthonormal basis of its
    known normal space gives an equivalent positive-definite solve. Dense
    covariance storage scales as (m*n)^4. Both modes must exceed one.
    """
    X = as_series(X, min_samples=3)
    m, n = X.shape[1:]
    limit = finite_scalar(condition_limit, "condition_limit")
    if limit <= 1:
        raise ValueError("condition_limit must exceed one")
    if min(m, n) <= 1:
        raise ValueError("both matrix dimensions must exceed one for a nontrivial test")
    if m * n > positive_int(max_dimension, "max_dimension"):
        raise ValueError("matrix vector dimension exceeds max_dimension for dense test")
    scale = float(np.max(np.abs(X))) or 1.0
    coefficient, xi = _var_moments(X / scale, limit)
    if np.max(abs(np.linalg.eigvals(coefficient))) >= 1:
        raise ValueError("unrestricted VAR fit is not stationary")
    A, B = nearest_kronecker_product(coefficient, (m, n))
    _condition(A, limit, "projected left coefficient")
    _condition(B, limit, "projected right coefficient")
    alpha, beta = A.ravel(order="F"), B.ravel(order="F").copy()
    beta /= np.linalg.norm(beta)
    qa = np.linalg.svd(alpha[None], full_matrices=True)[2][1:].T
    qb = np.linalg.svd(beta[None], full_matrices=True)[2][1:].T
    normal = np.kron(qb, qa)
    indices = _rearrangement_indices(m, n)
    xi = xi[np.ix_(indices, indices)]
    projected = _positive_definite(
        normal.T @ xi @ normal, limit, "normal-space covariance"
    )
    deviation = _rearrange_phi(coefficient - np.kron(B, A), m, n).ravel(order="F")
    residual = normal.T @ deviation
    statistic = float((len(X) - 1) * residual @ np.linalg.solve(projected, residual))
    df = (m * m - 1) * (n * n - 1)
    return MARSpecificationResult(
        statistic,
        float(chi2.sf(statistic, df)),
        df,
        A,
        B,
        coefficient,
        _condition(projected, limit, "normal-space covariance"),
        len(X) - 1,
    )
