"""Spike-and-slab EM variable selection for Gaussian MAR(1).

This implements the continuous normal-mixture EMVS posterior-mode branch
of the reference below, not MCMC or MAR*(P). Arbitrary
factor rescaling in the paper's equation (8) is omitted: it changes the
specified prior and can decrease the posterior. See docs/sparse-notes.md.
Result support masks and forecasts are plug-in summaries of this model.

References
----------
Celani, A., Pagnottoni, P. and Jones, G. (2024), "Bayesian Variable Selection
for Matrix Autoregressive Models", Sections 3--4 and Appendix C,
https://doi.org/10.1007/s11222-024-10402-y.
"""

from dataclasses import dataclass, field

import numpy as np

from ._validation import as_series, finite_scalar, positive_int
from .autoregression import fit_mar


def _positive(value, name):
    value = finite_scalar(value, name, minimum=0)
    if value == 0:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _pair(value, name, minimum=0):
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain two numbers") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must contain two numbers")
    return tuple(finite_scalar(v, name, minimum=minimum) for v in values)


def _matrix_pair(value, name):
    try:
        values = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain two matrices") from exc
    if len(values) != 2:
        raise ValueError(f"{name} must contain two matrices")
    return values


def _matrix(value, dim, name, *, positive_definite=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in "biuf" or np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real numeric")
    result = np.array(raw, dtype=float, copy=True)
    if result.shape != (dim, dim) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape {(dim, dim)}")
    if positive_definite:
        if not np.allclose(result, result.T, rtol=1e-12, atol=1e-14):
            raise ValueError(f"{name} must be symmetric positive definite")
        result = result / 2 + result.T / 2
        try:
            np.linalg.cholesky(result)
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"{name} must be symmetric positive definite") from exc
    return result


def _mixture(coefficient, theta, spike, slab):
    """Conditional slab probabilities and log marginal prior, in log space."""
    # Endpoints are legitimate Beta(1,b) conditional modes.
    log_theta = -np.inf if theta == 0 else np.log(theta)
    log_other = -np.inf if theta == 1 else np.log1p(-theta)
    log_spike = log_other - 0.5 * (np.log(2 * np.pi * spike) + coefficient**2 / spike)
    log_slab = log_theta - 0.5 * (np.log(2 * np.pi * slab) + coefficient**2 / slab)
    log_prior = np.logaddexp(log_spike, log_slab)
    probability = np.exp(log_slab - log_prior)
    return probability, float(np.sum(log_prior))


def _log_likelihood(residuals, row_cov, col_cov):
    """Separable Gaussian likelihood with Cholesky whitening."""
    N, m, n = residuals.shape
    lr, lc = np.linalg.cholesky(row_cov), np.linalg.cholesky(col_cov)
    white = np.linalg.solve(lr, residuals)
    white = np.linalg.solve(lc, white.transpose(0, 2, 1))
    return float(
        -0.5
        * (
            N * m * n * np.log(2 * np.pi)
            + 2 * N * n * np.log(np.diag(lr)).sum()
            + 2 * N * m * np.log(np.diag(lc)).sum()
            + np.sum(white**2)
        )
    )


def _coefficient_update(Y, Z, other, row_cov, col_cov, precision):
    """Solve one exact penalized GLS block by SVD-compressed augmented LS.

    No vectorized time-by-matrix design, inverse, or squared-condition-number
    normal equations are formed. The largest dense solve has m*m columns.
    The transposed data give the corresponding right-factor update.
    """
    n_obs, m, n = Y.shape
    col_root = np.linalg.cholesky(col_cov)
    row_inverse_root = np.linalg.solve(np.linalg.cholesky(row_cov), np.eye(m))
    # Multiplication on the right by inv(L_col.T) whitens columns.
    white_y = np.linalg.solve(col_root, Y.transpose(0, 2, 1)).transpose(0, 2, 1)
    design = Z @ other.T
    white_d = np.linalg.solve(col_root, design.transpose(0, 2, 1)).transpose(0, 2, 1)
    D = white_d.transpose(0, 2, 1).reshape(n_obs * n, m)
    target = white_y.transpose(0, 2, 1).reshape(n_obs * n, m)
    u, singular, vh = np.linalg.svd(D, full_matrices=False)
    reduced_d = singular[:, None] * vh
    reduced_y = (u.T @ target) @ row_inverse_root.T
    design = np.kron(reduced_d, row_inverse_root)
    design = np.concatenate((design, np.diag(np.sqrt(precision.ravel(order="F")))))
    response = np.concatenate((reduced_y.ravel(), np.zeros(m * m)))
    return np.linalg.lstsq(design, response, rcond=None)[0].reshape(m, m, order="F")


def _covariance_mode(residuals, other_cov, omega, xi, df):
    """Exact IW conditional mode for the row covariance."""
    N, m, n = residuals.shape
    white = np.linalg.solve(
        np.linalg.cholesky(other_cov), residuals.transpose(0, 2, 1)
    ).transpose(0, 2, 1)
    scatter = np.einsum("tij,tkj->ik", white, white)
    covariance = (scatter + xi * omega) / (N * n + df + m + 1)
    return covariance / 2 + covariance.T / 2


def _scale_mode(covariances, omegas, dfs, shape, rate):
    numerator = (
        shape - 1 + 0.5 * sum(c.shape[0] * nu for c, nu in zip(covariances, dfs))
    )
    denominator = rate + 0.5 * sum(
        np.trace(np.linalg.solve(cov, omega)) for cov, omega in zip(covariances, omegas)
    )
    return float(numerator / denominator)


def _balance_coefficients(A, B, precision_a, precision_b):
    """Maximize the current EM surrogate along (c*A, B/c), c > 0."""
    energy_a = np.sum(precision_a * A**2)
    energy_b = np.sum(precision_b * B**2)
    if energy_a == 0 or energy_b == 0:
        return A, B
    c = np.exp(0.25 * (np.log(energy_b) - np.log(energy_a)))
    return A * c, B / c


def _balance_covariances(covs, omegas, dfs, xi):
    """Maximize IW prior along (c*Sigma_row, Sigma_col/c)."""
    row, col = covs
    m, n = len(row), len(col)
    a = (m * (dfs[0] + m + 1) - n * (dfs[1] + n + 1)) / xi
    r = np.trace(np.linalg.solve(row, omegas[0]))
    b = np.trace(np.linalg.solve(col, omegas[1]))
    # The root is invariant to a common scale; avoid overflowing b*r.
    scale = max(abs(a), r, b)
    a, r, b = a / scale, r / scale, b / scale
    root = np.hypot(a, 2 * np.sqrt(r) * np.sqrt(b))
    c = 2 * r / (root + a) if a >= 0 else (root - a) / (2 * b)
    return row * c, col / c


def _log_beta_kernel(theta, alpha, beta):
    if (theta == 0 and alpha > 1) or (theta == 1 and beta > 1):
        return -np.inf
    return float(
        (0 if alpha == 1 else (alpha - 1) * np.log(theta))
        + (0 if beta == 1 else (beta - 1) * np.log1p(-theta))
    )


def _log_posterior(Y, Z, A, B, covs, theta, xi, spike, slab, beta, omegas, dfs, gamma):
    """Observed log posterior, including only parameter-dependent terms.

    Includes the likelihood constant and mixture-normal constants. Drops
    only fixed Beta/Gamma/IW normalization constants, so values should only
    be compared between iterations with the same data and hyperparameters.
    """
    result = _log_likelihood(Y - A @ Z @ B.T, *covs)
    for coefficient, weight in zip((A, B), theta):
        result += _mixture(coefficient, weight, spike, slab)[1]
        result += _log_beta_kernel(weight, *beta)
    for covariance, omega, nu in zip(covs, omegas, dfs):
        d = len(covariance)
        result += 0.5 * nu * d * np.log(xi)
        result -= 0.5 * (nu + d + 1) * np.linalg.slogdet(covariance)[1]
        result -= 0.5 * xi * np.trace(np.linalg.solve(covariance, omega))
    result += (gamma[0] - 1) * np.log(xi) - gamma[1] * xi
    return float(result)


@dataclass
class SparseMARResult:
    """Local EMVS mode and conditional, plug-in inclusion probabilities.

    Inclusion probabilities condition on the fitted coefficients and other
    parameters; they are not integrated posterior inclusion probabilities.
    A/B are continuously shrunk, generally dense; support masks apply the
    0.5 slab-probability rule without changing forecasts. Proper priors set
    the factor scales. The likelihood alone identifies only B kron A and
    column_covariance kron row_covariance.
    Model-derived summaries and forecasts inherit the reference and
    corrected-EMVS scope of :func:`fit_sparse_mar` and this module.
    """

    A: np.ndarray
    B: np.ndarray
    row_covariance: np.ndarray
    column_covariance: np.ndarray
    row_inclusion: np.ndarray
    column_inclusion: np.ndarray
    inclusion_weights: np.ndarray
    covariance_scale: float
    fitted_values: np.ndarray
    residuals: np.ndarray
    log_posterior_history: np.ndarray
    converged: bool
    n_iter: int
    spike_variance: float
    slab_variance: float
    _history: np.ndarray = field(repr=False)

    @property
    def row_support(self):
        return self.row_inclusion > 0.5

    @property
    def column_support(self):
        return self.column_inclusion > 0.5

    @property
    def coefficients(self):
        """Column-major VAR(1) coefficient with a leading lag axis."""
        return np.kron(self.B, self.A)[None]

    @property
    def spectral_radius(self):
        return float(
            np.max(np.abs(np.linalg.eigvals(self.A)))
            * np.max(np.abs(np.linalg.eigvals(self.B)))
        )

    @property
    def is_stable(self):
        """Stationarity diagnostic; stability is not imposed during fitting."""
        return self.spectral_radius < 1

    @property
    def log_likelihood(self):
        return _log_likelihood(
            self.residuals, self.row_covariance, self.column_covariance
        )

    def forecast(self, steps, history=None):
        """Recursive conditional-mean forecasts, without thresholding A/B."""
        steps = positive_int(steps, "steps")
        observed = (
            self._history if history is None else as_series(history, min_samples=1)
        )
        if observed.shape[1:] != self.fitted_values.shape[1:]:
            raise ValueError("history spatial dimensions must match the fitted series")
        last = observed[-1].copy()
        output = np.empty((steps,) + last.shape)
        for t in range(steps):
            last = self.A @ last @ self.B.T
            if not np.isfinite(last).all():
                raise FloatingPointError(
                    "forecast overflow; check fitted stability and history scale"
                )
            output[t] = last
        return output


def fit_sparse_mar(
    X,
    *,
    spike_variance=0.01,
    slab_variance=4.0,
    inclusion_prior=(1.0, 1.0),
    covariance_df=None,
    covariance_scale=None,
    scale_prior=(1.0, 1.0),
    initial=None,
    max_iter=500,
    tol=1e-7,
):
    """Fit zero-mean MAR(1) using continuous spike-and-slab ECM/EMVS.

    ``X`` is finite and time-first, with shape ``(T,m,n)``. Coefficient
    priors are independent mixtures N(0, spike_variance)/N(0, slab_variance)
    in both modes; these arguments are variances, not standard deviations.
    Each mode has an estimated slab weight with Beta(*inclusion_prior).
    Both Beta parameters must be >= 1 (finite conditional modes).

    Covariance priors are Sigma_k|xi ~ IW(xi*Omega_k, nu_k), with default
    Omega=(I_m/m,I_n/n), nu=(m+2,n+2); ``covariance_scale`` supplies the two
    SPD Omega matrices, ``covariance_df`` supplies their degrees of freedom.
    The shared xi has Gamma(shape, RATE), specified by ``scale_prior``.
    Priors operate in the supplied data units: preprocessing changes the
    statistical problem unless the covariance priors are also transformed.

    ``initial=(A,B)`` optionally supplies the initial coefficients; otherwise
    an unpenalized ALS fit initializes ECM. The scale of this starting pair
    is balanced once for initialization only. Subsequent iterations retain
    prior-selected scales, and exact conditional updates monotonically
    increase the observed log posterior up to numerical error. ``tol`` is
    the relative posterior-increment stopping tolerance. Local modes and
    sensitivity to initial values/prior settings should be assessed by the
    caller. There is no stationarity constraint or fitted intercept.

    This is the MAR(1) EMVS branch of Celani et al. (2024), with mathematical
    corrections described in docs/sparse-notes.md. MCMC posterior draws,
    credible intervals, and the paper's MAR*(P) branch are not implemented.

    References
    ----------
    Celani, A., Pagnottoni, P. and Jones, G. (2024), "Bayesian Variable
    Selection for Matrix Autoregressive Models", Sections 3--4 and Appendix C,
    https://doi.org/10.1007/s11222-024-10402-y.
    Exact conditional scale maximizations replace the paper's arbitrary
    post-update rescaling, which would change the prior. The default uniform
    Beta inclusion prior also differs from its simulation-specific settings.
    """
    X = as_series(X, min_samples=3)
    if np.array_equal(X, np.broadcast_to(X[0], X.shape)):
        raise ValueError(
            "temporally constant data cannot identify a nondegenerate innovation covariance"
        )
    max_iter = positive_int(max_iter, "max_iter")
    tol = _positive(tol, "tol")
    spike = _positive(spike_variance, "spike_variance")
    slab = _positive(slab_variance, "slab_variance")
    if spike >= slab:
        raise ValueError("spike_variance must be less than slab_variance")
    beta = _pair(inclusion_prior, "inclusion_prior", minimum=1)
    gamma = _pair(scale_prior, "scale_prior")
    if min(gamma) <= 0:
        raise ValueError("scale_prior shape and rate must be strictly positive")
    m, n = X.shape[1:]
    dfs = (
        (m + 2.0, n + 2.0)
        if covariance_df is None
        else _pair(covariance_df, "covariance_df")
    )
    if any(nu <= d - 1 for nu, d in zip(dfs, (m, n))):
        raise ValueError("covariance_df must exceed dimension minus one")
    if gamma[0] - 1 + 0.5 * (m * dfs[0] + n * dfs[1]) <= 0:
        raise ValueError(
            "scale prior and covariance_df must give a positive interior xi mode"
        )
    if covariance_scale is None:
        omegas = (np.eye(m) / m, np.eye(n) / n)
    else:
        covariance_scale = _matrix_pair(covariance_scale, "covariance_scale")
        omegas = tuple(
            _matrix(cov, d, "covariance_scale", positive_definite=True)
            for cov, d in zip(covariance_scale, (m, n))
        )
    if initial is None:
        start = fit_mar(X, method="als", max_iter=50)
        A, B = start.A.copy(), start.B.copy()
        norm_a, norm_b = np.linalg.norm(A), np.linalg.norm(B)
        if norm_a > 0 and norm_b > 0:
            balance = np.sqrt(norm_b / norm_a)
            A, B = A * balance, B / balance
    else:
        initial = _matrix_pair(initial, "initial")
        A, B = (_matrix(v, d, "initial") for v, d in zip(initial, (m, n)))
    covs = (np.eye(m), np.eye(n))
    xi = gamma[0] / gamma[1]
    theta = np.full(2, beta[0] / (beta[0] + beta[1]))
    Y, Z = X[1:], X[:-1]

    def objective():
        return _log_posterior(
            Y, Z, A, B, covs, theta, xi, spike, slab, beta, omegas, dfs, gamma
        )

    objectives = [objective()]
    if not np.isfinite(objectives[-1]):
        raise FloatingPointError(
            "initial sparse MAR posterior is not finite; check data and prior scales"
        )
    converged = False
    for iteration in range(1, max_iter + 1):
        p_a = _mixture(A, theta[0], spike, slab)[0]
        p_b = _mixture(B, theta[1], spike, slab)[0]
        precision_a = (1 - p_a) / spike + p_a / slab
        precision_b = (1 - p_b) / spike + p_b / slab
        A = _coefficient_update(Y, Z, B, *covs, precision_a)
        B = _coefficient_update(
            Y.transpose(0, 2, 1),
            Z.transpose(0, 2, 1),
            A,
            covs[1],
            covs[0],
            precision_b,
        )
        A, B = _balance_coefficients(A, B, precision_a, precision_b)
        theta = np.array(
            [
                (prob.sum() + beta[0] - 1) / (prob.size + beta[0] + beta[1] - 2)
                for prob in (p_a, p_b)
            ]
        )
        residuals = Y - A @ Z @ B.T
        row = _covariance_mode(residuals, covs[1], omegas[0], xi, dfs[0])
        col = _covariance_mode(residuals.transpose(0, 2, 1), row, omegas[1], xi, dfs[1])
        covs = _balance_covariances((row, col), omegas, dfs, xi)
        xi = _scale_mode(covs, omegas, dfs, *gamma)
        value = objective()
        if not np.isfinite(value):
            raise FloatingPointError(
                "sparse MAR posterior became nonfinite; check data and prior scales"
            )
        improvement = value - objectives[-1]
        if improvement < -1e-9 * (1 + abs(objectives[-1])):
            raise FloatingPointError(
                "sparse MAR ECM posterior decreased beyond numerical tolerance"
            )
        objectives.append(value)
        if abs(improvement) <= tol * (1 + abs(objectives[-2])):
            converged = True
            break
    # The joint sign change preserves both centered coefficient priors.
    if A.flat[np.argmax(np.abs(A))] < 0:
        A, B = -A, -B
    fitted = A @ Z @ B.T
    return SparseMARResult(
        A=A,
        B=B,
        row_covariance=covs[0],
        column_covariance=covs[1],
        row_inclusion=_mixture(A, theta[0], spike, slab)[0],
        column_inclusion=_mixture(B, theta[1], spike, slab)[0],
        inclusion_weights=theta,
        covariance_scale=xi,
        fitted_values=fitted,
        residuals=Y - fitted,
        log_posterior_history=np.asarray(objectives),
        converged=converged,
        n_iter=iteration,
        spike_variance=spike,
        slab_variance=slab,
        _history=X[-1:].copy(),
    )
