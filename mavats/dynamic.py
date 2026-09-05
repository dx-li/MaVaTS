"""Additive two-way dynamic factors of Yuan et al. (2023).

The estimator follows Algorithms 1/2 and supplement Algorithm A1 in
https://doi.org/10.1093/jrsssb/qkad077. Its additive covariance quasi-likelihood,
conditional factor scores, and pooled scalar autoregressions are different
from bilinear matrix PCA followed by a VAR. See docs/dynamic-notes.md.
"""

from dataclasses import dataclass, field

import numpy as np

from ._validation import as_series, finite_scalar, positive_int


def _positive(value, name):
    value = finite_scalar(value, name, minimum=0)
    if value == 0:
        raise ValueError(f"{name} must be strictly positive")
    return value


def _pair(value, name, shape=None):
    try:
        value = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain two positive integers") from exc
    if len(value) != 2:
        raise ValueError(f"{name} must contain two positive integers")
    value = tuple(positive_int(v, name) for v in value)
    if shape is not None and any(v >= d for v, d in zip(value, shape)):
        raise ValueError(f"{name} must be strictly smaller than spatial dimensions")
    return value


def _sym(A):
    return A / 2 + A.T / 2


def _leading(A, rank):
    values, vectors = np.linalg.eigh(_sym(A))
    return vectors[:, -rank:][:, ::-1], np.maximum(values[::-1], 0)


def _normalize_signs(loadings):
    """Paper first-row convention; largest-entry fallback at a zero anchor."""
    anchors = loadings[0].copy()
    zero = anchors == 0
    columns = np.arange(loadings.shape[1])
    anchors[zero] = loadings[np.argmax(np.abs(loadings), axis=0), columns][zero]
    return loadings * np.where(anchors < 0, -1, 1)


def _spectral_basis(loadings, covariance):
    """Eigenbasis of loading @ covariance @ loading.T without large eigh."""
    d, rank = loadings.shape
    values, rotation = np.linalg.eigh(_sym(covariance))
    if values[0] <= 0:
        raise FloatingPointError("factor covariance is not positive definite")
    complete = np.linalg.qr(loadings, mode="complete")[0]
    basis = np.column_stack((loadings @ rotation / np.sqrt(d), complete[:, rank:]))
    return basis, np.r_[d * values, np.zeros(d - rank)]


def _spectrum(L, Lam, psi_f, psi_g, noise):
    V, a = _spectral_basis(L, psi_f)
    U, b = _spectral_basis(Lam, psi_g)
    denominator = b[:, None] + a[None, :] + noise
    return U, V, denominator


def _inverse_action(Y, spectrum):
    U, V, denominator = spectrum
    return U @ ((U.T @ Y @ V) / denominator) @ V.T


def _objective(Y, L, Lam, psi_f, psi_g, noise):
    U, V, denominator = _spectrum(L, Lam, psi_f, psi_g, noise)
    rotated = U.T @ Y @ V
    return float(-np.log(denominator).sum() - np.sum(rotated**2 / denominator) / len(Y))


def _scores(Y, L, Lam, psi_f, psi_g, noise):
    W = _inverse_action(Y, _spectrum(L, Lam, psi_f, psi_g, noise))
    return W @ L @ psi_f, W.transpose(0, 2, 1) @ Lam @ psi_g


def _floor_spd(covariance, floor):
    values, vectors = np.linalg.eigh(_sym(covariance))
    active = bool(values[0] < floor)
    return _sym((vectors * np.maximum(values, floor)) @ vectors.T), active


def _em_step(Y, L, Lam, psi_f, psi_g, noise, floor=0):
    """Full latent Gaussian conditional-moment EM, including cross-effects.

    Partial traces of the inverse covariance produce the sum of marginal
    factor conditional covariances. No nm-by-nm or joint latent covariance
    is formed. Noise uncertainty uses Var(signal|Y)=s I-s² K^-1, which
    includes the conditional dependence between F and G.
    """
    T, n, m = Y.shape
    spectrum = _spectrum(L, Lam, psi_f, psi_g, noise)
    U, V, denominator = spectrum
    inverse = 1 / denominator
    W = _inverse_action(Y, spectrum)
    F = W @ L @ psi_f
    G = W.transpose(0, 2, 1) @ Lam @ psi_g
    # Partial trace over all observation rows/columns, respectively.
    trace_rows = (V * inverse.sum(axis=0)) @ V.T
    trace_columns = (U * inverse.sum(axis=1)) @ U.T
    conditional_f = n * psi_f - psi_f @ L.T @ trace_rows @ L @ psi_f
    conditional_g = m * psi_g - psi_g @ Lam.T @ trace_columns @ Lam @ psi_g
    new_f = np.einsum("tir,tis->rs", F, F) / (n * T) + conditional_f / n
    new_g = np.einsum("tir,tis->rs", G, G) / (m * T) + conditional_g / m
    residual = Y - F @ L.T - Lam @ G.transpose(0, 2, 1)
    # s * (1-s/denominator) is more stable than s-s²/denominator.
    uncertainty = np.sum(noise * ((denominator - noise) / denominator))
    new_noise = float((np.sum(residual**2) / T + uncertainty) / (n * m))
    new_f, active_f = _floor_spd(new_f, floor)
    new_g, active_g = _floor_spd(new_g, floor)
    active = active_f or active_g or new_noise < floor
    return new_f, new_g, max(new_noise, floor), active


def _covariance_em(Y, L, Lam, psi_f, psi_g, noise, floor, max_iter, tol):
    old = _objective(Y, L, Lam, psi_f, psi_g, noise)
    history, regularized, converged = [old], False, False
    for iteration in range(1, max_iter + 1):
        psi_f, psi_g, noise, active = _em_step(Y, L, Lam, psi_f, psi_g, noise, floor)
        regularized |= active
        new = _objective(Y, L, Lam, psi_f, psi_g, noise)
        _check_ascent(old, new)
        history.append(new)
        if abs(new - old) <= tol * (1 + abs(old)):
            converged = True
            break
        old = new
    return psi_f, psi_g, noise, np.asarray(history), converged, regularized


def _check_ascent(old, new):
    if not np.isfinite(new):
        raise FloatingPointError("two-way dynamic objective is nonfinite")
    if new < old - 1e-9 * (1 + abs(old)):
        raise FloatingPointError("two-way dynamic objective decreased beyond roundoff")


def _loading_moments(Y, other, own_variances, other_variances, noise):
    """Equation (10), evaluated without subtracting nearly equal inverses.

    Y is T by n by m and own loading has m rows. On an eigenvector of the
    other-mode signal covariance with eigenvalue b, W_s has eigenvalue
    f_s/((noise+b)*(noise+b+m*f_s)).
    """
    T, _, m = Y.shape
    U, b = _spectral_basis(other, np.diag(other_variances))
    rotated = U.T @ Y
    moments = []
    for f in own_variances:
        weights = (f / (noise + b)) / (noise + b + m * f)
        moment = np.einsum("tij,i,tik->jk", rotated, weights, rotated) / T
        moments.append(_sym(moment))
    return np.asarray(moments)


def _loading_update(moments, initial, max_iter, tol):
    """Heterogeneous quadratic maximization by shifted polar iterations."""
    d, rank = initial.shape
    if rank == 1:
        loading = np.sqrt(d) * _leading(moments[0], 1)[0]
        return loading, 1, True
    smallest = min(np.linalg.eigvalsh(moment)[0] for moment in moments)
    rho = smallest - 0.01  # Paper's positive-definite shift; objective invariant.
    shifted = moments - rho * np.eye(d)[None]
    loading = initial.copy()

    def score(Q):
        return float(np.einsum("is,sij,js->", Q, moments, Q))

    old, converged = score(loading), False
    for iteration in range(1, max_iter + 1):
        product = np.einsum("sij,js->is", shifted, loading)
        u, _, vh = np.linalg.svd(product, full_matrices=False)
        updated = np.sqrt(d) * (u @ vh)
        new = score(updated)
        _check_ascent(old, new)
        loading = updated
        if abs(new - old) <= tol * (1 + abs(old)):
            converged = True
            break
        old = new
    return loading, iteration, converged


def _rotate(L, covariance):
    values, vectors = np.linalg.eigh(_sym(covariance))
    return _normalize_signs(L @ vectors[:, ::-1]), np.diag(values[::-1])


def _stepwise(Y, ranks, max_iter, tol):
    """Supplement Algorithm A1 initialization via alternating residual PCA."""
    _, n, m = Y.shape
    c, r = ranks
    L = np.sqrt(m) * _leading(np.einsum("tij,tik->jk", Y, Y), r)[0]
    F = Y @ L / m
    old = np.inf
    history = []
    for _ in range(max_iter):
        residual = Y - F @ L.T
        Lam = np.sqrt(n) * _leading(np.einsum("tij,tkj->ik", residual, residual), c)[0]
        G = residual.transpose(0, 2, 1) @ Lam / n
        residual = Y - Lam @ G.transpose(0, 2, 1)
        L = np.sqrt(m) * _leading(np.einsum("tij,tik->jk", residual, residual), r)[0]
        F = residual @ L / m
        error = float(np.mean((Y - F @ L.T - Lam @ G.transpose(0, 2, 1)) ** 2))
        history.append(error)
        if abs(error - old) <= tol * (1 + abs(old)) and np.isfinite(old):
            break
        old = error
    return L, Lam, np.asarray(history)


def _select_ranks(Y, bounds, ridge, max_iter):
    """Paper Algorithm 2; ridge is already in the units of Y's second moments."""
    T, n, m = Y.shape
    cmax, rmax = bounds
    row_basis = _leading(np.einsum("tij,tkj->ik", Y, Y), n)[0]
    column_basis = _leading(np.einsum("tij,tik->jk", Y, Y), m)[0]
    c, r = bounds
    history, converged, cycle = [(c, r)], False, False
    spectra = None
    delta = max(n**-0.5, m**-0.5, T**-0.5)
    for _ in range(max_iter):
        Q = row_basis[:, :c]
        residual = Y - Q @ (Q.T @ Y)
        _, column_values = _leading(np.einsum("tij,tik->jk", residual, residual) / T, m)
        ratios = column_values[:rmax] / (column_values[1 : rmax + 1] + ridge * delta)
        r = int(np.argmax(ratios)) + 1
        Q = column_basis[:, :r]
        residual = Y - (Y @ Q) @ Q.T
        _, row_values = _leading(np.einsum("tij,tkj->ik", residual, residual) / T, n)
        ratios = row_values[:cmax] / (row_values[1 : cmax + 1] + ridge * delta)
        c = int(np.argmax(ratios)) + 1
        spectra = (row_values, column_values)
        current = (c, r)
        if current == history[-1]:
            converged = True
        elif current in history:
            cycle = True
        history.append(current)
        if converged or cycle:
            break
    return (c, r), np.asarray(history), spectra, converged, cycle


def _pooled_ar(scores, order):
    """Equation (7), scalar AR per factor pooled across independent units."""
    T, units, rank = scores.shape
    coefficients = np.empty((order, rank))
    innovations = np.empty(rank)
    residuals = np.empty((T - order, units, rank))
    for j in range(rank):
        design = np.stack(
            [
                scores[order - lag : T - lag, :, j].ravel()
                for lag in range(1, order + 1)
            ],
            axis=1,
        )
        target = scores[order:, :, j].ravel()
        coefficient, _, effective_rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if effective_rank < order:
            raise ValueError("pooled factor AR design is rank deficient; reduce orders")
        coefficients[:, j] = coefficient
        error = target - design @ coefficient
        residuals[:, :, j] = error.reshape(T - order, units)
        innovations[j] = np.sum(error**2) / (units * T)  # Paper equation (8).
    return coefficients, innovations, residuals


def _ar_radii(coefficients):
    order, rank = coefficients.shape
    result = np.empty(rank)
    for j in range(rank):
        companion = np.zeros((order, order))
        companion[0] = coefficients[:, j]
        if order > 1:
            companion[1:, :-1] = np.eye(order - 1)
        result[j] = np.max(np.abs(np.linalg.eigvals(companion)))
    return result


@dataclass
class TwoWayDynamicResult:
    """Additive factor estimates and pooled diagonal AR dynamics.

    Public ranks/orders use (row, column) order: paper (c,r)/(q,p).
    F has shape (T,n,r), G (T,m,c); signal = mean + F L.T + Lambda G.T.
    Covariances/innovation variances and scores use physical data units.
    Objective histories and rank spectra use X/data_scale units. Add
    -2*n*m*log(data_scale) to objective history for the physical quasi-
    likelihood. This objective omits fixed constants and is not the joint
    temporal likelihood. History tolerances do not certify global optima.
    """

    row_loadings: np.ndarray
    column_loadings: np.ndarray
    F: np.ndarray
    G: np.ndarray
    signal: np.ndarray
    residuals: np.ndarray
    mean: np.ndarray
    row_factor_covariance: np.ndarray
    column_factor_covariance: np.ndarray
    noise_variance: float
    row_ar: np.ndarray
    column_ar: np.ndarray
    row_innovation_variance: np.ndarray
    column_innovation_variance: np.ndarray
    objective_history: np.ndarray
    em_objective_histories: tuple
    inner_iterations: np.ndarray
    inner_converged: np.ndarray
    rank_history: np.ndarray
    rank_spectra: tuple | None
    rank_converged: bool
    rank_cycle: bool
    initialization_history: np.ndarray
    converged: bool
    n_iter: int
    variance_regularized: bool
    data_scale: float
    _psi_f: np.ndarray = field(repr=False)
    _psi_g: np.ndarray = field(repr=False)
    _noise: float = field(repr=False)

    @property
    def ranks(self):
        return (self.row_loadings.shape[1], self.column_loadings.shape[1])

    @property
    def orders(self):
        return (len(self.row_ar), len(self.column_ar))

    @property
    def loadings(self):
        return (self.row_loadings, self.column_loadings)

    @property
    def is_stable(self):
        return bool(
            np.all(_ar_radii(self.row_ar) < 1) and np.all(_ar_radii(self.column_ar) < 1)
        )

    def transform(self, X):
        """Contemporaneous conditional F/G scores using training parameters.

        These use only each supplied matrix, not a Kalman smoother or data
        from the future. F/G represent deviations from the training mean.
        """
        X = as_series(X, min_samples=1)
        if X.shape[1:] != self.mean.shape:
            raise ValueError("X spatial dimensions must match the fitted model")
        Y = X / self.data_scale - self.mean / self.data_scale
        F, G = _scores(
            Y,
            self.column_loadings,
            self.row_loadings,
            self._psi_f,
            self._psi_g,
            self._noise,
        )
        return F * self.data_scale, G * self.data_scale

    def inverse_transform(self, F, G):
        """Reconstruct the additive signal from F and G scores."""
        F, G = as_series(F, min_samples=1), as_series(G, min_samples=1)
        n, m = self.mean.shape
        c, r = self.ranks
        if F.shape[1:] != (n, r) or G.shape != (len(F), m, c):
            raise ValueError("F and G score dimensions must match the fitted model")
        return (
            self.mean
            + F @ self.column_loadings.T
            + self.row_loadings @ G.transpose(0, 2, 1)
        )

    def reconstruct(self, X):
        return self.inverse_transform(*self.transform(X))

    def forecast(self, steps, history=None):
        """Recursive plug-in common-component forecasts in observation units.

        Uses the last p/q contemporaneous scores, predicting the residual
        component as zero. Under serially correlated idiosyncratic errors
        these need not equal full conditional-mean observation forecasts.
        """
        steps = positive_int(steps, "steps")
        q, p = self.orders
        if history is None:
            F, G = self.F[-p:], self.G[-q:]
        else:
            history = as_series(history, min_samples=max(p, q))
            F, G = self.transform(history)
        output = []
        f_history, g_history = list(F[-p:]), list(G[-q:])
        for _ in range(steps):
            f = sum(
                coefficient * f_history[-lag]
                for lag, coefficient in enumerate(self.column_ar, start=1)
            )
            g = sum(
                coefficient * g_history[-lag]
                for lag, coefficient in enumerate(self.row_ar, start=1)
            )
            prediction = self.inverse_transform(f[None], g[None])[0]
            if not np.isfinite(prediction).all():
                raise FloatingPointError(
                    "two-way dynamic forecast overflow; inspect AR stability"
                )
            output.append(prediction)
            f_history.append(f)
            g_history.append(g)
        return np.asarray(output)


def fit_two_way_dynamic(
    X,
    ranks=None,
    *,
    orders=(1, 1),
    rank_bounds=None,
    ratio_ridge=0.01,
    center=False,
    max_iter=100,
    tol=1e-7,
    em_max_iter=100,
    loading_max_iter=100,
    rank_max_iter=20,
    variance_floor=1e-10,
    initial=None,
):
    """Fit Yuan et al.'s additive two-way dynamic factor model.

    X is time-first (T,n,m); ranks=(c,r) and orders=(q,p) follow spatial
    (row,column) order. Both ranks are positive and strictly below their
    corresponding dimensions. If ranks=None, Algorithm 2 selects them
    within rank_bounds (default half each dimension), with denominator
    addition ratio_ridge*max(n**-.5,m**-.5,T**-.5) in original data units.
    This selects among positive ranks; it is not a no-factor test.

    Algorithm A1 provides initialization; Algorithm 1 combines heterogeneous
    quadratic loading updates and full latent-moment covariance EM. The
    final factor scores are conditional Gaussian linear predictors under
    the working additive covariance, followed by pooled scalar AR fits.
    Each AR coefficient is shared across units within its factor column.

    center=False uses the paper's zero-mean model. center=True estimates
    and removes a training mean (an explicit extension). initial optionally
    supplies (row_loadings,column_loadings) with L.T@L=dimension*I.
    variance_floor is a relative lower bound on all covariance eigenvalues
    and noise variance, as a fraction of mean(Y**2) after centering; this
    computational safeguard restricts the parameter space and is reported.
    It may be zero to disable regularization on nondegenerate data.

    max_iter, em_max_iter and loading_max_iter bound outer and inner loops;
    tol is a relative objective-increment tolerance. Inner loops use tol/10.
    All convergence/rank diagnostics are retained. Stability is diagnosed,
    not enforced. This is a local quasi-likelihood solution, not a global
    optimization certificate or a fit of the full temporal likelihood.
    """
    X = as_series(X, min_samples=3)
    if not isinstance(center, (bool, np.bool_)):
        raise ValueError("center must be boolean")
    T, n, m = X.shape
    if min(n, m) < 2:
        raise ValueError("both spatial dimensions must exceed one")
    orders = _pair(orders, "orders")
    if max(orders) >= T - 1:
        raise ValueError("orders must leave at least two response observations")
    max_iter = positive_int(max_iter, "max_iter")
    em_max_iter = positive_int(em_max_iter, "em_max_iter")
    loading_max_iter = positive_int(loading_max_iter, "loading_max_iter")
    rank_max_iter = positive_int(rank_max_iter, "rank_max_iter")
    tol = _positive(tol, "tol")
    ratio_ridge = _positive(ratio_ridge, "ratio_ridge")
    variance_floor = finite_scalar(variance_floor, "variance_floor", minimum=0)
    scale = float(np.max(np.abs(X)))
    if scale == 0:
        raise ValueError("zero data have no positive innovation scale")
    Y = X / scale
    mean = Y.mean(axis=0) if center else np.zeros((n, m))
    Y = Y - mean
    energy = float(np.mean(Y**2))
    if energy <= np.finfo(float).tiny:
        raise ValueError("centered data have no identifiable variation")
    floor = variance_floor * energy
    if floor == 0:
        floor = np.finfo(float).tiny
    bounds = (
        (max(1, n // 2), max(1, m // 2))
        if rank_bounds is None
        else _pair(rank_bounds, "rank_bounds", (n, m))
    )
    if ranks is None:
        # Preserve paper ridge units under purely numerical data scaling.
        ridge = (ratio_ridge / scale) / scale
        if not np.isfinite(ridge) or ridge == 0:
            raise ValueError(
                "rank-selection ridge is not representable at this data scale; supply explicit ranks"
            )
        ranks, rank_history, spectra, rank_converged, rank_cycle = _select_ranks(
            Y, bounds, ridge, rank_max_iter
        )
    else:
        ranks = _pair(ranks, "ranks", (n, m))
        rank_history, spectra, rank_converged, rank_cycle = (
            np.asarray([ranks]),
            None,
            True,
            False,
        )
    c, r = ranks
    if initial is None:
        L, Lam, initialization_history = _stepwise(Y, ranks, 50, 1e-6)
    else:
        try:
            raw_row, raw_column = initial
        except (ValueError, TypeError) as exc:
            raise ValueError("initial must contain row and column loadings") from exc
        matrices = []
        for raw, d, k in ((raw_row, n, c), (raw_column, m, r)):
            raw = np.asarray(raw)
            if raw.dtype.kind not in "biuf" or np.iscomplexobj(raw):
                raise ValueError("initial loadings must be real numeric")
            value = np.array(raw, dtype=float, copy=True)
            if (
                value.shape != (d, k)
                or not np.isfinite(value).all()
                or not np.allclose(
                    value.T @ value, d * np.eye(k), rtol=1e-8, atol=1e-10
                )
            ):
                raise ValueError(
                    "initial loadings must match ranks and dimension-scaled orthogonality"
                )
            matrices.append(value)
        Lam, L = matrices
        initialization_history = np.empty(0)
    f0 = Y @ L / m
    g0 = Y.transpose(0, 2, 1) @ Lam / n
    psi_f = np.diag(np.maximum(np.mean(f0**2, axis=(0, 1)), floor))
    psi_g = np.diag(np.maximum(np.mean(g0**2, axis=(0, 1)), floor))
    noise = max(
        float(np.mean((Y - f0 @ L.T - Lam @ g0.transpose(0, 2, 1)) ** 2)), floor
    )
    L, psi_f = _rotate(L, psi_f)
    Lam, psi_g = _rotate(Lam, psi_g)
    history = [_objective(Y, L, Lam, psi_f, psi_g, noise)]
    em_histories, inner_iterations, inner_converged = [], [], []
    converged, regularized = False, False
    for iteration in range(1, max_iter + 1):
        moments = _loading_moments(Y, Lam, np.diag(psi_f), np.diag(psi_g), noise)
        L, it_f, done_f = _loading_update(moments, L, loading_max_iter, tol / 10)
        moments = _loading_moments(
            Y.transpose(0, 2, 1), L, np.diag(psi_g), np.diag(psi_f), noise
        )
        Lam, it_g, done_g = _loading_update(moments, Lam, loading_max_iter, tol / 10)
        psi_f, psi_g, noise, em_history, done_em, active = _covariance_em(
            Y, L, Lam, psi_f, psi_g, noise, floor, em_max_iter, tol / 10
        )
        regularized |= active
        L, psi_f = _rotate(L, psi_f)
        Lam, psi_g = _rotate(Lam, psi_g)
        new = _objective(Y, L, Lam, psi_f, psi_g, noise)
        _check_ascent(history[-1], new)
        em_histories.append(em_history)
        inner_iterations.append((it_g, it_f, len(em_history) - 1))
        inner_converged.append((done_g, done_f, done_em))
        improvement = abs(new - history[-1])
        history.append(new)
        if (
            improvement <= tol * (1 + abs(history[-2]))
            and done_g
            and done_f
            and done_em
        ):
            converged = True
            break
    F, G = _scores(Y, L, Lam, psi_f, psi_g, noise)
    column_ar, innovation_f, _ = _pooled_ar(F, orders[1])
    row_ar, innovation_g, _ = _pooled_ar(G, orders[0])
    signal = (mean + F @ L.T + Lam @ G.transpose(0, 2, 1)) * scale
    # Physical variances may overflow even though the scaled fit is valid.
    physical_f = (psi_f * scale) * scale
    physical_g = (psi_g * scale) * scale
    physical_noise = (noise * scale) * scale
    if not (
        np.isfinite(physical_f).all()
        and np.isfinite(physical_g).all()
        and np.isfinite(physical_noise)
    ):
        raise FloatingPointError("physical variances overflow; rescale observations")
    if (
        np.any(np.diag(physical_f) <= 0)
        or np.any(np.diag(physical_g) <= 0)
        or physical_noise <= 0
    ):
        raise FloatingPointError("physical variances underflow; rescale observations")
    return TwoWayDynamicResult(
        row_loadings=Lam,
        column_loadings=L,
        F=F * scale,
        G=G * scale,
        signal=signal,
        residuals=X - signal,
        mean=mean * scale,
        row_factor_covariance=physical_g,
        column_factor_covariance=physical_f,
        noise_variance=physical_noise,
        row_ar=row_ar,
        column_ar=column_ar,
        row_innovation_variance=(innovation_g * scale) * scale,
        column_innovation_variance=(innovation_f * scale) * scale,
        objective_history=np.asarray(history),
        em_objective_histories=tuple(em_histories),
        inner_iterations=np.asarray(inner_iterations),
        inner_converged=np.asarray(inner_converged),
        rank_history=rank_history,
        rank_spectra=spectra,
        rank_converged=rank_converged,
        rank_cycle=rank_cycle,
        initialization_history=initialization_history,
        converged=converged,
        n_iter=iteration,
        variance_regularized=regularized,
        data_scale=scale,
        _psi_f=psi_f,
        _psi_g=psi_g,
        _noise=noise,
    )
