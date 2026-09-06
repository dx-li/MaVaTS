"""Local Grassmann optimization of the EMAR profile, equations (15)--(16).

Samadi and De Alwis (2026), https://doi.org/10.1080/07350015.2025.2537404.
QR retraction, transported conjugate gradients, Armijo search and multistarts
are numerical choices, not a claim to reproduce the authors' software.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def _spd(matrix, name):
    matrix = matrix / 2 + matrix.T / 2
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite and positive definite")
    try:
        factor = cho_factor(matrix, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"{name} must be positive definite; no covariance flooring is used"
        ) from exc
    return matrix, factor


def _logdet(factor):
    return 2 * np.log(np.diag(factor[0])).sum()


def _profile_value_gradient(basis, residual, response_factor):
    """Published logdet profile and its horizontal (Grassmann) gradient."""
    mu = residual @ basis
    su = cho_solve(response_factor, basis, check_finite=False)
    _, fm = _spd(basis.T @ mu, "projected residual moment")
    _, fs = _spd(basis.T @ su, "projected inverse response moment")
    gradient = 2 * (
        cho_solve(fm, mu.T, check_finite=False).T
        + cho_solve(fs, su.T, check_finite=False).T
    )
    gradient -= basis @ (basis.T @ gradient)
    return float(_logdet(fm) + _logdet(fs)), gradient


def _grassmann_run(initial, residual, response_factor, max_iter, tol):
    basis = initial.copy()
    value, gradient = _profile_value_gradient(basis, residual, response_factor)
    direction = -gradient
    history = [value]
    reason = "max_iter"
    for _ in range(max_iter):
        norm = float(np.linalg.norm(gradient))
        if norm <= tol:
            reason = "gradient_tolerance"
            break
        slope = float(np.sum(gradient * direction))
        if slope >= -1e-3 * norm**2:
            direction, slope = -gradient, -(norm**2)
        step = min(1.0, 1.0 / max(np.linalg.norm(direction), 1.0))
        accepted = False
        for _ in range(50):
            candidate, triangular = np.linalg.qr(basis + step * direction)
            # Align QR column signs with the old basis for vector transport.
            candidate *= np.where(np.diag(triangular) < 0, -1.0, 1.0)
            new_value, new_gradient = _profile_value_gradient(
                candidate, residual, response_factor
            )
            if new_value <= value + 1e-4 * step * slope:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            reason = "line_search_failed"
            break
        transported = gradient - candidate @ (candidate.T @ gradient)
        beta = max(
            0.0, float(np.sum(new_gradient * (new_gradient - transported))) / norm**2
        )
        direction = -new_gradient + beta * (
            direction - candidate @ (candidate.T @ direction)
        )
        basis, value, gradient = candidate, new_value, new_gradient
        history.append(value)
    norm = float(np.linalg.norm(gradient))
    converged = norm <= tol
    if converged:
        reason = "gradient_tolerance"
    return basis, {
        "objective": value,
        "gradient_norm": norm,
        "converged": converged,
        "stop_reason": reason,
        "n_iter": len(history) - 1,
        "objective_history": tuple(history),
    }


def _fit_envelope(residual, response, rank, *, initial, starts, rng, max_iter, tol):
    """Optimize the published profile; retain the previous space as a start.

    Independent positive moment rescalings change only an additive constant.
    All recorded profile objectives use these scaled moments, so compare them
    only within one call, not between outer sweeps.
    """
    residual, _ = _spd(residual, "residual moment")
    response, _ = _spd(response, "response moment")
    residual = residual / np.max(np.abs(residual))
    response, factor = _spd(response / np.max(np.abs(response)), "response moment")
    dimension = len(response)
    if rank == dimension:
        basis = np.eye(dimension)
        value, _ = _profile_value_gradient(basis, residual, factor)
        run = dict(
            objective=value,
            gradient_norm=0.0,
            converged=True,
            stop_reason="full_dimension",
            n_iter=0,
            objective_history=(value,),
        )
        return basis, {"selected_start": 0, "runs": (run,), **run}
    candidates = [] if initial is None else [initial.copy()]
    for index in range(starts):
        if index < 2:
            moment = residual if index == 0 else response
            candidates.append(np.linalg.eigh(moment)[1][:, :rank])
        else:
            candidates.append(np.linalg.qr(rng.normal(size=(dimension, rank)))[0])
    results = [
        _grassmann_run(basis, residual, factor, max_iter, tol) for basis in candidates
    ]
    best = min(range(len(results)), key=lambda i: results[i][1]["objective"])
    basis, diagnostic = results[best]
    return basis, {
        **diagnostic,
        "selected_start": best,
        "runs": tuple(result[1] for result in results),
    }
