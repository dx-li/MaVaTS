"""Shared validation; public methods accept real, finite, time-first arrays."""

from numbers import Integral

import numpy as np


def positive_int(value, name, *, minimum=1):
    """Validate an integer without silently rounding floats or accepting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def as_series(X, min_samples=2, ndim=3):
    """Return a finite float64 view/copy; never mutate the caller's array."""
    raw = np.asarray(X)
    if np.iscomplexobj(raw) or raw.dtype.kind not in "biuf":
        raise ValueError("X must contain real numeric values")
    X = np.asarray(raw, dtype=np.float64)
    if (ndim is not None and X.ndim != ndim) or X.ndim < 2:
        raise ValueError(f"X must have {ndim or 'at least 2'} dimensions, time first")
    if X.shape[0] < min_samples or any(d == 0 for d in X.shape[1:]):
        raise ValueError(
            f"X needs at least {min_samples} observations and nonempty modes"
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    return X


def ranks_tuple(ranks, shape):
    """Validate one rank per spatial mode (or return None for auto selection)."""
    if ranks is None:
        return None
    try:
        ranks = tuple(ranks)
    except TypeError as exc:
        raise ValueError("ranks must be a sequence, one per spatial mode") from exc
    if len(ranks) != len(shape):
        raise ValueError("ranks must have one entry per spatial mode")
    result = tuple(positive_int(r, "rank") for r in ranks)
    if any(r > d for r, d in zip(result, shape)):
        raise ValueError("ranks cannot exceed the spatial dimensions")
    return result


def random_generator(random_state=None):
    """Create/reuse a Generator without affecting NumPy's global RNG."""
    return np.random.default_rng(random_state)


def finite_scalar(value, name, *, minimum=None):
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if not np.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}")
    return value
