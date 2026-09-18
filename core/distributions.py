"""
Shared statistical distribution utilities for OptiRND simulation engines.

Design goals:
    * DRY: single vectorized PERT implementation replaces five copy-pasted
      engine-local copies.
    * Concurrency isolation: sampling uses a dependency-injected
      numpy.random.Generator. The legacy engines called np.random.seed(42)
      in __init__, mutating PROCESS-WIDE random state - that breaks
      concurrent Streamlit sessions and stochastic testing. This module
      never touches the global numpy RNG.
    * Reproducibility: a deterministic stream is created ONLY when an
      explicit seed is provided.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

ArrayLike = Union[int, float]

# Default shape parameter of the PERT (Beta) distribution.
PERT_GAMMA_SHAPE: float = 4.0


def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Create an isolated numpy Generator.

    - seed=None  -> fresh entropy (independent streams per engine instance).
    - seed=42    -> deterministic, reproducible stream for tests/regression,
                    WITHOUT altering numpy's global random state.
    """
    return np.random.default_rng(seed)


def validate_triplet(low: ArrayLike, likely: ArrayLike, high: ArrayLike) -> Tuple[float, float, float]:
    """
    Validate a (low, likely, high) triplet.

    Raises:
        ValueError: if inputs are non-finite or the ordering low <= likely <= high
                    is violated.
    Returns:
        The triplet as plain floats.
    """
    low_f, likely_f, high_f = float(low), float(likely), float(high)
    for name, value in (("low", low_f), ("likely", likely_f), ("high", high_f)):
        if not np.isfinite(value):
            raise ValueError(f"PERT parameter '{name}' must be finite, got {value!r}.")
    if not (low_f <= likely_f <= high_f):
        raise ValueError(
            f"PERT parameters violate ordering: require low <= likely <= high, "
            f"got low={low_f}, likely={likely_f}, high={high_f}."
        )
    return low_f, likely_f, high_f


def generate_pert_samples(
    low: ArrayLike,
    likely: ArrayLike,
    high: ArrayLike,
    size: int = 10_000,
    rng: Optional[np.random.Generator] = None,
    gamma_shape: float = PERT_GAMMA_SHAPE,
) -> np.ndarray:
    """
    Vectorized PERT sampling via the Beta distribution.

    Parameters
    ----------
    low, likely, high:
        PERT triplet. Inverted ranges raise ValueError (validation happens
        BEFORE any sampling - callers cannot silently sample nonsense).
    size:
        Number of samples to draw.
    rng:
        Optional numpy.random.Generator (dependency injection). A fresh
        isolated Generator is created when omitted. The global numpy RNG
        is NEVER used or mutated.
    gamma_shape:
        PERT shape parameter (classic value: 4.0).

    Returns
    -------
    np.ndarray of shape (size,) with values in [low, high].
    For low == high a degenerate (constant) array is returned.
    """
    low_f, likely_f, high_f = validate_triplet(low, likely, high)

    if int(size) <= 0:
        raise ValueError(f"Sample size must be positive, got {size!r}.")
    size = int(size)

    if low_f == high_f:
        # Degenerate distribution: all mass at the single point.
        return np.full(size, low_f, dtype=float)

    generator = rng if rng is not None else make_rng()
    if not isinstance(generator, np.random.Generator):
        raise TypeError(
            f"rng must be a numpy.random.Generator instance, got {type(generator).__name__}."
        )

    alpha = 1.0 + gamma_shape * (likely_f - low_f) / (high_f - low_f)
    beta_val = 1.0 + gamma_shape * (high_f - likely_f) / (high_f - low_f)

    beta_samples = generator.beta(alpha, beta_val, size)
    return low_f + beta_samples * (high_f - low_f)
