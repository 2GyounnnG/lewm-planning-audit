"""Lossless seed dtype compatibility at the World.reset boundary only.

The pinned SWM may forward NumPy integer scalars to Gymnasium, which requires
Python integers. This changes their representation, never their value or the
official environment source. Missing seeds remain missing.
"""
from numbers import Integral
import numpy as np

SEED_COMPATIBILITY = 'LOSSLESS_PYTHON_INT_UINT32_RESET_BOUNDARY_V1'


def _scalar(seed):
    if seed is None:
        return None
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise TypeError('Reset seed must be an integer or None; no float coercion')
    value = int(seed)
    if not 0 <= value <= 2**32 - 1:
        raise ValueError('Reset seed must be within the uint32 range')
    return value


def normalize_reset_seed(seed):
    """Return Python int/None or a flat list thereof, preserving every value."""
    if isinstance(seed, np.ndarray):
        if seed.ndim == 0:
            return _scalar(seed.item())
        if seed.ndim != 1:
            raise TypeError('Reset seed array must be scalar or one-dimensional')
        return [_scalar(value) for value in seed]
    if isinstance(seed, (list, tuple)):
        return [_scalar(value) for value in seed]
    return _scalar(seed)


normalize_seed = normalize_reset_seed
