"""Shared bootstrap sampling utilities for statistical inference."""
from __future__ import annotations

import numpy as np


def circular_block_bootstrap_sample(
    data: np.ndarray,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a circular block bootstrap sample of the same length as *data*.

    Wraps data circularly so blocks near the end wrap to the beginning,
    ensuring every observation has equal probability in any position.
    """
    n = len(data)
    if n == 0:
        return data.copy()
    block_length = max(1, min(block_length, n))
    # Wrap data for circular blocking
    wrapped = np.concatenate([data, data[:block_length - 1]])
    result = np.empty(n, dtype=data.dtype)
    pos = 0
    while pos < n:
        start = rng.integers(0, n)
        end = min(pos + block_length, n)
        take = end - pos
        result[pos:end] = wrapped[start : start + take]
        pos = end
    return result


def stationary_block_bootstrap(
    data: np.ndarray,
    n_steps: int,
    expected_block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stationary block bootstrap (Politis & Romano 1994).

    Block lengths are geometrically distributed with mean *expected_block_length*.
    Returns an array of length *n_steps*.
    """
    n = len(data)
    if n == 0:
        return np.empty(0, dtype=data.dtype)
    ebl = max(1, expected_block_length)
    p = 1.0 / ebl  # probability of starting a new block

    result = np.empty(n_steps, dtype=data.dtype)
    pos = 0
    idx = rng.integers(0, n)  # initial random start
    while pos < n_steps:
        result[pos] = data[idx % n]
        pos += 1
        if pos < n_steps:
            if rng.random() < p:
                # Start a new block at a random position
                idx = rng.integers(0, n)
            else:
                idx += 1
    return result
