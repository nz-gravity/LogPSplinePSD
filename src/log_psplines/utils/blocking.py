"""Helpers for selecting block structures for time series workflows."""

from __future__ import annotations

import numpy as np


def infer_time_blocks(n: int, *, max_blocks: int) -> int:
    """Infer a reasonable number of time blocks for Wishart averaging."""

    if n <= 0:
        raise ValueError("n must be positive")
    if max_blocks <= 0:
        raise ValueError("max_blocks must be positive")

    target = max(1, 2 ** int(np.round(np.log2(n / (24 * 7)))))
    while target > 1 and n % target != 0:
        target //= 2
    while target > max_blocks:
        target //= 2
    return max(1, target)


__all__ = ["infer_time_blocks"]
