"""Axis-aligned box cluster generator."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from ..utils import FloatOrRange, sample_float
from .base import ClusterGenerator


class BoxGenerator(ClusterGenerator):
    """Axis-aligned box with uniform point distribution.

    Parameters
    ----------
    width:
        Extent along X, in ε units.
    height:
        Extent along Y, in ε units.
    depth:
        Extent along Z, in ε units.
    """

    def __init__(
        self,
        width: FloatOrRange,
        height: FloatOrRange,
        depth: FloatOrRange,
    ) -> None:
        self.width = width
        self.height = height
        self.depth = depth

    def generate(self, n_points: int, eps: float, rng: Generator) -> np.ndarray:
        width_eps  = sample_float(self.width, rng)
        height_eps = sample_float(self.height, rng)
        depth_eps  = sample_float(self.depth, rng)

        self._warn_small_dimension("width",  width_eps)
        self._warn_small_dimension("height", height_eps)
        self._warn_small_dimension("depth",  depth_eps)

        w = width_eps  * eps
        h = height_eps * eps
        d = depth_eps  * eps

        x = rng.uniform(-w / 2.0, w / 2.0, n_points)
        y = rng.uniform(-h / 2.0, h / 2.0, n_points)
        z = rng.uniform(-d / 2.0, d / 2.0, n_points)

        return np.column_stack([x, y, z]).astype(np.float32)
