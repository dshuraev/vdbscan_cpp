"""Solid cylinder cluster generator."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from ..utils import FloatOrRange, sample_float
from .base import ClusterGenerator


class CylinderGenerator(ClusterGenerator):
    """Solid cylinder with uniform point distribution.

    Wider and shorter than a tube — described by radius and height.

    Parameters
    ----------
    radius:
        Cross-section radius, in ε units.
    height:
        Cylinder height along the Z axis, in ε units.
    """

    def __init__(self, radius: FloatOrRange, height: FloatOrRange) -> None:
        self.radius = radius
        self.height = height

    def generate(self, n_points: int, eps: float, rng: Generator) -> np.ndarray:
        radius_eps = sample_float(self.radius, rng)
        height_eps = sample_float(self.height, rng)

        self._warn_small_dimension("height", height_eps)

        radius_w = radius_eps * eps
        height_w = height_eps * eps

        theta = rng.uniform(0.0, 2.0 * np.pi, n_points)
        r = np.sqrt(rng.uniform(0.0, 1.0, n_points)) * radius_w
        z = rng.uniform(-height_w / 2.0, height_w / 2.0, n_points)

        x = r * np.cos(theta)
        y = r * np.sin(theta)

        return np.column_stack([x, y, z]).astype(np.float32)
