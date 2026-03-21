"""Crescent cluster generator.

A crescent is a partial arc of a circle with a tubular cross-section.
Points are sampled uniformly within a tube of radius *thickness/2* that
follows the arc.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from ..utils import FloatOrRange, sample_float
from .base import ClusterGenerator


class CrescentGenerator(ClusterGenerator):
    """Partial circular arc with a circular cross-section.

    Parameters
    ----------
    radius:
        Distance from the origin to the arc center-line, in ε units.
    thickness:
        Diameter of the tubular cross-section, in ε units.
    arc_fraction:
        Portion of a full circle covered by the arc (1.0 = full circle).
    """

    def __init__(
        self,
        radius: FloatOrRange,
        thickness: FloatOrRange,
        arc_fraction: FloatOrRange,
    ) -> None:
        self.radius = radius
        self.thickness = thickness
        self.arc_fraction = arc_fraction

    def generate(self, density_factor: float, min_pts: int, eps: float, rng: Generator) -> np.ndarray:
        import math
        radius_eps = sample_float(self.radius, rng)
        thickness_eps = sample_float(self.thickness, rng)
        arc_fraction = sample_float(self.arc_fraction, rng)

        self._warn_small_dimension("thickness", thickness_eps)

        # Volume: arc_fraction × 2π × radius × π × (thickness/2)²  (all in ε units)
        volume_eps = arc_fraction * math.pi ** 2 * radius_eps * thickness_eps ** 2 / 2.0
        n_points = self._compute_n_points(volume_eps, density_factor, min_pts)

        radius_w = radius_eps * eps
        tube_radius = (thickness_eps * eps) / 2.0

        # ── Angular positions along the arc ──────────────────────────────────
        theta = rng.uniform(0.0, arc_fraction * 2.0 * np.pi, n_points)

        # ── Cross-section: uniform disk of radius tube_radius ─────────────────
        # Use sqrt for uniform-area distribution in polar coords
        phi_tube = rng.uniform(0.0, 2.0 * np.pi, n_points)
        r_tube = np.sqrt(rng.uniform(0.0, 1.0, n_points)) * tube_radius

        radial_offset = r_tube * np.cos(phi_tube)  # in the arc's radial direction
        z_offset = r_tube * np.sin(phi_tube)        # perpendicular (out of the plane)

        # ── Assemble world coordinates ────────────────────────────────────────
        r_actual = radius_w + radial_offset
        x = r_actual * np.cos(theta)
        y = r_actual * np.sin(theta)
        z = z_offset

        return np.column_stack([x, y, z]).astype(np.float32)
