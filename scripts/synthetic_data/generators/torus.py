"""Solid torus cluster generator.

A solid torus (donut with a filled tube) is parameterized by:
  R — major radius (center of tube to center of torus)
  r — minor radius (tube radius) = R / aspect_ratio

Uniform sampling uses rejection to correct for the varying circumference of
the inner vs. outer regions of the torus.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from ..utils import FloatOrRange, sample_float
from .base import ClusterGenerator


class TorusGenerator(ClusterGenerator):
    """Solid torus with uniform point distribution.

    Parameters
    ----------
    major_radius:
        Major radius R, in ε units.
    aspect_ratio:
        R / r.  Larger values make a thinner, more ring-like torus.
    """

    def __init__(
        self,
        major_radius: FloatOrRange,
        aspect_ratio: FloatOrRange,
    ) -> None:
        self.major_radius = major_radius
        self.aspect_ratio = aspect_ratio

    def generate(self, density_factor: float, min_pts: int, eps: float, rng: Generator) -> np.ndarray:
        import math
        r_major_eps = sample_float(self.major_radius, rng)
        aspect = sample_float(self.aspect_ratio, rng)

        r_major = r_major_eps * eps
        r_minor = r_major / aspect  # minor (tube) radius in world units

        self._warn_small_dimension("minor_radius (R / aspect_ratio)", r_minor / eps)

        # Volume of solid torus: 2π² R r²  (in ε³ units)
        volume_eps = 2.0 * math.pi ** 2 * r_major_eps * (r_major_eps / aspect) ** 2
        n_points = self._compute_n_points(volume_eps, density_factor, min_pts)

        # ── Uniform sampling in a solid torus via rejection ───────────────────
        # Volume element in toroidal coords: r' * (r_major + r'cos φ) dr' dφ dθ
        # Over-sample; accept with probability ∝ (r_major + r'·cos φ) / max_weight
        max_weight = r_major + r_minor
        collected: list[np.ndarray] = []
        n_collected = 0

        while n_collected < n_points:
            batch = max(n_points - n_collected, 128) * 3

            theta = rng.uniform(0.0, 2.0 * np.pi, batch)
            phi   = rng.uniform(0.0, 2.0 * np.pi, batch)

            # Uniform in disk cross-section (sqrt trick)
            r_frac  = np.sqrt(rng.uniform(0.0, 1.0, batch))
            r_tube  = r_frac * r_minor

            # Jacobian-based rejection for uniform volume distribution
            weights = r_major + r_tube * np.cos(phi)
            accept  = rng.uniform(0.0, max_weight, batch) < weights

            theta_a = theta[accept]
            phi_a   = phi[accept]
            r_a     = r_tube[accept]

            x = (r_major + r_a * np.cos(phi_a)) * np.cos(theta_a)
            y = (r_major + r_a * np.cos(phi_a)) * np.sin(theta_a)
            z = r_a * np.sin(phi_a)

            pts = np.column_stack([x, y, z])
            collected.append(pts)
            n_collected += len(pts)

        return np.vstack(collected)[:n_points].astype(np.float32)
