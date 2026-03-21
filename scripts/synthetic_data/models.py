"""Pydantic configuration models for the synthetic data generator."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

# ── Range type aliases ────────────────────────────────────────────────────────
# These mirror what is written in the JSON config.
# A scalar means a fixed value; a 2-element list means sample from [min, max].

FloatOrRange = float | list[float]   # e.g. 0.35  or  [0.15, 0.35]

# Quaternion [w, x, y, z], or the strings "none" / "random"
RotationSpec = Literal["none", "random"] | list[float]


# ── DBSCAN oracle parameters ──────────────────────────────────────────────────

class DbscanConfig(BaseModel):
    eps: float
    min_pts: int


# ── Per-cluster-group configuration ──────────────────────────────────────────
# Each entry in the top-level "clusters" list describes a *group* of clusters
# of the same type.  n_clusters controls how many individual clusters are
# instantiated from that group.

class _BaseClusterConfig(BaseModel):
    type: str
    n_clusters: int
    rotation: RotationSpec = "random"
    density_factor: FloatOrRange = 2.0


class GaussianClusterConfig(_BaseClusterConfig):
    type: Literal["gaussian"] # pyright: ignore[reportIncompatibleVariableOverride]
    std: FloatOrRange


class CrescentClusterConfig(_BaseClusterConfig):
    """Partial arc of a circle with a tubular cross-section."""
    type: Literal["crescent"] # pyright: ignore[reportIncompatibleVariableOverride]
    radius: FloatOrRange
    thickness: FloatOrRange
    arc_fraction: FloatOrRange  # fraction of a full circle, 1 = full circle


class TorusClusterConfig(_BaseClusterConfig):
    type: Literal["torus"] # pyright: ignore[reportIncompatibleVariableOverride]
    major_radius: FloatOrRange          # R  (center of tube to center of torus)
    aspect_ratio: FloatOrRange          # R / r  →  minor_radius r = R / aspect_ratio


class TubeClusterConfig(_BaseClusterConfig):
    """Solid, thin cylinder — like a pencil or rod."""
    type: Literal["tube"] # pyright: ignore[reportIncompatibleVariableOverride]
    length: FloatOrRange
    radius: FloatOrRange


class CylinderClusterConfig(_BaseClusterConfig):
    """Solid cylinder with uniform point distribution."""
    type: Literal["cylinder"] # pyright: ignore[reportIncompatibleVariableOverride]
    radius: FloatOrRange
    height: FloatOrRange


class BallClusterConfig(_BaseClusterConfig):
    """Solid sphere with uniform point distribution."""
    type: Literal["ball"] # pyright: ignore[reportIncompatibleVariableOverride]
    radius: FloatOrRange


class BoxClusterConfig(_BaseClusterConfig):
    """Axis-aligned box with uniform point distribution."""
    type: Literal["box"] # pyright: ignore[reportIncompatibleVariableOverride]
    width: FloatOrRange
    height: FloatOrRange
    depth: FloatOrRange


# Discriminated union — pydantic selects the right model based on "type"
ClusterConfig = Annotated[
    GaussianClusterConfig
    | CrescentClusterConfig
    | TorusClusterConfig
    | TubeClusterConfig
    | CylinderClusterConfig
    | BallClusterConfig
    | BoxClusterConfig,
    Field(discriminator="type"),
]


# ── Noise configuration ───────────────────────────────────────────────────────

class UniformNoiseConfig(BaseModel):
    """Uniform background noise spread across the scene bounding box."""
    type: Literal["uniform"]
    ratio: float  # fraction of total cluster point count


class BoundaryNoiseConfig(BaseModel):
    """Noise points placed near cluster boundaries."""
    type: Literal["boundary"]
    ratio: float                            # fraction of total cluster point count
    distance_from_cluster: FloatOrRange     # in ε units


NoiseConfig = Annotated[
    UniformNoiseConfig | BoundaryNoiseConfig,
    Field(discriminator="type"),
]


# ── Top-level scene configuration ─────────────────────────────────────────────

class SceneConfig(BaseModel):
    id: str
    seed: int
    dbscan: DbscanConfig
    cluster_distance: FloatOrRange  # distance between cluster centers, in eps units
    clusters: list[ClusterConfig]
    noise: list[NoiseConfig] = []
