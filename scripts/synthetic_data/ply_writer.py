"""Write point clouds to PLY files using the plyfile library.

Output vertex properties (in this order):
    x, y, z  — float32 coordinates
    label    — int32 DBSCAN label  (-1 = noise)
    red, green, blue — uint8 RGB color
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def write_ply(
    path: Path,
    points: np.ndarray,
    labels: np.ndarray,
    colors: np.ndarray,
    comments: list[str] | None = None,
) -> None:
    """Write a PLY point cloud file.

    Parameters
    ----------
    path:
        Destination file path (created or overwritten).
    points:
        (N, 3) float32 array of XYZ coordinates.
    labels:
        (N,) int32 array of DBSCAN labels.
    colors:
        (N, 3) uint8 array of RGB values.
    comments:
        Optional list of strings written as PLY header comment lines.
    """
    n = len(points)
    assert labels.shape == (n,), "labels must be 1-D with length == n_points"
    assert colors.shape == (n, 3), "colors must be (N, 3)"

    dtype = np.dtype([
        ("x",     np.float32),
        ("y",     np.float32),
        ("z",     np.float32),
        ("label", np.int32),
        ("red",   np.uint8),
        ("green", np.uint8),
        ("blue",  np.uint8),
    ])

    vertices = np.empty(n, dtype=dtype)
    vertices["x"]     = points[:, 0].astype(np.float32)
    vertices["y"]     = points[:, 1].astype(np.float32)
    vertices["z"]     = points[:, 2].astype(np.float32)
    vertices["label"] = labels.astype(np.int32)
    vertices["red"]   = colors[:, 0]
    vertices["green"] = colors[:, 1]
    vertices["blue"]  = colors[:, 2]

    element = PlyElement.describe(vertices, "vertex")
    PlyData([element], text=False, comments=comments or []).write(str(path))
