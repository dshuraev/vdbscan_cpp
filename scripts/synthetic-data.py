#!/usr/bin/env python3
"""Generate synthetic 3-D point clouds for DBSCAN benchmarking.

Usage
-----
    synthetic-data.py cloud_config.json
    synthetic-data.py cloud_config.json -o /tmp/clouds

The script executes the following pipeline steps:

  1. Parse config — load and validate the JSON configuration file.
  2. Generate clusters — create per-cluster point clouds in local space,
     apply rotation, and translate each cluster to its world-space center.
  3. Place clusters — distribute cluster centers in 3-D space, centered
     around (0, 0, 0), respecting the minimum-separation constraint.
  4. Generate noise — produce uniform background noise and/or boundary noise
     that does not overlap cluster space or other noise types.
  5. Run DBSCAN — apply scikit-learn DBSCAN to the full point cloud to
     obtain ground-truth cluster labels.
  6. Assign colors — map each DBSCAN label to a distinct RGB color using
     the golden-ratio hue sequence; noise is always grey (128, 128, 128).
  7. Write PLY — save the final point cloud as a binary PLY file.

Note: the JSON config uses // comments in the README examples for documentation
purposes only.  The actual config files must be valid JSON (no comments).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from sklearn.cluster import DBSCAN

from synthetic_data.coloring import assign_colors
from synthetic_data.generators import apply_rotation, create_generator
from synthetic_data.models import SceneConfig
from synthetic_data.noise import generate_all_noise
from synthetic_data.placement import place_cluster_centers
from synthetic_data.ply_writer import write_ply
from synthetic_data.utils import sample_float

console = Console(stderr=False)
err_console = Console(stderr=True)

app = typer.Typer(
    help=__doc__,
    add_completion=False,
    rich_markup_mode="rich",
)


# ── Step helpers ──────────────────────────────────────────────────────────────


def _step1_parse_config(config_path: Path) -> SceneConfig:
    """Load and validate the JSON configuration file."""
    raw = config_path.read_text()
    data = json.loads(raw)
    return SceneConfig.model_validate(data)


def _step2_generate_cluster_points(
    config: SceneConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Generate all cluster points in *local* space (no placement yet).

    Returns
    -------
    local_groups : list of (n_points, 3) arrays, one per individual cluster.
    group_meta   : list of (group_index, cluster_index_within_group) pairs.
    """
    local_groups: list[np.ndarray] = []

    for _g_idx, cluster_group_cfg in enumerate(config.clusters):
        generator = create_generator(cluster_group_cfg)

        for _c_idx in range(cluster_group_cfg.n_clusters):
            density_factor = sample_float(cluster_group_cfg.density_factor, rng)

            local_pts = generator.generate(density_factor, config.dbscan.min_pts, config.dbscan.eps, rng)
            rotated   = apply_rotation(local_pts, cluster_group_cfg.rotation, rng)

            local_groups.append(rotated)

    return local_groups # pyright: ignore[reportReturnType]


def _step3_place_clusters(
    config: SceneConfig,
    local_groups: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Place cluster centers and return all cluster points in world space.

    Returns
    -------
    (N_total, 3) float32 array of all cluster points in world space.
    """
    n_clusters = len(local_groups)
    centers = place_cluster_centers(
        n_total=n_clusters,
        cluster_distance=config.cluster_distance,
        eps=config.dbscan.eps,
        rng=rng,
    )

    world_groups: list[np.ndarray] = []
    for local_pts, center in zip(local_groups, centers, strict=False):
        world_groups.append((local_pts + center).astype(np.float32))

    if not world_groups:
        return np.zeros((0, 3), dtype=np.float32)
    return np.vstack(world_groups)


def _step4_generate_noise(
    config: SceneConfig,
    cluster_points: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate all noise types, each avoiding the already-occupied space."""
    return generate_all_noise(config, cluster_points, rng)


def _step5_run_dbscan(
    all_points: np.ndarray,
    config: SceneConfig,
) -> np.ndarray:
    """Run DBSCAN on the full point cloud and return integer label array."""
    db = DBSCAN(eps=config.dbscan.eps, min_samples=config.dbscan.min_pts, n_jobs=-1)
    labels = db.fit_predict(all_points)
    return labels.astype(np.int32)


def _step6_assign_colors(labels: np.ndarray) -> np.ndarray:
    """Map DBSCAN labels to RGB colors via the golden-ratio hue sequence."""
    return assign_colors(labels)


def _step7_write_ply(
    config: SceneConfig,
    output_dir: Path,
    all_points: np.ndarray,
    labels: np.ndarray,
    colors: np.ndarray,
    comments: list[str] | None = None,
) -> Path:
    """Write the PLY file and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{config.id}.ply"
    write_ply(out_path, all_points, labels, colors, comments=comments)
    return out_path


# ── CLI entry point ───────────────────────────────────────────────────────────


def _print_warnings(caught: list) -> None:
    for w in caught:
        err_console.print(f"  [bold yellow]⚠[/bold yellow]  {w.message}")


def _step(n: int, label: str) -> None:
    console.print(f"[bold cyan][{n}/7][/bold cyan] {label}")


@app.command()
def main(
    config_path: Path = typer.Argument(
        ...,
        help="Path to the cloud_config.json file.",
        exists=True,
        readable=True,
    ),
    output: Path = typer.Option(
        Path("."),
        "--output",
        "-o",
        help="Directory where the .ply file is written (created if absent).",
    ),
) -> None:
    # ── Step 1: Parse ────────────────────────────────────────────────────────
    _step(1, f"Parsing config: [bold]{config_path}[/bold]")
    config = _step1_parse_config(config_path)
    rng = np.random.default_rng(config.seed)

    total_clusters = sum(g.n_clusters for g in config.clusters)

    info = Table.grid(padding=(0, 2))
    info.add_row("[dim]id[/dim]",           config.id)
    info.add_row("[dim]seed[/dim]",         str(config.seed))
    info.add_row("[dim]eps[/dim]",          str(config.dbscan.eps))
    info.add_row("[dim]min_pts[/dim]",      str(config.dbscan.min_pts))
    info.add_row("[dim]clusters[/dim]",     str(total_clusters))
    info.add_row("[dim]noise types[/dim]",  str(len(config.noise)))
    console.print(info)

    # ── Step 2: Generate local cluster points ─────────────────────────────
    _step(2, f"Generating [bold]{total_clusters}[/bold] cluster(s) in local space …")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        local_groups = _step2_generate_cluster_points(config, rng)

    _print_warnings(caught)
    total_cluster_pts = sum(len(g) for g in local_groups)
    console.print(f"  [green]✓[/green] {total_cluster_pts:,} cluster points")

    # ── Step 3: Place clusters in world space ─────────────────────────────
    _step(3, "Placing cluster centers around (0, 0, 0) …")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cluster_points = _step3_place_clusters(config, local_groups, rng) # pyright: ignore[reportArgumentType]

    _print_warnings(caught)

    # ── Step 4: Generate noise ────────────────────────────────────────────
    if config.noise:
        _step(4, f"Generating noise ({len(config.noise)} type(s)) …")
        noise_points = _step4_generate_noise(config, cluster_points, rng)
        console.print(f"  [green]✓[/green] {len(noise_points):,} noise points")
    else:
        _step(4, "[dim]No noise configured — skipping.[/dim]")
        noise_points = np.zeros((0, 3), dtype=np.float32)

    # ── Combine cluster + noise ───────────────────────────────────────────
    all_points = (
        np.vstack([cluster_points, noise_points]) if len(noise_points) > 0
        else cluster_points
    )

    # ── Step 5: Run DBSCAN ────────────────────────────────────────────────
    _step(5, f"Running DBSCAN on [bold]{len(all_points):,}[/bold] points …")
    labels = _step5_run_dbscan(all_points, config)

    n_found = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    n_noise_dbscan = int((labels == -1).sum())
    console.print(
        f"  [green]✓[/green] {n_found} cluster(s) found, "
        f"{n_noise_dbscan:,} noise point(s)"
    )

    if n_found != total_clusters:
        err_console.print(
            f"  [bold yellow]⚠[/bold yellow]  Expected {total_clusters} cluster(s) "
            f"but DBSCAN found {n_found}.  "
            "Consider adjusting [italic]density_factor[/italic] or cluster parameters."
        )

    # Build metadata comment for the PLY header so C++ tests can read
    # eps/min_pts (to reproduce the DBSCAN run) and the ground-truth
    # cluster sizes/noise count (to compare against vdbscan output).
    cluster_ids, cluster_sizes_arr = np.unique(
        labels[labels >= 0], return_counts=True
    )
    del cluster_ids
    sorted_sizes = ",".join(str(s) for s in sorted(cluster_sizes_arr.tolist()))
    ply_comment = (
        f"vdbscan"
        f" eps={config.dbscan.eps}"
        f" min_pts={config.dbscan.min_pts}"
        f" n_clusters={n_found}"
        f" n_noise={n_noise_dbscan}"
        f" sizes={sorted_sizes}"
    )

    # ── Step 6: Assign colors ─────────────────────────────────────────────
    _step(6, "Assigning colors …")
    colors = _step6_assign_colors(labels)

    # ── Step 7: Write PLY ─────────────────────────────────────────────────
    _step(7, f"Writing PLY to [bold]{output}[/bold] …")
    out_path = _step7_write_ply(
        config, output, all_points, labels, colors, comments=[ply_comment]
    )
    console.print(f"  [green]✓[/green] [bold]{out_path}[/bold]")


if __name__ == "__main__":
    app()
