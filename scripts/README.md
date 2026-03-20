# `vdbscan_cpp` helper scripts

## `synthetic-data.py`

Usage: `synthetic-data.py cloud_config.json [-o|--output DIRECTORY_PATH]`

Used to generate synthetic 3D volumetric cloud data based on generation parameters
described in `.json` configuration file (see below). Will output `.ply` file
to current directory or `DIRECTORY_PATH` is specified with the name based on the
`id` specified in configuration.

The script will first generate a point cloud based on the parameters, then apply
`sklearn.cluster.DBSCAN` to generate clusters. Note that as DBSCAN labels are not
stable across runs or implementation; only the noise label is stable (`-1` in scikit-learn).
Therefore, `label` field should only be used to determine cluster membership, not
compare clusters label values across the runs.

The `ply` file will have the following properties:

```txt
property float x
property float y
property float z
property label int
property uchar red
property uchar green
property uchar blue
```

Every cluster will be assigned its own color. Noise is grey `(128,128,128)`.

### Configuration

```json
{
  "id": "test_1", // will output `test_1.ply`
  "seed": 42, // RNG seed
  // DBSCAN parameters for oracle to generate ground truths
  "dbscan": {
    "eps": 0.35,
    "min_pts": 8
  },
  "cluster_distance": [3.0, 10.0], // minimal distance between two cluster centers
  // Description of cluster generator parameters
  "clusters": [
    {
      "type": "gaussian", // cluster generator type: isotropic gaussian cloud
      "n_clusters": 3, // number of clusters for this generator
      "points_per_cluster": 30, // number of points in the cluster: INTEGER or [MIN, MAX]
      "std": [0.15, 0.35] // standard deviation: FLOAT or [MIN, MAX]
    },
    {
      "type": "crescent",
      "n_clusters": 2,
      "points_per_cluster": 70,
      "radius": [1.0, 1.6],
      "thickness": [0.08, 0.18],
      "arc_fraction": [0.4, 0.7] // how much of an circle, 1 = full circle
    },
    {
      "type": "torus",
      "n_clusters": 2,
      "points_per_cluster": [120, 220],
      "major_radius": [0.8, 1.5],
      "aspect_ratio": 2.5 // Ratio R/r, with major radius R (center of tube to center of torus) and minor radius r (tube radius)
    },
    {
      "type": "tube", // solid
      "n_clusters": 2,
      "points_per_cluster": [60, 140],
      "length": [1.5, 4.0],
      "radius": [0.03, 0.12]
    },
    {
      "type": "cylinder", // solid, uniform point distribution
      "n_clusters": 2,
      "points_per_cluster": [80, 160],
      "radius": [0.4, 0.9],
      "height": [0.1, 0.5]
    },
    {
      "type": "ball", // solid, uniform point distribution
      "n_clusters": 2,
      "points_per_cluster": [80, 160],
      "radius": [0.4, 0.9]
    },
    {
      "type": "box",
      "n_clusters": 1,
      "points_per_cluster": 40,
      "width": 8.5,
      "height": 4.3,
      "depth": 2.1
    }
  ],
  // parameters of noise generators
  "noise": [
    {
      "type": "uniform", // uniform background noise
      "ratio": 0.1 // 10% of CLUSTER points
    },
    {
      "type": "boundary", // generates noise around clusters
      "ratio": 0.05, // 5% of CLUSTER points
      "distance_from_cluster": [0.8, 1.5] // distance from cluster
    }
  ]
}
```

NOTE: all geometric parameters (`height`, `radius`, `distance`, etc.) are generated relative to `epsilon`,
so `"height": 1.2` means $\text{height}:=1.2*\varepsilon$.

Every `clusters` also supports optional:

- `"rotation": [w, x, y, z]`: rotation indicated by quaternions, as well as `"rotation": "none"` and `"rotation": "random"` (default)
- `"density_factor": FLOAT | [MIN, MAX]`: expressing parameter $\alpha$ of cluster density (see below). Default is 2.

### Cluster Density

Let's derive approximate density of a cluster in terms of DBSCAN parameters such as $\varepsilon$:

```math
N\approx\rho V\\
V_{\varepsilon}=\frac{4}{3}\pi\varepsilon^3\\
E[k]\approx\rho\frac{4}{3}\pi\varepsilon^3\\
```

so

```math
\rho\approx\frac{E[k]}{\frac{4}{3}\pi\varepsilon^3}
```

where

- $N$ - number of points in cluster
- $V$ - cluster volume
- $\rho$ - average cluster density
- $V_\varepsilon$ - volume of $\varepsilon$ neighborhood around the point
- $E[k]$ - expected number of points within $\varepsilon$ neighborhood
- $K$ - minimum points for a given point to be considered a core point.

The expected number of points $E[k]$ should be greater or equal to $K$, the `min_points`
parameter of the DBSCAN, for a cluster to be considered a true cluster.
For practical considerations let's express this as:

```math
E[k] = \alpha K
```

- $\alpha = 1$ would be _idealized_ theoretical threshold for a given cluster to be considered a true cluster
- $\alpha < 1$ results in clusters sparser than threshold
- $\alpha > 1$ results in clusters denser than threshold

It is important to note that $E[k]$ is expected number of points in the neighborhood,
so, by nature of DBSCAN we should pick $\alpha$ with a safety factor to ensure that
the cluster is identified as true cluster or a noise cluster.

Recommended tuning parameters:

- for true clusters: $\alpha \ge 2$
- for noise clusters: $\alpha \le 0.5$

### Cluster Volume and Number of Points

Cluster volume is defined in relation to $\varepsilon$: `"height": 1.2` implies
$\text{height}:=1.2\varepsilon$. This is made to allow easier reasoning about the
size of clusters when varying $\varepsilon$.

Number of points in cluster is

```math
N \approx V\frac{\alpha K}{\frac{4}{3}\pi\varepsilon^3}
```

It is important to note that if a cluster is _too small_, there may not be enough
points within $\varepsilon$ for a cluster to be considered a true cluster even with
$\alpha > 1$.

The generation algorithm will produce a warning when a given cluster dimension
(`height`, `thickness`, etc.) is below 2 (corresponding to $2\varepsilon$).
