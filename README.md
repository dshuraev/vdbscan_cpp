# `vdbscan_cpp`

High-performance DBSCAN point-cloud clustering for C++17, accelerated by a Morton-order voxel spatial index.

## Table of Contents

- [`vdbscan_cpp`](#vdbscan_cpp)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Theory](#theory)
    - [DBSCAN](#dbscan)
    - [Voxel-Morton Acceleration](#voxel-morton-acceleration)
  - [Requirements](#requirements)
  - [Building](#building)
    - [CMake options](#cmake-options)
  - [Usage](#usage)
    - [Library API](#library-api)
    - [CLI](#cli)
      - [Input formats](#input-formats)
      - [Output modes](#output-modes)
      - [Examples](#examples)
  - [Testing](#testing)
    - [Benchmarking on KITTI data](#benchmarking-on-kitti-data)
  - [Performance](#performance)
  - [Roadmap](#roadmap)
  - [License](#license)

---

## Overview

`vdbscan_cpp` clusters 3D point clouds using the DBSCAN algorithm with a spatial index that reduces per-point neighborhood queries from O(n) to O(1). On typical LiDAR data (KITTI autonomous-driving scans) it sustains throughput in the tens of millions of points per second.

**Key properties**:

- Expected **O(n)** time complexity for uniform and clustered data
- Automatic **noise labelling** — no need to know the number of clusters upfront
- Configurable via two intuitive parameters: neighborhood radius `epsilon` and `min_pts`
- Accepts **.xyz**, **.ply**, and **.bin** (KITTI) point-cloud formats
- Clean C++17 library API + standalone CLI tool

---

## Theory

### DBSCAN

[DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) (_Density-Based Spatial Clustering of Applications with Noise_) groups points by local density:

- A **core point** has at least `min_pts` neighbors within distance `epsilon`.
- A **border point** is within `epsilon` of a core point but is not core itself.
- Any point that is neither core nor reachable from a core point is **noise**.

Clusters grow by recursively expanding from core points to all density-reachable neighbors. Unlike k-means, DBSCAN discovers clusters of arbitrary shape and marks outliers explicitly — both very desirable properties for real-world sensor data.

The naive implementation checks every pair of points, giving O(n²) time. A spatial index reduces the neighborhood query to O(log n) (k-d tree) or O(1) expected (hash-based), making the overall algorithm O(n log n) or O(n).

### Voxel-Morton Acceleration

This library uses a custom spatial index built in five passes:

**1. Quantization** — The 3D space is divided into a regular grid of cubic voxels with side length `epsilon`. Each point's floating-point coordinates are snapped to integer voxel indices. Because the voxel side equals `epsilon`, any two points in the same or adjacent voxels are candidates for the ε-neighborhood; points two or more voxels apart are guaranteed to be farther than `epsilon`.

**2. Morton encoding** — Each voxel's (x, y, z) integer coordinates are interleaved bitwise into a single 64-bit [Morton code](https://en.wikipedia.org/wiki/Z-order_curve) (Z-order curve). Morton codes have the property that spatially close voxels produce numerically close codes, so sorting by Morton code groups nearby voxels together in memory.

**3. Radix sort** — All points are sorted by their Morton code in O(n) time using an 8-bit radix sort (8 passes). After sorting, points belonging to the same voxel are contiguous.

**4. Run-length encoding** — Consecutive points with equal Morton codes are compressed into `VoxelSpan` records (`start`, `count`). This produces one compact entry per occupied voxel.

**5. 3x3x3 neighbor lookup table (LUT)** — For every occupied voxel, the 26 neighboring voxels (plus itself) are located via binary search on the Morton-sorted span list and stored in a fixed-size 27-element array. This one-time O(n) construction amortizes all future neighborhood queries.

At query time, finding all $\varepsilon$-neighbors of a point costs exactly **27 span lookups** — constant time regardless of dataset size. Distance checks are then performed only within the (small) candidate set, and the structure-of-arrays memory layout enables SIMD vectorization of the inner distance loop.

---

## Requirements

| Tool                                                                       | Version | Purpose                                        |
| -------------------------------------------------------------------------- | ------- | ---------------------------------------------- |
| CMake                                                                      | ≥ 3.20  | Build system                                   |
| C++ compiler                                                               | C++17   | GCC or Clang                                   |
| [task](https://taskfile.dev/)                                              | any     | Convenient build targets                       |
| [uv](https://docs.astral.sh/uv/)                                           | any     | `test:synthetic` and synthetic data generation |
| `gcov` / `llvm-cov gcov`                                                   | —       | `test:coverage` (matches your toolchain)       |
| [`cargo-flamegraph`](https://github.com/flamegraph-rs/flamegraph) + `perf` | —       | `perf:flamegraph` targets                      |

Dependencies (Boost headers, libmorton, Catch2, CLI11, Google Benchmark, Tracy) are fetched automatically by CMake via `FetchContent`.

## Building

```sh
task build:cli:release   # CLI executable + library (recommended)
task build:lib:release   # Library only

task clean               # Remove all build artifacts and .cache/
task --list              # Full list of available targets
```

To switch to GCC (default is Clang):

```sh
task build:cli:release TOOLCHAIN=gcc
```

### CMake options

| Option                    | Default | Description                                                               |
| ------------------------- | ------- | ------------------------------------------------------------------------- |
| `VDBSCAN_BUILD_CLI`       | ON      | Build the CLI executable                                                  |
| `VDBSCAN_BENCHMARK_KITTI` | OFF     | Build the KITTI benchmark                                                 |
| `VDBSCAN_TEST`            | OFF     | Build unit and integration tests                                          |
| `VDBSCAN_TRACY`           | OFF     | Enable [Tracy](https://github.com/wolfpld/tracy) profiler instrumentation |
| `VDBSCAN_COVERAGE`        | OFF     | Enable gcov/llvm-cov coverage                                             |

## Usage

### Library API

```cpp
#include <vdbscan/vdbscan.hpp>

// Build a point cloud
PointCloud cloud;
cloud.push(x, y, z);   // add individual points ...
// ... see PointCloud API for initialization methods

// Run clustering
float epsilon  = 0.5f;  // neighborhood radius (same units as your coordinates)
size_t min_pts = 8;     // minimum neighbors to be a core point

Clustering result = dbscan(cloud, epsilon, min_pts);

// result.cloud  — points reordered by Morton code (same set, different order)
// result.labels — parallel array: 0 = noise, 1..N = cluster ID
```

`PointCloud` stores coordinates in a structure-of-arrays layout (`vx`, `vy`, `vz`) for cache and SIMD efficiency.

### CLI

```text
vdbscan_cli -i <input> -o <output> -e <epsilon> -m <min_pts>
```

| Flag              | Description                                      |
| ----------------- | ------------------------------------------------ |
| `-i`, `--input`   | Input file (`.ply`, `.xyz`, `.bin`)              |
| `-o`, `--output`  | Output file (`.ply`, `.xyz`) or directory        |
| `-e`, `--epsilon` | Neighborhood radius (must be > 0)                |
| `-m`, `--min-pts` | Minimum neighbors for a core point (must be > 0) |

#### Input formats

- `.xyz` — ASCII, one `x y z` triplet per line
- `.bin` — KITTI binary format (4×float32 per point; 4th value ignored)
- `.ply` — ASCII or binary PLY with `x`, `y`, `z` vertex properties

#### Output modes

- **Single file** (`.ply` or `.xyz`): All points colored by cluster. Noise is gray; clusters get distinct colors via golden-ratio HSV distribution.
- **Directory**: Separate files per cluster (`cluster_1.xyz`, `cluster_2.xyz`, ...) plus `noise.xyz`.

#### Examples

```sh
# Cluster a LiDAR scan and write a colored PLY
vdbscan_cli -i scan.ply -o colored.ply -e 0.5 -m 8

# Write clusters to separate files in a directory
vdbscan_cli -i scan.bin -o clusters/ -e 0.75 -m 4

# ASCII input/output
vdbscan_cli -i scan.xyz -o result.xyz -e 0.25 -m 6
```

---

## Testing

```sh
task test:unit:run       # Unit tests (Catch2)
task test:synthetic:run  # Integration tests vs. sklearn ground truth
task test:all:run        # Both
task test:coverage       # HTML + text coverage report
```

Unit tests cover edge cases including: empty clouds, single points, exact-distance boundaries, border-point semantics, duplicate coordinates, insertion-order independence, quantization boundaries, and extreme coordinate clamping.

Integration tests generate synthetic point clouds (Gaussian blobs, toroids, crescents, etc.), cluster them with both `vdbscan` and scikit-learn's DBSCAN, and compare cluster counts and per-cluster sizes within a 5% tolerance.

### Benchmarking on KITTI data

```sh
task build:bench:kitti KITTI_PATH=/path/to/kitti/velodyne
task bench:kitti:run
```

**Flamegraph** (requires `cargo-flamegraph` + `perf`):

```sh
task perf:flamegraph:kitti KITTI_PATH=/path/to/kitti/velodyne

# Optional: restrict to a specific benchmark case
task perf:flamegraph:kitti KITTI_PATH=/path/to/kitti/velodyne BENCH_FILTER=<regex>
```

**Tracy real-time profiler** — capture a trace:

```sh
# Step 1: start the capture process (writes a .trace file named after the current git SHA)
task tracy

# Step 2: in another terminal, run the benchmark (connect to the waiting capture)
task bench:kitti:tracy KITTI_PATH=/path/to/kitti/velodyne

# Optional: profile only a single epsilon/min_pts combination
task bench:kitti:tracy KITTI_PATH=/path/to/kitti/velodyne EPSILON=0.5 MINPTS=8
```

`task build:tracy:capture` compiles the bundled `tracy-capture` binary from the Tracy v0.11.1
sources fetched by CMake; run it once if `tracy-capture` is not already on `PATH`.

## Performance

The index construction is O(n) (radix sort dominates in practice) and neighborhood queries are O(1) per point, giving **O(n) overall** for clustered or uniform data.

Memory layout choices that help throughput:

- SoA coordinate storage for SIMD-vectorized distance loops
- Single pre-allocated circular buffer for the BFS expansion queue
- 27-element neighbor LUT avoids any per-query allocations

Profiler zones (visible in Tracy or flame graphs) map directly to the five index-construction passes and the two DBSCAN phases, making it straightforward to identify bottlenecks on your specific data.

## Roadmap

- [x] Library core
- [x] CLI
- [x] Benchmarks (KITTI dataset)
- [x] Unit tests
- [x] Integration tests against scikit-learn oracle
- [x] Perf traces / flamegraph
- [x] Tracy profiler instrumentation
- [ ] Fuzzing

## License

MIT. See [LICENSE](./LICENSE).
