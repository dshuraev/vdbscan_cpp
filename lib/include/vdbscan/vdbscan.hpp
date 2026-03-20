#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace vdbscan {

/// A 3-D point cloud stored in Structure-of-Arrays (SoA) layout.
///
/// Keeping X, Y, and Z coordinates in separate contiguous vectors enables
/// SIMD-friendly vectorization during spatial indexing and distance
/// computations.
class PointCloud {
public:
  PointCloud() = default;

  /// Constructs an empty cloud with storage pre-reserved for \p capacity points.
  explicit PointCloud(std::size_t capacity);

  /// Constructs a cloud by taking ownership of existing coordinate vectors.
  /// \throws std::invalid_argument if the vectors do not all have the same size.
  PointCloud(std::vector<float> x, std::vector<float> y, std::vector<float> z);

  /// Constructs a cloud from initializer lists of coordinates.
  /// \throws std::invalid_argument if the lists do not all have the same size.
  PointCloud(std::initializer_list<float> x, std::initializer_list<float> y,
             std::initializer_list<float> z);

  /// Appends point (\p x, \p y, \p z) to the cloud.
  void push(float x, float y, float z);

  /// Returns the number of points in the cloud.
  [[nodiscard]] std::size_t length() const noexcept;

  std::vector<float> vx; ///< X coordinates.
  std::vector<float> vy; ///< Y coordinates.
  std::vector<float> vz; ///< Z coordinates.
};

/// The result of a DBSCAN clustering run.
///
/// \c cloud is a Morton-sorted copy of the input (see \c dbscan).
/// \c labels[i] is the cluster label for \c cloud's i-th point:
///   - \c 0  — noise (no cluster).
///   - \c ≥1 — cluster identifier.
class Clustering {
public:
  Clustering() = default;

  /// Constructs a result from an already-sorted cloud and its per-point labels.
  Clustering(PointCloud sorted_cloud, std::vector<size_t> labels);

  PointCloud cloud;           ///< Points in Morton (Z-order) sorted order.
  std::vector<size_t> labels; ///< Per-point cluster label; 0 = noise.
};

/// Clusters a 3-D point cloud using the DBSCAN algorithm, accelerated by a
/// voxel-Morton spatial index.
///
/// Points are quantized into cubic voxels of side length \p epsilon and sorted
/// along a Morton (Z-order) space-filling curve.  Neighbor queries during both
/// core-point detection and BFS cluster expansion are answered via a
/// precomputed 3x3x3 voxel look-up table, limiting per-point search cost to
/// O(27 x avg_voxel_density) instead of O(n).
///
/// \param cloud    Input point cloud (unmodified).
/// \param epsilon  Neighborhood radius; must be > 0.
/// \param min_pts  Minimum number of points within \p epsilon (inclusive) for a
///                 point to be considered a core point.
/// \returns        A \c Clustering whose \c cloud is in Morton order and whose
///                 \c labels assign each point to a cluster (≥ 1) or noise (0).
[[nodiscard]] Clustering dbscan(const PointCloud &cloud, float epsilon,
                                uint_fast16_t min_pts);

} // namespace vdbscan
