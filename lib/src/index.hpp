#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <vdbscan/vdbscan.hpp>

/// A contiguous run of Morton-sorted points that all fall within the same
/// voxel cell.  Voxels are identified uniquely by their \c morton_code.
struct VoxelSpan {
  std::size_t start;    ///< Index of the first point in the sorted cloud.
  std::size_t len;      ///< Number of points in this voxel.
  uint64_t morton_code; ///< 64-bit Morton (Z-order) code of this voxel.
  int32_t x;            ///< Quantized voxel X coordinate.
  int32_t y;            ///< Quantized voxel Y coordinate.
  int32_t z;            ///< Quantized voxel Z coordinate.
};

/// Maximum voxels in the 3x3x3 neighborhood of any given voxel.
inline constexpr std::size_t kMaxNeighbors = 27;

/// Fixed-size array of up to \c kMaxNeighbors voxel-span indices.
/// Slots that correspond to absent (out-of-bounds or empty) neighbors are
/// set to \c INVALID_SPAN.
using NeighborList = std::array<std::size_t, kMaxNeighbors>;

/// Sentinel index indicating that a neighbor voxel does not exist in the index.
inline constexpr std::size_t INVALID_SPAN =
    std::numeric_limits<std::size_t>::max();

/// Voxel-Morton spatial index built from a \c PointCloud.
///
/// Construction proceeds in five steps:
///  1. Quantize each point's floating-point coordinates into integer voxel
///     coordinates using the supplied \p epsilon as the voxel side length.
///  2. Morton-encode the voxel coordinates into 64-bit Z-order codes.
///  3. Radix-sort all points by Morton code; store the reordered points in
///     \c sorted_cloud.
///  4. Run-length encode the sorted codes into \c voxel_spans and populate
///     \c point_to_voxel.
///  5. Precompute the 3x3x3 voxel neighborhood for every span into
///     \c voxel_neighbor_lut.
///
/// After construction, DBSCAN neighbor lookups touch at most 27 voxels per
/// point regardless of dataset size.
class Index {
public:
  vdbscan::PointCloud sorted_cloud;             ///< Points reordered by Morton code.
  std::vector<VoxelSpan> voxel_spans;           ///< All occupied voxels in Morton order.
  std::vector<NeighborList> voxel_neighbor_lut; ///< 27-entry neighbor table per voxel.
  std::vector<std::size_t> point_to_voxel;      ///< Maps sorted-point index → voxel index.

  /// Builds the spatial index from \p cloud using \p epsilon as the voxel
  /// side length.
  /// \pre \p epsilon > 0.
  Index(const vdbscan::PointCloud &cloud, float epsilon);

  /// Returns \c true if the Euclidean distance between sorted-cloud points
  /// \p a and \p b is at most \c sqrt(eps_squared).
  [[nodiscard]] inline bool are_within(std::size_t a, std::size_t b,
                                       float eps_squared) const {
    auto ax = sorted_cloud.vx[a];
    auto ay = sorted_cloud.vy[a];
    auto az = sorted_cloud.vz[a];

    auto bx = sorted_cloud.vx[b];
    auto by = sorted_cloud.vy[b];
    auto bz = sorted_cloud.vz[b];

    auto dx = ax - bx;
    auto dy = ay - by;
    auto dz = az - bz;

    return (dx * dx + dy * dy + dz * dz) <= eps_squared;
  }
};
