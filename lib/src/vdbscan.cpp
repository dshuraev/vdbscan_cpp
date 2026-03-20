#include <boost/circular_buffer.hpp>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include <vdbscan/vdbscan.hpp>

#include "index.hpp"
#include "tracy_fwd.hpp"

namespace {

/// Throws \c std::invalid_argument if the three given lengths are not all equal.
void validate_lengths(const std::size_t x_len, const std::size_t y_len,
                      const std::size_t z_len) {
  if (x_len != y_len || x_len != z_len) {
    throw std::invalid_argument(
        "PointCloud coordinate vectors must have the same length");
  }
}

} // namespace

namespace vdbscan {

PointCloud::PointCloud(const std::size_t capacity) {
  vx.reserve(capacity);
  vy.reserve(capacity);
  vz.reserve(capacity);
}

PointCloud::PointCloud(std::vector<float> x, std::vector<float> y,
                       std::vector<float> z)
    : vx(std::move(x)), vy(std::move(y)), vz(std::move(z)) {
  validate_lengths(vx.size(), vy.size(), vz.size());
}

PointCloud::PointCloud(std::initializer_list<float> x,
                       std::initializer_list<float> y,
                       std::initializer_list<float> z)
    : PointCloud(std::vector<float>(x), std::vector<float>(y),
                 std::vector<float>(z)) {}

void PointCloud::push(const float x, const float y, const float z) {
  vx.push_back(x);
  vy.push_back(y);
  vz.push_back(z);
}

std::size_t PointCloud::length() const noexcept { return vx.size(); }

Clustering::Clustering(PointCloud sorted_cloud, std::vector<size_t> labels)
    : cloud(std::move(sorted_cloud)), labels(std::move(labels)) {}

Clustering dbscan(const PointCloud &cloud, float epsilon,
                  uint_fast16_t min_pts) {
  ZoneScoped;
  assert(epsilon > 0);
  auto eps2 = epsilon * epsilon;
  auto cloud_index = Index(cloud, epsilon);

  // Phase 1: Determine which points are core points.
  //
  // A point is a core point if at least min_pts points (including itself) lie
  // within epsilon.  The 27-voxel LUT bounds the search without scanning the
  // full dataset; the goto exits the inner loops as soon as the threshold is
  // met, avoiding unnecessary distance checks.
  auto is_core = std::vector<uint_fast8_t>(cloud_index.sorted_cloud.length());

  {
  ZoneScopedN("dbscan/CoreDetect");
  for (size_t i = 0; i < cloud_index.sorted_cloud.length(); i++) {
    uint_fast16_t neighbors_in_range = 0;
    size_t voxel_idx = cloud_index.point_to_voxel[i];
    const NeighborList &neighbors = cloud_index.voxel_neighbor_lut[voxel_idx];

    for (size_t neighbor_idx : neighbors) {
      if (neighbor_idx == INVALID_SPAN) {
        continue;
      }

      const VoxelSpan &span = cloud_index.voxel_spans[neighbor_idx];

      for (size_t point_idx = span.start; point_idx < span.start + span.len;
           point_idx++) {

        if (cloud_index.are_within(i, point_idx, eps2)) {
          neighbors_in_range += 1;
        }

        if (neighbors_in_range >= min_pts) {
          goto MAYBE_ADD_COREPOINT;
        }
      }
    }
  MAYBE_ADD_COREPOINT:
    if (neighbors_in_range >= min_pts) {
      is_core[i] = 1;
    }
  }
  } // dbscan/CoreDetect

  // Phase 2: BFS cluster expansion from each unvisited core point.
  //
  // A circular buffer (pre-allocated to cloud size) serves as the BFS queue,
  // avoiding per-expansion heap allocations.  Non-core points reachable within
  // epsilon are assigned to the cluster but not enqueued (they cannot seed
  // further expansion).  Points not reached by any BFS retain label 0 (noise).
  auto queue = boost::circular_buffer<size_t>(is_core.size());
  auto visited = std::vector<uint_fast8_t>(cloud_index.sorted_cloud.length());
  auto labels = std::vector<size_t>(cloud_index.sorted_cloud.length());
  size_t cluster_label = 1; // Labels start at 1; 0 is reserved for noise.

  {
  ZoneScopedN("dbscan/BFS");
  for (size_t i = 0; i < cloud_index.sorted_cloud.length(); i++) {
    if (!is_core[i] || visited[i]) {
      continue;
    }
    queue.clear();
    queue.push_back(i);
    visited[i] = 1;
    labels[i] = cluster_label;
    while (!queue.empty()) {
      // Get core point from queue, lookup voxel neighbors
      const size_t point_index = queue.front();
      queue.pop_front();
      size_t voxel_idx = cloud_index.point_to_voxel[point_index];
      const NeighborList &neighbors =
          cloud_index.voxel_neighbor_lut[voxel_idx];
      {
      ZoneScopedN("dbscan/BFS/SpanScan");
      const VoxelSpan &qvoxel = cloud_index.voxel_spans[voxel_idx];
      const float qx = cloud_index.sorted_cloud.vx[point_index];
      const float qy = cloud_index.sorted_cloud.vy[point_index];
      const float qz = cloud_index.sorted_cloud.vz[point_index];
      const float cell = cloud_index.voxel_size;
      const float rx = qx - qvoxel.x * cell;
      const float ry = qy - qvoxel.y * cell;
      const float rz = qz - qvoxel.z * cell;
      for (std::size_t s = 0; s < kMaxNeighbors; ++s) {
        // We iterate over every neighbor of the voxel and bail out early if
        // - span is invalid (contains no points), or
        // - the minimum distance from the point to the voxel is greater than epsilon
        const std::size_t span_idx = neighbors[s];
        if (span_idx == INVALID_SPAN) {
          continue;
        }
        const NeighborOffset &off = kNeighborOffsets[s];
        if (off.filterable) {
          const float gx =
              (off.dx == 0) ? 0.f : (off.dx > 0) ? (cell - rx) : rx;
          const float gy =
              (off.dy == 0) ? 0.f : (off.dy > 0) ? (cell - ry) : ry;
          const float gz =
              (off.dz == 0) ? 0.f : (off.dz > 0) ? (cell - rz) : rz;
          if (gx * gx + gy * gy + gz * gz > eps2) {
            continue;
          }
        }
        const VoxelSpan &span = cloud_index.voxel_spans[span_idx];
        for (size_t nidx = span.start; nidx < span.start + span.len; nidx++) {
          // The span may contain interesting points, so we have to check every 
          // non-visited and enqueue core points
          if (!visited[nidx] &&
              cloud_index.are_within(qx, qy, qz, nidx, eps2)) {
            visited[nidx] = 1;
            labels[nidx] = cluster_label;
            if (is_core[nidx]) {
              queue.push_back(nidx);
            }
          }
        }
      }
      } // dbscan/BFS/SpanScan
    }
    cluster_label += 1;
  }
  } // dbscan/BFS
  return Clustering(std::move(cloud_index.sorted_cloud), std::move(labels));
}

} // namespace vdbscan
