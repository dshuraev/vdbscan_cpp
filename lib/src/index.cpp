#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <libmorton/morton.h>
#include <vector>

#include "index.hpp"

namespace {

/// Search radius (in voxels) of the precomputed neighborhood; 1 → 3x3x3.
const uint_fast8_t NEIGHBOR_RADIUS = 1;

/// Bias added to signed voxel coordinates before Morton encoding so that the
/// signed range [−2^20, 2^20−1] maps to the unsigned range [0, 2^21−1].
/// With 21 bits per axis and 3 axes, Morton codes fit in 63 bits (uint64_t).
const int32_t MORTON_BIAS = 1 << 20;

/// Minimum and maximum valid quantized voxel coordinate (inclusive).
const int32_t MORTON_COORD_MIN = -(1 << 20);
const int32_t MORTON_COORD_MAX = (1 << 20) - 1;

/// Compiler hint for non-aliased pointer parameters, enabling better
/// auto-vectorization in hot loops.
#if defined(__clang__) || defined(__GNUC__)
#define VDBSCAN_RESTRICT __restrict__
#else
#define VDBSCAN_RESTRICT
#endif

/// Maps a single floating-point coordinate to its integer voxel index.
/// \p inv is the precomputed reciprocal of epsilon (voxel side length).
/// The result is clamped to the Morton-encodable range.
[[nodiscard]] inline int32_t quantize(float coord, float inv) {
  return std::clamp((int32_t)std::floor((coord * inv)), MORTON_COORD_MIN,
                    MORTON_COORD_MAX);
}

/// Quantizes all \p len elements of the float array \p src into the int32_t
/// array \p dst using \p inv as the reciprocal of the voxel side length.
/// The loop is annotated for SIMD vectorization.
inline void quantize_axis(const float *VDBSCAN_RESTRICT src,
                          int32_t *VDBSCAN_RESTRICT dst, std::size_t len,
                          float inv) {
  assert(src != nullptr);
  assert(dst != nullptr);

#if defined(__clang__)
#pragma clang loop vectorize(enable) interleave(enable)
#endif
  for (std::size_t i = 0; i < len; ++i) {
    dst[i] = quantize(src[i], inv);
  }
}

/// Converts a signed voxel coordinate to the unsigned range required by
/// libmorton by adding \c MORTON_BIAS.
/// \pre \p coord ∈ [MORTON_COORD_MIN, MORTON_COORD_MAX].
[[nodiscard]] inline uint32_t morton_coord(int32_t coord) {
  assert(coord >= MORTON_COORD_MIN);
  assert(coord <= MORTON_COORD_MAX);
  return static_cast<uint32_t>(coord + MORTON_BIAS);
}

/// Returns the 64-bit Morton (Z-order) code for voxel (\p x, \p y, \p z).
[[nodiscard]] inline uint64_t morton_encode_one(int32_t x, int32_t y,
                                                int32_t z) {
  return static_cast<uint64_t>(libmorton::morton3D_64_encode(
      morton_coord(x), morton_coord(y), morton_coord(z)));
}

/// Associates a point's Morton code and quantized voxel position with its
/// original (pre-sort) index in the input cloud.
struct MortonRef {
  uint64_t morton_code; ///< 64-bit Z-order code for this point's voxel.
  std::size_t index;    ///< Original point index in the unsorted cloud.
  int32_t x;            ///< Quantized voxel X coordinate.
  int32_t y;            ///< Quantized voxel Y coordinate.
  int32_t z;            ///< Quantized voxel Z coordinate.
};

/// Fills \p morton_refs with Morton codes and quantized coordinates for all
/// \p len points defined by the integer coordinate arrays \p vx, \p vy, \p vz.
/// The loop is annotated for SIMD vectorization.
inline void morton_encode(const int32_t *VDBSCAN_RESTRICT vx,
                          const int32_t *VDBSCAN_RESTRICT vy,
                          const int32_t *VDBSCAN_RESTRICT vz,
                          MortonRef *VDBSCAN_RESTRICT morton_refs,
                          std::size_t len) {
  assert(vx != nullptr);
  assert(vy != nullptr);
  assert(vz != nullptr);
  assert(morton_refs != nullptr);

  for (std::size_t i = 0; i < len; ++i) {
    morton_refs[i] = MortonRef{morton_encode_one(vx[i], vy[i], vz[i]), i, vx[i],
                               vy[i], vz[i]};
  }
}

/// Sorts \p refs in ascending order of \c MortonRef::morton_code using an
/// in-place 8-bit radix sort (8 passes over 64-bit keys).
/// O(n) time, O(n) auxiliary space.
///
/// The even number of passes (8) ensures the final sorted data ends up back
/// in \p refs rather than in the scratch buffer.
inline void radix_sort_morton_refs(std::vector<MortonRef> &refs) {
  constexpr std::size_t RADIX_BITS = 8;
  constexpr std::size_t RADIX_SIZE = 1u << RADIX_BITS;
  constexpr uint64_t RADIX_MASK = RADIX_SIZE - 1;

  const std::size_t len = refs.size();
  if (len < 2) {
    return;
  }

  auto scratch = std::vector<MortonRef>(len);
  auto counts = std::array<std::size_t, RADIX_SIZE>{};

  MortonRef *src = refs.data();
  MortonRef *dst = scratch.data();

  for (std::size_t shift = 0; shift < 64; shift += RADIX_BITS) {
    counts.fill(0);

    for (std::size_t i = 0; i < len; ++i) {
      const auto bucket =
          static_cast<std::size_t>((src[i].morton_code >> shift) & RADIX_MASK);
      ++counts[bucket];
    }

    std::size_t offset = 0;
    for (std::size_t bucket = 0; bucket < RADIX_SIZE; ++bucket) {
      const auto count = counts[bucket];
      counts[bucket] = offset;
      offset += count;
    }

    for (std::size_t i = 0; i < len; ++i) {
      const auto bucket =
          static_cast<std::size_t>((src[i].morton_code >> shift) & RADIX_MASK);
      dst[counts[bucket]++] = src[i];
    }

    std::swap(src, dst);
  }
}

/// Run-length encodes the Morton-sorted \p sorted_refs into a vector of
/// \c VoxelSpan entries (one per distinct Morton code) and populates
/// \p point_to_voxel so that each sorted-point index maps to its voxel index.
[[nodiscard]] inline std::vector<VoxelSpan>
build_voxel_spans(const std::vector<MortonRef> &sorted_refs,
                  std::vector<std::size_t> &point_to_voxel) {
  const std::size_t len = sorted_refs.size();
  auto spans = std::vector<VoxelSpan>{};
  point_to_voxel.clear();
  point_to_voxel.resize(len);
  if (len == 0) {
    return spans;
  }

  spans.reserve(len);
  std::size_t span_idx = 0;
  std::size_t run_start = 0;
  uint64_t run_code = sorted_refs[0].morton_code;

  for (std::size_t i = 1; i < len; ++i) {
    if (sorted_refs[i].morton_code == run_code) {
      continue;
    }

    spans.push_back(
        VoxelSpan{run_start, i - run_start, run_code, sorted_refs[run_start].x,
                  sorted_refs[run_start].y, sorted_refs[run_start].z});
    for (std::size_t point_idx = run_start; point_idx < i; ++point_idx) {
      point_to_voxel[point_idx] = span_idx;
    }
    ++span_idx;
    run_start = i;
    run_code = sorted_refs[i].morton_code;
  }

  spans.push_back(VoxelSpan{run_start, len - run_start, run_code,
                            sorted_refs[run_start].x, sorted_refs[run_start].y,
                            sorted_refs[run_start].z});
  for (std::size_t point_idx = run_start; point_idx < len; ++point_idx) {
    point_to_voxel[point_idx] = span_idx;
  }
  spans.shrink_to_fit();
  return spans;
}

/// Binary-searches \p spans (sorted by Morton code) for the span whose
/// \c morton_code equals \p morton_code.
/// Returns the span's index, or \c INVALID_SPAN if no such span exists.
[[nodiscard]] inline std::size_t
find_voxel_span(const std::vector<VoxelSpan> &spans, uint64_t morton_code) {
  const auto it = std::lower_bound(spans.begin(), spans.end(), morton_code,
                                   [](const VoxelSpan &span, uint64_t code) {
                                     return span.morton_code < code;
                                   });
  if (it == spans.end() || it->morton_code != morton_code) {
    return INVALID_SPAN;
  }
  return static_cast<std::size_t>(it - spans.begin());
}

/// Builds the 3x3x3 voxel neighborhood lookup table for every span in \p spans.
///
/// For each voxel, iterates over all 27 relative offsets
/// (dx, dy, dz) ∈ {−1, 0, 1}³, Morton-encodes the neighbor's coordinates,
/// and binary-searches for the corresponding span.  Missing or out-of-range
/// neighbors are recorded as \c INVALID_SPAN.
[[nodiscard]] inline std::vector<NeighborList>
build_voxel_neighbor_lut(const std::vector<VoxelSpan> &spans) {
  auto lut = std::vector<NeighborList>(spans.size());

  for (std::size_t i = 0; i < spans.size(); ++i) {
    auto &neighbors = lut[i];
    neighbors.fill(INVALID_SPAN);

    std::size_t neighbor_idx = 0;
    for (int32_t dz = -static_cast<int32_t>(NEIGHBOR_RADIUS);
         dz <= static_cast<int32_t>(NEIGHBOR_RADIUS); ++dz) {
      for (int32_t dy = -static_cast<int32_t>(NEIGHBOR_RADIUS);
           dy <= static_cast<int32_t>(NEIGHBOR_RADIUS); ++dy) {
        for (int32_t dx = -static_cast<int32_t>(NEIGHBOR_RADIUS);
             dx <= static_cast<int32_t>(NEIGHBOR_RADIUS); ++dx) {
          const int32_t nx = spans[i].x + dx;
          const int32_t ny = spans[i].y + dy;
          const int32_t nz = spans[i].z + dz;

          if (nx < MORTON_COORD_MIN || nx > MORTON_COORD_MAX ||
              ny < MORTON_COORD_MIN || ny > MORTON_COORD_MAX ||
              nz < MORTON_COORD_MIN || nz > MORTON_COORD_MAX) {
            neighbors[neighbor_idx++] = INVALID_SPAN;
            continue;
          }

          neighbors[neighbor_idx++] =
              find_voxel_span(spans, morton_encode_one(nx, ny, nz));
        }
      }
    }
  }

  return lut;
}
} // namespace

Index::Index(const PointCloud &cloud, float epsilon) {
  assert(epsilon > 0.0);
  const auto len = cloud.length();

  if (len == 0) {
    sorted_cloud = PointCloud(len);
    return;
  }

  // Compute the reciprocal of epsilon, nudged slightly towards zero so that
  // voxels are fractionally larger than epsilon.  This prevents points that
  // are exactly epsilon apart from landing in adjacent voxels when they should
  // share one.
  const float inv = std::nextafter(1.0f / epsilon, 0.0f);

  // Step 1: Voxelize — quantize each axis into integer voxel coordinates.
  auto qx = std::vector<int32_t>(len);
  auto qy = std::vector<int32_t>(len);
  auto qz = std::vector<int32_t>(len);

  quantize_axis(cloud.vx.data(), qx.data(), len, inv);
  quantize_axis(cloud.vy.data(), qy.data(), len, inv);
  quantize_axis(cloud.vz.data(), qz.data(), len, inv);

  // Step 2: Morton-encode every point's voxel coordinates.
  auto morton_refs = std::vector<MortonRef>(len);
  morton_encode(qx.data(), qy.data(), qz.data(), morton_refs.data(), len);

  // Step 3: Sort by Z-order code; scatter original points into sorted_cloud.
  radix_sort_morton_refs(morton_refs);

  sorted_cloud = PointCloud();
  sorted_cloud.vx.resize(len);
  sorted_cloud.vy.resize(len);
  sorted_cloud.vz.resize(len);

  for (std::size_t i = 0; i < len; ++i) {
    const auto point_idx = morton_refs[i].index;
    sorted_cloud.vx[i] = cloud.vx[point_idx];
    sorted_cloud.vy[i] = cloud.vy[point_idx];
    sorted_cloud.vz[i] = cloud.vz[point_idx];
  }

  // Step 4: Run-length encode sorted Morton codes into voxel spans.
  voxel_spans = build_voxel_spans(morton_refs, point_to_voxel);

  // Step 5: Precompute the 3x3x3 voxel neighborhood for every span.
  voxel_neighbor_lut = build_voxel_neighbor_lut(voxel_spans);
}
