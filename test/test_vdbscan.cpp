#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <vdbscan/vdbscan.hpp>

using namespace vdbscan;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static std::set<std::size_t>
unique_clusters(const std::vector<std::size_t> &labels) {
  std::set<std::size_t> s;
  for (auto l : labels)
    if (l != 0)
      s.insert(l);
  return s;
}

static std::size_t count_noise(const std::vector<std::size_t> &labels) {
  return static_cast<std::size_t>(
      std::count(labels.begin(), labels.end(), std::size_t{0}));
}

// Returns the sorted-cloud index of the first point with the given x
// coordinate, or SIZE_MAX if not found.
static std::size_t find_x(const Clustering &c, float x) {
  for (std::size_t i = 0; i < c.cloud.length(); ++i)
    if (c.cloud.vx[i] == x)
      return i;
  return SIZE_MAX;
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. PointCloud constructor validation
//    lib/src/vdbscan.cpp : validate_lengths (line 15)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("PointCloud vector constructor throws on mismatched lengths",
          "[pointcloud][constructor]") {
  // x longer than y / z
  REQUIRE_THROWS_AS(
      PointCloud(std::vector<float>{1.f, 2.f}, std::vector<float>{1.f},
                 std::vector<float>{1.f, 2.f}),
      std::invalid_argument);
  // y and z mismatch while x == y
  REQUIRE_THROWS_AS(
      PointCloud(std::vector<float>{1.f}, std::vector<float>{1.f},
                 std::vector<float>{1.f, 2.f, 3.f}),
      std::invalid_argument);
}

TEST_CASE(
    "PointCloud initializer-list constructor throws on mismatched lengths",
    "[pointcloud][constructor]") {
  REQUIRE_THROWS_AS(
      PointCloud(std::initializer_list<float>{1.f, 2.f},
                 std::initializer_list<float>{1.f},
                 std::initializer_list<float>{1.f, 2.f}),
      std::invalid_argument);
}

TEST_CASE("PointCloud constructors accept equal-length inputs",
          "[pointcloud][constructor]") {
  REQUIRE_NOTHROW(PointCloud(std::vector<float>{1.f, 2.f},
                             std::vector<float>{3.f, 4.f},
                             std::vector<float>{5.f, 6.f}));
  REQUIRE_NOTHROW(
      PointCloud(std::initializer_list<float>{1.f},
                 std::initializer_list<float>{2.f},
                 std::initializer_list<float>{3.f}));
  // zero-length is valid
  REQUIRE_NOTHROW(PointCloud(std::vector<float>{}, std::vector<float>{},
                             std::vector<float>{}));
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. PointCloud::push and length() consistency
//    lib/src/vdbscan.cpp : push (line 44), length (line 50)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("PointCloud::push and length() are consistent",
          "[pointcloud][push]") {
  PointCloud pc;
  REQUIRE(pc.length() == 0);

  pc.push(1.f, 2.f, 3.f);
  REQUIRE(pc.length() == 1);
  REQUIRE(pc.vx[0] == 1.f);
  REQUIRE(pc.vy[0] == 2.f);
  REQUIRE(pc.vz[0] == 3.f);

  pc.push(4.f, 5.f, 6.f);
  REQUIRE(pc.length() == 2);
  REQUIRE(pc.vx[1] == 4.f);
  REQUIRE(pc.vy[1] == 5.f);
  REQUIRE(pc.vz[1] == 6.f);
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Empty cloud behavior
//    lib/src/index.cpp : Index::Index (line 259)
//    lib/src/vdbscan.cpp : dbscan (line 55)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("dbscan on empty cloud returns empty sorted cloud and empty labels",
          "[dbscan][empty]") {
  PointCloud empty;
  auto result = dbscan(empty, 1.0f, 2);
  REQUIRE(result.cloud.length() == 0);
  REQUIRE(result.labels.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. Single-point and tiny-cloud cases
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Single point with min_pts=1 forms a cluster", "[dbscan][single]") {
  PointCloud pc{{0.f}, {0.f}, {0.f}};
  auto result = dbscan(pc, 1.0f, 1);
  REQUIRE(result.labels.size() == 1);
  REQUIRE(unique_clusters(result.labels).size() == 1);
  REQUIRE(count_noise(result.labels) == 0);
}

TEST_CASE("Single point with min_pts=2 is noise", "[dbscan][single]") {
  PointCloud pc{{0.f}, {0.f}, {0.f}};
  auto result = dbscan(pc, 1.0f, 2);
  REQUIRE(result.labels.size() == 1);
  REQUIRE(result.labels[0] == 0);
}

TEST_CASE("Two close points with min_pts=2 form one cluster",
          "[dbscan][tiny]") {
  PointCloud pc{{0.f, 0.5f}, {0.f, 0.f}, {0.f, 0.f}};
  auto result = dbscan(pc, 1.0f, 2);
  REQUIRE(result.labels.size() == 2);
  REQUIRE(unique_clusters(result.labels).size() == 1);
  REQUIRE(count_noise(result.labels) == 0);
}

TEST_CASE("Two close points with min_pts=3 are both noise", "[dbscan][tiny]") {
  PointCloud pc{{0.f, 0.5f}, {0.f, 0.f}, {0.f, 0.f}};
  auto result = dbscan(pc, 1.0f, 3);
  REQUIRE(result.labels.size() == 2);
  REQUIRE(count_noise(result.labels) == 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Exact-distance boundary: neighbor inclusion uses <=
//    lib/src/index.hpp : are_within (line 57)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Points at distance exactly epsilon are neighbors (inclusive <=)",
          "[dbscan][boundary]") {
  // (0,0,0) and (1,0,0) are exactly eps=1 apart; should be mutual neighbors.
  const float eps = 1.0f;
  PointCloud pc{{0.f, 1.0f}, {0.f, 0.f}, {0.f, 0.f}};
  auto result = dbscan(pc, eps, 2);
  REQUIRE(result.labels.size() == 2);
  REQUIRE(count_noise(result.labels) == 0);
  REQUIRE(unique_clusters(result.labels).size() == 1);
}

TEST_CASE("Points just beyond epsilon are not neighbors", "[dbscan][boundary]") {
  const float eps = 1.0f;
  // nextafter(1.0f, 2.0f) is the smallest float strictly > 1
  const float beyond = std::nextafter(1.0f, 2.0f);
  PointCloud pc{{0.f, beyond}, {0.f, 0.f}, {0.f, 0.f}};
  auto result = dbscan(pc, eps, 2);
  REQUIRE(result.labels.size() == 2);
  REQUIRE(count_noise(result.labels) == 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Border-point semantics: reachable non-core points join a cluster but
//    do not expand it
//    lib/src/vdbscan.cpp : BFS expansion (line 130)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Border point is assigned to cluster but dangling point stays noise",
          "[dbscan][border]") {
  // eps=0.6, min_pts=5
  //
  // Core cluster (P0–P4): five tight points, all within 0.6 of each other.
  //   P0=(0,0,0)  P1=(0.2,0,0)  P2=(0,0.2,0)  P3=(0.2,0.2,0)  P4=(0.1,0.1,0)
  //   Each has ≥ 5 neighbors → all core.
  //
  // Border (P5=(0.7,0,0)):
  //   Neighbors within 0.6: self, P1 (dist=0.5), P3 (dist≈0.539), P6 (dist=0.6)
  //   Total = 4 < 5  → NOT core.
  //   P5 is within 0.6 of core point P1, so it is reached by BFS and assigned
  //   to the cluster, but it is never enqueued.
  //
  // Dangling (P6=(1.3,0,0)):
  //   Only neighbor within 0.6: self + P5 (dist=0.6) = 2 < 5 → NOT core.
  //   P6 is reachable from P5 (within eps) but P5 is not enqueued.
  //   Therefore P6 must remain noise (label 0).

  const float eps = 0.6f;
  PointCloud pc{
      {0.0f, 0.2f, 0.0f, 0.2f, 0.1f, 0.7f, 1.3f}, // x
      {0.0f, 0.0f, 0.2f, 0.2f, 0.1f, 0.0f, 0.0f}, // y
      {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}  // z
  };

  auto result = dbscan(pc, eps, 5);

  REQUIRE(result.labels.size() == 7);
  REQUIRE(unique_clusters(result.labels).size() == 1);

  // Dangling point (x=1.3) must be noise.
  const std::size_t dangling = find_x(result, 1.3f);
  REQUIRE(dangling != SIZE_MAX);
  REQUIRE(result.labels[dangling] == 0);

  // Everything except the dangling point is in the cluster.
  REQUIRE(count_noise(result.labels) == 1);
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Two dense clusters separated by more than epsilon
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Two dense groups separated by > epsilon produce two clusters",
          "[dbscan][clusters]") {
  PointCloud pc;
  // Group A around (0, 0, 0)
  pc.push(0.0f, 0.0f, 0.0f);
  pc.push(0.3f, 0.0f, 0.0f);
  pc.push(0.0f, 0.3f, 0.0f);
  // Group B around (10, 0, 0) — separation >> eps=0.5
  pc.push(10.0f, 0.0f, 0.0f);
  pc.push(10.3f, 0.0f, 0.0f);
  pc.push(10.0f, 0.3f, 0.0f);

  auto result = dbscan(pc, 0.5f, 2);
  REQUIRE(unique_clusters(result.labels).size() == 2);
  REQUIRE(count_noise(result.labels) == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. Non-core bridge point within epsilon of both groups keeps them separate
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Non-core bridge point does not merge two dense clusters",
          "[dbscan][bridge]") {
  // eps=2.5, min_pts=5
  //
  // Group A: five points forming a plus in the YZ plane at x=0.
  //   All pairwise distances ≤ 1 < 2.5  → all core.
  //
  // Group B: same geometry at x=5.
  //   Groups A and B are 5 apart; A-to-B distances all > 2.5 (nearest pair = 5).
  //
  // Bridge (2.5, 0, 0):
  //   dist to (0,0,0) = 2.5 ≤ eps  ✓
  //   dist to (5,0,0) = 2.5 ≤ eps  ✓
  //   dist to every other A/B point = sqrt(6.25 + 0.25) ≈ 2.55 > eps  ✗
  //   Neighbors = self + A-center + B-center = 3 < 5  → NOT core.
  //
  // BFS from group A will reach the bridge and assign it to cluster A.
  // The bridge is not enqueued, so group B is never reached from it.
  // Group B therefore starts a fresh BFS → cluster B.
  // Result: exactly 2 clusters, 0 noise.

  const float eps = 2.5f;
  PointCloud pc;
  // Group A (x=0 plane)
  pc.push(0.f, 0.f, 0.f);
  pc.push(0.f, 0.f, 0.5f);
  pc.push(0.f, 0.5f, 0.f);
  pc.push(0.f, 0.f, -0.5f);
  pc.push(0.f, -0.5f, 0.f);
  // Group B (x=5 plane)
  pc.push(5.f, 0.f, 0.f);
  pc.push(5.f, 0.f, 0.5f);
  pc.push(5.f, 0.5f, 0.f);
  pc.push(5.f, 0.f, -0.5f);
  pc.push(5.f, -0.5f, 0.f);
  // Bridge
  pc.push(2.5f, 0.f, 0.f);

  auto result = dbscan(pc, eps, 5);

  REQUIRE(result.labels.size() == 11);
  REQUIRE(unique_clusters(result.labels).size() == 2);
  REQUIRE(count_noise(result.labels) == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. Duplicate points at identical coordinates
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Duplicate points at identical coordinates form one cluster",
          "[dbscan][duplicates]") {
  PointCloud pc;
  for (int i = 0; i < 5; ++i)
    pc.push(0.f, 0.f, 0.f);

  auto result = dbscan(pc, 0.1f, 5);
  REQUIRE(result.labels.size() == 5);
  REQUIRE(unique_clusters(result.labels).size() == 1);
  REQUIRE(count_noise(result.labels) == 0);
}

TEST_CASE("Duplicate points below min_pts threshold are all noise",
          "[dbscan][duplicates]") {
  PointCloud pc;
  for (int i = 0; i < 3; ++i)
    pc.push(0.f, 0.f, 0.f);

  auto result = dbscan(pc, 0.1f, 5);
  REQUIRE(result.labels.size() == 3);
  REQUIRE(count_noise(result.labels) == 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// 10. Order-insensitivity: cluster membership is stable despite Morton sorting
//     lib/src/index.cpp : radix_sort_morton_refs (line 283)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Cluster membership is identical regardless of input order",
          "[dbscan][order]") {
  // Two clear clusters.  Push them in forward order then in reverse and verify
  // that every point (identified by its coordinates) receives the same cluster
  // label in both runs.
  //
  // Note: points sharing a voxel may appear in different relative positions
  // inside the sorted cloud depending on their original push order, so raw
  // vector equality is not guaranteed.  We compare per-coordinate labels.

  std::vector<std::tuple<float, float, float>> pts = {
      {0.f, 0.f, 0.f},  {0.2f, 0.f, 0.f}, {0.f, 0.2f, 0.f},
      {10.f, 0.f, 0.f}, {10.2f, 0.f, 0.f},{10.f, 0.2f, 0.f},
  };

  auto build = [&](bool reverse) {
    auto p = pts;
    if (reverse)
      std::reverse(p.begin(), p.end());
    PointCloud pc;
    for (auto &[x, y, z] : p)
      pc.push(x, y, z);
    return pc;
  };

  auto r1 = dbscan(build(false), 0.5f, 2);
  auto r2 = dbscan(build(true), 0.5f, 2);

  // Same number of clusters and noise points.
  REQUIRE(unique_clusters(r1.labels).size() == unique_clusters(r2.labels).size());
  REQUIRE(count_noise(r1.labels) == count_noise(r2.labels));

  // Build coordinate → label maps and compare them directly.
  // Both runs produce the same Morton order so the cluster IDs are stable.
  auto make_map = [](const Clustering &c) {
    std::map<std::tuple<float, float, float>, std::size_t> m;
    for (std::size_t i = 0; i < c.cloud.length(); ++i)
      m[{c.cloud.vx[i], c.cloud.vy[i], c.cloud.vz[i]}] = c.labels[i];
    return m;
  };

  REQUIRE(make_map(r1) == make_map(r2));
}

// ─────────────────────────────────────────────────────────────────────────────
// 11. Quantization / voxel-boundary behavior near exact multiples of epsilon
//     lib/src/index.cpp : nextafter adjustment (line 264)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Points at exact multiples of epsilon form correct neighborhoods",
          "[dbscan][quantization]") {
  // With eps=1.0, the reciprocal is nudged by nextafter so that floor(n * inv)
  // does not jump to a new voxel at x = n * eps for integer n.
  // Concretely: x=0, x=1, x=2, x=3 should each be within eps of their
  // immediate neighbours and together form a single chain cluster.

  const float eps = 1.0f;
  PointCloud pc{{0.f, 1.f, 2.f, 3.f}, {0.f, 0.f, 0.f, 0.f},
                {0.f, 0.f, 0.f, 0.f}};
  auto result = dbscan(pc, eps, 2);

  // Interior points (x=1, x=2) have 3 neighbours each → core.
  // Endpoints (x=0, x=3) have 2 neighbours each → core (min_pts=2).
  // All four are connected → one cluster.
  REQUIRE(result.labels.size() == 4);
  REQUIRE(unique_clusters(result.labels).size() == 1);
  REQUIRE(count_noise(result.labels) == 0);
}

TEST_CASE("Chain with min_pts=3: only interior points are core but all join",
          "[dbscan][quantization]") {
  // Same chain: x=0,1,2,3.  With min_pts=3, endpoints have only 2 neighbours
  // and are non-core, but they lie within eps of interior core points and so
  // are assigned to the cluster as border points.
  const float eps = 1.0f;
  PointCloud pc{{0.f, 1.f, 2.f, 3.f}, {0.f, 0.f, 0.f, 0.f},
                {0.f, 0.f, 0.f, 0.f}};
  auto result = dbscan(pc, eps, 3);

  REQUIRE(result.labels.size() == 4);
  REQUIRE(unique_clusters(result.labels).size() == 1);
  REQUIRE(count_noise(result.labels) == 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// 12. Coordinate clamping at extreme magnitudes
//     lib/src/index.cpp : quantize / clamp (line 37)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Extreme coordinates are clamped without crash or misclassification",
          "[dbscan][clamp]") {
  // A normal 2-point cluster near the origin, plus one point at 1e30 that
  // clamps to MORTON_COORD_MAX.  The cluster must still form correctly and the
  // extreme point must be noise (unreachable from origin at eps=0.5).
  PointCloud pc;
  pc.push(0.f, 0.f, 0.f);
  pc.push(0.3f, 0.f, 0.f);
  pc.push(1e30f, 0.f, 0.f);

  auto result = dbscan(pc, 0.5f, 2);

  REQUIRE(result.labels.size() == 3);
  REQUIRE(unique_clusters(result.labels).size() == 1);
  REQUIRE(count_noise(result.labels) == 1);

  const std::size_t extreme = find_x(result, 1e30f);
  REQUIRE(extreme != SIZE_MAX);
  REQUIRE(result.labels[extreme] == 0);
}

TEST_CASE("Two identical extreme-coordinate points share a voxel and cluster",
          "[dbscan][clamp]") {
  // Two points at exactly the same extreme coordinate both clamp to
  // MORTON_COORD_MAX.  Their Euclidean distance is 0, so they are neighbours.
  // The origin is isolated and becomes noise.
  PointCloud pc;
  pc.push(1e30f, 0.f, 0.f);
  pc.push(1e30f, 0.f, 0.f); // identical extreme value
  pc.push(0.f, 0.f, 0.f);   // origin — far from the clamped pair

  auto result = dbscan(pc, 0.5f, 2);

  REQUIRE(result.labels.size() == 3);
  REQUIRE(unique_clusters(result.labels).size() == 1);
  REQUIRE(count_noise(result.labels) == 1);

  const std::size_t origin = find_x(result, 0.f);
  REQUIRE(origin != SIZE_MAX);
  REQUIRE(result.labels[origin] == 0);
}
