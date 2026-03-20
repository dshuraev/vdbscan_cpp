#include <benchmark/benchmark.h>

#ifdef TRACY_ENABLE
#  include <tracy/Tracy.hpp>
#else
#  define FrameMark
#endif

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <vector>

#include <vdbscan/vdbscan.hpp>

#ifndef KITTI_DATASET_PATH
#define KITTI_DATASET_PATH ""
#endif

#ifndef KITTI_STRIDE
#define KITTI_STRIDE 1
#endif

#ifndef KITTI_MAX_FILES
#define KITTI_MAX_FILES 0
#endif

namespace fs = std::filesystem;

using namespace vdbscan;

namespace {

constexpr std::size_t kStride = static_cast<std::size_t>(KITTI_STRIDE);
constexpr std::size_t kMaxFiles = static_cast<std::size_t>(KITTI_MAX_FILES);

struct LoadedScan {
  PointCloud cloud;
};

[[nodiscard]] PointCloud read_kitti_bin(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open KITTI scan: " + path.string());
  }

  PointCloud cloud;
  std::array<float, 4> point{};

  while (input.read(reinterpret_cast<char *>(point.data()),
                    static_cast<std::streamsize>(sizeof(point)))) {
    cloud.push(point[0], point[1], point[2]);
  }

  if (!input.eof()) {
    throw std::runtime_error("invalid KITTI .bin payload in " + path.string());
  }

  return cloud;
}

[[nodiscard]] std::vector<fs::path> selected_scan_paths() {
  std::vector<fs::path> paths;
  for (const fs::directory_entry &entry :
       fs::directory_iterator(KITTI_DATASET_PATH)) {
    if (entry.is_regular_file() && entry.path().extension() == ".bin") {
      paths.push_back(entry.path());
    }
  }

  std::sort(paths.begin(), paths.end());

  std::vector<fs::path> selected;
  selected.reserve(kMaxFiles > 0 ? kMaxFiles : paths.size());

  for (std::size_t i = 0; i < paths.size(); i += kStride) {
    if (kMaxFiles > 0 && selected.size() >= kMaxFiles) {
      break;
    }
    selected.push_back(paths[i]);
  }

  if (selected.empty()) {
    throw std::runtime_error("no KITTI .bin scans selected");
  }

  return selected;
}

[[nodiscard]] std::vector<LoadedScan> load_selected_scans() {
  const std::vector<fs::path> paths = selected_scan_paths();
  std::vector<LoadedScan> scans;
  scans.reserve(paths.size());

  for (const fs::path &scan_path : paths) {
    scans.push_back(LoadedScan{read_kitti_bin(scan_path)});
  }

  return scans;
}

void BM_DBSCAN_KITTI(benchmark::State &state) {
  const float epsilon = static_cast<float>(state.range(0)) / 100.0f;
  const auto min_pts = static_cast<uint_fast16_t>(state.range(1));

  const std::vector<LoadedScan> &dataset = load_selected_scans();
  std::size_t points_per_pass = 0;
  for (const LoadedScan &scan : dataset) {
    points_per_pass += scan.cloud.length();
  }

  state.SetLabel(KITTI_DATASET_PATH);
  state.counters["files"] = static_cast<double>(dataset.size());
  state.counters["points_per_pass"] = static_cast<double>(points_per_pass);
  state.counters["dots_per_second"] =
      benchmark::Counter(static_cast<double>(points_per_pass),
                         benchmark::Counter::kIsIterationInvariantRate);

  for (auto _ : state) {
    for (const LoadedScan &scan : dataset) {
      const Clustering clustering = dbscan(scan.cloud, epsilon, min_pts);
      benchmark::DoNotOptimize(clustering.labels.data());
      benchmark::DoNotOptimize(clustering.cloud.length());
      FrameMark;
    }
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(points_per_pass));
}

} // namespace

#if defined(KITTI_EPSILON) && defined(KITTI_MINPTS)
BENCHMARK(BM_DBSCAN_KITTI)->Args({KITTI_EPSILON, KITTI_MINPTS});
#else
BENCHMARK(BM_DBSCAN_KITTI)->Args({50, 4})->Args({75, 4})->Args({100, 4});
#endif
