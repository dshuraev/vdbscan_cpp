#include <CLI/CLI.hpp>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

#include <vdbscan/vdbscan.hpp>

#include "codecs.hpp"

namespace fs = std::filesystem;

using namespace vdbscan;

int main(int argc, char *argv[]) {
  CLI::App app{"Cluster point clouds with voxel-accelerated DBSCAN."};
  app.set_help_all_flag("--help-all", "Show all help, including hidden details.");
  app.footer(
      "Examples:\n"
      "  vdbscan_cli -i scan.bin -o clusters --epsilon 0.8 --min-pts 8\n"
      "  vdbscan_cli -i scan.xyz -o colored.ply -e 0.5 -m 10\n"
      "  vdbscan_cli -i scan.ply -o labeled.xyz -e 0.25 -m 6");

  fs::path input_path;
  fs::path output_path;
  float epsilon = 0.0f;
  std::uint16_t min_pts = 0;

  app.add_option("-i,--input", input_path, "Input point cloud (.ply, .xyz, .bin).")
      ->required()
      ->check(CLI::ExistingFile);
  app.add_option("-o,--output", output_path,
                 "Output file or directory. Files ending in .xyz or .ply write a "
                 "single colored file; any other path is treated as a directory "
                 "for split cluster .xyz files.")
      ->required();
  app.add_option("-e,--epsilon", epsilon,
                 "DBSCAN neighborhood radius. Must be greater than zero.")
      ->required();
  app.add_option("-m,--min-pts,--min-points", min_pts,
                 "Minimum number of neighboring points for a core point.")
      ->required();

  try {
    app.parse(argc, argv);

    if (epsilon <= 0.0f) {
      throw CLI::ValidationError("--epsilon", "must be greater than zero");
    }
    if (min_pts == 0) {
      throw CLI::ValidationError("--min-pts", "must be greater than zero");
    }

    const PointCloud cloud = codecs::read_point_cloud(input_path);
    const Clustering clustering = dbscan(cloud, epsilon, min_pts);

    if (codecs::writes_single_output(output_path)) {
      codecs::write_single_output(output_path, clustering);
    } else {
      codecs::write_split_output(output_path, clustering);
    }

    const std::size_t cluster_count = codecs::cluster_count(clustering);
    std::cout << "points=" << clustering.cloud.length()
              << " clusters=" << cluster_count
              << " output=" << output_path.string() << '\n';
    return 0;
  } catch (const CLI::ParseError &e) {
    return app.exit(e);
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
