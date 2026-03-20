#pragma once

#include <cstddef>
#include <filesystem>

#include <vdbscan/vdbscan.hpp>

namespace codecs {

[[nodiscard]] vdbscan::PointCloud read_point_cloud(const std::filesystem::path &path);

[[nodiscard]] bool writes_single_output(const std::filesystem::path &path);

void write_single_output(const std::filesystem::path &path,
                         const vdbscan::Clustering &clustering);

void write_split_output(const std::filesystem::path &path,
                        const vdbscan::Clustering &clustering);

[[nodiscard]] std::size_t cluster_count(const vdbscan::Clustering &clustering) noexcept;

} // namespace codecs
