#include "codecs.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

using namespace vdbscan;

namespace {

using Rgb = std::array<std::uint8_t, 3>;

[[nodiscard]] std::string lower_copy(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return value;
}

[[nodiscard]] std::string extension_of(const fs::path &path) {
  return lower_copy(path.extension().string());
}

[[nodiscard]] Rgb hsv_to_rgb(const float hue_degrees, const float saturation,
                             const float value) {
  const float chroma = value * saturation;
  const float hue_sector = hue_degrees / 60.0f;
  const float x = chroma * (1.0f - std::fabs(std::fmod(hue_sector, 2.0f) - 1.0f));

  float r1 = 0.0f;
  float g1 = 0.0f;
  float b1 = 0.0f;

  if (hue_sector < 1.0f) {
    r1 = chroma;
    g1 = x;
  } else if (hue_sector < 2.0f) {
    r1 = x;
    g1 = chroma;
  } else if (hue_sector < 3.0f) {
    g1 = chroma;
    b1 = x;
  } else if (hue_sector < 4.0f) {
    g1 = x;
    b1 = chroma;
  } else if (hue_sector < 5.0f) {
    r1 = x;
    b1 = chroma;
  } else {
    r1 = chroma;
    b1 = x;
  }

  const float match = value - chroma;
  const auto to_byte = [match](float channel) {
    return static_cast<std::uint8_t>(
        std::lround(std::clamp((channel + match) * 255.0f, 0.0f, 255.0f)));
  };

  return {to_byte(r1), to_byte(g1), to_byte(b1)};
}

[[nodiscard]] Rgb color_for_label(const std::size_t label) {
  if (label == 0) {
    return {96, 96, 96};
  }

  constexpr float golden_ratio_conjugate = 0.6180339887498948f;
  const float hue = std::fmod(static_cast<float>(label) * golden_ratio_conjugate, 1.0f) *
                    360.0f;
  return hsv_to_rgb(hue, 0.75f, 0.95f);
}

[[nodiscard]] PointCloud read_xyz(const fs::path &path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open input file: " + path.string());
  }

  PointCloud cloud;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) {
      continue;
    }

    std::istringstream row(line);
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
    if (!(row >> x >> y >> z)) {
      throw std::runtime_error("invalid .xyz row in " + path.string());
    }
    cloud.push(x, y, z);
  }
  return cloud;
}

[[nodiscard]] PointCloud read_bin(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open input file: " + path.string());
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

struct PlyHeader {
  struct Property {
    std::string type_name;
    std::string property_name;
  };

  std::size_t vertex_count = 0;
  bool is_ascii = false;
  std::vector<Property> vertex_properties;
};

[[nodiscard]] PlyHeader read_ply_header(std::istream &input) {
  std::string line;
  if (!std::getline(input, line) || line != "ply") {
    throw std::runtime_error("invalid .ply header");
  }

  PlyHeader header;
  bool in_vertex_element = false;
  while (std::getline(input, line)) {
    if (line == "end_header") {
      break;
    }

    std::istringstream row(line);
    std::string keyword;
    row >> keyword;

    if (keyword == "format") {
      std::string format;
      row >> format;
      header.is_ascii = (format == "ascii");
      if (!header.is_ascii && format != "binary_little_endian") {
        throw std::runtime_error("unsupported .ply format: " + format);
      }
    } else if (keyword == "element") {
      std::string element_name;
      std::size_t count = 0;
      row >> element_name >> count;
      in_vertex_element = (element_name == "vertex");
      if (in_vertex_element) {
        header.vertex_count = count;
        header.vertex_properties.clear();
      }
    } else if (keyword == "property" && in_vertex_element) {
      std::string type_name;
      std::string property_name;
      row >> type_name >> property_name;
      if (type_name == "list") {
        throw std::runtime_error("list properties are not supported in vertex elements");
      }
      header.vertex_properties.push_back(PlyHeader::Property{
          std::move(type_name), std::move(property_name)});
    }
  }

  if (header.vertex_count == 0) {
    throw std::runtime_error(".ply file does not contain vertex data");
  }
  return header;
}

[[nodiscard]] std::size_t property_index(const PlyHeader &header,
                                         const std::string_view name) {
  for (std::size_t i = 0; i < header.vertex_properties.size(); ++i) {
    if (header.vertex_properties[i].property_name == name) {
      return i;
    }
  }
  throw std::runtime_error("required .ply property missing: " + std::string(name));
}

[[nodiscard]] double read_ascii_scalar(std::istream &input,
                                       const std::string &type_name) {
  if (type_name == "float" || type_name == "float32" || type_name == "double" ||
      type_name == "float64") {
    double value = 0.0;
    if (!(input >> value)) {
      throw std::runtime_error("invalid ascii .ply scalar value");
    }
    return value;
  }

  long long value = 0;
  if (!(input >> value)) {
    throw std::runtime_error("invalid ascii .ply scalar value");
  }
  return static_cast<double>(value);
}

[[nodiscard]] PointCloud read_ply_ascii(std::istream &input, const PlyHeader &header) {
  const std::size_t x_idx = property_index(header, "x");
  const std::size_t y_idx = property_index(header, "y");
  const std::size_t z_idx = property_index(header, "z");

  PointCloud cloud(header.vertex_count);
  std::string line;
  for (std::size_t i = 0; i < header.vertex_count; ++i) {
    if (!std::getline(input, line)) {
      throw std::runtime_error("unexpected end of .ply vertex data");
    }

    std::istringstream row(line);
    std::vector<double> values(header.vertex_properties.size());
    for (std::size_t j = 0; j < values.size(); ++j) {
      values[j] = read_ascii_scalar(row, header.vertex_properties[j].type_name);
    }
    cloud.push(static_cast<float>(values[x_idx]), static_cast<float>(values[y_idx]),
               static_cast<float>(values[z_idx]));
  }
  return cloud;
}

template <typename T> [[nodiscard]] T read_binary_value(std::istream &input) {
  T value{};
  input.read(reinterpret_cast<char *>(&value), sizeof(T));
  if (!input) {
    throw std::runtime_error("unexpected end of binary .ply vertex data");
  }
  return value;
}

[[nodiscard]] double read_binary_scalar(std::istream &input,
                                        const std::string &type_name) {
  if (type_name == "char" || type_name == "int8") {
    return static_cast<double>(read_binary_value<std::int8_t>(input));
  }
  if (type_name == "uchar" || type_name == "uint8") {
    return static_cast<double>(read_binary_value<std::uint8_t>(input));
  }
  if (type_name == "short" || type_name == "int16") {
    return static_cast<double>(read_binary_value<std::int16_t>(input));
  }
  if (type_name == "ushort" || type_name == "uint16") {
    return static_cast<double>(read_binary_value<std::uint16_t>(input));
  }
  if (type_name == "int" || type_name == "int32") {
    return static_cast<double>(read_binary_value<std::int32_t>(input));
  }
  if (type_name == "uint" || type_name == "uint32") {
    return static_cast<double>(read_binary_value<std::uint32_t>(input));
  }
  if (type_name == "float" || type_name == "float32") {
    return static_cast<double>(read_binary_value<float>(input));
  }
  if (type_name == "double" || type_name == "float64") {
    return read_binary_value<double>(input);
  }
  throw std::runtime_error("unsupported binary .ply scalar type: " + type_name);
}

[[nodiscard]] PointCloud read_ply_binary(std::istream &input, const PlyHeader &header) {
  if (header.vertex_properties.size() < 3) {
    throw std::runtime_error("binary .ply vertex layout must contain x y z");
  }

  const std::size_t x_idx = property_index(header, "x");
  const std::size_t y_idx = property_index(header, "y");
  const std::size_t z_idx = property_index(header, "z");

  PointCloud cloud(header.vertex_count);
  std::vector<double> values(header.vertex_properties.size());

  for (std::size_t i = 0; i < header.vertex_count; ++i) {
    for (std::size_t j = 0; j < values.size(); ++j) {
      values[j] = read_binary_scalar(input, header.vertex_properties[j].type_name);
    }
    cloud.push(static_cast<float>(values[x_idx]), static_cast<float>(values[y_idx]),
               static_cast<float>(values[z_idx]));
  }
  return cloud;
}

[[nodiscard]] PointCloud read_ply(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open input file: " + path.string());
  }

  const PlyHeader header = read_ply_header(input);
  return header.is_ascii ? read_ply_ascii(input, header)
                         : read_ply_binary(input, header);
}

void write_xyz_single(const fs::path &path, const Clustering &clustering) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to open output file: " + path.string());
  }

  output << std::fixed << std::setprecision(6);
  for (std::size_t i = 0; i < clustering.cloud.length(); ++i) {
    const Rgb color = color_for_label(clustering.labels[i]);
    output << clustering.cloud.vx[i] << ' ' << clustering.cloud.vy[i] << ' '
           << clustering.cloud.vz[i] << ' ' << static_cast<int>(color[0]) << ' '
           << static_cast<int>(color[1]) << ' ' << static_cast<int>(color[2]) << '\n';
  }
}

void write_ply_single(const fs::path &path, const Clustering &clustering) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to open output file: " + path.string());
  }

  output << "ply\n";
  output << "format ascii 1.0\n";
  output << "element vertex " << clustering.cloud.length() << '\n';
  output << "property float x\n";
  output << "property float y\n";
  output << "property float z\n";
  output << "property uchar red\n";
  output << "property uchar green\n";
  output << "property uchar blue\n";
  output << "end_header\n";

  output << std::fixed << std::setprecision(6);
  for (std::size_t i = 0; i < clustering.cloud.length(); ++i) {
    const Rgb color = color_for_label(clustering.labels[i]);
    output << clustering.cloud.vx[i] << ' ' << clustering.cloud.vy[i] << ' '
           << clustering.cloud.vz[i] << ' ' << static_cast<int>(color[0]) << ' '
           << static_cast<int>(color[1]) << ' ' << static_cast<int>(color[2]) << '\n';
  }
}

void write_xyz_points(const fs::path &path, const PointCloud &cloud,
                      const std::vector<std::size_t> &indices) {
  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("failed to open output file: " + path.string());
  }

  output << std::fixed << std::setprecision(6);
  for (const std::size_t index : indices) {
    output << cloud.vx[index] << ' ' << cloud.vy[index] << ' ' << cloud.vz[index] << '\n';
  }
}

} // namespace

namespace codecs {

PointCloud read_point_cloud(const fs::path &path) {
  const std::string extension = extension_of(path);

  if (extension == ".xyz") {
    return read_xyz(path);
  }
  if (extension == ".bin") {
    return read_bin(path);
  }
  if (extension == ".ply") {
    return read_ply(path);
  }

  throw std::runtime_error("unsupported input format: " + path.string());
}

bool writes_single_output(const fs::path &path) {
  const std::string extension = extension_of(path);
  return extension == ".xyz" || extension == ".ply";
}

void write_single_output(const fs::path &path, const Clustering &clustering) {
  const std::string extension = extension_of(path);

  if (extension == ".xyz") {
    write_xyz_single(path, clustering);
    return;
  }
  if (extension == ".ply") {
    write_ply_single(path, clustering);
    return;
  }

  throw std::runtime_error("unsupported single-file output format: " + path.string());
}

void write_split_output(const fs::path &path, const Clustering &clustering) {
  fs::create_directories(path);

  const std::size_t max_label = cluster_count(clustering);
  std::vector<std::vector<std::size_t>> grouped(max_label + 1);

  for (std::size_t i = 0; i < clustering.labels.size(); ++i) {
    grouped[clustering.labels[i]].push_back(i);
  }

  write_xyz_points(path / "noise.xyz", clustering.cloud, grouped[0]);
  for (std::size_t label = 1; label <= max_label; ++label) {
    write_xyz_points(path / ("cluster_" + std::to_string(label) + ".xyz"),
                     clustering.cloud, grouped[label]);
  }
}

std::size_t cluster_count(const Clustering &clustering) noexcept {
  std::size_t max_label = 0;
  for (const std::size_t label : clustering.labels) {
    max_label = std::max(max_label, label);
  }
  return max_label;
}

} // namespace codecs
