# `vdbscan_cpp`

General-purpose DBSCAN point-cloud clustering accelerated by Morton-order voxel index.

## Requirements

- `CMake >= 3.20`
- C++ compiler supporting C++17

## Building

Use [task](https://taskfile.dev/) for building the application:

```sh
task build:cli:release # to build CLI + library
task build:lib:release # to build library alone
```

or manually:

```sh
# CLI + library
cmake -S . -B ./build/directory \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVDBSCAN_BUILD_CLI=ON \
cmake --build ./build/directory --target vdbscan_cli -j

# library only
cmake -S . -B ./build/directory \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DVDBSCAN_BUILD_CLI=OFF \
cmake --build ./build/directory --target vdbscan -j
```

## Roadmap

- [x] Library core
- [x] CLI
- [ ] Unit tests
- [ ] Fuzzing
- [ ] Benchmarks
- [ ] Perf traces

## License

MIT. See [LICENSE](/mnt/archive/Work/Projects/vdbscan_cpp/LICENSE).
