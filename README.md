# `vdbscan_cpp`

General-purpose DBSCAN point-cloud clustering accelerated by Morton-order voxel index.

## Requirements

- `CMake >= 3.20`
- C++ compiler supporting C++17
- [`uv`](https://docs.astral.sh/uv/) for `test:all:run` and synthetic targets.
- `gcov` or `llvm-cov gcov` for `test:coverage` depending on toolchain
- [`cargo-flamegraph`](https://github.com/flamegraph-rs/flamegraph), `perf` for `perf:flamegraph:kitti` target

## Building

Use [task](https://taskfile.dev/) for building the application:

```sh
task build:cli:release # to build CLI + library
task build:lib:release # to build library alone
# for more options:
task --list
```

## Roadmap

- [x] Library core
- [x] CLI
- [x] Benchmarks
- [x] Unit tests
- [x] Testing on synthetic data (oracle [sklearn.cluster.DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html))
- [ ] Fuzzing
- [x] Perf traces
- [x] Tracy

## License

MIT. See [LICENSE](./LICENSE).
