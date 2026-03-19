# `vdbscan_cpp`

General-purpose DBSCAN point-cloud clustering accelerated by Morton-order voxel index.

## Requirements

- `CMake >= 3.20`
- C++ compiler supporting C++17
- `gcovr` for `test:coverage`

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
- [ ] Fuzzing
- [ ] Perf traces

## License

MIT. See [LICENSE](./LICENSE).
