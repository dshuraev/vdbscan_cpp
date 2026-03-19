# `vdbscan-cli`

Provides CLI to interact with the library core.

## Options

- `-i|--input PATH`: path to point cloud data; supported formats/extensions `.ply`, `.xyz`, `.bin` (KITTI dataset format)
- `-o|--output PATH`: output path.
  If the ends with `.xyz` or `.ply` outputs all clustering data in a single file, with a single color per cluster.
  Otherwise, creates a directory (if does not exist) and outputs every cluster in `xyz` format: `noise.xyz`, `cluster_1.xyz`, ...
- `-e|--epsilon FLOAT`: DBSCAN neighborhood radius, must be greater than zero
- `-m|--min-pts INT`: minimum number of neighboring points for a core point
