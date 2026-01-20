# Fast BCC on GPUs

GPU-based implementation of Biconnected Components (BCC) algorithms for graph processing.

## Project Structure

```
.
├── baselines/          # Baseline implementations for comparison
│   ├── cpu/           # CPU-based implementation using ParlayLib
│   ├── wk-bcc-2017/   # Wadekar-Kothapalli BCC 2017 baseline
│   └── wk-bcc-2018/   # Wadekar-Kothapalli BCC 2018 baseline
|
├── depth/             # Calculate approximate diameter of input graph using serial BFS
├── external/          # External memory implementations
│   ├── streams/       # Version with CUDA streams
│   └── without_streams/ # Version without CUDA streams
└── gpu/               # Main GPU implementations
    ├── with_filter/   # Implementation with filtering optimization
    └── without_filter/ # Implementation without filtering
```

## Components

### GPU Implementations
The project includes multiple GPU implementations with different optimization strategies:
- **In-memory**: Direct GPU memory implementations with/without filtering optimizations
- **External**: Out-of-core implementations for large graphs using CUDA streams
- **Filtering**: Edge filtering to reduce computation overhead

## Building

### Build All Projects

From the root directory:

```bash
make
```

This will build all subprojects automatically.

### Build Individual Projects

Each subdirectory contains its own Makefile:

```bash
cd <directory>
make
```

For example:
```bash
cd gpu/with_filter
make
```

### Clean Build Artifacts

To clean all build artifacts:

```bash
make clean
```

## Running

### Quick Start - Run All Implementations

From the root directory:

```bash
bash run_all.sh
```

This script builds everything and runs each implementation against available datasets.

**Configuration** (optional environment variables):
```bash
CPU_ROUNDS=5 GPU_SHARE=0.8 BATCH_SIZE=500000 bash run_all.sh
```

### Run Individual Implementations

#### 1. CPU Baseline (ParlayLib)
```bash
cd baselines/cpu/src
./FAST_BCC <graph_file> [num_rounds]
```
Example:
```bash
./FAST_BCC ../../../datasets/input.txt 3
```
Parameters:
- `num_rounds`: Number of benchmark rounds (default: 3)

#### 2. Wadekar-Kothapalli BCC 2017
```bash
cd baselines/wk-bcc-2017
./bin/cuda_bcc -i <graph_file> -a ebcc [-o output_dir] [-d device]
```
Example:
```bash
./bin/cuda_bcc -i datasets/graph.txt -a ebcc -o output/
```
Options:
- `-a`: Algorithm (`cv`=cut vertex, `ce`=cut edges, `ibcc`=implicit BCC, `ebcc`=explicit BCC)
- `-o`: Output directory (optional)
- `-d`: CUDA device number (default: 0)

#### 3. Wadekar-Kothapalli BCC 2018
```bash
cd baselines/wk-bcc-2018
./bin/main <graph_file> [k] [verbose]
```
Example:
```bash
./bin/main datasets/graph.txt 2 0
```
Parameters:
- `k`: Max edges per vertex to sample (default: 2)
- `verbose`: Enable verbose output, 0 or 1 (default: 0)

#### 4. GPU with Filter (In-Memory)
```bash
cd gpu/with_filter
./main <graph_file>
```
Example:
```bash
./main ../../depth/datasets/input_100.txt
```

#### 5. GPU without Filter (In-Memory)
```bash
cd gpu/without_filter
./main <graph_file>
```
Example:
```bash
./main ../../depth/datasets/input_100.txt
```

#### 6. External with Streams
```bash
cd external/streams
./ext-bcc <graph_file> <gpu_share> <batch_size>
```
Example:
```bash
./ext-bcc input.txt 1.0 1048576
```
Parameters:
- `gpu_share`: GPU workload share (0.0 to 1.0)
- `batch_size`: Batch size for processing

#### 7. External without Streams
```bash
cd external/without_streams
./ext-bcc <graph_file> <gpu_share> <batch_size>
```
Example:
```bash
./ext-bcc input.txt 1.0 1048576
```

#### 8. Depth BFS (Graph Diameter)
```bash
cd depth
./bfs <graph_file> [source_vertex]
```
Example:
```bash
./bfs datasets/edges.txt 0
```

## Requirements

- **NVIDIA CUDA Toolkit** (version 11.0 or later recommended)
- **C++ compiler** with C++17 support (g++ or clang++)
- **CUDA-capable GPU** with compute capability 7.0 or higher
- **ParlayLib** (included as submodule for CPU baseline)

## Input Graph Format

All implementations expect graphs in edge list format:
```
num_vertices num_edges
u1 v1
u2 v2
...
```

- Vertices are 0-indexed
- Each line after the header represents an undirected edge
- For undirected graphs, include both (u,v) and (v,u) or just one (implementation handles both)

## Datasets

Test datasets are located in:
- `datasets/` - Main dataset directory
- `depth/datasets/` - Additional test graphs
- `baselines/wk-bcc-2017/datasets/` - Baseline datasets
- `baselines/wk-bcc-2018/datasets/` - Baseline datasets

## Troubleshooting

### CUDA Architecture Mismatch
If you get architecture errors, update the `-arch` flag in Makefiles to match your GPU:
```makefile
CXXFLAGS = -arch=sm_XX  # Replace XX with your GPU's compute capability
```

### Out of Memory
For large graphs:
- Use the external memory implementations (`external/streams` or `external/without_streams`)
- Adjust `GPU_SHARE` and `BATCH_SIZE` parameters

### Build Errors
Ensure CUDA toolkit is in your PATH:
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Contact & Support

For questions, issues, or contributions, please contact:
- **Maintainer**: -
- **GitHub Issues**: Report bugs via GitHub issues

When reporting issues, please include:
- GPU model and compute capability
- CUDA version (`nvcc --version`)
- Error messages or unexpected behavior
- Graph size (vertices/edges) if relevant
