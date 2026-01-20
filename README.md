# Fast BCC on GPUs

GPU-accelerated implementation of Biconnected Components (BCC) algorithms for graph processing.

## Project Structure

```
.
├── baselines/          # Baseline implementations for comparison
│   ├── cpu/           # CPU-based implementation using ParlayLib
│   ├── uvm/           # Unified Virtual Memory (UVM) GPU implementation
│   ├── wk-bcc-2017/   # Wadekar-Kothapalli BCC 2017 baseline
│   └── wk-bcc-2018/   # Wadekar-Kothapalli BCC 2018 baseline
├── depth/             # Depth-based approach implementation
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
- Filtering optimizations
- CUDA streams for concurrent execution
- Unified Virtual Memory (UVM) for simplified memory management

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
./FAST_BCC ../../datasets/input.txt 3
```

#### 2. UVM GPU Implementation
```bash
cd baselines/uvm
./main <graph_file>
```
Example:
```bash
./main g1.txt
```

#### 3. Wadekar-Kothapalli BCC 2017
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

#### 4. Wadekar-Kothapalli BCC 2018
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

#### 5. GPU with Filter
```bash
cd gpu/with_filter
./main <graph_file>
```
Example:
```bash
./main ../../depth/datasets/input_100.txt
```

#### 6. GPU without Filter
```bash
cd gpu/without_filter
./main <graph_file>
```
Example:
```bash
./main ../../depth/datasets/input_100.txt
```

#### 7. External with Streams
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

#### 8. External without Streams
```bash
cd external/without_streams
./ext-bcc <graph_file> <gpu_share> <batch_size>
```
Example:
```bash
./ext-bcc input.txt 1.0 1048576
```

#### 9. Depth BFS
```bash
cd depth
./bfs <graph_file> [source_vertex]
```
Example:
```bash
./bfs datasets/edges.txt 0
```

## Requirements

- NVIDIA CUDA Toolkit
- C++ compiler with C++11 support or later
- CUDA-capable GPU

## Datasets

Test datasets are located in:
- `depth/datasets/`
- `baselines/wk-bcc-2017/datasets/`

## References

This implementation builds upon research in parallel BCC algorithms and GPU computing.
