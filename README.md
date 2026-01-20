# Fast BCC on GPUs

GPU-accelerated implementation of Biconnected Components (BCC) algorithms for graph processing.

## Project Structure

```
.
├── baselines/          # Baseline implementations for comparison
│   ├── cpu/           # CPU-based implementation using ParlayLib
│   ├── uvm/           # Unified Virtual Memory (UVM) GPU implementation
│   ├── wk-bcc-2017/   # Wang-Koorapaty BCC 2017 baseline
│   └── wk-bcc-2018/   # Wang-Koorapaty BCC 2018 baseline
├── depth/             # Depth-based approach implementation
├── external/          # External memory implementations
│   ├── streams/       # Version with CUDA streams
│   └── without_streams/ # Version without CUDA streams
└── gpu/               # Main GPU implementations
    ├── with_filter/   # Implementation with filtering optimization
    └── without_filter/ # Implementation without filtering
```

## Components

### Core Algorithms
- **Eulerian Tour**: Graph traversal technique
- **List Ranking**: Parallel list ranking algorithm
- **Spanning Forest/Tree**: Graph spanning structures
- **Sparse Table**: Range query data structures (min/max)
- **Connected Components (CC)**: Graph connectivity analysis
- **Root Smallest Tree (RST)**: Tree-based graph representation

### GPU Implementations
The project includes multiple GPU implementations with different optimization strategies:
- Filtering optimizations
- CUDA streams for concurrent execution
- Unified Virtual Memory (UVM) for simplified memory management

## Building

Each subdirectory contains its own Makefile. To build a specific implementation:

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

Most implementations include runner scripts:

```bash
./runner.sh
```

Or run the executable directly:
```bash
./main <graph_file>
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
