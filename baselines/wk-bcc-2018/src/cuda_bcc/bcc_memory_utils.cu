#include "cuda_bcc/bcc_memory_utils.cuh"
#include "cuda_utility.cuh"

gpu_bcc::gpu_bcc(int vertices, long edges) : numVert(vertices), numEdges(edges) {
    
    E = 2 * numEdges; // Two times the original edges count

    // size_t size = E * sizeof(int);
    size_t vert_size = numVert * sizeof(int);

    // Allocate arrays for spanning tree output
    CUDA_CHECK(cudaMalloc(&d_parent, numVert * sizeof(int)), "Failed to allocate parent array.");
    CUDA_CHECK(cudaMalloc(&d_level,  numVert * sizeof(int)), "Failed to allocate level array.");

    // Allocate arrays for the Fundamental cycle traversal
    CUDA_CHECK(cudaMalloc(&d_isSafe,        vert_size), "Failed to allocate memory for d_isSafe");
    CUDA_CHECK(cudaMalloc(&d_cut_vertex,    vert_size), "Failed to allocate memory for d_cut_vertex");
    CUDA_CHECK(cudaMalloc(&d_imp_bcc_num,   vert_size), "Failed to allocate memory for d_imp_bcc_num");
    CUDA_CHECK(cudaMalloc(&d_isPartofFund,  vert_size), "Failed to allocate memory for d_isPartofFund");
    CUDA_CHECK(cudaMalloc(&d_is_baseVertex, vert_size), "Failed to allocate memory for d_is_baseVertex");
    CUDA_CHECK(cudaMalloc(&d_nonTreeEdgeId, numVert * sizeof(long)), "Failed to allocate memory for d_nonTreeEdgeId");

    // For connected Components
    CUDA_CHECK(cudaMalloc(&d_rep,        vert_size), "Failed to allocate memory for d_rep");
    CUDA_CHECK(cudaMalloc(&d_baseU,      numEdges * sizeof(int)), "Failed to allocate memory for d_baseU");
    CUDA_CHECK(cudaMalloc(&d_baseV,      numEdges * sizeof(int)), "Failed to allocate memory for d_baseV");
    CUDA_CHECK(cudaMalloc(&d_baseVertex, numEdges * sizeof(int)), "Failed to allocate memory for d_baseVertex");

    // For making the bcc numbers continuous 
    CUDA_CHECK(cudaMalloc(&d_bcc_flag, vert_size), "Failed to allocate memory for d_bcc_flag");

    // Common flag for bfs and cc
    CUDA_CHECK(cudaMalloc(&d_flag,                  sizeof(int)), "Unable to allocate flag value");
}

void gpu_bcc::init(int numVert, long num_non_tree_edges) {
    size_t vert_size = numVert * sizeof(int);
    
    // Initialize GPU memory
    CUDA_CHECK(cudaMemset(d_level,        -1, vert_size), "Failed to initialize level array.");
    CUDA_CHECK(cudaMemset(d_isSafe,        0, vert_size), "Failed to memset d_isSafe");
    CUDA_CHECK(cudaMemset(d_cut_vertex,    0, vert_size), "Failed to memset d_cut_vertex");
    CUDA_CHECK(cudaMemset(d_isPartofFund,  0, vert_size), "Failed to memset d_isPartofFund");
    CUDA_CHECK(cudaMemset(d_is_baseVertex, 0, vert_size), "Failed to memset d_is_baseVertex");
}