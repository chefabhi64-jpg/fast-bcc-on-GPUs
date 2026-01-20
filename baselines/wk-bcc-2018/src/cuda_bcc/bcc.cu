#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

// Helper functions
#include "cuda_bcc/utility.hpp"
#include "cuda_utility.cuh"

#include "cuda_bcc/bcc.cuh"
#include "cuda_bcc/bcc_memory_utils.cuh"

#include "cuda_bcc/bfs.cuh"

#include "cuda_bcc/lca.cuh"
#include "cuda_bcc/cut_vertex.cuh"

// #define DEBUG

__global__ 
void update_bcc_flag_kernel(int* d_cut_vertex, int* d_bcc_num, int* d_flag, int numVert) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < numVert) {
		if(!d_cut_vertex[i]) {
			d_flag[d_bcc_num[i]] = 1;
		}
	}
}

__global__ 
void update_bcc_number_kernel(int* d_cut_vertex, int* d_bcc_num, int* bcc_ps, int* cut_ps, int numVert) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// if(i == 0) update_bcg_num_vert;
	if(i < numVert) {
		if(!d_cut_vertex[i]) {
			d_bcc_num[i] = bcc_ps[d_bcc_num[i]] - 1;
		}
		else
			d_bcc_num[i] = bcc_ps[numVert - 1] + cut_ps[i] - 1;
	}
}

// inclusive prefix_sum
void incl_scan(
	int*& d_in, 
	int*& d_out, 
	int& num_items) {

    size_t temp_storage_bytes = 0;
    cudaError_t status;
	void* d_temp_storage = nullptr;
	// Determine temporary device storage requirements
    status = cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    CUDA_CHECK(status, "Error in CUB InclusiveSum");
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes), "Failed to allocate temporary storage for InclusiveSum");
	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize before InclusiveSum");

    // Run inclusive prefix sum
    status = cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    CUDA_CHECK(status, "Error in CUB InclusiveSum");
	CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize after InclusiveSum");

	// Free temporary storage
	CUDA_CHECK(cudaFree(d_temp_storage), "Failed to free temporary storage for InclusiveSum");

}

int update_bcc_numbers(gpu_bcc& g_bcc_ds, int numVert) {
    int* d_bcc_flag 	= 	g_bcc_ds.d_bcc_flag;
    int* d_bcc_ps 		= 	g_bcc_ds.d_rep;   // reusing few arrays
    int* d_cut_ps 		= 	g_bcc_ds.d_level; // reusing few arrays
    
    int* d_cut_vertex 	= 	g_bcc_ds.d_cut_vertex;
    int* d_bcc_num 		= 	g_bcc_ds.d_imp_bcc_num;

    int numThreads = static_cast<int>(maxThreadsPerBlock);
    size_t numBlocks = (numVert + numThreads - 1) / numThreads;

	#ifdef DEBUG
		std::vector<int> h_cut_vertex(numVert);
		std::vector<int> h_bcc_num(numVert);
		CUDA_CHECK(cudaMemcpy(h_cut_vertex.data(), d_cut_vertex, numVert * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_cut_vertex to host");
		CUDA_CHECK(cudaMemcpy(h_bcc_num.data(), d_bcc_num, numVert * sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy d_bcc_num to host");
		print(h_cut_vertex, "Cut Vertex array before updating BCC numbers");
		print(h_bcc_num, "BCC number array before updating BCC numbers");
	#endif

    update_bcc_flag_kernel<<<numBlocks, numThreads>>>(d_cut_vertex, d_bcc_num, d_bcc_flag, numVert);
    CUDA_CHECK(cudaDeviceSynchronize(), "..");

    incl_scan(d_bcc_flag,   d_bcc_ps, numVert);
    incl_scan(d_cut_vertex, d_cut_ps, numVert);

    int* h_max_ps_bcc = new int;
	int* h_max_ps_cut_vertex = new int;

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize stream after copying max prefix sums");

    CUDA_CHECK(cudaMemcpy(h_max_ps_bcc, &d_bcc_ps[numVert - 1], sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back max_ps_bcc.");
    CUDA_CHECK(cudaMemcpy(h_max_ps_cut_vertex, &d_cut_ps[numVert - 1], sizeof(int), cudaMemcpyDeviceToHost), "Failed to copy back max_ps_cut_vertex.");
    
    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize stream after copying max prefix sums");

	// std::cout << "max_ps_bcc: " << *h_max_ps_bcc << "\n";
	// std::cout << "max_ps_cut_vertex: " << *h_max_ps_cut_vertex << "\n";

    int bcg_num_vert = *h_max_ps_bcc + *h_max_ps_cut_vertex;

    update_bcc_number_kernel<<<numBlocks, numThreads>>>(
    	d_cut_vertex, 
    	d_bcc_num, 
    	d_bcc_ps, 
    	d_cut_ps, 
    	numVert
    );

    CUDA_CHECK(cudaDeviceSynchronize(), "Failed to synchronize stream after copying max prefix sums");

    if(g_verbose) {
    	std::cout << "BCC numbers:" << std::endl;
    	kernelPrintArray(d_bcc_num, numVert);
    	std::cout << "Cut vertex status:" << std::endl;
    	kernelPrintArray(d_cut_vertex, numVert);
    }

    return bcg_num_vert;
}

void cuda_bcc(gpu_bcc& g_bcc_ds, bool last_bcc) {

	long numEdges 	= g_bcc_ds.numEdges; // numEdges is unique edge count (only (2,1), not (1,2)).
	int numVert 	= g_bcc_ds.numVert;

	long E = 2 * numEdges; // Two times the original edges count (0,1) and (1,0).
    
    // csr data-structures
    long* d_vertices = g_bcc_ds.d_vertices;
    int* d_edges = g_bcc_ds.d_edges;

	int *d_parent = g_bcc_ds.d_parent;
	int *d_level = g_bcc_ds.d_level;

	// Create a random device and seed it
    std::random_device rd;
    std::mt19937 gen(rd());

    // Create a distribution in the range [0, numVert]
    std::uniform_int_distribution<> distrib(0, numVert - 1);

    // Generate a random root value
    int root = distrib(gen);
    int root_level = 0;
    root = 0;
    // Output the random root value
    // std::cout << "Random root value: " << root << std::endl;

	int child_of_root = -1;

	CUDA_CHECK(cudaMemcpy(&d_level[root], &root_level, sizeof(int), cudaMemcpyHostToDevice), "Failed to set root level.");

	auto start = std::chrono::high_resolution_clock::now();	
	// Step 1: Construct a rooted spanning tree
	constructSpanningTree(numVert, E, d_vertices, d_edges, d_level, d_parent, root, child_of_root);
	
	auto end = std::chrono::high_resolution_clock::now();
	auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	
	// std::cout <<"Spanning tree creation finished in: " << dur <<" ms.\n";

	#ifdef DEBUG
		std::vector<int> h_parent(numVert);
		std::vector<int> h_level(numVert);

		CUDA_CHECK(cudaMemcpy(h_parent.data(), d_parent, numVert * sizeof(int), cudaMemcpyDeviceToHost), 
	    				"Failed to copy back parent array to host.");

		CUDA_CHECK(cudaMemcpy(h_level.data(), d_level, numVert * sizeof(int), cudaMemcpyDeviceToHost), 
	    				"Failed to copy back level array to host.");

		print(h_parent, "parent array");
		print(h_level,   "level array");
	#endif
	start = std::chrono::high_resolution_clock::now();
	// Step 3 & 4 : Find LCA and Base Vertices, then apply connected Comp
    naive_lca(g_bcc_ds, root, child_of_root);

    // Step 5: Propagate safness to representative & parents
    // Step 6: Update cut vertex status and cut - edge status
    // Step 7: Update implicit bcc labels
    assign_cut_vertex_BCC(g_bcc_ds, root, child_of_root);

	if(!last_bcc) {
		g_bcc_ds.numVert = update_bcc_numbers(g_bcc_ds, numVert);

		end = std::chrono::high_resolution_clock::now();
		dur += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		add_function_time("Step 2: BCC Computation", dur);
		// std::cout <<"BCC computation finished in: " << dur <<" ms.\n";
	} else {
		end = std::chrono::high_resolution_clock::now();
		dur += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		add_function_time("Step 4: Last BCC Computation", dur);
		// std::cout <<"BCC computation finished in: " << dur <<" ms.\n";
	}
}
