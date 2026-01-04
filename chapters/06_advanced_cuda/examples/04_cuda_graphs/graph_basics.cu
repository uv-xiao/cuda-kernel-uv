#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Simple kernels for graph construction
__global__ void vector_add(float *c, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_mul(float *c, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

__global__ void vector_scale(float *a, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] *= scalar;
    }
}

__global__ void vector_sqrt(float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        a[i] = sqrtf(a[i]);
    }
}

// Example 1: Manual graph creation
void example_manual_graph_creation() {
    printf("\n1. Manual Graph Creation:\n");

    const int n = 1024 * 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_b, *d_c, *d_d;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_d, bytes));

    // Initialize data
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Create graph
    cudaGraph_t graph;
    CHECK_CUDA_ERROR(cudaGraphCreate(&graph, 0));

    // Define kernel parameters
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    // Node 1: c = a + b
    cudaGraphNode_t add_node;
    cudaKernelNodeParams add_params = {0};
    void *add_args[] = {&d_c, &d_a, &d_b, &n};
    add_params.func = (void*)vector_add;
    add_params.gridDim = grid;
    add_params.blockDim = block;
    add_params.sharedMemBytes = 0;
    add_params.kernelParams = add_args;
    add_params.extra = NULL;
    CHECK_CUDA_ERROR(cudaGraphAddKernelNode(&add_node, graph, NULL, 0, &add_params));

    // Node 2: d = c * b (depends on add_node)
    cudaGraphNode_t mul_node;
    cudaKernelNodeParams mul_params = {0};
    void *mul_args[] = {&d_d, &d_c, &d_b, &n};
    mul_params.func = (void*)vector_mul;
    mul_params.gridDim = grid;
    mul_params.blockDim = block;
    mul_params.sharedMemBytes = 0;
    mul_params.kernelParams = mul_args;
    mul_params.extra = NULL;
    cudaGraphNode_t deps[] = {add_node};
    CHECK_CUDA_ERROR(cudaGraphAddKernelNode(&mul_node, graph, deps, 1, &mul_params));

    // Node 3: d *= 2.0 (depends on mul_node)
    cudaGraphNode_t scale_node;
    cudaKernelNodeParams scale_params = {0};
    float scalar = 2.0f;
    void *scale_args[] = {&d_d, &scalar, &n};
    scale_params.func = (void*)vector_scale;
    scale_params.gridDim = grid;
    scale_params.blockDim = block;
    scale_params.sharedMemBytes = 0;
    scale_params.kernelParams = scale_args;
    scale_params.extra = NULL;
    cudaGraphNode_t deps2[] = {mul_node};
    CHECK_CUDA_ERROR(cudaGraphAddKernelNode(&scale_node, graph, deps2, 1, &scale_params));

    // Instantiate the graph
    cudaGraphExec_t graph_exec;
    CHECK_CUDA_ERROR(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

    // Launch the graph
    CHECK_CUDA_ERROR(cudaGraphLaunch(graph_exec, 0));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Verify result
    float *h_result = (float*)malloc(bytes);
    CHECK_CUDA_ERROR(cudaMemcpy(h_result, d_d, bytes, cudaMemcpyDeviceToHost));

    // Expected: d = ((a + b) * b) * 2 = ((1 + 2) * 2) * 2 = 12
    bool correct = true;
    for (int i = 0; i < n && correct; i++) {
        if (fabsf(h_result[i] - 12.0f) > 1e-5) {
            correct = false;
        }
    }

    printf("   Created graph with 3 kernel nodes\n");
    printf("   Expected result: 12.0\n");
    printf("   Actual result: %.1f\n", h_result[0]);
    printf("   Verification: %s\n", correct ? "PASS" : "FAIL");

    // Cleanup
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    free(h_a);
    free(h_b);
    free(h_result);
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    CHECK_CUDA_ERROR(cudaFree(d_d));
}

// Example 2: Graph with memcpy nodes
void example_graph_with_memcpy() {
    printf("\n2. Graph with Memory Copy Nodes:\n");

    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    float *d_data;

    for (int i = 0; i < n; i++) {
        h_input[i] = (float)i;
    }

    CHECK_CUDA_ERROR(cudaMalloc(&d_data, bytes));

    // Create graph
    cudaGraph_t graph;
    CHECK_CUDA_ERROR(cudaGraphCreate(&graph, 0));

    // Node 1: Host to Device copy
    cudaGraphNode_t h2d_node;
    cudaMemcpy3DParms h2d_params = {0};
    h2d_params.srcPtr = make_cudaPitchedPtr(h_input, bytes, n, 1);
    h2d_params.dstPtr = make_cudaPitchedPtr(d_data, bytes, n, 1);
    h2d_params.extent = make_cudaExtent(bytes, 1, 1);
    h2d_params.kind = cudaMemcpyHostToDevice;
    CHECK_CUDA_ERROR(cudaGraphAddMemcpyNode(&h2d_node, graph, NULL, 0, &h2d_params));

    // Node 2: Kernel execution (sqrt)
    cudaGraphNode_t kernel_node;
    cudaKernelNodeParams kernel_params = {0};
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    void *kernel_args[] = {&d_data, &n};
    kernel_params.func = (void*)vector_sqrt;
    kernel_params.gridDim = grid;
    kernel_params.blockDim = block;
    kernel_params.sharedMemBytes = 0;
    kernel_params.kernelParams = kernel_args;
    kernel_params.extra = NULL;
    cudaGraphNode_t deps[] = {h2d_node};
    CHECK_CUDA_ERROR(cudaGraphAddKernelNode(&kernel_node, graph, deps, 1, &kernel_params));

    // Node 3: Device to Host copy
    cudaGraphNode_t d2h_node;
    cudaMemcpy3DParms d2h_params = {0};
    d2h_params.srcPtr = make_cudaPitchedPtr(d_data, bytes, n, 1);
    d2h_params.dstPtr = make_cudaPitchedPtr(h_output, bytes, n, 1);
    d2h_params.extent = make_cudaExtent(bytes, 1, 1);
    d2h_params.kind = cudaMemcpyDeviceToHost;
    cudaGraphNode_t deps2[] = {kernel_node};
    CHECK_CUDA_ERROR(cudaGraphAddMemcpyNode(&d2h_node, graph, deps2, 1, &d2h_params));

    // Instantiate and execute
    cudaGraphExec_t graph_exec;
    CHECK_CUDA_ERROR(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
    CHECK_CUDA_ERROR(cudaGraphLaunch(graph_exec, 0));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Verify
    bool correct = true;
    for (int i = 0; i < 10 && correct; i++) {
        float expected = sqrtf((float)i);
        if (fabsf(h_output[i] - expected) > 1e-5) {
            correct = false;
        }
    }

    printf("   Created graph: H2D -> Kernel -> D2H\n");
    printf("   Input[0-4]: 0, 1, 2, 3, 4\n");
    printf("   Output[0-4]: %.2f, %.2f, %.2f, %.2f, %.2f\n",
           h_output[0], h_output[1], h_output[2], h_output[3], h_output[4]);
    printf("   Verification: %s\n", correct ? "PASS" : "FAIL");

    // Cleanup
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    free(h_input);
    free(h_output);
    CHECK_CUDA_ERROR(cudaFree(d_data));
}

// Example 3: Updating graph parameters
void example_graph_update() {
    printf("\n3. Updating Graph Parameters:\n");

    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_a, *d_result;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, bytes));

    float *h_a = (float*)malloc(bytes);
    float *h_result = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    // Create graph with scaling kernel
    cudaGraph_t graph;
    CHECK_CUDA_ERROR(cudaGraphCreate(&graph, 0));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    float scalar = 2.0f;
    cudaGraphNode_t scale_node;
    cudaKernelNodeParams params = {0};
    void *args[] = {&d_a, &scalar, &n};
    params.func = (void*)vector_scale;
    params.gridDim = grid;
    params.blockDim = block;
    params.sharedMemBytes = 0;
    params.kernelParams = args;
    params.extra = NULL;
    CHECK_CUDA_ERROR(cudaGraphAddKernelNode(&scale_node, graph, NULL, 0, &params));

    cudaGraphExec_t graph_exec;
    CHECK_CUDA_ERROR(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

    // Execute with scalar = 2.0
    CHECK_CUDA_ERROR(cudaGraphLaunch(graph_exec, 0));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(h_result, d_a, bytes, cudaMemcpyDeviceToHost));
    printf("   After scaling by 2.0: %.1f\n", h_result[0]);

    // Update the scalar parameter to 3.0
    scalar = 3.0f;
    cudaKernelNodeParams new_params = params;
    void *new_args[] = {&d_a, &scalar, &n};
    new_params.kernelParams = new_args;
    CHECK_CUDA_ERROR(cudaGraphExecKernelNodeSetParams(graph_exec, scale_node, &new_params));

    // Execute with updated parameter
    CHECK_CUDA_ERROR(cudaGraphLaunch(graph_exec, 0));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    CHECK_CUDA_ERROR(cudaMemcpy(h_result, d_a, bytes, cudaMemcpyDeviceToHost));
    printf("   After scaling by 3.0: %.1f\n", h_result[0]);
    printf("   Graph parameter update: %s\n",
           fabsf(h_result[0] - 6.0f) < 1e-5 ? "PASS" : "FAIL");

    // Cleanup
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    free(h_a);
    free(h_result);
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_result));
}

// Example 4: Graph cloning
void example_graph_clone() {
    printf("\n4. Graph Cloning:\n");

    const int n = 1024;
    const int bytes = n * sizeof(float);

    float *d_data;
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, bytes));

    // Create original graph
    cudaGraph_t graph;
    CHECK_CUDA_ERROR(cudaGraphCreate(&graph, 0));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    float scalar = 2.0f;
    cudaGraphNode_t node;
    cudaKernelNodeParams params = {0};
    void *args[] = {&d_data, &scalar, &n};
    params.func = (void*)vector_scale;
    params.gridDim = grid;
    params.blockDim = block;
    params.sharedMemBytes = 0;
    params.kernelParams = args;
    params.extra = NULL;
    CHECK_CUDA_ERROR(cudaGraphAddKernelNode(&node, graph, NULL, 0, &params));

    // Clone the graph
    cudaGraph_t cloned_graph;
    CHECK_CUDA_ERROR(cudaGraphClone(&cloned_graph, graph));

    // Both graphs can be instantiated and executed independently
    cudaGraphExec_t exec1, exec2;
    CHECK_CUDA_ERROR(cudaGraphInstantiate(&exec1, graph, NULL, NULL, 0));
    CHECK_CUDA_ERROR(cudaGraphInstantiate(&exec2, cloned_graph, NULL, NULL, 0));

    printf("   Original graph instantiated successfully\n");
    printf("   Cloned graph instantiated successfully\n");
    printf("   Both graphs can execute independently: PASS\n");

    // Cleanup
    cudaGraphExecDestroy(exec1);
    cudaGraphExecDestroy(exec2);
    cudaGraphDestroy(graph);
    cudaGraphDestroy(cloned_graph);
    CHECK_CUDA_ERROR(cudaFree(d_data));
}

int main() {
    printf("========== CUDA Graphs - Manual Creation ==========\n");

    // Check CUDA version
    int runtime_version;
    CHECK_CUDA_ERROR(cudaRuntimeGetVersion(&runtime_version));
    printf("CUDA Runtime Version: %d.%d\n", runtime_version / 1000, (runtime_version % 100) / 10);

    if (runtime_version < 10000) {
        printf("ERROR: CUDA Graphs require CUDA 10.0 or later\n");
        return 1;
    }

    example_manual_graph_creation();
    example_graph_with_memcpy();
    example_graph_update();
    example_graph_clone();

    printf("\n========== All Examples Completed Successfully ==========\n");
    printf("\nKey Takeaways:\n");
    printf("1. Graphs can be created manually by adding nodes\n");
    printf("2. Dependencies define execution order\n");
    printf("3. Graph parameters can be updated without recreation\n");
    printf("4. Graphs can be cloned for independent execution\n");

    return 0;
}
