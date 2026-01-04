#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void saxpy(float *y, const float *x, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

__global__ void vector_scale(float *x, float scalar, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] *= scalar;
    }
}

__global__ void vector_add(float *c, const float *a, const float *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// Benchmark standard kernel launches vs graph execution
float benchmark_standard_launches(float *d_x, float *d_y, float *d_z,
                                   int n, int iterations) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    float a = 2.0f;
    float scalar = 1.5f;

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        saxpy<<<grid, block>>>(d_y, d_x, a, n);
        vector_scale<<<grid, block>>>(d_y, scalar, n);
        vector_add<<<grid, block>>>(d_z, d_x, d_y, n);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return time_ms;
}

// Benchmark graph execution
float benchmark_graph_execution(float *d_x, float *d_y, float *d_z,
                                int n, int iterations, cudaGraphExec_t graph_exec) {
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA_ERROR(cudaGraphLaunch(graph_exec, 0));
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float time_ms;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time_ms, start, stop));

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return time_ms;
}

int main() {
    printf("========== CUDA Graphs - Stream Capture ==========\n");

    const int n = 1024 * 1024;
    const int bytes = n * sizeof(float);
    const int iterations = 1000;

    // Allocate memory
    float *h_x = (float*)malloc(bytes);
    float *h_y = (float*)malloc(bytes);
    float *h_z = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
    }

    float *d_x, *d_y, *d_z;
    CHECK_CUDA_ERROR(cudaMalloc(&d_x, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_y, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_z, bytes));

    CHECK_CUDA_ERROR(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    // Create stream for capture
    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    float a = 2.0f;
    float scalar = 1.5f;

    printf("\n1. Stream Capture Example:\n");

    // Begin stream capture
    cudaGraph_t graph;
    CHECK_CUDA_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Record operations into the stream (these are captured, not executed)
    saxpy<<<grid, block, 0, stream>>>(d_y, d_x, a, n);
    vector_scale<<<grid, block, 0, stream>>>(d_y, scalar, n);
    vector_add<<<grid, block, 0, stream>>>(d_z, d_x, d_y, n);

    // End capture
    CHECK_CUDA_ERROR(cudaStreamEndCapture(stream, &graph));

    printf("   Captured 3 kernel launches into graph\n");

    // Instantiate the graph
    cudaGraphExec_t graph_exec;
    CHECK_CUDA_ERROR(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

    printf("   Graph instantiated successfully\n");

    // Execute the graph
    CHECK_CUDA_ERROR(cudaGraphLaunch(graph_exec, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Verify result
    CHECK_CUDA_ERROR(cudaMemcpy(h_z, d_z, bytes, cudaMemcpyDeviceToHost));

    // Expected: z = x + (a*x + y)*scalar = 1 + (2*1 + 2)*1.5 = 1 + 6 = 7
    float expected = 1.0f + (2.0f * 1.0f + 2.0f) * 1.5f;
    bool correct = (fabsf(h_z[0] - expected) < 1e-5);

    printf("   Expected result: %.1f\n", expected);
    printf("   Actual result: %.1f\n", h_z[0]);
    printf("   Verification: %s\n", correct ? "PASS" : "FAIL");

    // Reset data for benchmarking
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    printf("\n2. Performance Comparison (%d iterations):\n", iterations);

    // Benchmark standard launches
    float standard_time = benchmark_standard_launches(d_x, d_y, d_z, n, iterations);

    // Reset data
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    // Benchmark graph execution
    float graph_time = benchmark_graph_execution(d_x, d_y, d_z, n, iterations, graph_exec);

    printf("   Standard kernel launches: %.3f ms\n", standard_time);
    printf("   Graph execution:          %.3f ms\n", graph_time);
    printf("   Speedup:                  %.2fx\n", standard_time / graph_time);
    printf("   Per-iteration overhead saved: %.2f us\n",
           (standard_time - graph_time) * 1000.0f / iterations);

    // Example 3: Nested capture
    printf("\n3. Stream Capture with cudaMemcpyAsync:\n");

    cudaGraph_t graph2;
    CHECK_CUDA_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Capture memory copies and kernels
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_x, h_x, bytes, cudaMemcpyHostToDevice, stream));
    saxpy<<<grid, block, 0, stream>>>(d_y, d_x, a, n);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_z, d_y, bytes, cudaMemcpyDeviceToHost, stream));

    CHECK_CUDA_ERROR(cudaStreamEndCapture(stream, &graph2));

    cudaGraphExec_t graph_exec2;
    CHECK_CUDA_ERROR(cudaGraphInstantiate(&graph_exec2, graph2, NULL, NULL, 0));

    printf("   Captured H2D copy + kernel + D2H copy\n");

    // Execute
    CHECK_CUDA_ERROR(cudaGraphLaunch(graph_exec2, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    // Verify (y = a*x + y = 2*1 + 2 = 4)
    expected = 2.0f * 1.0f + 2.0f;
    correct = (fabsf(h_z[0] - expected) < 1e-5);

    printf("   Expected result: %.1f\n", expected);
    printf("   Actual result: %.1f\n", h_z[0]);
    printf("   Verification: %s\n", correct ? "PASS" : "FAIL");

    // Example 4: Graph re-execution with updated data
    printf("\n4. Graph Re-execution:\n");

    // Update input data
    for (int i = 0; i < n; i++) {
        h_x[i] = 3.0f;
        h_y[i] = 4.0f;
    }
    CHECK_CUDA_ERROR(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    // Re-execute the same graph with new data
    CHECK_CUDA_ERROR(cudaGraphLaunch(graph_exec, stream));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));

    CHECK_CUDA_ERROR(cudaMemcpy(h_z, d_z, bytes, cudaMemcpyDeviceToHost));

    // Expected: z = x + (a*x + y)*scalar = 3 + (2*3 + 4)*1.5 = 3 + 15 = 18
    expected = 3.0f + (2.0f * 3.0f + 4.0f) * 1.5f;
    correct = (fabsf(h_z[0] - expected) < 1e-5);

    printf("   Re-executed graph with new input data\n");
    printf("   Expected result: %.1f\n", expected);
    printf("   Actual result: %.1f\n", h_z[0]);
    printf("   Verification: %s\n", correct ? "PASS" : "FAIL");

    // Cleanup
    cudaGraphExecDestroy(graph_exec);
    cudaGraphExecDestroy(graph_exec2);
    cudaGraphDestroy(graph);
    cudaGraphDestroy(graph2);
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream));

    free(h_x);
    free(h_y);
    free(h_z);
    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUDA_ERROR(cudaFree(d_y));
    CHECK_CUDA_ERROR(cudaFree(d_z));

    printf("\n========== All Examples Completed Successfully ==========\n");
    printf("\nKey Takeaways:\n");
    printf("1. Stream capture automatically records operations into a graph\n");
    printf("2. Graphs significantly reduce kernel launch overhead\n");
    printf("3. Graphs can include memory copies and kernel launches\n");
    printf("4. Graphs can be re-executed with updated data\n");
    printf("5. Best for workloads with many small kernels or repeated patterns\n");

    return 0;
}
