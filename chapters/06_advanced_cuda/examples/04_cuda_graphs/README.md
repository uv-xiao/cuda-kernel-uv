# CUDA Graphs Examples

CUDA Graphs allow you to define GPU work as a graph of operations, significantly reducing kernel launch overhead for recurring workloads.

## Overview

Traditional CUDA programming launches kernels one at a time. Each launch incurs CPU-GPU synchronization overhead (10-50 microseconds). CUDA Graphs bundle multiple operations into a single launchable object, executing the entire sequence with one API call.

## Performance Benefits

- **Reduced Launch Overhead**: 10-30% speedup for workloads with many kernels
- **Better Optimization**: Driver can optimize entire graph
- **Lower CPU Usage**: Fewer API calls
- **Deterministic Execution**: Fixed dependency structure

## When to Use CUDA Graphs

**Good Candidates:**
- Recurring workloads (same operations repeated)
- Many small kernels
- Fixed dependency structure
- Iterative algorithms
- Deep learning inference

**Poor Candidates:**
- One-time operations
- Dynamic workload patterns
- Conditional execution that changes
- Large kernels (launch overhead is negligible)

## Graph Creation Methods

### 1. Manual Creation (graph_basics.cu)

Explicitly create nodes and define dependencies:

```cuda
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);

// Add kernel node
cudaGraphNode_t node1;
cudaKernelNodeParams params = {...};
cudaGraphAddKernelNode(&node1, graph, NULL, 0, &params);

// Add dependent node
cudaGraphNode_t node2;
cudaGraphNode_t deps[] = {node1};
cudaGraphAddKernelNode(&node2, graph, deps, 1, &params2);
```

**Advantages:** Full control, clear dependencies
**Disadvantages:** Verbose, manual dependency management

### 2. Stream Capture (graph_capture.cu)

Record operations from a stream:

```cuda
cudaStream_t stream;
cudaStreamCreate(&stream);

cudaGraph_t graph;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

// Record operations
kernel1<<<grid, block, 0, stream>>>(...);
kernel2<<<grid, block, 0, stream>>>(...);
cudaMemcpyAsync(..., stream);

cudaStreamEndCapture(stream, &graph);
```

**Advantages:** Automatic dependency capture, less code
**Disadvantages:** Less explicit control

## Graph Execution Workflow

1. **Create** graph (manual or capture)
2. **Instantiate** graph → executable graph
3. **Launch** executable graph (can reuse many times)
4. **Update** parameters if needed (optional)
5. **Destroy** when done

```cuda
// Create
cudaGraph_t graph = create_graph();

// Instantiate
cudaGraphExec_t graph_exec;
cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);

// Launch multiple times
for (int i = 0; i < 1000; i++) {
    cudaGraphLaunch(graph_exec, stream);
}

// Cleanup
cudaGraphExecDestroy(graph_exec);
cudaGraphDestroy(graph);
```

## Graph Node Types

- **Kernel nodes**: GPU kernel execution
- **Memcpy nodes**: Host-device, device-device copies
- **Memset nodes**: Memory initialization
- **Host nodes**: CPU function calls
- **Child graph nodes**: Nested graphs
- **Empty nodes**: Dependency-only nodes

## Updating Graphs

Update parameters without recreating:

```cuda
cudaKernelNodeParams new_params;
cudaGraphExecKernelNodeSetParams(graph_exec, node, &new_params);
```

**What can be updated:**
- Kernel parameters
- Memcpy sources/destinations (same size)
- Node parameters

**What cannot be updated:**
- Graph structure (add/remove nodes)
- Node types
- Dependencies

To change structure: create new graph

## Performance Comparison

### Typical Results (1000 iterations, 3 kernels each)

```
Standard launches: 45.2 ms
Graph execution:   32.1 ms
Speedup:           1.41x
Overhead saved:    13.1 us per iteration
```

### Why Faster?

1. **Single submission**: One GPU command instead of 3000
2. **No host sync**: GPU executes entire graph independently
3. **Driver optimization**: Can optimize entire sequence
4. **Reduced queuing**: Less work for GPU scheduler

## Best Practices

1. **Reuse graphs**: Create once, launch many times
2. **Use stream capture** for complex workflows
3. **Update parameters** instead of recreating
4. **Profile first**: Ensure launch overhead is significant
5. **Check dependencies**: Ensure correct execution order
6. **Consider memory**: Graphs have memory footprint

## Building and Running

```bash
mkdir build && cd build
cmake ..
make
./graph_basics
./graph_capture
```

## Requirements

- CUDA 10.0 or later
- All NVIDIA GPUs (any compute capability)
- For nested graphs: CUDA 11.0+

## References

- [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [Getting Started with CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/)
