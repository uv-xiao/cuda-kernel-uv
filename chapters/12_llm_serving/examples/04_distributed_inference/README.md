# Example 4: Distributed Inference

Multi-GPU LLM inference with tensor parallelism using NCCL.

## Overview

Tensor parallelism splits model weights across GPUs:
- **Column parallel**: Split weight columns (for QKV projection, FFN up)
- **Row parallel**: Split weight rows (for attention output, FFN down)
- **NCCL collectives**: All-reduce, all-gather for synchronization

## Tensor Parallelism Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    TENSOR PARALLELISM (TP=4)                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Column Parallel Linear (QKV Projection):                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Weight: [hidden × 3*hidden]                                    │   │
│  │           Split by columns                                       │   │
│  │                                                                  │   │
│  │  GPU 0: W[:, 0:N/4]    → Q0, K0, V0                            │   │
│  │  GPU 1: W[:, N/4:N/2]  → Q1, K1, V1                            │   │
│  │  GPU 2: W[:, N/2:3N/4] → Q2, K2, V2                            │   │
│  │  GPU 3: W[:, 3N/4:N]   → Q3, K3, V3                            │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Row Parallel Linear (Output Projection):                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Each GPU has partial output → All-Reduce (SUM)                 │   │
│  │                                                                  │   │
│  │  GPU 0: out0 ─┐                                                 │   │
│  │  GPU 1: out1 ─┼──▶ NCCL All-Reduce ──▶ out_sum (on all GPUs)   │   │
│  │  GPU 2: out2 ─┤                                                 │   │
│  │  GPU 3: out3 ─┘                                                 │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Vocabulary Parallel Embedding:                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Vocab: [V × hidden] split by rows                              │   │
│  │                                                                  │   │
│  │  GPU 0: vocab[0:V/4]        (mask out-of-range tokens)         │   │
│  │  GPU 1: vocab[V/4:V/2]                                         │   │
│  │  GPU 2: vocab[V/2:3V/4]                                        │   │
│  │  GPU 3: vocab[3V/4:V]                                          │   │
│  │                                                                  │   │
│  │  After lookup: All-Reduce (SUM) to get full embedding          │   │
│  │                                                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## NCCL Communication Wrapper

### PyNCCL Implementation

**File:** `code-repos/mini-sglang/python/minisgl/kernel/csrc/src/pynccl.cu`
**Lines:** 72-175

```cpp
class PyNCCLCommunicator {
public:
    PyNCCLCommunicator(int tp_rank, int tp_size, ncclUniqueId id, size_t max_bytes)
        : m_max_bytes(max_bytes) {
        // Create NCCL communicator
        NCCL_CHECK(ncclCommInitRank(&m_comm, tp_size, id, tp_rank));

        // Allocate symmetric memory buffer for small tensors
        if (max_bytes > 0) {
            cudaMalloc(&m_sym_mem, max_bytes);
        }
    }

    auto all_reduce(tvm::ffi::TensorView t, std::string op) const -> void {
        auto stream = get_stream(t.device());
        auto* data_ptr = t.data();
        auto size_bytes = t.nbytes();

        ncclRedOp_t reduce_op = (op == "sum") ? ncclSum : ncclMax;

        if (size_bytes <= m_max_bytes && m_sym_mem) {
            // Use symmetric memory for small tensors
            cudaMemcpyAsync(m_sym_mem, data_ptr, size_bytes, cudaMemcpyD2D, stream);
            ncclAllReduce(m_sym_mem, m_sym_mem, size_dim, dtype, reduce_op, m_comm, stream);
            cudaMemcpyAsync(data_ptr, m_sym_mem, size_bytes, cudaMemcpyD2D, stream);
        } else {
            // In-place all-reduce for large tensors
            ncclAllReduce(data_ptr, data_ptr, size_dim, dtype, reduce_op, m_comm, stream);
        }
    }

    auto all_gather(tvm::ffi::TensorView dst, tvm::ffi::TensorView src) const -> void {
        auto stream = get_stream(dst.device());
        ncclAllGather(src.data(), dst.data(), src_size, dtype, m_comm, stream);
    }

private:
    ncclComm_t m_comm;
    void* m_sym_mem = nullptr;
    size_t m_max_bytes;
};
```

## Python Integration

### Distributed Communicator

**File:** `code-repos/mini-sglang/python/minisgl/distributed/impl.py`
**Lines:** 44-91

```python
@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    comm: PyNCCLCommunicator

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")
        return x

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info
        world_size = get_tp_info().size
        output_shape = list(x.shape)
        output_shape[0] *= world_size
        result = x.new_empty(output_shape)
        self.comm.all_gather(result, x)
        return result


def enable_pynccl_distributed(
    tp_info: DistributedInfo,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_bytes: int
) -> None:
    """Enable PyNCCL for tensor parallelism."""
    if tp_info.size == 1:
        return

    from minisgl.kernel import init_pynccl

    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))
```

## Usage in Model Layers

### Embedding with Vocabulary Parallelism

**File:** `code-repos/mini-sglang/python/minisgl/layers/embedding.py`
**Lines:** 14-43

```python
class VocabParallelEmbedding(BaseOP):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        tp_info = get_tp_info()
        tp_rank = tp_info.rank
        self.tp_size = tp_info.size

        # Each GPU gets vocab_size/tp_size rows
        self.num_embeddings_tp = divide_up(num_embeddings, self.tp_size)
        start_idx = self.num_embeddings_tp * tp_rank
        finish_idx = min(start_idx + self.num_embeddings_tp, num_embeddings)
        self.vocab_range = (start_idx, finish_idx - start_idx)

        self.weight = torch.empty(self.num_embeddings_tp, embedding_dim)
        self._comm = DistributedCommunicator()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from minisgl.kernel import indexing

        # Lookup with masking for out-of-range indices
        y = indexing(
            weights=self.weight,
            indices=x,
            vocab_range=self.vocab_range if self.tp_size > 1 else None,
        )

        # All-reduce to combine results from all GPUs
        return self._comm.all_reduce(y) if self.tp_size > 1 else y
```

---

## Running Multi-GPU Test

### Communication Test

```bash
cd /home/uvxiao/mlkb/code-repos/mini-sglang
python tests/kernel/test_comm.py  # Requires 4 GPUs
```

### Profile NCCL

```bash
nsys profile --trace=cuda,nvtx,nccl \
    -o nccl_trace \
    python tests/kernel/test_comm.py
```

### Multi-GPU Inference

```bash
# 4-GPU tensor parallel inference
python -m minisgl \
    --model "meta-llama/Llama-3.1-70B-Instruct" \
    --tp 4 \
    --port 8000
```

---

## Performance Considerations

### All-Reduce Bandwidth

| TP Size | A100 NVLink BW | Expected Latency (8KB) |
|---------|----------------|------------------------|
| 2 | 300 GB/s | ~5 us |
| 4 | 300 GB/s | ~10 us |
| 8 | 300 GB/s | ~20 us |

### Optimization: Symmetric Memory

For small tensors (< 8KB), use pre-allocated symmetric memory:
- Avoids dynamic allocation overhead
- Better for repeated all-reduce in tight loops

---

## Summary

| Operation | NCCL Collective | Use Case |
|-----------|-----------------|----------|
| Attention output | All-Reduce | Combine partial attention outputs |
| FFN output | All-Reduce | Combine partial FFN outputs |
| LM head | All-Gather | Collect full logits from vocab parallel |

---

## Next Steps

- Complete exercises to solidify understanding
- Try different TP sizes and measure latency
- Explore pipeline parallelism (not covered in mini-sglang)
