# CUDA Kernel Development Tutorial

A comprehensive, hands-on tutorial for learning CUDA kernel development—from fundamentals to state-of-the-art LLM kernels including FlashAttention, DeepSeek Sparse Attention, and MoE accelerators.

## Who Is This For?

This tutorial is designed for developers aiming to become expert GPU kernel developers. You should have:

### Prerequisites
- **C++ proficiency**: Comfortable with templates, modern C++ (C++17)
- **Python knowledge**: For Triton and high-level DSL chapters
- **Basic GPU understanding**: Know what a GPU is and why it's fast for parallel workloads
- **Linear algebra**: Matrix operations, dot products, attention mechanisms

### Hardware Requirements
- NVIDIA GPU (Volta or newer recommended: V100, A100, RTX 30xx/40xx, H100)
- Minimum 8GB VRAM for most examples
- 16GB+ VRAM recommended for advanced chapters

## Software Setup

> **Note**: You need an NVIDIA GPU driver installed at the system level. Verify with `nvidia-smi`.

### Recommended: Micromamba/Conda (One-Command Setup)

The easiest way to set up the environment is using micromamba or conda:

```bash
# Install micromamba (if not already installed)
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)

# Create and activate environment
micromamba create -f environment.yml
micromamba activate cuda-tutorial

# Verify installation
nvcc --version
python -c "import torch; print(torch.cuda.is_available())"
```

This installs CUDA toolkit, CMake, Python dependencies, and PyTorch in one step.

<details>
<summary><b>Alternative: Manual Installation</b></summary>

If you prefer manual installation or need system-wide CUDA:

#### 1. CUDA Toolkit
```bash
# Ubuntu/Debian
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
sudo sh cuda_12.4.0_550.54.14_linux.run

# Verify installation
nvcc --version
```

#### 2. Build Tools
```bash
# CMake 3.18+
sudo apt-get install cmake ninja-build
```

#### 3. Python Environment
```bash
python -m venv cuda-tutorial-env
source cuda-tutorial-env/bin/activate
pip install -r requirements.txt
```

</details>

### Optional: CUTLASS

For chapters 4-5 (CUTLASS & CuTe):

```bash
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
mkdir build && cd build
cmake .. -DCUTLASS_NVCC_ARCHS="80;89;90"
make -j$(nproc)
```

### Profiling Tools

- **Nsight Compute**: Kernel-level profiling ([Download](https://developer.nvidia.com/nsight-compute))
- **Nsight Systems**: System-wide profiling ([Download](https://developer.nvidia.com/nsight-systems))

## Repository Structure

```
cuda-kernel-tutorial/
├── chapters/
│   ├── 01_introduction/      # GPU architecture & first kernels
│   ├── 02_cuda_basics/       # Memory hierarchy & indexing
│   ├── 03_profiling/         # Nsight profiling & optimization
│   ├── 04_cutlass_cute/      # CUTLASS & CuTe/CuteDSL
│   ├── 05_deepgemm/          # FP8 GEMM & MoE patterns
│   ├── 06_advanced_cuda/     # Warp primitives & cooperative groups
│   ├── 07_triton/            # Triton kernel development
│   ├── 08_tilelang/          # TileLang & high-level DSLs
│   ├── 09_sparse_attention/  # FlashAttention & sparse patterns
│   ├── 10_moe_accelerators/  # MoE optimization & SonicMoE
│   └── 11_capstone/          # Full projects
├── common/                   # Shared utilities
├── benchmarks/               # Performance benchmarks
└── tests/                    # Test harnesses
```

## Curriculum

### Part 1: Foundations (Chapters 1-3)

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 01 | [GPU & CUDA Introduction](chapters/01_introduction/) | Threads, blocks, grids, memory model |
| 02 | [CUDA Basics & Memory](chapters/02_cuda_basics/) | Shared memory, synchronization, indexing |
| 03 | [Profiling & Optimization](chapters/03_profiling/) | Nsight Compute, memory coalescing, bank conflicts |

### Part 2: High-Performance Libraries (Chapters 4-5)

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 04 | [CUTLASS & CuTe](chapters/04_cutlass_cute/) | Tensor cores, CuTe layouts, CuteDSL |
| 05 | [DeepGEMM](chapters/05_deepgemm/) | FP8 GEMM, grouped GEMM, MoE patterns |

### Part 3: Advanced Techniques (Chapter 6)

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 06 | [Advanced CUDA](chapters/06_advanced_cuda/) | Warp primitives, cooperative groups, CUDA graphs |

### Part 4: High-Level DSLs (Chapters 7-8)

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 07 | [Triton](chapters/07_triton/) | Block-based kernels, autotuning, fusion |
| 08 | [TileLang](chapters/08_tilelang/) | Tile-centric DSL, pipelining |

### Part 5: LLM Kernels (Chapters 9-10)

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 09 | [Sparse Attention](chapters/09_sparse_attention/) | FlashAttention, DeepSeek sparse attention |
| 10 | [MoE Accelerators](chapters/10_moe_accelerators/) | Tile-aware optimization, SonicMoE |

### Part 6: Integration (Chapter 11)

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 11 | [Capstone Projects](chapters/11_capstone/) | End-to-end LLM inference, kernel comparison |

## Building & Running

### Build All Examples
```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;89;90"
make -j$(nproc)
```

### Build Specific Chapter
```bash
cd chapters/01_introduction
mkdir build && cd build
cmake ..
make
```

### Run Tests
```bash
# C++/CUDA tests
./build/tests/test_all

# Python tests
pytest tests/ -v
```

### Profiling
```bash
# Nsight Compute (kernel analysis)
ncu --set full -o profile ./build/chapters/03_profiling/examples/matmul

# Nsight Systems (timeline)
nsys profile -o timeline ./build/chapters/03_profiling/examples/matmul
```

## Learning Path

### Suggested Order
1. **Complete beginners**: Start with Chapter 01 and proceed sequentially
2. **Some CUDA experience**: Skim 01-02, focus on 03-06
3. **Experienced developers**: Jump to 04-05 for CUTLASS, then 07-10 for modern techniques

### Time Estimates
- Chapters 01-03: 2-3 hours each
- Chapters 04-06: 4-6 hours each
- Chapters 07-10: 6-8 hours each
- Chapter 11: 10+ hours (project-based)

## Key Resources

### Official Documentation
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
- [CUTLASS Documentation](https://docs.nvidia.com/cutlass/)
- [Triton Documentation](https://triton-lang.org/)

### Open Source References
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) - FP8 GEMM from DeepSeek
- [SonicMoE](https://github.com/Dao-AILab/sonic-moe) - Tile-aware MoE optimization
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) - Memory-efficient attention
- [TileLang](https://github.com/tile-ai/tilelang) - High-level kernel DSL
- [LeetCUDA](https://github.com/xlite-dev/LeetCUDA) - 200+ CUDA exercises

### Research Papers
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [SonicMoE Paper](https://arxiv.org/abs/2512.14080)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)

### Tutorials
- [CUDA MatMul Optimization (siboehm)](https://siboehm.com/articles/22/CUDA-MMM)
- [GPU MODE Lectures](https://github.com/cuda-mode/lectures)

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

**Happy kernel writing!** 🚀

*Start your journey: [Chapter 01 - GPU & CUDA Introduction](chapters/01_introduction/)*
