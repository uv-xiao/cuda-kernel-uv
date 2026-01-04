# CLAUDE.md - Project Guidelines for CUDA Kernel Tutorial Repository

## Project Overview

This is a comprehensive, GitHub-ready tutorial repository for systematic learning of CUDA kernel development, profiling & tuning, CUTLASS & CuteDSL, and modern LLM-scale kernels including DeepSeek V3.2 Sparse Attention and MoE accelerators.

**Target Audience**: Developers aiming to become expert GPU kernel developers, progressing from basics to state-of-the-art attention and MoE implementations.

## Repository Structure

```
cuda-kernel-tutorial/
├── CLAUDE.md                 # This file - project guidelines
├── README.md                 # Main entry point with setup instructions
├── LICENSE                   # MIT License
├── .gitignore               # Git ignore patterns
├── CMakeLists.txt           # Top-level CMake configuration
├── requirements.txt         # Python dependencies
├── setup.sh                 # Environment setup script
├── chapters/
│   ├── 01_introduction/     # GPU & CUDA Introduction
│   ├── 02_cuda_basics/      # CUDA Basics & Memory
│   ├── 03_profiling/        # Profiling & Optimization
│   ├── 04_cutlass_cute/     # CUTLASS and CuTe/CuteDSL
│   ├── 05_deepgemm/         # DeepGEMM & Advanced GEMM
│   ├── 06_advanced_cuda/    # Advanced CUDA Features
│   ├── 07_triton/           # Triton Kernel Design
│   ├── 08_tilelang/         # TileLang & High-Level DSLs
│   ├── 09_sparse_attention/ # Sparse Attention Kernels
│   ├── 10_moe_accelerators/ # Tile-Aware MoE Accelerators
│   └── 11_capstone/         # Capstone Projects
├── common/                   # Shared utilities and helpers
│   ├── include/             # Common headers
│   ├── src/                 # Common source files
│   └── python/              # Python utilities
├── benchmarks/              # Benchmark scripts and results
└── tests/                   # Test harnesses
```

## Chapter Structure

Each chapter folder should contain:

```
XX_chapter_name/
├── README.md                # Learning goals, outcomes, reading materials
├── materials/               # PDFs, links, additional reading
├── examples/                # Runnable code examples
│   ├── 01_example/
│   │   ├── CMakeLists.txt
│   │   ├── example.cu
│   │   └── README.md
│   └── ...
├── exercises/               # Practice problems
│   ├── 01_exercise/
│   │   ├── problem.md
│   │   ├── starter.cu
│   │   ├── solution.cu
│   │   └── test.py
│   └── ...
├── profiling/               # Profiler traces and analysis
└── benchmarks/              # Performance checkpoints
```

## Code Standards

### CUDA C++ (.cu, .cuh)
- Use modern C++17 features where appropriate
- Include comprehensive error checking with `cudaGetLastError()`
- Use descriptive variable names and comments
- Follow NVIDIA coding conventions
- Include timing and bandwidth measurements

### Python (.py)
- Use type hints
- Follow PEP 8 style guidelines
- Include docstrings for all functions
- Use pytest for testing

### Triton (.py with Triton)
- Clear block size and grid annotations
- Include performance comparisons with PyTorch/CUDA baselines

## Performance Targets

Each exercise should specify:
1. Correctness criteria (numerical tolerance)
2. Performance targets (GFLOP/s, bandwidth utilization %)
3. Profiler metrics to examine

## Build System

- CMake 3.18+ for CUDA projects
- Support for CUDA 11.8+ and 12.x
- Compute capabilities: sm_70, sm_80, sm_89, sm_90

## Testing

- Unit tests for all kernels
- Numerical correctness validation
- Performance regression tests
- CI/CD compatible test scripts

## Documentation Standards

- Use clear, concise Markdown
- Include diagrams (Mermaid, ASCII art, or images)
- Cite authoritative sources (NVIDIA docs, papers)
- Provide expected outputs for all examples

## Key Resources

### Official Documentation
- NVIDIA CUDA Programming Guide
- NVIDIA Nsight Compute Documentation
- CUTLASS Documentation
- Triton Documentation

### Research Papers
- DeepSeek-V3 Technical Report
- SonicMoE Paper
- FlashAttention Papers

### Open Source References
- DeepGEMM (DeepSeek)
- Triton-Puzzles-Lite
- LeetCUDA
- vLLM, SGLang kernels

## Commands

### Building
```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;89;90"
make -j$(nproc)
```

### Running Tests
```bash
pytest tests/ -v
./build/tests/test_all
```

### Profiling
```bash
ncu --set full -o profile ./build/examples/matmul
nsys profile -o timeline ./build/examples/attention
```

## Contributing Guidelines

1. All code must compile without warnings
2. Include tests for new kernels
3. Document performance characteristics
4. Update chapter READMEs with new content
