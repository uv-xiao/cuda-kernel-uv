# Chapter 11: Capstone Projects

Welcome to the capstone chapter! This chapter ties together everything you've learned throughout this tutorial by providing three comprehensive projects that simulate real-world kernel development scenarios.

## Overview

Each capstone project is designed to:
- Integrate multiple concepts from previous chapters
- Simulate realistic production scenarios
- Provide hands-on experience with end-to-end kernel development
- Build your portfolio with demonstrable work

## Project Options

### Project A: Mini LLM Inference Engine
**Difficulty**: Advanced
**Duration**: 2-3 weeks
**Focus**: End-to-end system integration, performance optimization

Build a minimal but functional LLM inference engine with optimized CUDA kernels for the critical path operations. This project emphasizes system-level thinking and practical performance optimization.

**Key Components**:
- Flash Attention or sparse attention kernel
- Mixture of Experts (MoE) layer with optimized GEMM
- Complete inference pipeline
- Comprehensive benchmarking suite

**Skills Demonstrated**:
- Advanced memory management
- Multi-kernel optimization
- System-level performance tuning
- Real-world benchmarking

[Go to Project A →](./project_a_inference_engine/)

---

### Project B: Kernel Implementation Comparison
**Difficulty**: Intermediate to Advanced
**Duration**: 1-2 weeks
**Focus**: Cross-framework expertise, analytical thinking

Implement the same GEMM kernel across four different frameworks (CUDA, Triton, TileLang, CUTLASS) and conduct a thorough comparative analysis. This project emphasizes understanding tradeoffs and making informed technology choices.

**Key Components**:
- GEMM implementation in CUDA C++
- GEMM implementation in Triton
- GEMM implementation in TileLang
- GEMM implementation using CUTLASS
- Comprehensive profiling and analysis
- Technical report with recommendations

**Skills Demonstrated**:
- Multi-framework proficiency
- Performance analysis
- Technical writing
- Technology evaluation

[Go to Project B →](./project_b_kernel_comparison/)

---

### Project C: Custom Sparse Attention Pattern
**Difficulty**: Advanced
**Duration**: 2-3 weeks
**Focus**: Novel algorithm design, research-oriented

Design and implement a novel sparse attention pattern tailored for a specific use case (long-context modeling, multimodal fusion, or hierarchical attention). This project emphasizes innovation and research skills.

**Key Components**:
- Novel sparsity pattern design
- Efficient CUDA kernel implementation
- Quality metrics (attention accuracy)
- Performance benchmarks
- Research-style documentation

**Skills Demonstrated**:
- Algorithm design
- Research methodology
- Innovation in kernel optimization
- Rigorous evaluation

[Go to Project C →](./project_c_custom_sparse_attention/)

---

## Expected Deliverables

All projects should include:

### 1. Working Code
- Clean, well-documented source code
- All code should compile and run without errors
- Comprehensive test suite
- Build scripts / setup instructions

### 2. Benchmarks
- Performance measurements on at least 2 problem sizes
- Comparison with baseline implementations
- Profiling results (nsys, ncu)
- Clear visualization of results

### 3. Documentation
- README with setup and usage instructions
- Code comments explaining key optimizations
- Technical report (2-4 pages) covering:
  - Approach and methodology
  - Key design decisions
  - Performance results
  - Lessons learned
  - Future improvements

### 4. Presentation
- 10-15 minute presentation covering your work
- Slides highlighting key results
- Live demo (optional but recommended)

---

## Evaluation Criteria

Your project will be evaluated on the following dimensions:

### Correctness (30%)
- Does the implementation produce correct results?
- Are edge cases handled properly?
- Is there adequate test coverage?

**Grading Scale**:
- 90-100%: Comprehensive tests, handles all edge cases, bit-exact when applicable
- 75-89%: Good test coverage, handles most cases, minor numerical issues
- 60-74%: Basic functionality works, some edge cases fail
- Below 60%: Fundamental correctness issues

### Performance (30%)
- How does performance compare to baselines?
- Are the implementations properly optimized?
- Is the optimization approach justified?

**Target Performance** (compared to cuBLAS/cuDNN baselines):
- Excellent: >80% of baseline throughput
- Good: 60-80% of baseline throughput
- Acceptable: 40-60% of baseline throughput
- Needs Work: <40% of baseline throughput

### Code Quality (20%)
- Is the code well-organized and readable?
- Are optimizations clearly documented?
- Is the build/test infrastructure sound?

**Criteria**:
- Clear structure and modularity
- Meaningful variable/function names
- Explanatory comments for non-obvious optimizations
- Consistent style
- No code smells (magic numbers, duplication, etc.)

### Analysis & Documentation (20%)
- Is the approach clearly explained?
- Are results properly analyzed?
- Are limitations acknowledged?
- Is the documentation helpful?

**Strong Documentation Includes**:
- Clear problem statement
- Design rationale
- Performance analysis with visualizations
- Honest discussion of limitations
- Suggestions for future work

---

## Resources

### Essential Tools
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) - v11.8 or later
- [Nsight Systems](https://developer.nvidia.com/nsight-systems) - For timeline profiling
- [Nsight Compute](https://developer.nvidia.com/nsight-compute) - For kernel profiling
- [Triton](https://github.com/openai/triton) - For Triton implementations
- [CUTLASS](https://github.com/NVIDIA/cutlass) - For CUTLASS implementations

### Reference Materials
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass/tree/main/media/docs)
- [Triton Documentation](https://triton-lang.org/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [Mixture of Experts Paper](https://arxiv.org/abs/1701.06538)

### Additional Resources
- [Profiling Checklist](./resources/profiling_checklist.md)
- [Optimization Tips](./resources/optimization_tips.md)
- [Presentation Template](./resources/presentation_template.md)

---

## Tips for Success

### Start Early
- These projects are substantial - don't underestimate the time needed
- Begin with the starter code to understand the scope
- Set up your development environment early

### Iterate
- Start with a simple, correct implementation
- Profile to identify bottlenecks
- Optimize incrementally
- Validate correctness after each optimization

### Ask Questions
- If you're stuck, consult the previous chapters
- Review reference implementations, but understand don't copy
- Reach out to the community or instructors

### Document as You Go
- Keep notes on design decisions
- Record benchmark results immediately
- Screenshot profiling results
- Maintain a development log

### Test Thoroughly
- Write tests before optimizing
- Test edge cases (size 0, size 1, non-power-of-2)
- Validate numerical accuracy
- Test on different GPU architectures if possible

### Focus on Learning
- The goal is to deepen your understanding
- It's okay if you don't achieve peak performance
- Document what you learned, especially from failures
- Compare your approaches with reference implementations

---

## Timeline Suggestions

### Week 1: Setup & Baseline
- Set up development environment
- Understand the problem thoroughly
- Implement naive/baseline version
- Establish correctness tests
- Run initial benchmarks

### Week 2: Optimization
- Profile the baseline implementation
- Identify key bottlenecks
- Implement optimizations incrementally
- Validate correctness after each change
- Document optimization strategies

### Week 3: Polish & Documentation
- Final performance tuning
- Complete test coverage
- Write technical report
- Prepare presentation
- Review and refine all deliverables

---

## Getting Help

If you encounter issues:

1. **Review Previous Chapters**: Most concepts have been covered earlier
2. **Check Reference Implementations**: Use them to verify your understanding
3. **Profile Early**: Don't guess at bottlenecks - measure
4. **Start Simple**: Get something working before optimizing
5. **Ask Specific Questions**: "Why is my kernel slow?" is too broad; "Why does my kernel have 80% L2 cache miss rate?" is specific

---

## Submission Checklist

Before submitting your capstone project:

- [ ] All code compiles without warnings
- [ ] All tests pass
- [ ] Benchmarks run successfully
- [ ] README includes setup instructions
- [ ] Technical report is complete (2-4 pages)
- [ ] Code is well-commented
- [ ] Profiling results are included
- [ ] Presentation is prepared
- [ ] All deliverables are organized in proper directory structure

---

## Academic Integrity

These projects are designed for individual work. While you may:
- Discuss high-level approaches with peers
- Reference public implementations and papers
- Use the provided starter and reference code

You must NOT:
- Copy code from peers or online sources without attribution
- Share your complete solutions with others
- Misrepresent work as your own

When in doubt, cite your sources and explain your understanding.

---

## What's Next?

After completing your capstone project, you will have:
- Portfolio-ready code demonstrating advanced CUDA skills
- Deep understanding of kernel optimization
- Experience with real-world performance engineering
- Foundation for contributing to production ML systems

Consider:
- Contributing to open-source projects (PyTorch, TensorFlow, Triton)
- Exploring research in ML systems
- Applying these skills in production environments
- Sharing your learnings through blog posts or talks

Good luck, and enjoy building!
