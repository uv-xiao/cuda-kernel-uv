# Capstone Project Presentation Template

**Duration**: 10-15 minutes
**Format**: Slides + Optional Demo

---

## Slide 1: Title Slide

**Project Title**: [Your Project Name]

**Subtitle**: [One-line description]

**Your Name**
**Date**

---

## Slide 2: Problem Statement

**What problem are you solving?**

- Brief context
- Why is this problem important?
- What are the challenges?

**Example (Project A - Inference Engine)**:
> Problem: LLM inference is slow due to quadratic attention and expensive MoE routing
>
> Challenges:
> - Attention: O(N²) memory and compute
> - MoE: Routing overhead and load balancing
> - Goal: Build optimized inference pipeline

---

## Slide 3: Approach Overview

**High-level architecture diagram**

Show the main components of your solution:
- For Project A: Pipeline diagram (Attention → MoE → Output)
- For Project B: Framework comparison matrix
- For Project C: Sparsity pattern visualization

**Key design decisions**:
- [Decision 1]
- [Decision 2]
- [Decision 3]

---

## Slide 4: Technical Implementation (1/2)

**Component 1**: [e.g., Flash Attention Kernel]

**Implementation highlights**:
- Approach: [Tiling strategy, memory optimization, etc.]
- Key optimizations:
  - Optimization 1
  - Optimization 2
  - Optimization 3

**Code snippet** (optional, keep it short):
```cuda
// Show 5-10 lines of key kernel code
```

---

## Slide 5: Technical Implementation (2/2)

**Component 2**: [e.g., MoE Layer]

**Implementation highlights**:
- Approach: [Routing strategy, batching, etc.]
- Key optimizations:
  - Optimization 1
  - Optimization 2

**Challenges encountered**:
- Challenge 1 → How you solved it
- Challenge 2 → How you solved it

---

## Slide 6: Performance Results

**Benchmark Configuration**:
- GPU: [Model]
- Test cases: [Sizes tested]
- Baselines: [What you compared against]

**Performance Table**:

| Configuration | Your Implementation | Baseline | Speedup |
|---------------|---------------------|----------|---------|
| Config 1 | X ms | Y ms | Z.Zx |
| Config 2 | X ms | Y ms | Z.Zx |
| Config 3 | X ms | Y ms | Z.Zx |

**Key metrics**:
- Throughput: [tokens/s, TFLOPS, etc.]
- Memory usage: [GB]
- Efficiency: [% of peak]

---

## Slide 7: Performance Visualization

**Graph**: Performance across different configurations

**Suggestions**:
- Bar chart comparing your implementation vs baselines
- Line graph showing scaling with input size
- Speedup chart

**Highlight**:
- Where you do well
- Where there's room for improvement

---

## Slide 8: Profiling Insights

**What did profiling reveal?**

**Timeline Analysis** (nsys):
- Screenshot or diagram of kernel timeline
- Key observation: [e.g., "Attention kernel takes 60% of time"]

**Kernel Analysis** (ncu):
- SM Utilization: [X%]
- Memory Bandwidth: [Y%]
- Key bottleneck: [Description]

**Surprising finding**:
[Something unexpected you discovered]

---

## Slide 9: Comparative Analysis

**For Project A/C**: Compare components

| Metric | Component A | Component B |
|--------|-------------|-------------|
| Performance | | |
| Memory | | |
| Complexity | | |

**For Project B**: Compare frameworks

| Framework | Development Time | Performance | Ease of Use |
|-----------|------------------|-------------|-------------|
| CUDA | | | |
| Triton | | | |
| TileLang | | | |
| CUTLASS | | | |

---

## Slide 10: Quality Evaluation (If Applicable)

**For Project C (Sparse Attention)**:

**Quality Metrics**:
- Metric 1: [e.g., Perplexity on long documents]
- Metric 2: [e.g., Accuracy on downstream task]

**Comparison with baselines**:

| Pattern | Quality Metric | Performance | Quality/Perf Trade-off |
|---------|----------------|-------------|------------------------|
| Dense | X | Y | Z |
| Your Pattern | X | Y | Z |

---

## Slide 11: Lessons Learned

**What went well?**
- Success 1
- Success 2
- Success 3

**What was challenging?**
- Challenge 1: [How you overcame it]
- Challenge 2: [What you learned]

**What would you do differently?**
- Insight 1
- Insight 2

---

## Slide 12: Key Takeaways

**Main findings** (3-4 bullet points):

1. [Key finding 1]
2. [Key finding 2]
3. [Key finding 3]
4. [Key finding 4]

**Impact**:
- Performance improvement: [Summary number]
- Learning outcome: [What you learned]

---

## Slide 13: Future Work

**If you had more time, what would you improve?**

**Near-term improvements**:
- Optimization 1
- Optimization 2

**Long-term possibilities**:
- Extension 1
- Extension 2

**Research directions**:
- Question 1
- Question 2

---

## Slide 14: Recommendations

**For Project B (Framework Comparison)**:

**When to use each framework**:
- Use CUDA when: [Scenario]
- Use Triton when: [Scenario]
- Use TileLang when: [Scenario]
- Use CUTLASS when: [Scenario]

**For Project A/C**:

**Best practices discovered**:
- Tip 1
- Tip 2
- Tip 3

---

## Slide 15: Demo (Optional)

**Live demonstration** or **recorded video**

Show:
- Running your implementation
- Comparing with baseline
- Profiling visualization
- Quality comparison (if applicable)

**Keep it short**: 2-3 minutes max

---

## Slide 16: Conclusion

**Summary**:
- Implemented: [What you built]
- Achieved: [Performance results]
- Learned: [Key insights]

**Thank you!**

**Questions?**

Contact: [Your email]
Code: [GitHub link if available]

---

## Appendix Slides (Backup)

**Have these ready for Q&A, but don't present unless asked**:

### A1: Detailed Architecture

[More detailed diagram of your implementation]

### A2: Additional Profiling Data

[Extra profiling screenshots or metrics]

### A3: Correctness Validation

[How you validated correctness]

### A4: Code Overview

[Brief code walkthrough if requested]

### A5: Alternative Approaches Considered

[What else you tried that didn't work]

---

## Presentation Tips

### Before the Presentation

- **Practice**: Rehearse 2-3 times
- **Timing**: Aim for 10-12 minutes (leave time for questions)
- **Test**: Ensure demo works, slides render correctly
- **Prepare**: Anticipate questions

### During the Presentation

- **Start strong**: Clear problem statement
- **Focus on insights**: Not just what you did, but what you learned
- **Use visuals**: Graphs > Tables > Text
- **Tell a story**: Problem → Approach → Results → Insights
- **Engage**: Make eye contact, speak clearly

### Handling Questions

- **Listen carefully**: Understand the question before answering
- **Be honest**: "I don't know, but here's what I think..." is fine
- **Refer to appendix**: "I have data on that in my backup slides"
- **Stay calm**: It's okay to pause and think

---

## Sample Q&A Preparation

**Likely questions for Project A**:

Q: Why did you choose Flash Attention over sparse attention?
A: [Your reasoning]

Q: What's the biggest bottleneck in your implementation?
A: [Based on profiling]

Q: How does your performance compare to production frameworks like vLLM?
A: [Honest comparison]

**Likely questions for Project B**:

Q: Which framework would you use for production?
A: [Your recommendation with reasoning]

Q: Did you encounter any bugs or issues with the frameworks?
A: [Your experience]

Q: How portable is each implementation?
A: [Analysis of portability]

**Likely questions for Project C**:

Q: How did you validate that your sparse pattern doesn't hurt quality?
A: [Your evaluation methodology]

Q: Could your pattern adapt dynamically during inference?
A: [Possibilities and challenges]

Q: How does your pattern compare to Longformer/BigBird?
A: [Comparison]

---

## Presentation Checklist

Before presenting:

- [ ] Slides are clear and readable (large fonts)
- [ ] Graphs have labeled axes and legends
- [ ] Code snippets are syntax-highlighted and readable
- [ ] Timing is 10-12 minutes (practiced)
- [ ] Demo is tested and working
- [ ] Backup slides prepared
- [ ] Questions anticipated
- [ ] Confidence level: high!

---

## Visual Design Tips

### Fonts
- Title: 36-44pt
- Headers: 28-32pt
- Body: 20-24pt
- Code: 16-18pt (monospace)

### Colors
- Use consistent color scheme
- High contrast for readability
- Color-blind friendly palette
- Highlight important numbers/results

### Graphs
- Clear axis labels
- Large legend
- Use different markers/patterns (not just colors)
- Annotate key points

### Layout
- Not too much text per slide (max 5-7 bullets)
- White space is good
- Consistent formatting
- Professional but not boring

---

## Example Slide Deck Structure

**For Project A: Inference Engine**
1. Title
2. Problem: LLM Inference is Slow
3. Approach: Optimized Attention + MoE
4. Flash Attention Implementation
5. MoE Layer Implementation
6. Performance Results (Table)
7. Performance Results (Graph)
8. Profiling Insights
9. End-to-End Analysis
10. Lessons Learned
11. Key Takeaways
12. Future Work
13. Demo (Optional)
14. Conclusion + Q&A

**For Project B: Framework Comparison**
1. Title
2. Problem: Choosing the Right Framework
3. Approach: Implement GEMM in 4 Frameworks
4. CUDA Implementation
5. Triton Implementation
6. TileLang & CUTLASS
7. Performance Comparison
8. Development Experience Comparison
9. Profiling Analysis
10. Framework Recommendations
11. Lessons Learned
12. Key Takeaways
13. Conclusion + Q&A

**For Project C: Sparse Attention**
1. Title
2. Problem: Dense Attention is Expensive
3. Use Case: [Your chosen use case]
4. Custom Sparsity Pattern Design
5. Pattern Visualization
6. Kernel Implementation
7. Performance Results
8. Quality Evaluation
9. Profiling Insights
10. Comparison with Other Patterns
11. Lessons Learned
12. Key Takeaways
13. Future Work
14. Conclusion + Q&A

---

Good luck with your presentation!
