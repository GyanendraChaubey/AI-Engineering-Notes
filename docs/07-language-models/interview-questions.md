# Language Models — Interview Questions

Role focus: **AI Researcher**

---

## Q1 — Compute Budget Allocation and Scaling Laws

**Question:** Given a fixed GPU budget, how do you decide the trade-off between model size, dataset size, and training steps? What role do scaling laws play, and when do you deviate from them?

**Short answer:** Scaling laws (particularly the Chinchilla result) give a principled compute-optimal frontier: roughly 20 tokens per parameter. But compute-optimal training is not always deployment-optimal — inference cost, data availability, and specific capability requirements often push decisions off the theoretical frontier.

---

### The core tension

With fixed compute C, you have a three-way trade-off:

- Train a **larger** model for **fewer** steps on **less** data
- Train a **smaller** model for **more** steps on **more** data
- Any configuration in between

Scaling laws let you predict which choice yields lower loss for a given compute budget.

---

### What the Chinchilla result says

The key insight from Chinchilla (2022): pre-2022 models were systematically undertrained — they were large but saw too few tokens. Compute-optimal training requires scaling dataset size proportionally with model size:

- ~20 training tokens per model parameter
- A 10B parameter model should see ~200B tokens

This reverses the prior intuition that "bigger model = better model." A 7B model trained on 140B tokens often outperforms a 70B model trained on 14B tokens at the same compute cost.

---

### When to deviate from compute-optimal training

**Inference cost matters more than training efficiency**

Compute-optimal training minimizes loss for a given training budget. But if you deploy a model at scale, inference cost (latency, memory, API calls) dominates. A 7B model costs ~10x less to serve than a 70B model. Deliberately over-training a smaller model — more tokens than compute-optimal — trades training efficiency for deployment efficiency.

This trade-off is explicit and quantifiable: "We accept a 5% worse training loss to get 10x cheaper inference."

**Data is the constraint, not compute**

Scaling laws assume you have enough high-quality tokens. When data is scarce:
- Smaller models that don't overfit are preferable
- Data repetition has diminishing returns (each repeat is worth less than fresh data)
- Data quality matters more than data quantity at this regime

**You need specific emergent capabilities**

Scaling laws predict average perplexity, not the presence of specific capabilities (multi-step reasoning, tool use, instruction following). Some capabilities appear suddenly at specific parameter counts. If you need a capability that only emerges at 70B+, compute-optimal theory is irrelevant — you need the scale.

---

### Decision framework

| Objective | What to optimize |
|-----------|-----------------|
| Best loss for fixed training compute | Follow Chinchilla scaling |
| Cheapest model at target quality | Over-train a smaller model |
| Best performance on specific tasks | Check capability emergence threshold |
| Limited data available | Smaller model, less over-training |

**Always validate with small-scale pilot runs.** Published scaling laws are empirical fits over specific conditions. Your data distribution, architecture, and hardware may produce different curves. Run 3–5 small experiments at 1% of your total budget to verify that your setup follows expected trends before committing.

---

## Q2 — Designing Novel Neural Architectures

**Question:** When building a new architecture for a specific task, what process do you follow from initial hypothesis through experimental validation?

**Short answer:** Architecture design proceeds through four phases — problem analysis (what does the task structurally require?), hypothesis generation, iterative prototyping with ablations, and characterization of when the architecture does and doesn't work.

---

### Phase 1: Problem analysis

Before touching any code, answer:

- **What does this task structurally require?** (Long-range dependencies? Local patterns? Permutation invariance? Multi-scale features?)
- **Why do existing architectures fail here?** (Quadratic attention cost? No temporal structure? Missing relational inductive bias?)
- **What hardware constraints apply?** (Memory budget, target throughput, deployment device)

This shapes the inductive bias your architecture should encode.

---

### Phase 2: Hypothesis generation

Draw from multiple sources to generate candidate designs:

- **Prior work:** What patterns have succeeded on structurally similar tasks?
- **Mathematical constraints:** What operations are theoretically expressive enough?
- **Efficiency requirements:** What operations are hardware-efficient on your target device?

Common structural patterns worth considering:

| Pattern | When to reach for it |
|---------|---------------------|
| Sparse/local attention | Long sequences where full attention is too expensive |
| Mixture of Experts | Conditional computation where different inputs need different processing |
| Hierarchical processing | Multi-resolution tasks (images, documents, time series) |
| Equivariant layers | Physical systems with known symmetries |
| External memory | Tasks requiring explicit fact storage and retrieval |

---

### Phase 3: Iterative prototyping

**Start with the simplest possible instantiation.** A complex architecture with a bug looks like a complex architecture that doesn't work. A simple architecture with a bug is easy to find and fix.

**Ablation sequence:**
1. Baseline: simplest architecture that could plausibly work
2. Add component A — does validation metric improve?
3. Add component B — does it improve further, or interact negatively with A?
4. Remove A, keep B — which component is doing the work?

Every component you add should earn its place through ablation. If removing it doesn't hurt, don't include it.

**Use synthetic data to test mechanistic hypotheses.** If your hypothesis is "this architecture handles long-range dependencies better," create a synthetic task with known long-range dependency structure. Measure directly. Benchmark performance on general tasks mixes in too many confounds.

---

### Phase 4: Validation and characterization

Two questions must be answered before claiming the architecture works:

**1. Does it outperform relevant baselines?**
Use the most recent and appropriate baselines, not ones selected to make the gap look large.

**2. Why does it outperform them?**
- Ablations should isolate which component produces the gains
- Synthetic experiments should verify the stated mechanism
- Out-of-distribution tests should show when the advantage holds and when it collapses

A result without a mechanism is a benchmark artifact. A result with a clear mechanism generalizes.

---

### Common failure modes

- **Overfitting to evaluation:** The architecture learns a quirk of the benchmark distribution
- **Computational impracticality:** Theoretically interesting but too slow to be useful
- **Implementation bugs:** Subtle errors (incorrect masking, wrong normalization order) that invalidate results — always implement a "sanity check" baseline from scratch

---

*Back to [Language Models →](README.md)*
