# LLM Deployment & Inference Optimization

> Production LLMs must be fast, cost-efficient, and reliable. Here's how to get there.

---

## 1. Why doesn't quantization significantly degrade LLM accuracy?

Quantization reduces the numerical precision of model weights (e.g., FP32 → INT8 → INT4). Intuitively, you'd expect significant accuracy loss — but in practice, LLMs are remarkably robust to this.

**Why LLMs tolerate quantization:**

1. **Redundancy in large models:** LLMs have billions of parameters with significant redundancy. Many weights contribute minimally to outputs — losing precision on them has negligible effect.

2. **Smooth weight distributions:** LLM weight distributions are roughly Gaussian and well-behaved, making them amenable to uniform quantization.

3. **Outlier handling:** Modern quantization methods (GPTQ, AWQ) identify and protect the small percentage of "outlier" weights that are critical to accuracy.

4. **Compensating activations:** When weights are quantized, the model's learned representations can partially compensate through the remaining full-precision computations.

**Quantization types and accuracy impact:**

| Precision | Memory (7B model) | Speed | Accuracy vs FP16 |
|---|---|---|---|
| FP32 | 28 GB | 1× | Baseline |
| BF16 | 14 GB | ~2× | ~identical |
| INT8 | 7 GB | ~2–3× | <0.5% degradation |
| INT4 (NF4) | 3.5 GB | ~3–4× | 1–3% degradation |
| INT2 | 1.75 GB | ~5× | Significant degradation |

```python
# GPTQ quantization (post-training, calibration-based)
from transformers import AutoModelForCausalLM, GPTQConfig

gptq_config = GPTQConfig(
    bits=4,
    dataset="c4",         # Calibration dataset
    tokenizer=tokenizer
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=gptq_config
)
# Weights quantized to 4-bit, activations remain float16
```

---

## 2. What techniques improve LLM inference throughput?

### 1. Continuous Batching (Dynamic Batching)
Traditional: wait for all requests in a batch to finish before starting new ones.  
Continuous: insert new requests as soon as a sequence finishes — GPUs stay fully utilized.

```
Static batching:    [Req1████░░░░░][Req2████████] → wait for both
Continuous batching: [Req1████] → immediately start [Req3] in that slot
```
**Result:** 2–4× throughput improvement. Used by vLLM, TGI by default.

### 2. PagedAttention (vLLM)
KV cache management inspired by OS virtual memory paging. Eliminates memory fragmentation, enabling larger batches.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3-8B-Instruct")
sampling_params = SamplingParams(temperature=0.8, max_tokens=256)

outputs = llm.generate(["Hello, tell me about AI."] * 100, sampling_params)
# Handles 100 concurrent requests efficiently via PagedAttention
```

### 3. Speculative Decoding
Use a small "draft" model to generate candidate tokens, then verify in parallel with the large model — net speedup of 2–3×.

```
Small model generates:  ["The", "cat", "sat", "on", "the"] (5 tokens in 1 step)
Large model verifies:   ✓ ✓ ✓ ✗ → accept 3 tokens, reject from position 4
Net: 3 tokens in the time it takes to run the large model once
```

### 4. Grouped Query Attention (GQA) / Multi-Query Attention (MQA)
Reduces KV cache size by sharing keys/values across attention heads:

```
Multi-Head Attention:   Q₁K₁V₁, Q₂K₂V₂, ..., Q₃₂K₃₂V₃₂  (32 KV heads)
Grouped Query Attention: Q₁..Q₄→K₁V₁, Q₅..Q₈→K₂V₂, ...  (8 KV heads)
Multi-Query Attention:  All Qₙ → K₁V₁                     (1 KV head)
```

GQA reduces KV cache by 4–8× with minimal quality loss — used in Llama 3, Mistral.

### 5. Flash Attention
Rewrites attention computation to minimize GPU HBM reads/writes using tiling — 3–8× speedup for attention layers.

### 6. Tensor Parallelism / Pipeline Parallelism
Split model across multiple GPUs for large models that don't fit on one GPU.

---

## 3. How do you speed up inference without architectural changes like GQA?

When you can't change the model architecture, these techniques apply:

### KV Cache Quantization
```python
# Quantize KV cache to INT8 (reduces KV memory by 2×)
# vLLM supports this natively
llm = LLM(model="...", kv_cache_dtype="fp8")
```

### Prefix Caching
If many requests share a common system prompt, cache the KV states for that prefix — skip recomputing it for every request.

```python
# vLLM automatic prefix caching
llm = LLM(model="...", enable_prefix_caching=True)

# All requests with the same system prompt reuse cached KV
requests = [
    "[SYSTEM] You are a helpful assistant. [USER] " + question
    for question in user_questions
]
# System prompt KV computed once, reused for all requests
```

### Prompt Compression
Compress long contexts to fewer tokens before sending to the LLM:

```python
from llmlingua import PromptCompressor

compressor = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank")
compressed = compressor.compress_prompt(
    long_context,
    rate=0.5  # Compress to 50% of original length
)
```

### Response Streaming
Don't wait for full generation — stream tokens to the client. Improves perceived latency (time-to-first-token) even if total time is the same.

```python
import openai

stream = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Explain attention mechanisms"}],
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Batched Offline Inference
For non-interactive workloads, batch all requests and process with maximum GPU utilization:

```python
from vllm import LLM

llm = LLM(model="...", max_num_seqs=256)  # Process 256 sequences simultaneously
outputs = llm.generate(all_prompts)  # Fully batched
```

---

*Next: [Agent-Based Systems →](../13-agents/README.md)*
