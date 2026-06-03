# Miscellaneous LLM Engineering Topics

> Advanced topics: cost optimization, MoE models, low-precision training, KV cache math, and attention internals.

---

## 1. How do you optimize the overall cost of an LLM-based system?

Cost comes from two sources: **inference** (token costs) and **infrastructure** (compute/storage).

### Reduce Token Usage
```python
# 1. Prompt compression — remove redundant context
from llmlingua import PromptCompressor
compressor = PromptCompressor()
compressed = compressor.compress_prompt(long_prompt, rate=0.5)  # 50% fewer tokens

# 2. Truncate retrieved context intelligently
def smart_truncate(chunks, max_tokens=2000):
    """Keep most relevant chunks within token budget."""
    total = 0
    kept = []
    for chunk in sorted(chunks, key=lambda x: x.score, reverse=True):
        chunk_tokens = count_tokens(chunk.text)
        if total + chunk_tokens <= max_tokens:
            kept.append(chunk)
            total += chunk_tokens
    return kept

# 3. Cache frequent responses
import hashlib, redis
cache = redis.Redis()

def cached_llm_call(prompt: str) -> str:
    key = hashlib.sha256(prompt.encode()).hexdigest()
    cached = cache.get(key)
    if cached:
        return cached.decode()
    response = call_llm(prompt)
    cache.setex(key, 3600, response)  # Cache 1 hour
    return response
```

### Use the Right Model for Each Task
```python
# Route to cheaper model when possible
def smart_route(query: str) -> str:
    complexity = classify_complexity(query)
    if complexity == "simple":
        return call_model("gpt-4o-mini", query)     # ~10× cheaper
    else:
        return call_model("gpt-4o", query)          # Only when needed
```

### Optimize Infrastructure
- Use spot/preemptible instances for batch workloads (60–90% cheaper)
- Auto-scale down during off-peak hours
- Self-host for high-volume, predictable workloads (often 5–10× cheaper than API at scale)
- Use quantized models (INT4/INT8) to fit more requests per GPU

### Cost Tracking
```python
# Track per-request cost in production
def tracked_llm_call(prompt, model="gpt-4o"):
    response = call_llm(prompt, model)
    
    # Log costs
    input_tokens  = count_tokens(prompt)
    output_tokens = count_tokens(response)
    cost = calculate_cost(model, input_tokens, output_tokens)
    
    metrics.track("llm_cost", cost, tags={"model": model, "endpoint": endpoint})
    return response
```

---

## 2. What are Mixture of Experts (MoE) models?

MoE is an architecture where instead of one large FFN layer, there are N "expert" FFN networks, and only K of them (typically 2) are activated per token.

```
Standard Transformer FFN:
  token → [Linear(d, 4d) → GELU → Linear(4d, d)]   ← ALL params activated

MoE FFN:
  token → Router → selects 2 of 8 experts
         → Expert 1 (25% tokens) + Expert 2 (25% tokens) + ... 
         → Only 2/8 experts compute per token
```

**Key properties:**

| | Dense Model | MoE Model |
|---|---|---|
| **Total parameters** | 7B | 47B (Mixtral 8×7B) |
| **Active parameters per token** | 7B | ~13B (2 of 8 experts) |
| **Compute per token** | 7B equivalent | ~13B equivalent |
| **Quality** | 7B quality | ~34B quality |

```python
# Simplified MoE layer
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts=8, top_k=2):
        super().__init__()
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([FFN(d_model) for _ in range(num_experts)])
        self.top_k = top_k
    
    def forward(self, x):
        # x: (batch, seq, d_model)
        router_logits = self.router(x)           # (batch, seq, num_experts)
        weights, indices = router_logits.topk(self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = indices[..., k]         # Which expert for each token
            expert_weight = weights[..., k:k+1]  # Weight for this expert
            for i, expert in enumerate(self.experts):
                mask = (expert_idx == i)
                if mask.any():
                    output[mask] += expert_weight[mask] * expert(x[mask])
        return output
```

**Why MoE:** More parameters (more knowledge) at manageable compute cost. Mixtral 8×7B achieves GPT-3.5 quality using only ~2 of 8 experts per token.

**Challenge:** Load balancing — if all tokens route to the same expert, others are wasted. Solved with auxiliary load balancing loss.

---

## 3. How do you build a production-grade RAG system?

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRODUCTION RAG ARCHITECTURE                   │
└─────────────────────────────────────────────────────────────────┘

INGESTION PIPELINE (offline)
  Documents → Parser → Chunker → Embedder → Vector DB
              │                              │
              └── Metadata DB ──────────────┘

QUERY PIPELINE (online, <500ms target)
  User Query
      │
  [Query Classifier] → route to appropriate sub-pipeline
      │
  [Query Rewriter] → normalize, expand, decompose
      │
  [Hybrid Retriever] = Dense ANN + BM25 → Reciprocal Rank Fusion
      │
  [Re-ranker] → cross-encoder scores top-20, returns top-5
      │
  [Context Builder] → assemble prompt with metadata, citations
      │
  [LLM] → grounded response with citations
      │
  [Output Validator] → check faithfulness, detect hallucination
      │
  [Response] → with source references
```

**Component responsibilities:**

```python
class ProductionRAG:
    def __init__(self):
        self.query_classifier = QueryClassifier()    # Is this a retrieval or chitchat?
        self.query_rewriter = QueryRewriter()        # Normalize and expand
        self.dense_retriever = DenseRetriever()      # Embedding-based
        self.sparse_retriever = BM25Retriever()      # Keyword-based
        self.reranker = CrossEncoderReranker()       # Precision layer
        self.llm = OpenAI(model="gpt-4o")
        self.validator = FaithfulnessChecker()
    
    def query(self, user_question: str) -> dict:
        # 1. Classify
        if self.query_classifier.is_chitchat(user_question):
            return {"answer": self.llm.chat(user_question), "sources": []}
        
        # 2. Rewrite
        expanded_query = self.query_rewriter.expand(user_question)
        
        # 3. Retrieve (hybrid)
        dense_results  = self.dense_retriever.get(expanded_query, k=20)
        sparse_results = self.sparse_retriever.get(expanded_query, k=20)
        merged = reciprocal_rank_fusion([dense_results, sparse_results])
        
        # 4. Rerank
        top_chunks = self.reranker.rerank(user_question, merged, top_k=5)
        
        # 5. Generate
        context = build_context(top_chunks)
        answer = self.llm.generate_grounded(user_question, context)
        
        # 6. Validate
        if not self.validator.is_faithful(answer, context):
            answer = "I don't have enough information to answer reliably."
        
        return {"answer": answer, "sources": [c.metadata for c in top_chunks]}
```

---

## 4. What is FP8 and what are its advantages in LLM training?

**FP8** is an 8-bit floating point format introduced by NVIDIA for Hopper (H100) GPUs. It comes in two variants:
- **E4M3** (4 exponent bits, 3 mantissa): Higher precision, smaller dynamic range — used for weights/activations
- **E5M2** (5 exponent bits, 2 mantissa): Larger dynamic range — used for gradients

**Advantages:**

| Precision | Memory per param | Tensor Core TFLOPS (H100) |
|---|---|---|
| FP32 | 4 bytes | 67 TFLOPS |
| BF16 | 2 bytes | 989 TFLOPS |
| FP8 | 1 byte | 1979 TFLOPS |

- **2× memory reduction** vs BF16 → larger batch sizes or models
- **2× compute throughput** vs BF16 on H100
- **Minimal accuracy loss** with proper scaling (similar to INT8 for inference)

```python
# FP8 training with Transformer Engine (NVIDIA)
import transformer_engine.pytorch as te

# Replace standard layers with FP8-capable versions
model = te.Linear(1024, 4096)  # Uses FP8 matmuls automatically

# Or use with HuggingFace + fp8
from transformers import TrainingArguments
args = TrainingArguments(
    fp8=True,              # Enable FP8 training
    bf16=False,
    ...
)
```

---

## 5. How do you train LLMs with low precision without losing accuracy?

Key challenges: underflow (gradients vanish to zero) and overflow (inf/nan).

**Mixed Precision Training (BF16/FP16 + FP32 master weights):**

```python
# Standard approach used by virtually all LLM training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # Only needed for FP16, not BF16

for batch in dataloader:
    with autocast(dtype=torch.bfloat16):  # Forward in BF16
        loss = model(batch)
    
    # Backward in FP32 (via scaler for FP16, or directly for BF16)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # Note: optimizer stores FP32 master weights internally
    # Gradients accumulated in FP32 before applying to FP32 weights
    # Only forward/backward matmuls use low precision
```

**Why BF16 is preferred over FP16 for LLMs:**
- BF16 has the same exponent range as FP32 → no overflow/underflow
- FP16 has limited dynamic range → requires loss scaling tricks
- BF16 is natively supported on A100/H100/TPU

**FP8 training specifics:**
- Requires per-tensor scaling factors to keep values in FP8 range
- Gradient communication still in BF16
- Currently used for pre-training large models at scale (Llama 3, Gemini)

---

## 6. How do you calculate the size of the KV cache?

The KV cache stores key and value tensors for all previous tokens during autoregressive generation.

**Formula:**
```
KV Cache size = 2 × num_layers × num_kv_heads × head_dim × sequence_length × batch_size × bytes_per_element

Where:
  2        = one K matrix + one V matrix
  bytes    = 2 for BF16, 1 for INT8 FP8
```

**Example: Llama 3 8B**
```python
num_layers    = 32
num_kv_heads  = 8      # Uses GQA: 8 KV heads vs 32 query heads
head_dim      = 128    # d_model/num_heads = 4096/32
seq_len       = 8192   # Context length
batch_size    = 1
bytes         = 2      # BF16

kv_cache_bytes = 2 * 32 * 8 * 128 * 8192 * 1 * 2
               = 2 * 32 * 8 * 128 * 8192 * 2
               = 1,073,741,824 bytes ≈ 1 GB

# Scale to batch_size=32:
# = 32 GB — this is why large batch inference requires significant GPU memory
```

**Reducing KV cache:**
- GQA/MQA: share KV across query heads → 4–8× reduction
- KV quantization: INT8/FP8 → 2× reduction
- Sliding window: only cache last N tokens → bounded size
- PagedAttention (vLLM): eliminate memory fragmentation, fit more sequences

---

## 7. What are the dimensions of each component in a multi-head attention block?

For a transformer with `d_model=4096`, `n_heads=32`, `d_head=128` (d_model/n_heads):

```python
# Input tensor
X: (batch, seq_len, d_model)       # e.g., (2, 512, 4096)

# Weight matrices (per attention layer)
W_Q: (d_model, d_model)            # (4096, 4096)
W_K: (d_model, d_model)            # (4096, 4096) or (4096, n_kv_heads*d_head) for GQA
W_V: (d_model, d_model)            # (4096, 4096)
W_O: (d_model, d_model)            # (4096, 4096) output projection

# Per-head tensors (after splitting into heads)
Q_h: (batch, n_heads, seq, d_head) # (2, 32, 512, 128)
K_h: (batch, n_heads, seq, d_head) # (2, 32, 512, 128)
V_h: (batch, n_heads, seq, d_head) # (2, 32, 512, 128)

# Attention scores (the expensive part — quadratic!)
scores: (batch, n_heads, seq, seq)  # (2, 32, 512, 512) — 512² per head!

# Attention output per head
head_out: (batch, n_heads, seq, d_head)  # (2, 32, 512, 128)

# Concatenated and projected
output: (batch, seq, d_model)            # (2, 512, 4096)

# FFN layers
FFN_W1: (d_model, d_ff)            # (4096, 16384) — typically 4× expansion
FFN_W2: (d_ff, d_model)            # (16384, 4096)
```

---

## 8. How do you ensure the attention mechanism focuses on the correct parts of the input?

Attention naturally finds relevant tokens through dot-product similarity. But you can guide and improve focus:

**1. Positional Encoding Quality**
RoPE encodes relative positions directly in Q/K vectors, helping the model attend locally when appropriate and globally when needed.

**2. Attention Sink Phenomenon**
LLMs disproportionately attend to the first few tokens (the "sink" tokens) regardless of relevance. Mitigation: use attention sink tokens explicitly (StreamingLLM).

**3. Temperature in Attention (sharpening)**
Dividing by √d_k in `softmax(QKᵀ/√d_k)` prevents gradients from vanishing when d_k is large. Smaller d_k → sharper attention.

**4. Causal Masking (for decoders)**
```python
def causal_mask(seq_len):
    """Upper triangular mask ensures token i only attends to tokens 0..i"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.masked_fill(mask, float('-inf'))
```

**5. Training with Attention Supervision**
For tasks where you know which parts are relevant, you can add an attention supervision loss during fine-tuning that penalizes attending to irrelevant regions.

**6. Prompt Engineering for Focus**
```python
# Explicitly mark the relevant section
prompt = """
Read the following document carefully. The answer to the user's question
is most likely in the section marked [RELEVANT]:

...preamble...
[RELEVANT]
{key_section}
[/RELEVANT]
...rest of document...

Question: {question}
"""
```

**7. Lost-in-the-Middle Problem**
LLMs perform worse when relevant information is in the middle of a long context — they over-attend to the beginning and end. Fix: place key information at the beginning or end of the context, or use re-ranking to surface the most relevant chunk first.

---

*Next: [Case Studies →](../16-case-studies/README.md)*
