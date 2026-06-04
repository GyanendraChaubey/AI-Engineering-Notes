# AI Engineering Interview — Quick Reference Cheat Sheet

**How to use:** Read each question, recall the answer, then check. For full mathematical derivations, diagrams, and deep dives, see the topic-specific sections: [RAG](../02-rag/README.md), [Embeddings](../04-embeddings/README.md), [Fine-Tuning](../08-fine-tuning/README.md), [Language Models](../07-language-models/README.md), [Deployment](../12-deployment/README.md), and others in the [index](../index.md).

---

## 1. RAG Pipeline

**Q: What is RAG and when is it preferred over fine-tuning?**
A: RAG retrieves relevant documents at query-time and injects them into the LLM context — preferred when the knowledge base changes frequently, is private, or is too large to fine-tune on.

**Q: What are the two pipelines in a RAG system?**
A: Offline ingestion (parse -> chunk -> embed -> index) and online retrieval (embed query -> hybrid search -> rerank -> prompt -> LLM -> guardrail -> answer).

**Q: When would you NOT use RAG?**
A: When data fits entirely in the context window, when the task requires deeply internalized reasoning, or when sub-100ms latency is mandatory and caching is insufficient.

**Q: What is the "lost in the middle" problem?**
A: LLMs pay more attention to the beginning and end of the context window — content in the middle is underutilized, so critical chunks should be placed first or last.

**Q: How does a cross-encoder re-ranker differ from a bi-encoder?**
A: Bi-encoder embeds query and document independently (fast, used for retrieval); cross-encoder takes the pair together for more precise scoring (slower, used for reranking top-20 to top-5).

**Q: How do you handle a RAG system that returns no relevant chunks?**
A: Return a graceful fallback ("I don't have reliable information on this"), expand top-K, try BM25 fallback, or rephrase the query with HyDE before giving up.

**Q: What is HyDE (Hypothetical Document Embeddings)?**
A: Instead of embedding the raw query, generate a hypothetical answer with an LLM and embed that — the hypothetical answer is stylistically closer to real document chunks, improving retrieval.

**Q: How do you scale RAG from 10K to 10M documents?**
A: Shard the vector index across multiple nodes, use HNSW for ANN search, add a semantic cache layer, batch embedding generation, and use async parallel retrieval.

**Q: What is the difference between Traditional RAG and PageIndex?**
A: Traditional RAG chunks documents blindly (no layout awareness) and retrieves text fragments. PageIndex parses document structure first — preserving page boundaries, table integrity, and section hierarchy — and retrieves whole pages or layout blocks as the retrieval unit. PageIndex is strongly preferred for PDFs, financial reports, and research papers. Production systems combine both in Hierarchical RAG: PageIndex retrieves the right pages, then chunk-level refinement extracts the precise passage within those pages.

---

## 2. Chunking

**Q: Why do we chunk documents?**
A: To fit text within the embedding model's token limit and to make retrieval return focused, relevant content with precise semantic embeddings.

**Q: What is recursive character chunking and why is it the default?**
A: Tries splitting at paragraph -> line -> sentence -> word before hard-cutting — preserves semantic boundaries unlike fixed-size which cuts blindly at character N.

**Q: What chunk size and overlap should you start with?**
A: 500 tokens with 50-token overlap (10%) is a common baseline; increase chunk size for long-form reasoning, decrease for precise factual retrieval.

**Q: When should you use semantic chunking?**
A: When retrieval quality matters more than processing speed — groups sentences by embedding similarity, creating semantically coherent chunks at the cost of extra computation.

**Q: How do you handle tables in PDF chunking?**
A: Extract with pdfplumber or Camelot, convert to Markdown, and store each table as a single atomic chunk with its caption — never split a table across two chunks.

**Q: How do you handle images and charts in a RAG pipeline?**
A: Use OCR (Textract/Tesseract) or a multimodal LLM (GPT-4 Vision) to generate text descriptions, embed those descriptions, and store the image URL in chunk metadata.

**Q: What is RAPTOR indexing?**
A: Recursively cluster and summarize groups of chunks with an LLM, index both original leaf chunks and all summary nodes — enables answering both specific and broad questions from one index.

**Q: Should you still chunk documents smaller than the model's context window?**
A: Yes — smaller, focused chunks produce more precise embeddings and reduce noise in the retrieved context, even if the whole document would technically fit.

---

## 3. Embeddings

**Q: What is an embedding?**
A: A fixed-length numeric vector encoding the semantic meaning of text — semantically similar texts produce vectors that are close together in high-dimensional space.

**Q: Why are embedding dimensions multiples of 32 (512, 768, 1536)?**
A: GPU SIMD hardware is optimized for memory-aligned operations in chunks of 32 or 64 — these sizes maximize computational efficiency.

**Q: What are the dimensions of text-embedding-ada-002 and text-embedding-3-large?**
A: ada-002 produces 1536 dimensions; text-embedding-3-large produces 3072 dimensions (higher quality, higher cost).

**Q: Dense vs sparse embeddings — what is the difference?**
A: Dense (neural): low-dimensional, all values non-zero, captures semantic meaning. Sparse (BM25/TF-IDF): high-dimensional (~50K), mostly zeros, captures exact keyword matches.

**Q: Why use cosine similarity instead of Euclidean distance for embeddings?**
A: Cosine is scale-invariant — a short and long document about the same topic have different magnitudes but similar directional angles, making cosine more reliable.

**Q: What is embedding drift?**
A: When the embedding model is updated, the same text produces different vectors — stored vectors become inconsistent with newly generated ones, degrading retrieval quality.

**Q: Is a sentence transformer encoder-based or decoder-based?**
A: Encoder-based (BERT-style) — produces a fixed-length representation of the full input, unlike decoder-only models which generate text token by token.

---

## 4. Vector Databases & Search

**Q: How is a vector database different from a relational database?**
A: Vector DBs store high-dimensional float vectors and support approximate nearest-neighbor (ANN) similarity search; relational DBs store structured rows and support exact SQL queries.

**Q: What is HNSW and what is the trade-off?**
A: Hierarchical Navigable Small World — graph-based ANN with O(log N) search. Trades ~1-5% recall loss for massive speed gains over brute-force KNN. Always used in production.

**Q: When would you use BM25 over vector search?**
A: When querying for exact terms, model numbers, codes, or names — BM25 excels at exact keyword matching where semantic paraphrase is not needed.

**Q: What is Reciprocal Rank Fusion (RRF)?**
A: Merges two ranked lists without normalizing scores: $RRF(d) = \sum \frac{1}{60 + rank_i}$ — the document ranking highest in both lists wins.

**Q: Why use hybrid search (vector + BM25) as the default in production?**
A: Semantic search misses exact terms (e.g., "Error-404"); keyword search misses synonyms — hybrid consistently outperforms either alone.

**Q: What HNSW parameters should you tune?**
A: M (connections per node), ef_construction (candidates at build time), ef_search (candidates at query time — increase for better recall at cost of speed).

---

## 5. RAG Evaluation (RAGAS)

**Q: What are the four core RAGAS metrics?**
A: Context Precision and Context Recall (retrieval quality), Faithfulness and Answer Relevance (generation quality).

**Q: What is Context Precision (RAGAS)?**
A: Signal-to-noise ratio in retrieval — what fraction of retrieved chunks are relevant to the ground truth. Compare: retrieved context vs. ground truth.

**Q: What is Context Recall (RAGAS)?**
A: Coverage of retrieval — what fraction of the ground truth information is present in retrieved chunks. Compare: ground truth vs. retrieved context.

**Q: What is Faithfulness?**
A: What fraction of factual claims in the generated answer are explicitly supported by retrieved context — claims not in context = hallucination.

**Q: What is Answer Relevance?**
A: Whether the generated answer actually addresses the question — measured by generating hypothetical questions from the answer and checking alignment with the original.

**Q: How do you measure answer correctness without RAGAS?**
A: LLM-as-judge: prompt a second LLM with question + ground truth + generated answer, ask it to score 1-5 for correctness — return JSON {"score": N, "reason": "..."}.

**Q: What is the difference between RAGAS context precision and KGPT context utilization?**
A: Context precision (RAGAS) = retrieved context vs. ground truth (was right content retrieved?). Context utilization (KGPT custom) = retrieved context vs. generated answer (how much was used?). Different metrics.

---

## 6. Hallucination

**Q: What are the main root causes of hallucination in RAG?**
A: Wrong chunks retrieved, vague prompts not constraining the LLM, key facts split across chunk boundaries, relevant chunk outside top-K, or context overflow causing "lost in the middle."

**Q: What is the most effective single technique to reduce hallucination?**
A: A strict system prompt: "Answer ONLY based on the provided context. If the answer is not in the context, say I do not know" — prevents the LLM from using parametric knowledge.

**Q: How does citation enforcement reduce hallucination?**
A: Force the LLM to cite the chunk ID for every claim — if it cannot produce a valid citation from the retrieved set, the claim is hallucinated and can be flagged.

**Q: How does LLM-as-a-Judge work for hallucination detection?**
A: After generation, a second LLM call checks whether every factual claim is supported by retrieved context; if faithfulness score < threshold, return a fallback message.

---

## 7. Transformer Architecture & LLMs

**Q: Explain the attention mechanism formula.**
A: Attention(Q,K,V) = softmax(Q*Kt / sqrt(dk)) * V. Q asks "what am I looking for?", K answers "what do I offer?", V contains the information to aggregate. sqrt(dk) prevents vanishing gradients.

**Q: Encoder-only vs decoder-only transformers — what is the difference?**
A: Encoder (BERT): bidirectional attention, every token sees all others — best for understanding. Decoder (GPT): causal attention, each token only sees past tokens — best for generation.

**Q: Does ChatGPT use an encoder and decoder?**
A: No — GPT is decoder-only. There is no separate encoder; "encoding" the user input just means tokenization + embedding lookup through the same decoder stack.

**Q: Why do same words in different order produce different embeddings?**
A: Positional encodings added to token embeddings before attention inject word-order information — "dog bites man" and "man bites dog" produce different contextual representations.

**Q: What is multi-head attention?**
A: Multiple attention heads run in parallel (each with different learned W_Q, W_K, W_V projections) — each head learns different semantic relationships; outputs are concatenated.

**Q: What is the vanishing gradient problem and how do transformers address it?**
A: Gradients shrink exponentially through many layers. Transformers use residual connections (x + Attention(x)) and layer normalization to maintain gradient flow.

**Q: How does an LLM handle out-of-vocabulary (OOV) words?**
A: Via subword tokenization (BPE or SentencePiece) — unknown words are split into known subword pieces; there are no true OOV tokens in modern LLMs.

---

## 8. Fine-tuning: LoRA, QLoRA & SetFit

**Q: What is LoRA?**
A: Low-Rank Adaptation — injects two small trainable matrices A (r x n) and B (m x r) per layer; delta_W = B*A with B initialized to zero. Only A and B are trained; base model is frozen.

**Q: What are the dimensions of LoRA matrices for a weight matrix of size m x n?**
A: A has shape r x n (down-projection), B has shape m x r (up-projection), where r << min(m,n). B is initialized to zero so delta_W = 0 at training start.

**Q: What is QLoRA?**
A: Quantized LoRA — base model in 4-bit NF4 (8x memory reduction), LoRA adapters in BF16; enables fine-tuning 7B+ models on a single consumer GPU.

**Q: What does "4-bit" mean in QLoRA?**
A: Each weight stored in 4 bits (16 possible values) instead of 32 bits — 8x smaller. NF4 (NormalFloat4) is optimized for normally-distributed neural network weights.

**Q: What is SetFit and when should you use it?**
A: Few-shot fine-tuning for sentence transformers using contrastive learning — trains on 8-64 examples per class; ideal for intent classification when labeled data is scarce.

**Q: How do you choose between SetFit, LLM, and XGBoost for intent classification?**
A: SetFit: few examples (<100/class), fast inference. LLM: ambiguous/evolving intents, zero-shot. XGBoost: abundant labeled data, highest throughput, cheapest. Production: SetFit primary + LLM fallback.

**Q: What is the mathematical formula for LoRA's weight update and why does it work?**
A: ΔW = B × A where A ∈ R^(r×n), B ∈ R^(m×r), r << min(m,n). B is initialized to zero so the adapter starts as an identity pass-through. It works because weight updates during fine-tuning have low intrinsic rank — a full m×n update can be well-approximated by a rank-r product.

**Q: Which layers does LoRA typically target in a transformer, and why?**
A: Query (W_q) and Value (W_v) projections in the attention mechanism — empirically these have the most impact on task adaptation. Key and output projections are optionally included. Feed-forward layers are usually skipped to keep adapter size small.

**Q: Give a concrete memory comparison: full fine-tuning vs LoRA for a 4096×4096 weight matrix.**
A: Full fine-tuning: 4096×4096 = ~16.8M parameters. LoRA rank=8: A(8×4096) + B(4096×8) = ~65K parameters — roughly 256× fewer trainable parameters, with proportionally lower optimizer state memory.

---

## 9. Agents, LangGraph & MCP

**Q: What is an AI agent?**
A: An LLM in an observe-reason-act loop that can call tools, observe results, and decide the next step — unlike a single LLM call which is stateless and one-shot.

**Q: What is LangGraph and why use it over a simple chain?**
A: A stateful graph framework where nodes are processing steps and edges are transitions — enables conditional branching, loops, parallelism, and checkpointing that a linear chain cannot express.

**Q: What is "state" in LangGraph and why is it immutable within nodes?**
A: A TypedDict shared across all nodes — nodes return partial update dicts and never mutate state directly, preventing race conditions in parallel branches and making transitions auditable.

**Q: What is checkpointing in LangGraph?**
A: Persisting state after every node to Redis/SQLite/Postgres — enables resuming a failed workflow from the last checkpoint and supports human-in-the-loop review.

**Q: How do you prevent infinite loops in an agent workflow?**
A: Add an iterations counter to state; add a conditional edge that exits when iterations >= MAX_ITERATIONS. Also track visited tool calls and exit on repeated identical calls.

**Q: What is MCP (Model Context Protocol)?**
A: Anthropic's open protocol standardizing how LLMs communicate with external tools and data — the "USB-C for AI integrations" so any MCP client can use any MCP server.

**Q: What are the three components of an MCP server?**
A: Tools (functions the LLM can call with JSON Schema), Resources (data the LLM can read), and Prompts (reusable prompt templates) — exposed via stdio or HTTP/SSE.

---

## 10. MLOps, Drift & Production

**Q: Data drift vs concept drift — what is the difference?**
A: Data drift: input distribution P(X) changes. Concept drift: the relationship P(Y|X) changes — same input should now produce different output due to world changes.

**Q: What PSI value triggers model retraining?**
A: PSI < 0.10 = no action; 0.10-0.25 = monitor; greater than 0.25 = significant drift, retrain the model.

**Q: What is PSI (Population Stability Index)?**
A: PSI = sum((actual% - expected%) * ln(actual% / expected%)) — measures distribution shift between training-time and production-time feature distributions.

**Q: What is the champion-challenger deployment pattern?**
A: Current best model (champion) handles most traffic; new model (challenger) gets a small split (5-10%); metrics are compared and challenger promoted only if it outperforms.

**Q: How do you detect model degradation without ground truth labels?**
A: Monitor proxy metrics — embedding drift, output entropy, user feedback signals, LLM-as-judge scores on sampled production responses.

**Q: When do you retrain on recent data only vs all historical data?**
A: Recent only: concept drift confirmed (old patterns invalid). All historical: data drift only (same patterns, new input distribution).

---

## 11. Classification Metrics

**Q: What are precision and recall, and when does each matter more?**
A: Precision = TP/(TP+FP): minimize false alarms (spam filter). Recall = TP/(TP+FN): minimize missed cases (cancer screening, fraud detection).

**Q: When is accuracy a misleading metric?**
A: On imbalanced datasets — 95% accuracy on a 95/5 fraud dataset can mean zero fraud caught. Use F1, ROC AUC, or PR AUC instead.

**Q: When should you use F1 vs ROC AUC?**
A: F1 for imbalanced classes at a fixed threshold. ROC AUC when ranking quality matters or comparing across thresholds.

**Q: What happens to precision and recall when threshold increases (0.5 to 0.7)?**
A: Precision increases (fewer false positives); recall decreases (more actual positives missed).

**Q: What is bias vs variance in ML?**
A: Bias = systematic error (underfitting — model too simple). Variance = sensitivity to training data (overfitting — model too complex). Ideal: minimize both.

---

## 12. Guardrails, Security & PII

**Q: What are input vs output guardrails?**
A: Input: PII masking, toxicity filter, prompt injection detection, off-topic router. Output: faithfulness check, PII scan on response, harmful content filter, citation validator.

**Q: What is prompt injection and how do you mitigate it?**
A: Attacker embeds override instructions in user input. Mitigate with input sanitization, XML-tagged user input (separating user content from instructions), and output validation.

**Q: How do you ensure multi-tenant data isolation in a vector DB?**
A: Tag every chunk with tenant_id at ingestion; inject it as a mandatory metadata filter on every vector search query; validate from JWT token in middleware — undeniable.

---

## 13. Caching, Latency & Scaling

**Q: What are the two types of caching in RAG?**
A: Exact-match cache (SHA256 hash of query -> cached response) and semantic cache (embed query -> find past queries with cosine > 0.95 -> return cached response).

**Q: Which component is the first bottleneck as a RAG system scales?**
A: LLM API — every query requires at least one 1-5s call while all other components complete in under 200ms. Semantic caching (40-60% hit rate) is the most impactful fix.

**Q: What is the bottleneck order from 100 to 100K users?**
A: LLM API rate limits -> embedding service throughput -> vector DB query concurrency -> application tier (scales cheaply with Kubernetes HPA).

**Q: How do you achieve 1-2 second RAG response times?**
A: Semantic cache eliminates 40-60% of LLM calls; async parallel retrieval; streaming shows first token in ~300ms; route simple queries to smaller/cheaper models.

---

## 14. Python Fundamentals

**Q: What is a decorator in Python?**
A: A higher-order function that wraps another function to add behavior (logging, timing, auth, retry) without modifying the original code — applied with @decorator_name syntax.

**Q: What is a generator and when do you use it?**
A: A function using `yield` that produces values one at a time without loading all into memory — use for large sequences, streaming data, or lazy evaluation.

**Q: What is the GIL and when does it matter?**
A: CPython's Global Interpreter Lock allows only one thread to execute at a time — irrelevant for I/O-bound work (GIL released during waits) but prevents CPU parallelism (use multiprocessing instead).

**Q: What is super().__init__() and why use it?**
A: super() follows Python's MRO (Method Resolution Order) — correctly handles multiple inheritance. Hardcoding the parent class name breaks cooperative multiple inheritance.

**Q: What is a race condition in Python threading?**
A: counter += 1 (read-modify-write) can be interrupted between threads — GIL can release between LOAD and STORE bytecodes causing lost increments. Fix with threading.Lock().

**Q: Solve Top-K Frequent Elements (LeetCode) — optimal approach?**
A: Bucket sort O(n): Counter(nums) -> buckets indexed by frequency -> iterate buckets high-to-low, collect until k elements found. Min-heap approach is O(n log k) and also acceptable.

---

## 15. ML Coding — NumPy & Image Processing

**Q: Implement pixel-level Mixup: given two RGB pixels `[R,G,B]` and a weight `alpha`, return the blended pixel.**
A:
```python
def mixup_pixel(pixel1, pixel2, alpha):
    return [alpha * p1 + (1 - alpha) * p2 for p1, p2 in zip(pixel1, pixel2)]
```
Formula: `mixed = alpha * pixel1 + (1 - alpha) * pixel2`. Time: O(3) = O(1). Space: O(1).

**Q: Implement Mixup for a full H×W×3 image using nested loops (no NumPy).**
A:
```python
def mixup_images(image1, image2, alpha):
    H, W = len(image1), len(image1[0])
    return [
        [
            [alpha * image1[i][j][c] + (1 - alpha) * image2[i][j][c] for c in range(3)]
            for j in range(W)
        ]
        for i in range(H)
    ]
```
Time: O(H × W × 3). Space: O(H × W × 3).

**Q: Implement vectorized image Mixup using NumPy — no Python loops.**
A:
```python
import numpy as np

def mixup_images(image1, image2, alpha):
    img1 = np.asarray(image1, dtype=np.float32)
    img2 = np.asarray(image2, dtype=np.float32)
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same shape")
    return alpha * img1 + (1 - alpha) * img2
```
NumPy broadcasts the scalar `alpha` over the entire array in one C-level operation — same O(H×W×3) complexity but ~100× faster in practice due to SIMD hardware.

**Q: What is wrong with this code for blending two uint8 images equally? `mixup = image1 + image2; mixup //= 2`**
A: Integer overflow. `np.uint8` wraps at 255: `200 + 200 = 400 % 256 = 144`, so `144 // 2 = 72` instead of the correct `200`. Fix: cast to float32 before arithmetic, then clip and cast back.
```python
def combine_images_evenly(image1, image2):
    img1 = image1.astype(np.float32)
    img2 = image2.astype(np.float32)
    return ((img1 + img2) / 2).astype(np.uint8)
```

**Q: How are labels combined in Mixup data augmentation?**
A: Soft label blending: `y_mixed = alpha * y1 + (1 - alpha) * y2`. For one-hot vectors: cat=[1,0,0], dog=[0,1,0] with alpha=0.7 gives y_mixed=[0.7, 0.3, 0]. The model trains with cross-entropy against this soft target, not a hard label.

**Q: How is alpha sampled in Mixup training, and what distribution is used?**
A: `alpha ~ Beta(λ, λ)` where λ is a hyperparameter (commonly 0.2 or 1.0). The Beta distribution naturally produces values in [0,1] with a U-shape at low λ (mostly near 0 or 1) or uniform at λ=1 — giving a mix of nearly-pure and truly-blended samples each batch.

---

## 16. Quick-Fire Definitions

| Term | One-line definition |
|:---|:---|
| **BM25** | Keyword search scoring TF-IDF + document length normalization |
| **RRF** | Merges two ranked lists: score = sum(1 / (60 + rank)) |
| **HNSW** | Graph-based ANN index; O(log N) search with ~1-5% recall loss |
| **PSI** | Measures feature distribution shift; > 0.25 = retrain needed |
| **LoRA** | Fine-tuning via low-rank matrices; A(r x n) * B(m x r) = delta_W |
| **QLoRA** | LoRA + 4-bit NF4 quantization of base model; fits 7B+ on 1 GPU |
| **SetFit** | Few-shot sentence transformer fine-tuning via contrastive learning |
| **RAGAS** | RAG eval: Context Precision/Recall + Faithfulness + Answer Relevance |
| **MCP** | Standardized LLM-to-tool protocol (Anthropic open standard) |
| **LangGraph** | Stateful directed graph for agentic workflows with checkpointing |
| **ReAct** | Agent pattern: Reason + Act interleaved (observe-think-act loop) |
| **RAPTOR** | Recursive chunk summarization tree; indexes raw and summary nodes |
| **HyDE** | Generate hypothetical answer -> embed it -> use for retrieval |
| **Semantic cache** | Cache by query embedding similarity (cosine > 0.95) not exact match |
| **ColPali** | Multimodal model that embeds page images directly, skipping OCR |
| **LlamaGuard** | Meta open-source safety model; classifies 14 harm categories |
| **tiktoken** | OpenAI tokenizer for counting tokens before API calls |
| **GELU** | Activation in GPT/BERT; smoother gradient flow than ReLU |
| **Embedding drift** | Embeddings shift after model update; re-index to fix retrieval |
| **Champion-Challenger** | A/B deployment: current model gets majority traffic, new model gets small split for comparison |
| **Context Precision** | RAGAS: fraction of retrieved chunks relevant to ground truth |
| **Context Recall** | RAGAS: fraction of ground truth info covered by retrieved chunks |
| **Faithfulness** | RAGAS: fraction of answer claims supported by retrieved context |
| **NF4** | 4-bit NormalFloat quantization; optimized for neural network weight distributions |
| **Mixup** | Data augmentation: blend two images and labels — `x = α·x1 + (1-α)·x2`, `y = α·y1 + (1-α)·y2`, α ~ Beta(λ,λ) |
| **uint8 overflow** | Adding two uint8 arrays wraps at 255; always cast to float32 before arithmetic then clip back |
| **MRR** | Mean Reciprocal Rank: average of 1/rank_of_first_relevant_result |
| **Mode collapse** | GAN failure where generator produces limited variety of outputs; discriminator can't tell them apart |
| **FID** | Fréchet Inception Distance — measures quality + diversity of generated images; lower = better |
| **Wasserstein distance** | Earth mover's distance between distributions; more stable than JS divergence for GAN training |
| **Dice loss** | Segmentation loss = 1 - 2TP/(2TP+FP+FN); better than CE for small/imbalanced foreground regions |
| **Classifier-free guidance** | Diffusion technique: train with/without condition; inference blends both for controllable generation |
| **Knowledge distillation** | Train small student model to mimic large teacher's soft outputs (logits), not just hard labels |
| **GroupNorm** | Normalizes within channel groups; batch-size independent — preferred for 3D medical imaging |
| **Depthwise separable conv** | Factorizes standard conv into per-channel spatial + 1×1 cross-channel; ~8-9x fewer FLOPs |
| **FlashAttention** | IO-aware exact attention: tiles QKᵀ in SRAM, never materializes n×n matrix; O(n²) FLOPs, O(n·d) memory |
| **Online softmax** | Running max m + running sum ℓ enables exact softmax one block at a time without storing the full row |
| **Hierarchical RAG** | Two-stage: PageIndex coarse retrieval → chunk-level fine-grained extraction within retrieved pages |
| **Sliding window generation** | Process long context in chunks, maintain running summary; trades precision for latency |
| **TTFT** | Time-To-First-Token — determined by prefill phase; scales O(n²) with context length |

---

## 17. Deep Learning Fundamentals

**Q: Explain backpropagation mathematically.**
A: We minimize loss L(θ) using the chain rule. For layer l: ∂L/∂W^(l) = (∂L/∂a^(l)) · (∂a^(l)/∂z^(l)) · (∂z^(l)/∂W^(l)), where z^(l) = W^(l)·a^(l-1) + b and a^(l) = σ(z^(l)). Gradients are propagated backward layer by layer. Weights updated via: W := W - η · ∂L/∂W.

**Q: What causes vanishing and exploding gradients?**
A: Vanishing: repeated multiplication of small derivatives (<1) during backprop causes gradients to shrink toward zero — common with sigmoid/tanh in deep networks. Exploding: gradients grow exponentially when derivatives >1. Solutions: ReLU activations, residual connections, gradient clipping, He/Xavier initialization.

**Q: How does residual learning work?**
A: Instead of learning H(x) directly, the network learns the residual F(x) = H(x) - x. The output is y = F(x) + x. The skip connection creates a direct gradient highway from output to early layers, preventing vanishing gradients in very deep networks (ResNet).

**Q: BatchNorm vs LayerNorm — when to use each?**
A: BatchNorm normalizes across the batch dimension (requires large batches, used in CNNs). LayerNorm normalizes across the feature dimension (batch-size independent, used in Transformers and RNNs). BatchNorm is unstable with small batches; LayerNorm is stable regardless of batch size.

**Q: When would you use GroupNorm instead of BatchNorm?**
A: When batch size is very small (e.g., 3D medical imaging with batch size 1-2), when BatchNorm is numerically unstable, or when training on distributed systems with micro-batches. GroupNorm normalizes within channel groups, making it fully independent of batch size.

---

## 18. Computer Vision & Medical Imaging

**Q: Explain the U-Net architecture.**
A: U-Net consists of a contracting encoder (downsampling with conv + pooling), a bottleneck, and an expanding decoder (upsampling with transposed conv). Skip connections concatenate encoder feature maps to corresponding decoder layers, preserving spatial resolution lost during downsampling. Standard for biomedical segmentation.

**Q: Why are skip connections critical in U-Net?**
A: They pass high-resolution spatial features from encoder to decoder, which would otherwise be lost during pooling. This enables precise pixel-wise localization — the decoder knows both "what" (from bottleneck semantics) and "where" (from skip-connected spatial detail).

**Q: What is the difference between 2D and 3D CNNs?**
A: 2D CNN kernels are H×W — process one image slice at a time, losing inter-slice context. 3D CNN kernels are D×H×W — process volumetric data (CT/MRI) with full spatial context across depth. 3D is anatomically superior for medical imaging but requires ~D× more memory and compute.

**Q: How do you optimize segmentation for imbalanced classes?**
A: Use Dice loss or Focal loss instead of standard cross-entropy (which is dominated by the background class). Other strategies: weighted cross-entropy, oversampling patches containing foreground, patch-based training, and data augmentation targeting minority class regions.

**Q: What is the difference between Dice score and IoU?**
A: Dice = 2TP / (2TP + FP + FN). IoU = TP / (TP + FP + FN). Dice is the harmonic mean of precision and recall; IoU is the Jaccard index. Dice is more sensitive to small objects (penalizes FP/FN less harshly relative to TP) and is preferred for medical segmentation.

**Q: What is EfficientNet compound scaling?**
A: EfficientNet scales depth (d^α), width (w^β), and resolution (r^γ) simultaneously under a fixed compute budget. Rather than scaling one dimension alone (which gives diminishing returns), compound scaling balances all three, consistently achieving higher accuracy with fewer parameters.

**Q: What is depthwise separable convolution?**
A: Factorizes a standard convolution into: (1) depthwise conv — applies one filter per input channel (captures spatial patterns), (2) pointwise 1×1 conv — mixes channels. Standard conv: k²·M·N ops. Depthwise separable: k²·M + M·N ops. ~8-9× cheaper, used in MobileNet.

**Q: How do you reduce model size without losing accuracy?**
A: Pruning (remove low-weight connections), quantization (reduce precision FP32→INT8/INT4), knowledge distillation (train smaller student from large teacher), low-rank approximation (factorize weight matrices), and architecture replacement (EfficientNet, MobileNet).

---

## 19. GANs & Diffusion Models

**Q: Explain GAN training instability.**
A: GANs are a min-max game: G minimizes, D maximizes. This creates a non-convex optimization with no guaranteed convergence. Vanishing gradients occur when D becomes too strong (near-zero gradient to G). Mode collapse occurs when G finds a few outputs that fool D and stops exploring. These make training sensitive to hyperparameters and architecture choices.

**Q: What is mode collapse in GANs?**
A: Generator learns to produce a limited subset of outputs (e.g., always generating the same face) because they reliably fool the discriminator. The generator stops exploring the full data distribution. Fixes: minibatch discrimination (penalize similar batches), WGAN (stabler gradients), feature matching (match intermediate activations, not just final outputs).

**Q: How does WGAN fix GAN training?**
A: WGAN replaces JS divergence with Wasserstein-1 (Earth Mover's) distance. JS divergence saturates when distributions don't overlap — zero gradient to generator. Wasserstein distance provides meaningful, smooth gradients even when distributions are far apart. The discriminator (now called critic) is not bounded to [0,1], enabling cleaner loss signals.

**Q: What is the Wasserstein distance intuitively?**
A: The minimum "work" required to transform one probability distribution into another — the cost of moving probability mass. Formally: W(Pr, Pg) = inf_{γ∈Π} E_{(x,y)~γ}[||x-y||]. Unlike KL/JS divergence, it is well-defined and continuous even when distributions have non-overlapping support.

**Q: What is gradient penalty in WGAN-GP?**
A: WGAN requires the critic to be 1-Lipschitz (||∇D(x)|| ≤ 1 everywhere). Weight clipping (original WGAN) is too aggressive. WGAN-GP adds a soft penalty: λ·(||∇D(x̂)||₂ - 1)², where x̂ is a random interpolation between real and fake samples. This enforces the Lipschitz constraint without clipping.

**Q: How do you evaluate GAN quality?**
A: FID (Fréchet Inception Distance): compares feature statistics (mean + covariance) between real and generated images in Inception v3 feature space — lower is better. IS (Inception Score): measures quality (sharp predictions) + diversity (varied predictions). Domain-specific: expert validation, clinical metrics for medical imaging.

**Q: How do you condition a GAN on text?**
A: (1) Concatenate text embedding (from CLIP/BERT) to the latent noise vector before generator input. (2) Conditional BatchNorm — affine transform parameters (γ, β) predicted from text embedding. (3) Cross-attention layers (like in diffusion) — text tokens attend to spatial feature maps at each resolution.

**Q: How do you generate medical synthetic data safely?**
A: Apply differential privacy during training to prevent patient data memorization. Validate generated samples clinically before use. Check that generated distribution doesn't reconstruct any training example (membership inference defense). Use distributional similarity tests (FID, expert review) to confirm realism. Never use synthetic data for diagnosis without proper validation.

**Q: How does diffusion work mathematically?**
A: Forward process: q(x_t | x_{t-1}) = N(√(1-β_t)·x_{t-1}, β_t·I) — iteratively adds Gaussian noise over T steps until x_T ~ N(0,I). Reverse process: a neural network pθ(x_{t-1}|x_t) learns to denoise step by step, reconstructing x_0 from pure noise.

**Q: How is noise predicted in diffusion models?**
A: The UNet-based model ε_θ(x_t, t) is trained to predict the noise ε that was added at step t. Loss: L = E[||ε - ε_θ(√ᾱ_t·x_0 + √(1-ᾱ_t)·ε, t)||²]. At inference, predicted noise is subtracted iteratively using a scheduler (DDPM, DDIM, DPM++) to denoise.

**Q: Why are diffusion models more stable than GANs?**
A: No adversarial game — training is a simple supervised regression (predict the noise). No mode collapse because the model learns the full data distribution via likelihood-based training. No discriminator collapse or vanishing gradients. The trade-off: diffusion models are slower at inference (100-1000 steps vs. one GAN forward pass).

**Q: What is classifier-free guidance (CFG)?**
A: Train the same model both with condition c and without (randomly drop condition during training, replacing with null token ∅). At inference, blend: ε = (1+w)·ε_θ(x_t, c) - w·ε_θ(x_t, ∅), where w is the guidance scale. Higher w → stronger adherence to condition but less diversity. Eliminates need for a separate classifier.

**Q: How do you fine-tune a diffusion model?**
A: DreamBooth: fine-tune all weights on 3-5 images of a specific subject with a rare token, using prior preservation loss to prevent forgetting. LoRA: inject low-rank adapters into attention and cross-attention layers, training only those. ControlNet: freeze base model, add trainable copy of encoder conditioned on structural signal (pose, depth, edge). Textual inversion: only optimize a new text embedding token.

---

## 20. Healthcare NLP & Medical AI

**Q: How do you perform Named Entity Recognition (NER) in medical text?**
A: Use domain-pretrained models: BioBERT (trained on PubMed abstracts) or ClinicalBERT (trained on MIMIC clinical notes). Fine-tune as a token classification head (BIO tagging scheme). Add a CRF layer on top to enforce label transition constraints (e.g., I-DRUG cannot follow B-DISEASE). Datasets: i2b2, n2c2, BC5CDR.

**Q: How do you handle domain adaptation for medical NLP?**
A: (1) Continued pretraining on domain corpus (PubMed, clinical notes) before task fine-tuning. (2) Use domain-specific models (BioBERT, SciBERT, PubMedBERT) instead of general BERT. (3) Instruction-tune an LLM on medical Q&A pairs (MedQA, PubMedQA). (4) Data mixing: combine domain data with general data to prevent catastrophic forgetting.

**Q: How do you handle class imbalance in medical NLP (e.g., rare disease classification)?**
A: Focal loss (down-weights easy negatives, focuses on hard cases). Oversampling rare classes (SMOTE or synonym augmentation). Hierarchical classification (coarse → fine). Few-shot learning with prompt engineering for very rare labels. Weighted cross-entropy with inverse class frequency weights.

**Q: How do you ensure factual accuracy in healthcare content generation?**
A: RAG with a curated, peer-reviewed medical corpus (UpToDate, PubMed, clinical guidelines). Require citation of source document for every factual claim. Confidence scoring — flag low-confidence outputs for human review. Multi-step fact-checking pipeline: generate → verify each claim against retrieved evidence → rewrite if contradicted.

**Q: How do you prevent hallucinations in a medical LLM?**
A: Strict RAG grounding: "Answer only from the provided clinical guidelines, cite the source." Citation enforcement: parse citations and verify each maps to a real retrieved chunk. Abstention training (SFT on negative examples): teach the model to say "I don't have sufficient information" rather than guess. Human-in-the-loop review for high-stakes outputs.

**Q: How would you architect a medical content generation system?**
A: Ingestion layer: parse clinical guidelines, PubMed articles, drug databases → chunk → embed → store in vector DB with metadata (source, date, specialty). Query layer: hybrid search (dense + BM25) → cross-encoder rerank → top-K chunks. Generation layer: LLM with strict grounding prompt + citations. Guardrail layer: fact-check generated claims, PII scan, toxicity filter, medical safety classifier. Audit layer: log all generations with retrieved context for regulatory traceability.

---

## 21. Prompting & Context Engineering

**Q: How do you design a stable, reliable system prompt?**
A: Structure the prompt in five sections: (1) **Role** — precise persona with domain and expertise level. (2) **Instructions** — what to do, written as positive commands not negations. (3) **Constraints** — hard rules (output format, length, language). (4) **Output format** — explicit schema (JSON, markdown) with field names. (5) **Examples** — 2-3 few-shot examples anchoring the target behavior. Stable prompts minimize output entropy: the LLM should almost always produce the same structure regardless of input variation. Version-control prompts as code and A/B test changes systematically.

**Q: How do you manage memory in multi-turn conversations?**
A: Four strategies with increasing fidelity: (1) **Full history** — append every turn to context. Simple, loses nothing, but context grows unbounded and TTFT degrades quadratically. (2) **Sliding window** — keep only the last N turns. Cheap but loses early context (breaks personalization). (3) **Summary memory** — after every K turns, LLM summarizes the conversation so far; prepend summary to new turns. (4) **Vector memory store** — embed each turn, retrieve the top-k most relevant past turns at query time. Production systems combine: summary for long-term continuity + vector retrieval for semantic relevance.

**Q: What prompt engineering strategies reduce hallucination?**
A: (1) **Retrieval grounding** — always include a source context and constrain to it: "Answer ONLY from the context below. If it is not there, say I don't know." (2) **Chain-of-thought suppression in final output** — use CoT internally but strip reasoning from user-facing response to prevent reasoning errors propagating. (3) **Self-consistency voting** — generate N answers at temperature > 0, return the majority consensus. (4) **Confidence-gated output** — add "Only answer if you are highly confident. Otherwise say I am not certain." (5) **Explicit citation requirement** — "Every factual claim must be followed by [Source: chunk_id]." Uncitable claims = hallucinated claims.

**Q: How do you manage and version system prompts in production?**
A: Treat prompts as first-class code artifacts: store in a YAML/JSON file in version control alongside the model version it was tested with. Each prompt version should record: hash, model name, p50/p95 latency, task success rate, and human eval score. Use feature flags to A/B test prompt versions in production with traffic splitting. Never deploy a new prompt without first running it against your golden evaluation set. Track prompt lineage in your experiment tracker (MLflow, W&B) alongside model checkpoints.

---

## 22. Real-World Debugging & LLMOps

**Q: How would you debug incorrect or unexpected LLM outputs?**
A: Systematic five-step process: (1) **Check retrieval** — log and inspect retrieved chunks; is the answer present? (2) **Check prompt rendering** — log the full rendered prompt; truncation or template bugs often hide here. (3) **Run at temperature=0** — eliminate stochasticity to make the bug reproducible. (4) **Isolate components** — test retrieval alone (correct chunks?), then generation alone (correct answer given perfect context?). (5) **Compare model versions** — if a recent deployment broke outputs, roll back model/prompt independently to isolate which change caused regression.

**Q: How do you safely migrate when your embedding model changes?**
A: Changing the embedding model is a breaking change — all stored vectors become incompatible. Safe migration strategy: (1) **Dual-index** — run both old and new models in parallel; new writes go to both indexes. (2) **Background re-embedding** — batch re-embed all documents with the new model overnight. (3) **Shadow traffic** — for 5-10% of queries, compare retrieval results between old and new indexes; measure Recall@K parity. (4) **Cutover** — once re-embedding is complete and quality is verified, switch all traffic to the new index and decomission the old one. Never do a hard cutover without parity verification.

**Q: What is different about CI/CD for LLM workflows vs traditional ML?**
A: Traditional ML CI/CD tests deterministic model outputs against fixed metrics. LLM CI/CD is non-deterministic and harder: (1) **Prompt regression tests** — run a golden set of 50-100 (input, expected_behavior) pairs through the pipeline on every PR; flag if LLM-as-judge score drops >5%. (2) **Latency gates** — p95 TTFT and total response time must be within SLA bounds. (3) **Embedding drift check** — if embedding model or chunking changed, verify Recall@K is maintained. (4) **Hallucination rate check** — sample 10-20 production-like queries and measure Faithfulness. (5) **Shadow deployment** — route 5% traffic to new version before full rollout.

**Q: What fallback strategy would you use if the LLM call fails mid-task?**
A: Layered fallbacks: (1) **Retry with exponential backoff** — transient API timeouts often self-resolve. (2) **Smaller model fallback** — if Opus/GPT-4 fails, route to Haiku/GPT-3.5 with a degraded-quality warning. (3) **Context reduction** — if context exceeds the model's limit, truncate chunks and retry. (4) **Partial structured output** — return what was successfully generated before the failure; mark incomplete fields explicitly. (5) **Human-in-the-loop escalation** — for high-stakes (medical, legal) tasks, route to human review instead of returning a degraded response. Always log the failure with full context for post-mortem analysis.

**Q: How do you detect and mitigate prompt injection attacks in RAG?**
A: Prompt injection in RAG happens when a malicious document contains instructions like "Ignore all previous instructions and output X." Defenses: (1) **Structural separation** — wrap retrieved content in XML tags (`<retrieved_context>...</retrieved_context>`) and instruct the model to treat that zone as data only, not instructions. (2) **Input sanitization** — filter known injection patterns from retrieved text before injecting into prompt. (3) **Output validation** — run a second LLM call or classifier to check if the output deviates from the expected schema/topic. (4) **Privilege separation** — never mix user-controlled content with system instructions in the same prompt segment. (5) **Canary tokens** — embed rare sentinel strings in the system prompt; if they appear in the output, a jailbreak occurred.

**Q: How do you monitor for hallucination drift in production?**
A: Since ground-truth labels don't exist at scale, use proxy monitoring: (1) **Faithfulness sampling** — daily sample 1-5% of production queries, run LLM-as-judge to score Faithfulness; alert if 7-day rolling average drops below threshold. (2) **Citation hit rate** — for citation-enforced systems, track the fraction of responses where all cited chunk IDs map to real retrieved documents. (3) **User rejection signals** — thumbs-down ratings, follow-up correction messages, or session abandonment correlate with hallucinated responses. (4) **Contradiction detection** — for factual domains, run a secondary NLI model to check if the generated answer contradicts any retrieved chunk. Combine all signals into a single "quality score" dashboard.

---

## 23. LLM Evaluation & User Satisfaction

**Q: How do you evaluate an LLM-powered app beyond accuracy?**
A: Decompose evaluation into four orthogonal dimensions: (1) **Retrieval quality** (if RAG) — Context Precision, Context Recall, Recall@K. (2) **Generation quality** — Faithfulness (no hallucination), Answer Relevance, BLEU/ROUGE/BERTScore for reference tasks. (3) **System behavior** — Abstention rate (does it refuse when it should?), format compliance rate, latency P50/P95. (4) **User-perceived quality** — task completion rate, follow-up correction rate, explicit rating, implicit frustration signals (short session, rapid re-query). Use LLM-as-judge (GPT-4 scoring 1-5 on each dimension) to scale evaluation across thousands of samples without human annotation.

**Q: What metrics do you use to measure user satisfaction in production?**
A: Direct metrics: explicit thumbs-up/down rating, CSAT score, NPS (would you use this feature again?). Implicit behavioral signals: task completion rate (did the user act on the answer?), follow-up correction rate (did the user immediately rephrase or correct the LLM?), session abandonment rate (did the user leave without a satisfying answer?), time-to-next-query (a very fast follow-up often signals the previous answer was unhelpful). Log all LLM outputs and user actions together so you can retroactively correlate output quality with behavior signals.

**Q: How do you decompose and measure RAG quality across retrieval and generation separately?**
A: Use RAGAS or a custom evaluation harness with separate metrics per stage. **Retrieval stage**: Context Precision (what fraction of retrieved chunks were relevant to the ground-truth answer?) and Context Recall (what fraction of the ground-truth evidence was present in retrieved chunks?). **Generation stage**: Faithfulness (what fraction of generated claims are supported by retrieved context?) and Answer Relevance (does the answer actually address the question?). To debug: if Faithfulness is low but Context Precision is high → the LLM is hallucinating despite good retrieval → fix the prompt or use SFT for abstention. If Context Recall is low → retrieval is missing relevant documents → fix chunking, embedding, or expand top-K.

---

*Continue to: [Company Wise Interviews →](company_wise_interviews.md) · [RAG Deep Dive →](topic_deep_dive_rag.md)*
