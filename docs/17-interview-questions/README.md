# AI Engineering Interview — Quick Reference Cheat Sheet

**How to use:** Read each question, recall the answer, then check. For full answers + code examples, see `company_wise_interviews.md`. For diagrams + deep dives, see `topic_deep_dive_rag.md`.

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
A: Merges two ranked lists without normalizing scores: `RRF(d) = sum(1 / (60 + rank_i))` — the document ranking highest in both lists wins.

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

## 15. Quick-Fire Definitions

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
| **MRR** | Mean Reciprocal Rank: average of 1/rank_of_first_relevant_result |
