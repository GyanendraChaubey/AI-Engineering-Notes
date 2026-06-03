# Advanced Search Algorithms

> Retrieval quality is the single biggest lever in a RAG system. Poor retrieval = poor answers, regardless of LLM quality.

---

## 1. What are the main architectural patterns for information retrieval in LLM systems?

```
Pattern 1: Dense Retrieval (Semantic Search)
  Query → Embed → ANN Search → Top-K chunks

Pattern 2: Sparse Retrieval (Keyword Search)
  Query → Tokenize → BM25/TF-IDF → Top-K chunks

Pattern 3: Hybrid Search
  Dense + Sparse → Reciprocal Rank Fusion → Merged Top-K

Pattern 4: Multi-Stage Pipeline
  Fast recall (ANN) → Re-ranking (cross-encoder) → Top-N for LLM

Pattern 5: Agentic Retrieval
  LLM decomposes query → calls search tools iteratively → merges evidence
```

Each pattern adds complexity but improves recall and precision for different query types.

---

## 2. Why is retrieval quality so critical in production RAG systems?

The LLM is bounded by what you give it. If retrieval misses the relevant document:
- The LLM hallucinates or says "I don't know"
- No amount of prompt engineering or model size compensates

**The retrieval bottleneck:** In most production RAG failures, >70% of bad answers trace back to retrieval not finding the right chunk — not to the LLM failing to reason over it.

Analogy: You can hire the best analyst in the world, but if you give them the wrong files, they'll give you the wrong answer.

---

## 3. How do you achieve both speed and accuracy at scale in large-scale semantic search?

**Multi-stage architecture:**

```
Stage 1 — Recall (fast, approximate):
  ANN search with HNSW → retrieve top-100 candidates in <10ms

Stage 2 — Precision (slower, accurate):
  Cross-encoder re-ranker scores all 100 candidates → select top-5 in ~100ms

Stage 3 — Generation:
  LLM generates answer from top-5 chunks
```

Additional techniques:
- **Hardware acceleration:** GPU-based FAISS for sub-millisecond ANN
- **Caching:** Cache embeddings for frequent queries
- **Quantization:** INT8 quantized embeddings reduce memory and speed up SIMD distance computation
- **Sharding:** Distribute the vector index across multiple nodes

---

## 4. A client's RAG system has poor retrieval accuracy. How do you systematically fix it?

**Diagnostic framework:**

```python
# Step 1: Instrument retrieval — log what's being retrieved
for query in test_queries:
    results = retriever.get(query, k=10)
    log({"query": query, "retrieved_chunks": [r.text for r in results]})

# Step 2: Manually annotate — did the right chunk appear in top-10?
# Calculate Recall@10 = % of queries where relevant chunk is in top-10
```

**Fixes by root cause:**

| Root Cause | Fix |
|---|---|
| Query too short / ambiguous | Query expansion, HyDE |
| Wrong embedding model | Benchmark alternatives, fine-tune |
| Poor chunking | Smaller chunks, better boundaries |
| Missing keyword match | Add BM25 hybrid search |
| Relevant doc deeply buried | Increase k, add re-ranking |
| Metadata not leveraged | Add filtered search |
| Multi-hop question | Add query decomposition |

Work through these systematically — don't guess.

---

## 5. How does BM25 keyword-based retrieval work?

BM25 (Best Match 25) scores documents based on term frequency and inverse document frequency, with saturation to prevent spam:

```
BM25(q, d) = Σ IDF(tᵢ) × [tf(tᵢ,d) × (k₁+1)] / [tf(tᵢ,d) + k₁×(1-b+b×|d|/avgdl)]

Where:
  IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5))
  tf = term frequency in document
  |d| = document length, avgdl = avg document length
  k₁ ≈ 1.5 (term frequency saturation), b ≈ 0.75 (length normalization)
```

```python
from rank_bm25 import BM25Okapi
import nltk

# Index
tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# Query
query_tokens = nltk.word_tokenize("refund policy 30 days".lower())
scores = bm25.get_scores(query_tokens)
top_k_indices = scores.argsort()[-10:][::-1]
```

**Strength:** Exact keyword matches, works well for product names, IDs, technical terms.  
**Weakness:** No semantic understanding — "car" and "automobile" are unrelated to BM25.

---

## 6. How do you fine-tune a re-ranking model?

A re-ranker (cross-encoder) takes a (query, document) pair and outputs a single relevance score — much more accurate than bi-encoder similarity but slower.

```python
from sentence_transformers import CrossEncoder

# Load a pre-trained cross-encoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Score query-document pairs
pairs = [(query, doc) for doc in retrieved_docs]
scores = model.predict(pairs)
ranked = sorted(zip(scores, retrieved_docs), reverse=True)
```

**Fine-tuning with domain data:**

```python
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

# Training data: (query, doc, label) where label=1 if relevant, 0 if not
train_samples = [
    InputExample(texts=["What is the return policy?", 
                         "Items may be returned within 30 days."], label=1.0),
    InputExample(texts=["What is the return policy?", 
                         "Our CEO joined in 2020."], label=0.0),
]

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", num_labels=1)
model.fit(
    train_dataloader=DataLoader(train_samples, batch_size=16),
    epochs=3,
    output_path="./fine-tuned-reranker"
)
```

---

## 7. What are the most common information retrieval metrics and when do they fail?

| Metric | Formula | What it measures | Fails when |
|---|---|---|---|
| **Precision@K** | Relevant in top-K / K | Quality of top results | Doesn't reward ranking order |
| **Recall@K** | Relevant found / Total relevant | How many relevant docs found | Doesn't measure quality |
| **MRR** | 1/rank of first relevant | How quickly first relevant appears | Ignores results after first hit |
| **MAP** | Mean avg precision across queries | Overall ranking quality | Slow to compute, complex |
| **NDCG@K** | Normalized Discounted Cumulative Gain | Graded relevance + position | Requires graded labels |

**Most practical for RAG:** Recall@5 (did the right chunk appear in top 5?) + NDCG@5.

---

## 8. For a Q&A system like Quora, which evaluation metric would you choose?

**MRR (Mean Reciprocal Rank)** — because the user wants the first result to be the correct answer. In a Q&A context:

- Users scan from top to bottom
- The first highly relevant answer is what matters most
- Whether there are 5 or 50 good answers further down is less important

```python
def mean_reciprocal_rank(queries_results):
    """
    queries_results: list of lists, where each inner list is [is_relevant_1, is_relevant_2, ...]
    """
    rr_scores = []
    for results in queries_results:
        for rank, is_relevant in enumerate(results, start=1):
            if is_relevant:
                rr_scores.append(1 / rank)
                break
        else:
            rr_scores.append(0)
    return sum(rr_scores) / len(rr_scores)

# Example
results = [[0, 1, 0, 1], [1, 0, 0], [0, 0, 1]]
print(mean_reciprocal_rank(results))  # (1/2 + 1/1 + 1/3) / 3 ≈ 0.61
```

---

## 9. Which metric should you use to evaluate a recommendation system?

**NDCG@K (Normalized Discounted Cumulative Gain)** — because:
- Recommendations have **graded relevance** (loved, liked, clicked, ignored)
- **Position matters** — a great recommendation at rank 1 is worth more than at rank 10
- NDCG captures both aspects

```python
import numpy as np

def ndcg_at_k(relevance_scores, k):
    """
    relevance_scores: list of relevance scores (e.g., [3, 2, 0, 1, 3]) for ranked results
    """
    k = min(k, len(relevance_scores))
    dcg = sum(rel / np.log2(rank + 2) 
              for rank, rel in enumerate(relevance_scores[:k]))
    
    ideal = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum(rel / np.log2(rank + 2) 
               for rank, rel in enumerate(ideal))
    
    return dcg / idcg if idcg > 0 else 0

scores = [3, 2, 0, 1, 3]  # Relevance of recommended items
print(ndcg_at_k(scores, k=5))
```

---

## 10. How does hybrid search combine dense and sparse retrieval?

Hybrid search runs **both** BM25 and dense ANN search, then merges results:

```python
def hybrid_search(query, vectorstore, bm25_index, k=10, alpha=0.5):
    # Dense search
    query_emb = embed(query)
    dense_results = vectorstore.similarity_search_with_score(query_emb, k=k*2)
    
    # Sparse search
    tokens = tokenize(query)
    sparse_scores = bm25_index.get_scores(tokens)
    sparse_results = get_top_k(sparse_scores, k=k*2)
    
    # Merge with Reciprocal Rank Fusion
    return reciprocal_rank_fusion([dense_results, sparse_results], k=k)

def reciprocal_rank_fusion(result_lists, k=60):
    scores = {}
    for results in result_lists:
        for rank, (doc_id, score) in enumerate(results):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**When hybrid wins:**
- Queries with rare proper nouns or IDs → BM25 catches exact match
- Queries with semantic intent → Dense catches meaning
- Hybrid covers both cases

---

## 11. How do you merge rankings from multiple search methods?

**Reciprocal Rank Fusion (RRF)** is the standard approach — robust, no hyperparameter tuning needed:

```
RRF_score(doc) = Σ_list [ 1 / (k + rank_in_list) ]

Where k=60 is a smoothing constant
```

**Alternatives:**
- **Weighted score fusion:** `final = α × dense_score + (1-α) × sparse_score` — requires scores to be on the same scale (normalize first)
- **CombMNZ:** Sum of scores × number of lists the document appears in
- **Borda count:** Points based on rank position

RRF is preferred because it's immune to score scale differences between methods.

---

## 12. How do you handle multi-hop or multi-faceted queries?

A multi-hop query requires combining information from multiple documents:  
*"Who is the CEO of the company that acquired Slack?"*  
→ Hop 1: Which company acquired Slack? (Salesforce)  
→ Hop 2: Who is Salesforce's CEO? (Marc Benioff)

**Approaches:**

1. **Query decomposition** — Use an LLM to break the query into sub-questions

```python
decompose_prompt = """
Break this complex question into simpler sub-questions that can be answered independently.

Question: {question}

Return as a numbered list.
"""

sub_questions = llm(decompose_prompt.format(question=user_query))
# Sub-question 1: Which company acquired Slack?
# Sub-question 2: Who is the CEO of [company]?
```

2. **Iterative retrieval** — Retrieve → read → form follow-up query → retrieve again

3. **Graph-based retrieval** — Build a knowledge graph; traverse edges for multi-hop reasoning

4. **ReAct agent** — LLM decides when to call search and what to search for

---

## 13. What are the main techniques to improve retrieval quality?

| Technique | What it does |
|---|---|
| **HyDE** | Generate hypothetical answer → embed that instead of query |
| **Query expansion** | Add synonyms/related terms using LLM |
| **Query rewriting** | Rephrase query for better embedding alignment |
| **Hybrid search** | Combine dense + sparse |
| **Re-ranking** | Cross-encoder scores retrieved chunks more precisely |
| **Chunk parent retrieval** | Retrieve small chunks but return their larger parent for context |
| **Multi-vector retrieval** | Index multiple representations (summary + full text) |
| **Step-back prompting** | Ask a more abstract question first to retrieve broader context |
| **Self-query retrieval** | LLM generates metadata filters from natural language query |

```python
# HyDE implementation
def hyde_retrieve(query, retriever, llm):
    # Generate hypothetical answer
    hypothetical = llm(f"Write a 2-sentence answer to: {query}")
    # Embed and retrieve based on hypothetical answer
    return retriever.get_relevant_documents(hypothetical)
```

---

*Next: [Language Models Internal Working →](../07-language-models/README.md)*
