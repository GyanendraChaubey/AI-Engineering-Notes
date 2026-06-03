# Embedding Models

> Embeddings are the bridge between raw text and mathematical similarity — the foundation of semantic search and RAG.

---

## 1. What are vector embeddings and what does an embedding model do?

A **vector embedding** is a dense numerical representation of text (or images, audio, etc.) in a high-dimensional space, where semantically similar content is geometrically close.

An **embedding model** is a neural network that maps text → vector. It's trained so that:
- "dog" and "puppy" → nearby vectors
- "dog" and "quantum physics" → distant vectors

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "The cat sat on the mat.",
    "A feline rested on the rug.",   # semantically similar
    "The economy grew by 3% last quarter."  # unrelated
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 384)

# Cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(embeddings)
# sim[0][1] ≈ 0.85 (similar)
# sim[0][2] ≈ 0.12 (dissimilar)
```

---

## 2. How are embedding models used in LLM applications?

**Primary use cases:**

| Use Case | How Embeddings Are Used |
|---|---|
| **Semantic search** | Query → embed → find nearest document chunks |
| **RAG retrieval** | Embed chunks at index time; embed query at runtime |
| **Duplicate detection** | High cosine similarity between two texts → potential duplicate |
| **Clustering** | Group similar documents without labels |
| **Recommendation** | Find items similar to what a user has interacted with |
| **Classification** | Embed text → feed to a lightweight classifier head |

In RAG specifically:
1. At **index time**: each document chunk is embedded and stored
2. At **query time**: user query is embedded with the *same* model
3. **Distance** between query embedding and chunk embeddings determines relevance

---

## 3. What is the distinction between embedding short vs. long text?

| Aspect | Short Text | Long Text |
|---|---|---|
| **Token limit** | Fits easily (< 512 tokens) | May exceed model max (512–8192 tokens) |
| **Embedding quality** | High — model focuses on a clear semantic signal | Can degrade — model must compress many concepts |
| **Retrieval precision** | High | Lower — embedding averages over too much |
| **Strategy** | Direct embedding | Chunk first, embed chunks separately |

**Long-document embedding options:**
- **Chunking** (recommended): Split into pieces, embed each
- **Hierarchical embedding**: Embed paragraphs + sections + document; query at the right level
- **Late interaction models** (ColBERT): Embed each token separately, compute fine-grained similarity at query time

```python
# ColBERT-style: token-level embeddings
from colbert import Searcher
# Each document token gets its own embedding vector
# At query time: MaxSim aggregation over all token pairs
# Much more precise but 10x the storage
```

---

## 4. How do you benchmark embedding models on your own data?

Don't just rely on public leaderboards (MTEB) — they may not reflect your domain. Build a **domain-specific evaluation set**:

```python
# Step 1: Create a golden dataset
eval_set = [
    {
        "query": "What is our cancellation policy?",
        "relevant_doc_ids": ["doc_023", "doc_045"]  # ground truth
    },
    # ... 100-200 query/relevant pairs
]

# Step 2: Evaluate each candidate model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def evaluate_model(model_name, eval_set, corpus_embeddings, corpus_ids):
    model = SentenceTransformer(model_name)
    
    hits_at_5 = []
    for item in eval_set:
        query_emb = model.encode([item["query"]])
        scores = cosine_similarity(query_emb, corpus_embeddings)[0]
        top5_ids = [corpus_ids[i] for i in np.argsort(scores)[-5:][::-1]]
        
        hit = any(doc_id in top5_ids for doc_id in item["relevant_doc_ids"])
        hits_at_5.append(hit)
    
    return np.mean(hits_at_5)  # Recall@5

# Compare models
for model_name in ["all-MiniLM-L6-v2", "text-embedding-3-small", "bge-large-en-v1.5"]:
    score = evaluate_model(model_name, eval_set, corpus_embs, corpus_ids)
    print(f"{model_name}: Recall@5 = {score:.3f}")
```

**Key metrics:** Recall@K, NDCG@K, Mean Reciprocal Rank (MRR).

---

## 5. You're using an OpenAI embedding model and getting low retrieval accuracy. How do you improve it?

**Diagnosis → Fix:**

1. **Query-document mismatch** — Queries are short questions; documents are long answers. Fix: Use asymmetric embedding (different prompts for query vs. document).

```python
# OpenAI supports task-specific embeddings
query_embedding = embed(f"Represent this query for searching: {query}")
doc_embedding = embed(f"Represent this document for retrieval: {doc}")
```

2. **Domain gap** — Model wasn't trained on your vocabulary. Fix: Add a domain-specific fine-tuning layer (see Q6 below).

3. **Poor chunking** — Chunks are too large/small or split mid-sentence. Fix: Improve chunking strategy.

4. **Not enough results retrieved** — Retrieval k is too low. Fix: Increase k and use re-ranking.

5. **Metadata not used** — Relevant docs filtered out before embedding comparison. Fix: Combine keyword filter + semantic search.

6. **HyDE (Hypothetical Document Embedding)** — Instead of embedding the raw query, ask an LLM to generate a hypothetical answer, then embed that:

```python
# HyDE approach
hypothetical_answer = llm.generate(f"Answer this question in 1 paragraph: {query}")
query_embedding = embed(hypothetical_answer)  # More similar to real answer chunks
```

---

## 6. Walk through the steps for fine-tuning a sentence transformer for better embeddings.

**When to fine-tune:** Public models don't capture domain-specific terminology (medical, legal, code, internal jargon).

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Step 1: Prepare training pairs
# Format: (anchor, positive, [negative])
train_examples = [
    InputExample(texts=["What is our return policy?", 
                         "Items can be returned within 30 days with receipt."]),
    InputExample(texts=["How do I reset my password?", 
                         "Visit account settings and click 'Forgot Password'."]),
    # ... hundreds/thousands of pairs
]

# Step 2: Load base model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Step 3: Define loss
# MultipleNegativesRankingLoss: each positive pair, other batch items = negatives
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Step 4: Train
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="./fine-tuned-embedder"
)

# Step 5: Evaluate on held-out set before deploying
model = SentenceTransformer("./fine-tuned-embedder")
```

**Data generation tip:** Use an LLM to generate synthetic (query, relevant passage) pairs from your documents — this is called **GPL (Generative Pseudo Labeling)** and works very well with limited labeled data.

```python
# Generate synthetic training data
prompt = """
Given this document passage, generate 3 realistic search queries a user might ask 
that this passage would answer:

Passage: {passage}

Return as a JSON list of query strings.
"""
```

---

*Next: [Vector Databases →](../05-vector-databases/README.md)*
