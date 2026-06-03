# Vector Databases — Internal Working

> Understanding how vector databases index and search billions of embeddings efficiently.

---

## 1. What is a vector database?

A vector database is a storage system purpose-built for high-dimensional vector embeddings. It supports:
- **Storing** embedding vectors alongside metadata
- **Nearest neighbor search** — given a query vector, find the K most similar stored vectors
- **Filtered search** — combine vector similarity with metadata conditions
- **CRUD** operations on vectors

Popular options: **Pinecone**, **Weaviate**, **Qdrant**, **Chroma**, **Milvus**, **pgvector** (Postgres extension).

---

## 2. How does a vector database differ from a traditional relational database?

| Aspect | Relational DB | Vector DB |
|---|---|---|
| **Data model** | Rows and columns (structured) | High-dim float vectors + metadata |
| **Query type** | Exact match, range, JOIN | Approximate nearest neighbor (ANN) |
| **Index type** | B-tree, hash index | HNSW, IVF, PQ |
| **Scale** | Millions of rows easily | Billions of vectors with ANN |
| **Use case** | Transactions, reporting | Semantic search, RAG, recommendation |
| **Consistency** | ACID | Eventual / configurable |

A relational DB can't do "find the 5 most similar documents to this embedding" efficiently — that would require computing distance to every row. Vector DBs solve this with approximate indexing.

---

## 3. How does a vector database work internally?

```
Write path:
  Vector + Metadata → Normalization → Index structure update → Persistent storage

Read path:
  Query vector → ANN search on index → Top-K candidate IDs → 
  Optional re-ranking → Fetch metadata → Return results
```

The key component is the **vector index** — a data structure that enables sub-linear search time. Without an index, finding the nearest neighbor requires computing distance to every stored vector (O(n) — infeasible at scale).

---

## 4. What is the difference between a vector index, a vector database, and a vector plugin?

| Term | What it is |
|---|---|
| **Vector index** | Just the indexing algorithm (HNSW, IVF, etc.) — a library like FAISS |
| **Vector database** | Full system: index + storage + metadata + API + CRUD + filtering |
| **Vector plugin** | An extension added to an existing DB (e.g., pgvector for PostgreSQL) |

```
Vector Index (FAISS):
  - In-memory only
  - No persistence, no metadata, no filtering
  - Just: index.add(vectors), index.search(query, k)

Vector Database (Qdrant):
  - Persistent storage
  - Metadata + payload filtering
  - REST/gRPC API
  - Distributed / replicated
  - Built-in HNSW index
```

Use FAISS for prototyping and research; use a proper vector DB for production.

---

## 5. Small dataset, accuracy is top priority, speed is not. What search strategy would you use?

**Exact k-NN (brute-force search).**

With a small dataset (< 100K vectors), the overhead of an approximate index isn't worth it. Compute exact cosine similarity between the query and every stored vector.

```python
import faiss
import numpy as np

d = 384  # embedding dimension
index = faiss.IndexFlatL2(d)  # Exact L2 search — no approximation

# Add all vectors
index.add(corpus_embeddings.astype("float32"))

# Query
query_emb = model.encode(["What is the return policy?"])
distances, indices = index.search(query_emb.astype("float32"), k=5)
```

`IndexFlatL2` and `IndexFlatIP` are brute-force — 100% recall, but O(n) per query. At < 1M vectors this is fast enough on CPU.

---

## 6. How do clustering-based search and Locality-Sensitive Hashing work?

### Clustering (IVF — Inverted File Index)

1. **Training:** K-means cluster all vectors into N clusters (e.g., N=1024). Store cluster centroids.
2. **Indexing:** Assign each vector to its nearest centroid. Build an inverted list: centroid_id → [vector_ids].
3. **Search:** 
   - Embed query → find top `nprobe` nearest centroids
   - Search only those clusters (not all vectors)
   - Return top K from searched clusters

```python
import faiss

d, n_clusters = 384, 256
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, n_clusters)

index.train(all_vectors)   # Learn cluster centroids
index.add(all_vectors)     # Assign to clusters

index.nprobe = 16          # Search 16 nearest clusters (more = higher recall, slower)
D, I = index.search(query, k=10)
```

**Fails when:** Query falls near a cluster boundary — the true nearest neighbors may be in adjacent clusters not searched. Fix: Increase `nprobe`.

### Locality-Sensitive Hashing (LSH)

Uses hash functions where **similar vectors tend to hash to the same bucket**.

- Hash each vector → store in hash table
- At query time: hash query → look up same bucket → only compare those vectors
- Very fast for initial filtering, but recall < IVF in practice

---

## 7. How does clustering reduce search space and when does it fail?

**Reduction mechanism:** Instead of comparing to all N vectors, only compare to vectors in the `nprobe` nearest clusters. If N=1M, n_clusters=1000, nprobe=10 → compare to ~10,000 vectors (100× speedup).

**Failure cases:**
1. **Boundary queries** — Query sits between two clusters; relevant vectors are in the unchosen cluster
2. **Uneven clusters** — Some clusters have 100K vectors (slow to search), others have 10 (fast but often missed)
3. **High dimensionality** — Clustering becomes less effective as dimensions increase (curse of dimensionality)

**Mitigations:**
- Increase `nprobe` (slower but higher recall)
- Use HNSW instead (graph-based, handles boundaries better)
- Use Product Quantization combined with IVF (IVF-PQ) for better compression

---

## 8. What is a Random Projection index?

Random Projection (RP) reduces dimensionality while approximately preserving pairwise distances (Johnson-Lindenstrauss lemma).

```
Original: 1536-dim vector
↓ Multiply by random matrix R (1536 × 128)
Projected: 128-dim vector

Vectors that were similar in 1536-D are still similar in 128-D (approximately)
```

Used for: fast approximate search, dimensionality reduction before another index (IVF or HNSW).

```python
import numpy as np

def random_projection(vectors, target_dim):
    d = vectors.shape[1]
    R = np.random.randn(d, target_dim) / np.sqrt(target_dim)
    return vectors @ R

projected = random_projection(embeddings, target_dim=128)
```

**Tradeoff:** Faster search, less memory, but some recall loss.

---

## 9. What is Locality-Sensitive Hashing (LSH) and how does it index vectors?

LSH uses a family of hash functions where **collision probability ∝ similarity**.

For cosine similarity (SimHash):
1. Generate M random hyperplanes through the origin
2. For each vector, record which side of each hyperplane it falls on (binary: 0 or 1)
3. This creates an M-bit hash. Similar vectors → same hash bucket

```python
import numpy as np

def simhash(vectors, n_hyperplanes=64):
    d = vectors.shape[1]
    hyperplanes = np.random.randn(n_hyperplanes, d)
    projections = vectors @ hyperplanes.T   # (n, 64)
    return (projections > 0).astype(int)    # Binary hash: (n, 64)

hashes = simhash(embeddings)
# Vectors with same binary hash → candidate neighbors
```

**Limitation:** High false-positive rate (different vectors end up in the same bucket by chance). Often used as a first-pass filter followed by exact scoring.

---

## 10. What is Product Quantization (PQ)?

PQ dramatically compresses vector storage while enabling fast approximate distance computation.

1. **Split** each d-dim vector into M sub-vectors of d/M dimensions
2. **Cluster** each sub-space independently into K centroids (codebook)
3. **Encode** each sub-vector by its nearest centroid ID (1 byte if K=256)
4. **Storage:** A 1536-dim float32 vector (6KB) → 8-byte PQ code (750× compression)

```
Original:  [0.1, 0.3, ..., 0.8]  (1536 floats = 6144 bytes)
PQ-encoded: [42, 17, 203, 88, ...]  (8 bytes: 8 sub-spaces × 1 byte each)
```

Distance computation uses precomputed lookup tables — much faster than comparing raw floats.

```python
import faiss

# IVF + PQ (production-grade)
d, n_clusters, M, bits = 384, 256, 16, 8
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, n_clusters, M, bits)
index.train(training_vectors)
index.add(all_vectors)
```

---

## 11. Comparing vector index types — which one to choose for your use case?

| Index Type | Speed | Recall | Memory | When to Use |
|---|---|---|---|---|
| **Flat (exact)** | Slow | 100% | High | Small dataset (<500K), accuracy-critical |
| **IVF** | Fast | ~90% | Medium | Large dataset, some recall loss OK |
| **IVF-PQ** | Very fast | ~85% | Very low | Billion-scale, memory-constrained |
| **HNSW** | Fast | ~97% | High | Best recall/speed tradeoff in production |
| **LSH** | Very fast | ~80% | Low | Ultra-low latency, moderate accuracy |

**Production default: HNSW** — best recall for the speed budget.

```python
# Qdrant with HNSW
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient("localhost", port=6333)
client.create_collection(
    collection_name="knowledge_base",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE,
        hnsw_config={"m": 16, "ef_construct": 200}  # Higher = better recall, slower build
    )
)
```

---

## 12. How do you choose the right similarity metric for your use case?

| Metric | Formula | Best For |
|---|---|---|
| **Cosine similarity** | dot(a,b)/(‖a‖·‖b‖) | Most NLP/text tasks; magnitude-invariant |
| **Dot product** | dot(a,b) | When vectors are L2-normalized (= cosine then) |
| **Euclidean (L2)** | ‖a-b‖ | When magnitude matters; image embeddings |
| **Manhattan (L1)** | Σ\|aᵢ-bᵢ\| | Sparse vectors |

**For text embeddings:** Always use **cosine similarity** or normalize vectors and use dot product (equivalent, but faster).

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Equivalent (faster) with normalized vectors
a_norm = a / np.linalg.norm(a)
b_norm = b / np.linalg.norm(b)
sim = np.dot(a_norm, b_norm)
```

---

## 13. What types of filtering exist in vector databases and what are the challenges?

### Filter Types

1. **Pre-filter:** Apply metadata filter first → reduce candidate set → run ANN search on filtered set
2. **Post-filter:** Run ANN search → get top results → apply metadata filter on results
3. **In-filter (filtered HNSW):** Integrate filter into graph traversal — Qdrant and Weaviate do this

### Challenges

- **Pre-filter + small result set:** After filtering, the candidate set may be too small for ANN to find true neighbors → recall drops
- **Post-filter + high filter rate:** ANN returns K results, filter removes most → fewer than K valid results returned
- **Filter selectivity:** High-selectivity filters (rare attributes) break most approximate indexes

**Solution:** Use databases that support **filtered HNSW** (Qdrant, Weaviate) — they maintain the graph structure over the filtered subset at query time.

```python
# Qdrant filtered search
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="docs",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[FieldCondition(key="department", match=MatchValue(value="legal"))]
    ),
    limit=10
)
```

---

## 14. How do you decide which vector database is best for your project?

| Criterion | Recommendation |
|---|---|
| **Prototype / local dev** | Chroma (zero setup) or FAISS |
| **Production, fully managed** | Pinecone or Weaviate Cloud |
| **Self-hosted, best performance** | Qdrant (Rust, very fast, filtered search) |
| **Already on Postgres** | pgvector (no new infra) |
| **Billion-scale** | Milvus |
| **Multimodal (text + image)** | Weaviate |

**Key questions to ask:**
1. What scale? (10K docs vs 100M docs)
2. On-prem or managed cloud?
3. Need filtered search? (most use cases do)
4. Need real-time updates or batch only?
5. Existing infrastructure constraints?

---

*Next: [Advanced Search Algorithms →](../06-search-algorithms/README.md)*
