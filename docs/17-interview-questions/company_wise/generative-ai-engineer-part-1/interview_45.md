# Generative AI Engineer (Part 1) — Interview 45

**Q: Why do we need to do chunking in document processing for RAG pipelines?**

- Chunking is essential in RAG pipelines to break down large documents into smaller, semantically meaningful sections that can be efficiently processed, indexed, and retrieved.
- Most LLMs and embedding models have input size limitations (token or character limits), so chunking ensures each piece of content fits within these constraints.
- By splitting documents at logical boundaries (like headers or table structures), we preserve context and semantic relationships, which improves the relevance and accuracy of retrieval during question answering.
- Chunking also enables more granular search and retrieval, allowing the system to fetch only the most relevant sections instead of entire documents, which enhances both performance and user experience.
- In our pipeline, we use intelligent chunking strategies—splitting first at high-level headers (H1/H2), then recursively at lower-level headers (H3-H5), and consolidating small chunks—to maintain both structure and context, as described in our Knowledge GPT architecture documentation.

---

**Q: Should we still chunk documents if they are much smaller than the model’s context window?**

- Yes, even if a document is smaller than the model’s context window, chunking is still recommended in production RAG pipelines for several reasons:
 - **Semantic Granularity:** Chunking at logical boundaries (like headers or sections) ensures each chunk represents a coherent, self-contained idea, which improves retrieval relevance and answer quality.
 - **Efficient Retrieval:** Smaller, well-defined chunks allow the retrieval system to return only the most relevant sections, rather than entire documents, reducing noise and improving precision.
 - **Scalability:** As document sizes and corpus grow, consistent chunking strategies make the system robust to future changes and larger inputs.
 - **Metadata & Analytics:** Chunk-level metadata (such as section titles, headers, or creation timestamps) can be attached, enabling advanced filtering, analytics, and traceability.
 - **Pipeline Consistency:** Uniform chunking logic simplifies downstream processing, embedding, and indexing, regardless of document size.
- In our Knowledge GPT pipeline, we always apply chunking rules: if a document is under 25 KB, it is kept as a single chunk; otherwise, we split at H1/H2 headers, then recursively at lower headers, and consolidate small chunks. This ensures both efficiency and semantic integrity, even for smaller documents.

---

**Q: What do you know about the text-embedding-ada-002 embedding model?**

- The text-embedding-ada-002 is a state-of-the-art embedding model provided by OpenAI, widely used for semantic search, clustering, classification, and information retrieval tasks in production RAG pipelines.
- It generates a 1536-dimensional dense vector representation for any input text, capturing deep semantic relationships and contextual meaning.
- The model is highly efficient, supporting both short and long text inputs, and is optimized for low latency and high throughput, making it suitable for large-scale enterprise applications.
- In our Knowledge GPT pipeline, we use text-embedding-ada-002 to generate embeddings for document chunks and user queries. These embeddings are then indexed in OpenSearch to enable fast and accurate semantic retrieval.
- The model supports multilingual inputs, which is valuable for global enterprise use cases.
- According to OpenAI’s documentation and our internal benchmarks, text-embedding-ada-002 offers a strong balance of accuracy, speed, and cost-effectiveness compared to previous embedding models.
- In our architecture, we call the Azure OpenAI Embeddings API with chunk summaries to generate these 1536-dimensional vectors, which are then used for KNN vector search and hybrid retrieval strategies.

---

**Q: What is the relationship between chunking and the embedding model used in a RAG pipeline?**

- There is a direct relationship between chunking strategy and the embedding model in a RAG pipeline, as both impact retrieval quality and system performance.
- **Context Window Limit:** The embedding model (like text-embedding-ada-002) has a maximum input token limit. Chunking must ensure that each chunk fits within this limit, otherwise the embedding API will fail or truncate input, leading to loss of information.
- **Semantic Representation:** The quality of embeddings depends on the semantic coherence of each chunk. If chunking splits content arbitrarily (e.g., mid-sentence or mid-table), the resulting embeddings may not accurately capture the intended meaning, reducing retrieval relevance.
- **Chunk Size Optimization:** Chunk size should be chosen to maximize semantic completeness while staying within the embedding model’s input constraints. For example, in our Knowledge GPT pipeline, we:
 - Split documents at logical boundaries (headers, sections, tables).
 - Ensure each chunk is under 25 KB (well below the ~8K token limit of ada-002).
 - Summarize large chunks before embedding to further reduce size and improve focus.
- **Embedding Cost & Performance:** Smaller, well-chunked inputs reduce embedding costs and speed up processing, while also improving retrieval granularity.
- **Retrieval Precision:** Proper chunking aligned with embedding model limits ensures that each vector in the database represents a meaningful, retrievable unit of knowledge, directly impacting the accuracy and usefulness of RAG responses.

- In summary, chunking and embedding model selection must be coordinated to ensure each chunk is semantically meaningful, fits within model limits, and produces high-quality embeddings for effective retrieval.

---


**Q: How do you handle cases where fixed-size chunking splits a paragraph or semantic unit, potentially ruining the output?**

- Splitting a paragraph or semantic unit in the middle using fixed-size chunking can indeed degrade the quality of embeddings and retrieval, as it breaks the context and meaning.
- To address this, we avoid pure fixed-size chunking and instead use **semantic-aware, header-based chunking** strategies:
 - **Header-Based Splitting:** We first split documents at logical boundaries such as H1/H2 headers (using tools like LangChain’s MarkdownHeaderTextSplitter), ensuring each chunk aligns with a complete section or topic.
 - **Recursive Splitting:** If a chunk is still too large (>25 KB), we recursively split further at lower-level headers (H3, H4, H5), always respecting semantic boundaries.
 - **Chunk Consolidation:** If splitting results in very small chunks, we merge adjacent small chunks (using consolidate_chunks()) up to 25 KB, but only if they don’t cross higher-level headers, preserving semantic integrity.
 - **No Character-Level Splitting:** We never split within a paragraph, sentence, or table, and avoid breaking semantic units. If a section is still too large after all header-based splits, we keep it as-is rather than splitting arbitrarily.
- This approach ensures that each chunk is semantically meaningful, fits within the embedding model’s input size, and maintains high retrieval and answer quality.
- These strategies are documented and automated in our Knowledge GPT pipeline, as detailed in our architecture documentation, to ensure robust and context-preserving chunking for all document types.

---

**Q: Which search technique would you use for retrieving a specific, unseen model number (like "A1234567") in a RAG system, and why?**

- For queries involving specific model numbers or exact product codes—especially those that are non-English, alphanumeric, or previously unseen—the **lexical search (BM25)** technique is the most effective.
 - **Reason:** Lexical search matches exact keywords or phrases in the indexed document fields (like "chunk" and "title"). Model numbers are typically unique identifiers and do not have semantic meaning, so semantic/vector search may not retrieve them accurately.
 - **BM25 multi_match:** This approach excels at finding exact matches for product names, codes, or error numbers, regardless of language or context.
- In our Knowledge GPT pipeline, we default to BM25 lexical search for such cases, as documented in our architecture:
 - The BM25 query is run on both "chunk" and "title" fields to maximize the chance of matching the model number wherever it appears.
 - This ensures high precision and recall for queries involving unique identifiers.
- **Summary:** For queries like "What is A1234567?", I would use lexical (BM25) search because it is designed for exact keyword matching, which is ideal for retrieving documents containing specific, non-semantic identifiers such as model numbers.

---

**Q: What happens if you use only vector-based (semantic) search for queries like model numbers or unique identifiers?**

- Using only vector-based (semantic) search for queries involving model numbers, product codes, or unique identifiers is generally ineffective.
 - **Reason:** Semantic search relies on the conceptual similarity between the query and document content, using embeddings to capture meaning. However, model numbers and codes (e.g., "A1234567") do not have inherent semantic meaning—they are arbitrary strings.
 - **Embedding Limitations:** Embedding models like OpenAI’s ada-002 are trained on natural language and may not represent unique codes or identifiers accurately in vector space. As a result, the vector for "A1234567" may not be close to the vectors of relevant documents containing that code.
 - **Retrieval Failure:** This can lead to poor recall—relevant documents with the exact model number may not be retrieved, or irrelevant documents with similar-looking strings may be returned.
- **Industry Practice:** For such cases, lexical search (BM25) is always preferred, as it matches exact tokens and is language/context agnostic.
- **Summary:** Relying solely on semantic search for unique identifiers will likely result in missed matches and degraded retrieval quality. That’s why our Knowledge GPT pipeline always includes a lexical search component for such queries, as documented in our architecture.

---

**Q: What search algorithms are commonly used inside vector databases for similarity search?**

- Vector databases use specialized algorithms for efficient similarity search (nearest neighbor search) in high-dimensional spaces. The most common algorithms include:
 - **HNSW (Hierarchical Navigable Small World):**
 - Widely used in modern vector databases like OpenSearch, Milvus, and Pinecone.
 - Builds a multi-layer graph structure for fast approximate nearest neighbor (ANN) search.
 - Offers sub-linear query time and high recall, making it suitable for large-scale, real-time applications.
 - **IVF (Inverted File Index):**
 - Used in FAISS and other vector DBs.
 - Partitions the vector space into clusters (using k-means), then searches only within relevant clusters.
 - Good for batch processing and large datasets.
 - **Annoy (Approximate Nearest Neighbors Oh Yeah):**
 - Used in Spotify and some open-source projects.
 - Builds multiple random projection trees for fast ANN search.
 - Optimized for read-heavy workloads.
 - **Flat (Brute Force):**
 - Linear scan over all vectors.
 - Guarantees exact results but is slow for large datasets; used for small collections or as a baseline.
- In our Knowledge GPT pipeline (using OpenSearch), **HNSW** is the default algorithm for KNN vector search due to its balance of speed, scalability, and accuracy.
- These algorithms are configurable in most vector DBs, allowing tuning based on dataset size, latency, and recall requirements.

---

**Q: Do you know how HNSW (Hierarchical Navigable Small World) works for vector search?**

- Yes, I’m familiar with HNSW (Hierarchical Navigable Small World), which is a state-of-the-art algorithm for approximate nearest neighbor (ANN) search in high-dimensional vector spaces, widely used in vector databases like OpenSearch, Milvus, and Pinecone.
- **How HNSW Works:**
 - **Graph-Based Structure:** HNSW builds a multi-layer, navigable small-world graph where each node represents a vector, and edges connect nodes to their nearest neighbors.
 - **Hierarchical Layers:** The graph is organized into multiple layers. The top layers are sparse (fewer nodes, long-range links), while the bottom layer is dense (all nodes, short-range links).
 - **Insertion:** When a new vector is added, it is inserted at a randomly chosen top layer and then connected to its nearest neighbors at each layer as it descends. This maintains both local and global connectivity.
 - **Search Process:**
 - The search starts at the topmost layer with a random entry point.
 - At each layer, the algorithm greedily moves to the neighbor closest to the query vector.
 - Once it reaches the bottom layer, it performs a more exhaustive search among the closest nodes to find the approximate nearest neighbors.
 - **Efficiency:** This hierarchical, multi-layer approach allows HNSW to quickly traverse the graph, skipping large portions of the dataset and achieving sub-linear search times with high recall.
- **Industry Use:** HNSW is the default for KNN search in many production vector DBs due to its balance of speed, scalability, and accuracy, and we use it in our Knowledge GPT pipeline for efficient semantic retrieval.

---

**Q: How do you evaluate the retrieval quality in a RAG system?**

- In our Knowledge GPT RAG system, we use a comprehensive evaluation pipeline to measure retrieval quality, focusing on both quantitative metrics and qualitative analysis:
 - **Mean Reciprocal Rank (MRR):**
 - For each query, we check if the expected (ground truth) document appears in the retrieved results and at what rank.
 - Reciprocal rank is calculated as 1/rank (if found), or 0 if not found.
 - MRR is the average reciprocal rank across all queries, providing a robust measure of how high relevant documents appear in the results.
 - **Precision@1:**
 - Measures the fraction of queries where the top-ranked result is the correct (ground truth) document.
 - Indicates how often the system retrieves the most relevant document at the first position.
 - **Found/Not Found/No Document Breakdown:**
 - Tracks how many queries successfully retrieve the ground truth, how many do not, and how many return empty results.
 - **Answer Correctness & Relevance:**
 - Uses LLM-based evaluation (e.g., GPT-4o) to judge if the retrieved answer is factually correct and relevant to the query.
 - Also checks semantic similarity using embeddings and verifies cited URLs for health.
 - **Reporting & Release Management:**
 - All evaluation runs are logged and compared across releases, with detailed Excel and chart reports for MRR, Precision@1, and harmful content analysis.
 - Results are persisted in a structured S3 folder hierarchy for traceability and regression analysis.
- This multi-metric approach ensures we capture both the accuracy and reliability of retrieval, enabling continuous improvement and robust monitoring of the RAG system’s performance.

---

**Q: How are the final documents selected when combining results from both lexical (BM25) and embedding-based (vector) search in a hybrid search setup?**

- In our Knowledge GPT RAG system, we use a **hybrid search approach** where both lexical (BM25) and vector (KNN) searches are run in parallel.
- The results from both searchers are then combined using a **Reciprocal Rank Fusion (RRF)** algorithm, which is a rank-based, not score-based, method for merging results.
 - **How RRF Works:**
 - Each document retrieved by BM25 and KNN is assigned a rank based on its position in the respective result list.
 - For each unique document ID, the RRF score is calculated as: 
 `RRF_score = 1 / (rank_lexical + k) + 1 / (rank_vector + k)` 
 where `k` is a constant (e.g., 60) to smooth the influence of rank.
 - If a document appears in only one list, its rank in the other list defaults to the window size (e.g., 20).
 - All documents are then sorted by their combined RRF score in descending order.
 - The top N documents (as per the result limit) are selected as the final retrieval set.
- **Why RRF?**
 - RRF ensures that documents highly ranked by either search method are prioritized, balancing the strengths of both exact keyword matching and semantic similarity.
 - This approach is robust to score scale differences and helps surface the most relevant results from both search paradigms.
- This hybrid RRF-based selection is the default and recommended mode in our production pipeline, as documented in our Knowledge GPT architecture.

---
