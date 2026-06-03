# Retrieval Augmented Generation (RAG)

> RAG bridges the gap between static model knowledge and dynamic, verifiable, up-to-date information.

---

## 1. How can you make LLM answers more accurate, verifiable, and reliable?

LLMs hallucinate because they rely solely on parametric memory (weights). The solution is to **ground responses in retrieved evidence**:

1. **RAG (Retrieval Augmented Generation)** — Fetch relevant documents at query time and inject into the prompt
2. **Citations** — Force the model to reference specific source chunks
3. **Constrained generation** — "Only answer from the provided context"
4. **Verification loops** — Have the model cross-check its answer against sources

This makes answers verifiable (users can inspect sources), reduces hallucination, and allows knowledge to be updated without retraining.

---

## 2. How does a RAG pipeline work end to end?

```
User Query
    │
    ▼
[Query Encoder] → Query Embedding
    │
    ▼
[Vector Store] ──── similarity search ──── Top-K Document Chunks
    │
    ▼
[Context Assembly] → Prompt = System + Retrieved Chunks + Query
    │
    ▼
[LLM] → Grounded Response
```

**Detailed steps:**

1. **Indexing (offline):** Chunk documents → embed each chunk → store in vector DB
2. **Retrieval (online):** Embed user query → find top-K similar chunks via ANN search
3. **Augmentation:** Inject retrieved chunks into the LLM prompt as context
4. **Generation:** LLM generates an answer grounded in the retrieved context

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Build index
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())

# RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

result = qa_chain({"query": "What is our refund policy?"})
print(result["result"])
print(result["source_documents"])
```

---

## 3. What are the core advantages of a RAG architecture?

| Benefit | Explanation |
|---|---|
| **Up-to-date knowledge** | Add new documents without retraining the model |
| **Verifiable answers** | Users can inspect which chunks were used |
| **Reduced hallucination** | Model is constrained to retrieved context |
| **Domain adaptation** | Works on private/proprietary data out of the box |
| **Cost efficiency** | Cheaper than full fine-tuning for knowledge injection |
| **Scalable knowledge** | Index millions of documents; retrieve only what's needed |

---

## 4. When should you use fine-tuning instead of RAG?

Use this decision framework:

| Scenario | Recommended Approach |
|---|---|
| Need domain-specific **facts** | RAG |
| Need up-to-date / frequently changing knowledge | RAG |
| Need source attribution | RAG |
| Need a specific **style or tone** | Fine-tuning |
| Need a specific **task format** (e.g., JSON output) | Fine-tuning |
| Model needs to learn **reasoning patterns** | Fine-tuning |
| Latency-sensitive (no retrieval step) | Fine-tuning |
| Both knowledge AND style | RAG + Fine-tuning combined |

**Key insight:** RAG injects *what to know*; fine-tuning changes *how the model behaves*. They solve different problems.

---

## 5. What are the architectural patterns for customizing LLMs with proprietary data?

```
Pattern 1: Prompt Engineering Only
─────────────────────────────────
Proprietary data → Paste into context window
✓ Fast to implement  ✗ Limited by context length

Pattern 2: RAG
──────────────
Data → Chunk → Embed → Vector DB → Retrieve → Inject into prompt
✓ Scalable, verifiable  ✗ Retrieval quality matters

Pattern 3: Fine-Tuning Only
───────────────────────────
Data → Create training pairs → Fine-tune base model
✓ Bakes in style/format  ✗ Static knowledge, expensive

Pattern 4: RAG + Fine-Tuning
─────────────────────────────
Fine-tune for behavior → RAG for up-to-date knowledge
✓ Best of both worlds  ✗ Complex to maintain

Pattern 5: Long-Context Models
───────────────────────────────
Stuff entire knowledge base into 1M-token context
✓ Simple  ✗ Expensive, latency, needle-in-haystack issues

Pattern 6: Agent + Tool Use
────────────────────────────
Model decides when to call retrieval/search tools
✓ Dynamic, flexible  ✗ Harder to control
```

**Production recommendation:** Start with RAG (Pattern 2). Add fine-tuning (Pattern 4) once you've identified specific behavior gaps that retrieval alone can't fix.

---

*Next: [Document Digitization & Chunking →](../03-chunking/README.md)*
