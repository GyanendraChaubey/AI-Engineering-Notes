# Document Digitization & Chunking

> How you split your documents is often the single biggest factor in RAG retrieval quality.

---

## 1. What is chunking and why is it necessary?

**Chunking** is the process of splitting large documents into smaller, semantically meaningful pieces before indexing them in a vector store.

**Why it's needed:**
- LLMs have finite context windows — you can't inject an entire document
- Embedding models produce better vectors for focused, coherent text
- Smaller chunks allow more precise retrieval — you surface only the relevant section, not a 50-page report
- Granular chunks enable better citation and traceability

Without chunking, a 100-page PDF would produce one embedding that mixes everything — retrieval becomes a blunt instrument.

---

## 2. What factors influence the optimal chunk size?

| Factor | Impact on Chunk Size |
|---|---|
| **Embedding model context limit** | Max tokens the embedder can encode (e.g., 512 for many models) |
| **LLM context window** | Larger window → can afford larger or more chunks in prompt |
| **Document type** | Code → function-level; Legal → clause-level; FAQ → Q&A pair |
| **Query type** | Specific lookups → small chunks; Broad summaries → larger chunks |
| **Retrieval precision vs. recall** | Smaller = precise but may miss context; Larger = more context but noisier |

**Rule of thumb:** Start with 512 tokens with 10–20% overlap, then tune based on retrieval quality metrics.

---

## 3. What are the main chunking strategies?

### Fixed-Size Chunking
Split every N characters/tokens with optional overlap.
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_text(document_text)
```

### Semantic Chunking
Split at sentence boundaries where embedding similarity drops — each chunk is semantically coherent.
```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import OpenAIEmbeddings

chunker = SemanticChunker(OpenAIEmbeddings(), breakpoint_threshold_type="percentile")
chunks = chunker.create_documents([document_text])
```

### Document-Structure-Aware Chunking
Use document structure (headings, paragraphs, sections) as boundaries.
- Markdown → split at `##` headings
- HTML → split at `<section>`, `<article>` tags
- PDF → split at detected page breaks and heading styles

### Recursive Chunking
Try to split on paragraphs first, then sentences, then words — maintains natural boundaries while respecting size limits. This is the `RecursiveCharacterTextSplitter` approach above.

### Agentic / Proposition Chunking
Use an LLM to extract atomic propositions from text — each chunk is one standalone factual statement. High quality, but expensive.

```python
prompt = """
Extract all standalone factual propositions from this text.
Each proposition should be self-contained and independently verifiable.
Return as a JSON list of strings.

Text: {text}
"""
```

---

## 4. How do you determine the best chunk size experimentally?

```python
# Evaluation framework
from ragas import evaluate
from ragas.metrics import context_precision, context_recall

chunk_sizes = [256, 512, 1024]
results = {}

for size in chunk_sizes:
    # Build index with this chunk size
    index = build_index(docs, chunk_size=size)
    retriever = index.as_retriever(k=5)
    
    # Run evaluation dataset through RAG
    predictions = run_rag_pipeline(test_questions, retriever)
    
    # Score
    score = evaluate(predictions, metrics=[context_precision, context_recall])
    results[size] = score

# Pick chunk size with best context_precision + context_recall balance
```

Also measure end-to-end answer quality with LLM-as-judge, not just retrieval metrics.

---

## 5. What is the best approach to digitize and chunk complex documents like annual reports?

Annual reports have mixed content: text, tables, charts, footnotes, multi-column layouts. A naive PDF parser will scramble the reading order.

**Recommended pipeline:**

```
PDF → Document Intelligence (Azure DI / AWS Textract / Unstructured.io)
    → Structure extraction (headings, paragraphs, tables, figures)
    → Per-element processing:
         Text blocks  → semantic chunking
         Tables       → serialize to markdown, chunk by table
         Charts       → caption + AI description
         Footnotes    → attach to parent section as metadata
    → Enrich chunks with metadata (page, section, document, date)
    → Embed & index
```

```python
from unstructured.partition.pdf import partition_pdf

elements = partition_pdf(
    filename="annual_report.pdf",
    strategy="hi_res",          # Use layout model
    infer_table_structure=True, # Extract tables as HTML
    extract_images_in_pdf=True
)

for element in elements:
    if element.category == "Table":
        # Convert to markdown for embedding
        chunk = element.metadata.text_as_html
    elif element.category == "Image":
        # Use vision model to describe
        chunk = describe_image(element)
    else:
        chunk = element.text
```

---

## 6. How should tables be handled during chunking?

Tables are structured data — splitting them mid-row destroys their meaning.

**Strategies:**

1. **Keep each table as one chunk** — serialize to markdown or HTML, embed as a unit
2. **Table header repetition** — if a table spans multiple pages, prepend column headers to each row-group chunk
3. **Row-level chunking** — for very wide tables, each row becomes a chunk (with column names as context)
4. **Convert to natural language** — use an LLM to narrate each row: "In Q1, revenue was $2.3M, up 15% YoY"

```python
def table_to_markdown(table_html):
    """Convert HTML table to markdown for embedding."""
    import pandas as pd
    df = pd.read_html(table_html)[0]
    return df.to_markdown(index=False)

def table_to_nl_rows(df, table_title=""):
    """Convert each row to a natural language sentence."""
    rows = []
    for _, row in df.iterrows():
        sentence = f"[{table_title}] " + ", ".join(f"{col}: {val}" for col, val in row.items())
        rows.append(sentence)
    return rows
```

---

## 7. How do you handle very large tables that exceed the embedding model's context?

For tables with hundreds of rows:

1. **Vertical split by row groups** — chunk every N rows, always prepend headers
2. **Column selection** — embed only key columns; store full row as metadata for retrieval
3. **Summary + detail** — create a high-level summary chunk ("This table shows quarterly revenue by region from 2020–2024") + row-level chunks
4. **Hybrid approach** — use structured SQL/Pandas for table queries instead of vector search

```python
def chunk_large_table(df, rows_per_chunk=20):
    headers = df.columns.tolist()
    chunks = []
    for i in range(0, len(df), rows_per_chunk):
        sub_df = df.iloc[i:i+rows_per_chunk]
        # Always include headers
        chunk_text = sub_df.to_markdown(index=False)
        chunks.append({
            "text": chunk_text,
            "metadata": {"rows": f"{i}-{i+rows_per_chunk}", "type": "table"}
        })
    return chunks
```

---

## 8. How should list items be handled during the chunking process?

Lists are common in documents (bullet points, numbered steps, feature lists). The challenge: items lose meaning without their heading/context.

**Best practices:**

1. **Attach the parent heading** — prepend "Section: {heading}" to each list-item chunk
2. **Keep short lists together** — if a list has <10 items and fits in one chunk, don't split it
3. **Long lists** — split into groups of 5–7 items, each with the heading prepended
4. **Context-aware splitting** — don't split in the middle of a multi-part item

```python
def chunk_list_section(heading, items, max_items_per_chunk=7):
    chunks = []
    for i in range(0, len(items), max_items_per_chunk):
        group = items[i:i+max_items_per_chunk]
        text = f"Section: {heading}\n\n" + "\n".join(f"• {item}" for item in group)
        chunks.append(text)
    return chunks
```

---

## 9. How do you build a production-grade document processing and indexing pipeline?

```
┌─────────────────────────────────────────────────────┐
│                INGESTION LAYER                       │
│  File watcher / S3 trigger / API upload             │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              PARSING LAYER                           │
│  PDF: Unstructured / Azure DI                       │
│  DOCX: python-docx                                  │
│  HTML: BeautifulSoup / Trafilatura                  │
│  Images: Vision LLM (GPT-4V / Claude)               │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              CHUNKING LAYER                          │
│  Semantic / structure-aware chunking                 │
│  Metadata enrichment (source, date, section, page)  │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              EMBEDDING LAYER                         │
│  Batch embedding (parallelized)                     │
│  Rate limiting & retry logic                        │
│  Deduplication (hash-based)                         │
└─────────────────┬───────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────┐
│              STORAGE LAYER                           │
│  Vector DB (Pinecone / Weaviate / Qdrant)           │
│  Document store (for full text retrieval)           │
│  Metadata DB (PostgreSQL / MongoDB)                 │
└─────────────────────────────────────────────────────┘
```

**Production requirements:**
- Idempotent pipeline (re-running doesn't duplicate chunks)
- Versioning (track which document version each chunk came from)
- Monitoring (chunk count, embedding errors, index size)
- Incremental updates (only re-index changed documents)

```python
import hashlib

def get_doc_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

def should_reindex(doc_id: str, new_hash: str, metadata_db) -> bool:
    existing = metadata_db.get(doc_id)
    return existing is None or existing["hash"] != new_hash
```

---

## 10. How do you handle charts and graphs in a RAG pipeline?

Charts contain information that can't be extracted by text parsers. Options:

1. **Vision LLM description** — Use GPT-4V or Claude to generate a text description of the chart
2. **Caption extraction** — Extract figure captions (usually the most useful text near a chart)
3. **Alt text** — Use existing alt text if present
4. **Skip & flag** — For very complex charts, flag for human annotation

```python
import base64
from openai import OpenAI

client = OpenAI()

def describe_chart(image_path: str, surrounding_text: str = "") -> str:
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Describe the key data and insights in this chart. Context: {surrounding_text}"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_data}"}
                }
            ]
        }]
    )
    return response.choices[0].message.content
```

Store the generated description as a chunk with metadata `{"type": "chart_description", "page": N}`.

---

*Next: [Embedding Models →](../04-embeddings/README.md)*
