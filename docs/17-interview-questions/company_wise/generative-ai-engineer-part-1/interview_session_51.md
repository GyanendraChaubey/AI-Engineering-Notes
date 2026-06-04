# Generative AI Engineer (Part 1) — Interview Session 51 (Deep Dive)

**Q: How do you design and implement guardrails for data inflow and outflow in a RAG system?**

Guardrails operate at two layers: before the LLM sees the input and before the response reaches the user.

**Input Guardrails:**
- **PII Detection:** Microsoft Presidio or regex patterns; replace entities with `[NAME]`, `[EMAIL]` tokens.
- **Prompt Injection Detection:** Detect instruction-override patterns using keyword matching or classifier.
- **Toxicity Filter:** Fast classifier (AWS Comprehend, DistilBERT) scores toxicity before processing.
- **Off-topic Router:** Embed query; block if cosine similarity to valid-topics centroid is below threshold.

**Output Guardrails:**
- **Faithfulness Check (LLM-as-Judge):** Second LLM call: "Is every claim supported by context? Score 0-5." Block if < 3.
- **PII Leakage Scan:** Re-scan generated response with Presidio before returning to user.
- **Harmful Content Filter:** Azure Content Safety or custom keyword blocklist on output.
- **Source Citation Validator:** Verify cited chunk IDs exist in retrieved set.

**Without third-party libraries:** Regex PII patterns + blocked-phrase list for injection + LLM self-check call + JSON schema validation.

---

**Q: How do you develop an MCP (Model Context Protocol) server from scratch?**

**Development steps:**
1. **Define Tools:** Each tool needs `name`, `description` (LLM uses this to decide when to call), and `inputSchema` (JSON Schema).
2. **Define Resources:** Static or dynamic data the LLM can read (knowledge base docs, schema files).
3. **Implement Handlers:** Python/Node functions that execute when the LLM calls a tool.
4. **Transport:** `stdio` for local (Claude Desktop) or `HTTP/SSE` for remote clients.
5. **Auth:** OAuth 2.0 or API key validation. Store credentials in AWS Secrets Manager.
6. **Test:** MCP Inspector CLI or Claude Desktop.
7. **Harden:** Rate limiting, audit logging, circuit breakers, health checks.

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

app = Server("enterprise-mcp")

@app.list_tools()
async def list_tools():
    return [types.Tool(name="search_docs",
                       description="Search the knowledge base for relevant information",
                       inputSchema={"type": "object",
                                    "properties": {"query": {"type": "string"}},
                                    "required": ["query"]})]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "search_docs":
        results = vector_db.search(arguments["query"])
        return [types.TextContent(type="text", text=str(results))]
```

---

**Q: How do you handle high query volumes (200+ TPS) in an intent classification system?**

- **Horizontal scaling:** Multiple SageMaker endpoints behind a load balancer with auto-scaling.
- **Async inference:** FastAPI with `async def` prevents thread blocking during model inference.
- **Batching:** Group queries within a 50ms window into single batch inference -- reduces GPU overhead.
- **Semantic caching:** Return cached predictions from Redis for repeated queries instead of running inference.
- **Model optimization:** INT8/FP16 quantization via ONNX Runtime reduces latency 2-4x.
- **Queue-based decoupling:** SQS + ECS workers for spiky traffic.

At 250 TPS with 50ms latency: 250 x 0.05s = 13 concurrent requests minimum. Plan 3-4 replicas with batching.

---

**Q: How do you handle PII (Personally Identifiable Information) masking in production AI systems?**

**Pre-processing (before LLM):**
- Microsoft Presidio or AWS Comprehend to detect: names, emails, phones, SSNs, credit cards, addresses.
- Replace with tokens: `"John Smith" -> [PERSON_1]` (store mapping in secure session store if reversibility needed).
- Healthcare: HIPAA-compliant de-identification (18 safe harbor identifiers).

**During inference:** System prompt: "Never repeat or generate personal information." Temperature=0 for safety-critical responses.

**Post-processing:** Re-scan LLM output with Presidio -- models can regenerate PII from training data. Log only tokenized/masked versions.

**Compliance:** GDPR right-to-erasure deletion pipelines. Data residency: LLM API calls within required geographic boundaries.

---

**Q: How do you handle authentication and authorization in production AI deployments?**

**Authentication:**
- **JWT tokens:** Stateless, RS256-signed. Backend validates without database call.
- **OAuth 2.0 + OIDC:** Enterprise SSO (Azure AD, Okta). Users get JWT access token.
- **API keys:** Service-to-service calls. Store in AWS Secrets Manager, never in code.
- **mTLS:** Microservice-to-microservice mutual certificate validation.

**Authorization:**
- **RBAC:** Roles map to permissions (admin can delete docs; user can only query).
- **Document-level access:** Tag every chunk with `tenant_id` at ingestion; apply as mandatory metadata filter on every vector search.
- **Row-Level Security:** PostgreSQL + pgvector with RLS policies.
- **API Gateway:** Enforces auth before requests reach application code.

```python
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
import jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
```

---

**Q: How do you handle content safety in production LLM systems?**

**Multi-layer approach:**
1. **Input screening:** LlamaGuard, Azure Content Safety, or custom model for harmful intent classification.
2. **System prompt:** "Never discuss [prohibited topics]. If asked, respond: I cannot help with that."
3. **Output scanning:** Safety classifier on generated response; block/replace if score exceeds threshold.
4. **Threshold configuration:** Configurable per harm category (stricter for children's apps).
5. **Human review queue:** Log borderline cases (score 0.2-0.5) for continuous classifier improvement.
6. **Audit trail:** Log blocked queries (hash only, not raw content) for compliance.

**Tools:** Azure Content Safety, AWS Bedrock Guardrails, Meta LlamaGuard (open-source), NVIDIA NeMo Guardrails.

---

**Q: How do you route queries between document stores (SharePoint/S3) and structured databases (Snowflake)?**

**Intent-based routing:**
1. **Query classifier:** Determines type: Conceptual/explanatory -> document store; Analytical ("how many", "total") -> Snowflake (NL-to-SQL); Mixed -> run both.
2. **Keyword heuristics:** SQL-suggestive terms ("count", "sum", "trend") route to Snowflake as fast pre-filter.
3. **Hybrid fallback:** If classifier confidence < threshold, run both retrievals.

**Schema-aware Snowflake routing:**
- Store Snowflake schema metadata (table names, column descriptions) in PostgreSQL.
- Retrieve relevant schema snippets at query time; include in NL-to-SQL prompt.
- Validate generated SQL: allow-list of tables, block DML statements.

---

**Q: How do you handle scenarios where the LLM generates false responses due to insufficient retrieved context?**

- **Detection:** Faithfulness score < threshold (LLM-as-judge < 3/5) -> flag as unreliable.
- **Fallback:** Return: "I don't have enough reliable information to answer this accurately."
- **Improve retrieval first:** (1) Expand top-K from 5 to 10, (2) re-run with BM25 keyword search, (3) rephrase query via HyDE.
- **Log and alert:** Store low-confidence queries for evaluation; identify knowledge base gaps.
- **User transparency:** Show which source documents were used; if none are strong, state this explicitly.

---

**Q: How do you implement streaming responses in a RAG system?**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import openai

app = FastAPI()

async def stream_llm_response(query: str):
    chunks = retrieve_chunks(query)
    prompt = build_prompt(query, chunks)
    client = openai.AsyncOpenAI()
    stream = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        stream=True)
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield f"data: {delta}\n\n"  # Server-Sent Events format
    yield "data: [DONE]\n\n"

@app.get("/chat")
async def chat(query: str):
    return StreamingResponse(stream_llm_response(query), media_type="text/event-stream")
```

**Key points:** First token in ~300ms vs 2-5s for full response. Handle disconnects with `asyncio.CancelledError`. Frontend uses `EventSource` API. Citations appended after `[DONE]` signal.

---

**Q: How do you implement metadata management and chunk tracking in a RAG pipeline?**

**Chunk metadata schema:**
```json
{
  "chunk_id": "doc-001-chunk-042",
  "doc_id": "doc-001",
  "source_path": "s3://bucket/docs/manual.pdf",
  "page_number": 7,
  "section": "Installation",
  "token_count": 124,
  "content_hash": "sha256:abc123...",
  "created_at": "2024-01-15T10:30:00Z",
  "language": "en",
  "doc_type": "PDF"
}
```

**Storage:** Vectors + metadata in vector DB for fast filtered search. Document-level metadata in PostgreSQL for relational queries. Hash-based deduplication to skip re-embedding unchanged chunks. Soft-delete (`active=false`) preserves audit trail.

**Incremental updates:** Compute chunk hashes -> compare stored hashes -> re-embed only changed/new chunks.

---

**Q: How do you manage multi-turn dialogue in chatbot applications?**

**Three strategies:**
1. **Buffer Memory:** Store all messages; pass complete history. Context fills after 10-20 turns; cost grows linearly.
2. **Window Memory:** Keep last N turns (e.g., 5). Risk of losing critical early context.
3. **Summary Memory (best):** Periodically summarize old turns: "User asked about X, system explained Y." Keep summary + last N turns.

**Entity tracking:** Extract key entities (product, user preferences) and maintain in structured store for reference resolution across turns.

**Session management:** Redis with `session_id` key and TTL (24h). Use `user_id` for cross-device continuity.

---

**Q: How do you manage state in a LangGraph workflow?**

State is a `TypedDict` shared across all nodes. Nodes read from it and return partial updates.

```python
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
import operator

class AgentState(TypedDict):
    messages: Annotated[List[dict], operator.add]  # append-only
    context: List[str]
    answer: str
    iterations: int

def retrieve_node(state: AgentState) -> dict:
    docs = vector_search(state["messages"][-1]["content"])
    return {"context": docs}  # only update this key

def generate_node(state: AgentState) -> dict:
    answer = llm.invoke(build_prompt(state["messages"], state["context"]))
    return {"answer": answer, "iterations": state["iterations"] + 1}

def should_continue(state: AgentState) -> str:
    return END if (state["answer"] or state["iterations"] >= 3) else "retrieve"

graph = StateGraph(AgentState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("generate", generate_node)
graph.add_edge("retrieve", "generate")
graph.add_conditional_edges("generate", should_continue)
graph.set_entry_point("retrieve")
app = graph.compile(checkpointer=MemorySaver())
```

**Key:** Nodes return update dicts -- never mutate state directly. `checkpointer` persists state after each node for resume-after-failure and human-in-the-loop.

---

**Q: How do you manage token usage and control costs during LLM evaluation?**

- **Use cheaper judge model:** GPT-4o-mini or Claude Haiku for scoring -- 10-30x cheaper, sufficient for evaluation.
- **Batch offline:** Run as a batch job, not real-time. Process 100 samples in one job.
- **Limit context:** Send only question + context + answer to judge (not full conversation history).
- **Cache results:** Hash (question, context, answer); reuse score if unchanged.
- **Set max_tokens:** Judge calls need only 100-200 tokens for score + explanation.
- **Track and alert:** Log token counts; set per-experiment budgets.

Cost reference: 100 samples x 500 input x 100 output tokens with GPT-4o-mini = $0.01 total.

---

**Q: How do you measure the accuracy of a RAG pipeline?**

**Retrieval metrics:**
- Context Precision (RAGAS): what fraction of retrieved chunks are relevant to ground truth?
- Context Recall (RAGAS): what fraction of ground truth information is covered by retrieved chunks?
- MRR (Mean Reciprocal Rank): where does the first relevant chunk appear in the ranked list?

**Generation metrics:**
- Faithfulness (RAGAS): are all claims supported by retrieved context?
- Answer Relevance (RAGAS): does the answer address the question?
- Answer Correctness: LLM-as-judge comparison against ground truth.

**Process:** Prepare 50-200 curated question/ground-truth pairs -> run full pipeline -> score with RAGAS -> track mean scores over time -> alert if any metric drops > 5% from previous version.

---

**Q: How do you perform hyperparameter tuning for ML models?**

**Approaches ranked by efficiency:**
1. **Grid Search:** Exhaustive. Practical only for < 4 parameters.
2. **Random Search:** 2-3x faster with similar results.
3. **Bayesian Optimization (Optuna):** Focuses trials where improvement is likely. Best for expensive training.
4. **Early Stopping:** Abort bad configs when validation metric stops improving.

```python
import optuna
from sklearn.model_selection import cross_val_score
import xgboost as xgb

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
    }
    model = xgb.XGBClassifier(**params)
    return cross_val_score(model, X_train, y_train, cv=5, scoring="f1_weighted").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_params)
```

---

**Q: How do you run LLM evaluation for a RAG system?**

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

eval_data = Dataset.from_dict({
    "question": questions,
    "answer": generated_answers,
    "contexts": retrieved_contexts,  # list of lists
    "ground_truth": ground_truths
})
results = evaluate(eval_data, metrics=[faithfulness, answer_relevancy,
                                        context_precision, context_recall])
print(results)
# Output: {'faithfulness': 0.87, 'answer_relevancy': 0.91, ...}
```

**Custom LLM-as-judge** for answer correctness: prompt asks LLM to score 1-5 comparing generated answer to ground truth. Return JSON `{"score": N, "reason": "..."}`.

**Process:** Prepare 50-200 samples -> run full pipeline -> score with RAGAS + judge -> store in MLFlow -> alert on regression.

---

**Q: How do you solve the token limit problem when the LLM context window gets exhausted?**

**Strategies ranked by complexity:**
1. **Reduce top-K:** Retrieve top-3 instead of top-5 chunks.
2. **Chunk summarization:** Summarize each chunk before inserting (500 -> 100 tokens per chunk).
3. **Conversation compression:** Summarize older turns: "User asked X, system explained Y."
4. **Map-Reduce:** Summarize document sections in parallel, combine summaries.
5. **Dynamic truncation:** Count tokens with tiktoken; trim least relevant chunks to fit limit.

```python
import tiktoken

def fit_context_to_window(chunks: list, query: str, max_tokens: int = 3000) -> list:
    enc = tiktoken.encoding_for_model("gpt-4o")
    available = max_tokens - len(enc.encode(query)) - 500
    selected, used = [], 0
    for chunk in chunks:  # sorted by relevance descending
        tokens = len(enc.encode(chunk))
        if used + tokens > available:
            break
        selected.append(chunk)
        used += tokens
    return selected
```

---

**Q: How do you implement LLM-as-a-Judge to ensure responses are grounded in retrieved context?**

```python
import json, openai

JUDGE_PROMPT = (
    "RETRIEVED CONTEXT: {context}\n"
    "QUESTION: {question}\n"
    "GENERATED ANSWER: {answer}\n\n"
    "Evaluate if EVERY factual claim in the answer is explicitly supported by the context.\n"
    "Return JSON: {'score': <0-5>, 'unsupported_claims': [...], 'reasoning': '...'}\n"
    "5=fully grounded, 3=mostly grounded, 1=major hallucination"
)

def llm_judge(question: str, context: str, answer: str) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o-mini",  # cheaper model sufficient
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            context=context, question=question, answer=answer)}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    result = json.loads(response.choices[0].message.content)
    if result["score"] < 3:
        return "I cannot reliably answer this from the available knowledge base."
    return answer
```

**Best practices:** Use different/smaller model for judging (avoid self-evaluation bias). Temperature=0 for deterministic scores. Log unsupported claims for knowledge base gap analysis. Run on 100% of responses in dev, 10% sample in production.

---

**Q: What caching mechanisms are used in production RAG solutions?**

**Exact-match cache:** SHA256 hash of normalized query -> cached response. Hit rate: 20-60% for FAQ bots. TTL based on data freshness (news: 1h; product docs: 7 days).

**Semantic cache:** Embed incoming query; search Redis vector index for cosine similarity > 0.95 to past queries. Cache hit -> return cached response. Tools: GPTCache, RedisVL, LangChain SemanticSimilarityExactMatchCache. Threshold 0.95 (conservative, near-duplicates) vs. 0.85 (liberal, may return wrong answer for similar-but-different questions).

**Cache hierarchy:** L1: in-process Python dict (~1000 entries, fastest). L2: Redis cluster (shared across workers, sub-millisecond). L3: DynamoDB (longer TTL).

**Invalidation:** On document update, evict cache entries whose source documents changed.

---

**Q: What chunking mechanism is used for document processing in a RAG pipeline?**

| Document Type | Recommended Strategy | Reason |
|:---|:---|:---|
| Plain text / articles | RecursiveCharacterTextSplitter | Preserves sentence boundaries |
| Markdown / READMEs | MarkdownHeaderTextSplitter | Respects semantic sections |
| HTML / web pages | HTMLHeaderTextSplitter | Uses heading hierarchy |
| PDF with text | PyMuPDF -> Markdown -> MarkdownHeaderSplitter | Converts structure first |
| PDF with tables | Table-aware: keep table as single atomic chunk | Tables lose meaning if split |
| Code files | CodeTextSplitter | Respects function/class boundaries |
| High-accuracy RAG | SemanticChunker (embedding-based) | Groups semantically coherent content |

**Parameters:** chunk_size: 500-1000 tokens; chunk_overlap: 50-150 tokens (10-15% of chunk_size).

---

**Q: What chunking strategy should you use for documents containing images and tables?**

**Tables:** Extract with pdfplumber, Camelot, or PyMuPDF. Convert to Markdown table format. Treat each table as a **single atomic chunk** -- never split across two chunks. Include table caption as chunk prefix.

**Images and Charts:**
- OCR: AWS Textract or Tesseract -> extract text -> embed.
- Multimodal: GPT-4 Vision or LLaVA -> generate text description -> embed.
- ColPali: embeds entire page images directly (no OCR needed) -- retrieves pages via visual similarity.
- Store image URL in chunk metadata for display in retrieval results.

**Mixed documents:** Parse page -> separate regions -> chunk text normally -> keep table + surrounding text together -> generate image description as separate chunk with `chunk_type: "image"` -> index all types together.

---

**Q: What packages and libraries are commonly used for chunking, embedding, and indexing in a RAG pipeline?**

**Parsing:** `PyMuPDF (fitz)`, `pdfplumber`, `python-docx`, `unstructured` (universal), `camelot-py`, `pytesseract`, AWS Textract.

**Chunking:** `langchain.text_splitter` (RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, TokenTextSplitter), `llama-index` (SemanticSplitterNodeParser), `tiktoken` (token counting).

**Embedding:** `openai` (text-embedding-ada-002, text-embedding-3-large), `sentence-transformers` (all-MiniLM-L6-v2, BGE-m3), `torch` + `transformers` for custom models.

**Vector Indexing:** `opensearch-py`, `pinecone-client`, `chromadb`, `faiss`, `qdrant-client`, `pgvector`.

**Orchestration:** `langchain`, `llama-index`, `haystack`.

---

**Q: What is the RAPTOR approach to indexing in RAG systems?**

RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) indexes BOTH original chunks AND recursive LLM-generated summaries of grouped chunks.

**How it works:**
1. Start with leaf-level chunks (standard document chunks).
2. Cluster semantically similar chunks (Gaussian Mixture Models).
3. Summarize each cluster into a parent node via LLM.
4. Recursively cluster and summarize parent nodes until a single root summary.
5. Index ALL levels (leaf + summaries) in the same vector DB.

**At retrieval:** Search across all levels simultaneously. Specific queries hit leaf chunks; broad questions hit summary nodes.

**Best for:** Long documents, multi-hop questions, research Q&A. Trade-off: higher ingestion cost, more storage, but significantly better recall for complex queries.

---

**Q: What is Semantic Kernel and how does it compare to LangChain?**

Microsoft's open-source SDK for building AI-powered applications.

**Core concepts:** Kernel (orchestrator), Plugins (LLM + native functions), Planner (auto-creates execution plans), Memory (semantic vector + volatile).

| Aspect | Semantic Kernel | LangChain |
|:---|:---|:---|
| Origin | Microsoft | Community |
| Languages | Python, C#, Java | Python (primary) |
| Planning | Built-in auto-planner | Manual chains / AgentExecutor |
| Best for | Microsoft/Azure/.NET orgs | Python-first, open-source stack |

**When to use SK:** Existing .NET infrastructure, deep Azure integration. LangChain preferred for Python-first teams with broader ecosystem.

---

**Q: How do you use Weights & Biases (wandb) for observability during model fine-tuning?**

```python
import wandb
from transformers import TrainingArguments, Trainer

wandb.init(project="intent-classifier",
           config={"model": "all-MiniLM-L6-v2", "lr": 2e-5, "epochs": 3})

training_args = TrainingArguments(
    output_dir="./results",
    report_to="wandb",  # automatic logging
    logging_steps=10, eval_steps=50)
trainer = Trainer(model=model, args=training_args)
trainer.train()
wandb.finish()
```

**Auto-logged:** Training/validation loss, learning rate schedule, GPU utilization, gradient norms.
**Manual:** `wandb.log({"f1_weighted": 0.89})`.
**Sweeps (Bayesian hyperparameter search):** Define config with metric goal and parameter ranges; `wandb.agent` runs N trials.
**Limitations:** Requires internet; data uploaded to W&B servers (self-hosted option for air-gapped environments).

---

**Q: What are the common latency challenges in production GenAI/NLP systems?**

| Component | Typical Latency | Mitigation |
|:---|:---|:---|
| LLM API call | 1-10s | Streaming, semantic cache, smaller model routing |
| Embedding | 50-200ms | Local GPU model, async batching |
| Vector search (ANN) | 5-50ms | HNSW tuning, index sharding |
| PDF parsing | 100ms-5s | Pre-process at ingestion, cache parsed output |
| Reranking | 200ms-1s | Limit to top-10 candidates, quantize model |

**Most impactful (in order):** Semantic caching (eliminates 30-60% of LLM calls), streaming responses (reduces perceived latency 10x), smaller model routing for simple queries, async parallel retrieval.

---

**Q: What tools are available for managing and optimizing token usage in LLM workflows?**

- **tiktoken:** Count tokens before API call; calculate cost upfront; prevent truncation.
- **Graphify:** Converts documents to knowledge graphs; structured traversal replaces large-context retrieval.
- **LLMLingua / LLMLingua-2 (Microsoft):** Prompt compression -- 2-20x token reduction with < 5% accuracy drop.
- **LangChain OpenAICallbackHandler:** Auto-tracks token counts and cost per chain/agent run.
- **LangFuse / Helicone / Portkey:** Full LLM observability with token usage dashboards and budget alerts.
- **Strategies:** top-3 chunks via reranking, context summarization, small model routing for simple queries, hard `max_tokens` limit.

---

**Q: What frameworks are available for implementing guardrails in LLM systems?**

| Framework | Type | Key Feature |
|:---|:---|:---|
| Guardrails AI | Open-source | XML-based rail specs, output validation, retry on failure |
| NVIDIA NeMo Guardrails | Open-source | Colang language, conversation flow control |
| LlamaGuard (Meta) | Safety model | 14 harm categories, runs locally, free |
| Azure Content Safety | Cloud API | Multi-category, configurable thresholds |
| AWS Bedrock Guardrails | Cloud managed | Topic blocking, PII redaction, grounding checks |
| Custom pipeline | DIY | Regex + classifier + LLM-judge, full control |

**Production stack:** LlamaGuard (input safety) + Presidio (PII) + system prompt constraints + LLM-as-judge faithfulness check (output) + Azure Content Safety (output).

---

**Q: How do you choose between SetFit, Falcon/LLM, and XGBoost for intent classification?**

| Factor | SetFit | LLM (Falcon/GPT) | XGBoost |
|:---|:---|:---|:---|
| Training data | Few-shot (8-64/class) | Zero/few-shot | 100s/class |
| Inference speed | Fast (encoder only) | Slow (7B+ params) | Very fast |
| Cost | Very low | High | Negligible |
| Accuracy | High for clear intents | Highest, handles ambiguity | Good with enough data |
| New intent | Retrain required | Prompt update only | Retrain required |

**Decision:** < 100 samples/class -> SetFit. Ambiguous/evolving intents -> LLM. High throughput + sufficient data -> XGBoost. **Production hybrid:** SetFit primary; route low-confidence (< 0.7) to LLM fallback.

**Why cosine similarity for SetFit fine-tuning:** Contrastive learning pushes same-intent pairs together and different-intent pairs apart. Cosine similarity measures directional alignment independent of vector magnitude.

---

**Q: Why would you use LangGraph over a simple sequential RAG pipeline?**

A simple RAG pipeline is linear and stateless: `query -> embed -> retrieve -> prompt -> LLM -> answer`.

**Use LangGraph when you need:**
- **Conditional branching:** "If < 3 relevant chunks retrieved, do a web search fallback."
- **Loops / retry:** "If faithfulness < 3, reformulate query and retry (max 3 iterations)."
- **Persistent state:** Typed state object shared across all nodes without passing giant dicts manually.
- **Multi-agent coordination:** Orchestrator routes to sub-agents (research, summarization) with own tools.
- **Checkpointing:** Resume after failure; human-in-the-loop review at specific nodes.
- **Parallelism:** BM25 and vector search in parallel branches.

Keep simple pipeline for: single retrieval -> generation, no routing or retry logic needed.

---

**Q: Why use recursive chunking in a RAG pipeline?**

Recursive chunking (`RecursiveCharacterTextSplitter`) **preserves natural language boundaries** unlike fixed-size chunking.

- Tries `\n\n` (paragraph) -> `\n` (line) -> `.` (sentence) -> ` ` (word) before splitting at character N.
- Fixed-size may cut mid-sentence, breaking semantic meaning.
- Result: chunks end at paragraph/sentence boundaries; embeddings capture complete thoughts.
- Industry default: most widely used in production RAG (LangChain, LlamaIndex, Haystack).
- Configurable: set `chunk_size` and `chunk_overlap`; algorithm finds best natural boundary within limit.

vs. Semantic: recursive is faster/deterministic; semantic is more accurate but requires embedding every sentence.

---


---
