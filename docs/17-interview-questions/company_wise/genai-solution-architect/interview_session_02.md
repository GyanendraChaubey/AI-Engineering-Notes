# GenAI Solution Architect — Interview Session 2 (Deep Dive)

**Q: What tools are available for managing and optimizing token usage in large LLM workflows?**

- **tiktoken:** Count tokens before API call; avoid truncation; estimate cost upfront.
- **LLMLingua / LLMLingua-2 (Microsoft Research):** Prompt compression -- 2-20x token reduction with < 5% accuracy drop.
- **Graphify:** Converts document corpora into knowledge graphs; structured subgraph retrieval replaces raw text chunks -- reduces tokens while maintaining factual precision.
- **LangChain OpenAICallbackHandler:** Auto-tracks token counts and cost per chain/agent run.
- **LangFuse / Helicone / Portkey:** Full LLM observability -- every API call logged with token counts, latency, cost; dashboards for per-user and per-feature budgets.
- **Semantic caching:** Avoids LLM calls entirely for repeat queries -- biggest token saving possible.
- **Model routing:** Route simple queries to GPT-4o-mini (20x cheaper); reserve GPT-4 for complex reasoning.
- **Retrieval precision:** Cross-encoder reranker selects top-3 instead of top-10 chunks -> 70% token reduction with minimal quality loss.

---

**Q: What guardrails ensure safe and compliant LLM outputs in production?**

**Input guardrails (before LLM):**
- PII detection and masking (Presidio, AWS Comprehend).
- Toxicity / harmful intent classifier (LlamaGuard, fine-tuned DistilBERT).
- Prompt injection detection (Rebuff, regex for instruction-override patterns).
- Topic filtering: block queries outside allowed domain via semantic similarity check.

**Output guardrails (after LLM generates):**
- Faithfulness check via LLM-as-judge: every claim grounded in retrieved context?
- PII scan on output: re-run Presidio on generated text.
- Harmful content moderation: Azure Content Safety / AWS Bedrock Guardrails.
- Citation validation: verify cited sources exist in retrieval set.

**Compliance:** GDPR (no storage of personal queries, right-to-erasure). HIPAA (PHI detection, encryption, audit log). Financial services (no investment advice, disclaimer injection).

**Frameworks:** Guardrails AI, NeMo Guardrails, LlamaGuard, Azure Content Safety, AWS Bedrock Guardrails.

---


---
