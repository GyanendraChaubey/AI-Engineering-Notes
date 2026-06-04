# Generative AI Engineer (Part 1) — Interview 34

**Q: Walk through the end-to-end RAG (Retrieval-Augmented Generation) architecture for an enterprise RAG assistant.**

- The Knowledge GPT system is designed as an enterprise-grade RAG solution to enable natural language access to internal knowledge repositories.
- The architecture is modular, scalable, and optimized for both retrieval performance and LLM response quality.
- Here’s the end-to-end flow:

**1. Document Ingestion & Chunking**
 - Large enterprise documents are ingested and split into manageable chunks (up to 25 KB).
 - Each chunk is summarized using Azure OpenAI, and embeddings (1536-dim vectors) are generated for each summary.
 - Metadata (chunk ID, source, creation time, etc.) is attached to each chunk.
 - Chunks and their embeddings are bulk ingested into OpenSearch using the Bulk API, making them immediately searchable.

**2. Query Processing & Hybrid Search**
 - When a user submits a query, it is embedded using Azure OpenAI to generate a 1536-dim vector.
 - The system runs two searches in parallel:
 - Lexical search (BM25) for keyword-based matching.
 - Vector search (KNN) for semantic similarity.
 - Results from both searches are combined using Reciprocal Rank Fusion (RRF) to get the most relevant document chunks.

**3. Context Construction & Prompt Engineering**
 - The top-ranked chunks are consolidated and context data is built (including chunk, title, source, etc.).
 - A prompt template is loaded based on user/session context.
 - The prompt is constructed by injecting the context data, user query, and any relevant persona or language settings.

**4. LLM Generation**
 - The constructed prompt, along with conversation history, is sent to Azure OpenAI’s chat completion endpoint.
 - The LLM generates a natural language answer, citing the relevant document sources.

**5. API Layer & Integration**
 - The architecture uses three main microservices (Flask APIs):
 - Search API: Handles search queries and returns ranked chunks.
 - Completions API: Orchestrates the search, builds prompts, calls the LLM, and returns answers.
 - Feedback API: Collects user feedback for continuous improvement.
 - All APIs are exposed as REST endpoints (FastAPI in some deployments) for integration with enterprise applications.

**6. Security & Compliance**
 - PII detection is performed using AWS Comprehend for English/Spanish queries.
 - If PII is detected, the request is rejected for compliance.
 - All data flows are secured and monitored.

**7. Monitoring & Evaluation**
 - The system tracks response accuracy, relevance, and user feedback.
 - Continuous evaluation and retraining pipelines are in place to improve retrieval and generation quality.

**Summary of My Role:**
 - Led the end-to-end architecture and implementation.
 - Designed the hybrid search and RRF ranking logic.
 - Built the document processing, embedding, and ingestion pipelines.
 - Developed the API layer and integrated security/compliance checks.
 - Set up monitoring, evaluation, and feedback loops for continuous improvement.
 - Collaborated with cross-functional teams for deployment and integration.

This architecture ensures scalable, accurate, and secure retrieval-augmented generation for enterprise knowledge access.

---

**Q: Explain the initial stages of the enterprise RAG architecture, focusing on document ingestion, metadata extraction, and storage.**

- The architecture starts with a robust document ingestion pipeline designed for scalability and automation.
- All enterprise documents are stored in a central Knowledge as a System (KaaS) repository, which exposes APIs for document access.
- Every day, new or updated document metadata is incrementally pushed to AWS S3 as part of a scheduled process.
- An AWS Glue job runs daily to scan S3 for new metadata entries.
- For each new metadata entry, key fields like document ID, language, title, source, and creation time are extracted.
- Using the document ID, the system calls the KaaS API to retrieve the actual document link.
- The document is then downloaded from the provided link, supporting formats like PDF and HTML.
- All downloaded documents are stored in S3 in a unified format, ensuring consistency for downstream processing.
- This automated ingestion and normalization process ensures that the knowledge base is always up-to-date and ready for further processing, such as chunking, embedding, and indexing for retrieval.

This approach ensures high data quality, traceability, and seamless integration with the rest of the RAG pipeline.

---

**Q: What strategies do you apply for hallucination reduction in a RAG system?**

- Hallucination reduction is critical in RAG systems to ensure that LLM responses are grounded in actual enterprise knowledge and not fabricated.
- I use a combination of retrieval, prompt engineering, and post-processing strategies to minimize hallucinations:

**Retrieval-Level Strategies:**
- Use hybrid search (lexical + vector) to maximize the relevance and coverage of retrieved chunks.
- Apply Reciprocal Rank Fusion (RRF) to combine results from both search types, ensuring only the most contextually relevant chunks are passed to the LLM.
- Limit the number of retrieved chunks and enforce strict chunk size (e.g., 25 KB) to fit within the LLM’s context window, preserving context integrity.

**Prompt Engineering:**
- Design prompts that explicitly instruct the LLM to answer only using the provided context and to avoid speculation.
- Include clear instructions in the prompt to cite sources and ignore information not present in the retrieved context.
- Use persona and language settings to further constrain the LLM’s behavior.

**Context Construction:**
- Ensure that only high-confidence, top-ranked chunks are included in the prompt context.
- Summarize or highlight key sentences within chunks to focus the LLM on the most relevant information.

**Post-Processing & Evaluation:**
- Implement LLM-as-a-judge evaluation pipelines to assess answer correctness, context utilization, and relevance.
- Automatically flag or reject answers that do not cite sources or that reference information outside the retrieved context.
- Use harmful content and answer conformity checks to further validate LLM outputs.

**Continuous Monitoring & Feedback:**
- Collect user feedback and monitor answer quality through observability dashboards (Grafana, Athena, CloudWatch).
- Continuously retrain and fine-tune retrieval and prompt strategies based on evaluation metrics and feedback.

**Security & Compliance:**
- Integrate PII detection (AWS Comprehend) to prevent the LLM from generating or exposing sensitive information.

These combined strategies help ensure that the LLM’s answers are accurate, contextually grounded, and compliant with enterprise standards, significantly reducing hallucinations in production RAG systems.

---

**Q: Would you fully replace the ML intent classifier with an LLM, or would you build a hybrid architecture?**

- I would recommend a hybrid architecture instead of a full replacement.
- Here’s why and how I would approach it:

**Reasons for Hybrid Approach:**
- Traditional ML intent classifiers (like SetFit, CatBoost, XGBoost) are highly optimized for speed, cost, and predictable performance, especially at high throughput (e.g., 250 TPS).
- LLMs offer better generalization, handle ambiguous or unseen queries, and support multilingual scenarios, but are more expensive and have higher latency.
- A hybrid system leverages the strengths of both: efficiency and reliability from ML models, flexibility and coverage from LLMs.

**Hybrid Architecture Design:**
- **Primary Layer:** Use the existing ML intent classifier for the majority of queries, ensuring fast and cost-effective predictions.
- **Fallback Layer:** For low-confidence predictions, ambiguous queries, or unsupported languages, route the request to an LLM-based intent detection module.
 - This can be implemented as a fallback mechanism, as described in the Knowledge GPT architecture (using UIDS as fallback level 1).
 - The LLM can also generate synthetic data to augment the ML model, improving its coverage over time.
- **Continuous Evaluation:** Monitor intent prediction accuracy and user feedback. Use LLM outputs to retrain and enhance the ML model periodically.
- **API Orchestration:** Expose both models behind a unified API, with logic to select the appropriate backend based on confidence scores or business rules.

**Benefits:**
- Maintains high throughput and low cost for standard queries.
- Handles edge cases, new intents, and multilingual scenarios with LLM flexibility.
- Enables gradual transition and continuous improvement without risking production stability.

This approach ensures business continuity, cost efficiency, and the ability to leverage the latest advancements in LLMs without sacrificing reliability.

---

**Q: How do you design or architect a fallback mechanism in an enterprise AI system?**

- A robust fallback mechanism ensures reliability, business continuity, and graceful handling of failures or low-confidence scenarios in AI systems.
- In my recent projects (like Knowledge GPT), I designed a multi-level fallback architecture, leveraging both ML and LLM components.

**Key Steps in Fallback Mechanism Design:**

- **1. Define Fallback Levels:** 
 - Level 0: Full RAG flow (default) — user query goes through search, context construction, and LLM generation with citations.
 - Level 1: Intent Fallback — if the system detects low confidence or ambiguity, route the query to an intent detection API (like UIDS). The predicted intent replaces the user query, and the RAG flow is retried with this reformulated query.
 - Level 2: Search Fallback — if LLM or intent detection fails, return top search results as plain text links (no LLM call).

- **2. Fallback Routing Logic:** 
 - Use validators (PII detection, injection detection, product extraction) to check input safety and quality.
 - Based on validation results and confidence scores, set the `fallback_level` (e.g., from JWT token or internal logic).
 - Route the request to the appropriate handler (full RAG, intent fallback, or search fallback).

- **3. Modular Handlers:** 
 - Implement separate handler modules for each fallback level:
 - Full RAG handler: Orchestrates search, prompt construction, LLM call, and response formatting.
 - Intent fallback handler: Calls the intent detection API, reformulates the query, and re-invokes the RAG flow.
 - Search fallback handler: Returns search results directly, bypassing LLM.

- **4. Token and API Management:** 
 - For intent fallback, securely manage access tokens (e.g., cache UIDS API tokens for efficiency).
 - Ensure all API calls are authenticated and monitored.

- **5. Observability and Logging:** 
 - Log all fallback events, reasons, and outcomes for monitoring and continuous improvement.
 - Use dashboards (Grafana, Athena, CloudWatch) to visualize fallback rates and system health.

- **6. Response Formatting:** 
 - Clearly indicate in the response if a fallback path was used (e.g., set `finish_reason` to "fallback_intent_solution").

**Benefits:**
- Ensures the system always returns a meaningful response, even if the primary model fails.
- Maintains user experience and trust by handling errors gracefully.
- Supports continuous learning by analyzing fallback cases for future improvements.

This modular, multi-level fallback design is proven in production and aligns with best practices for enterprise AI reliability.

---

**Q: How do you detect data drift versus concept drift when intent classification accuracy drops in production?**

- When intent classification accuracy drops in production, it’s important to distinguish between data drift (changes in input data distribution) and concept drift (changes in the relationship between input and output).
- Here’s how I approach detection for both:

**Data Drift Detection:**
- **Statistical Monitoring:** 
 - Continuously monitor input features (user queries, language, length, etc.) using statistical tests (e.g., KS test, Chi-square) to compare current production data with training data distributions.
 - Use tools like AWS SageMaker Model Monitor or custom scripts to automate drift detection.
- **Embedding Distribution Analysis:** 
 - For NLP, compare the distribution of embeddings (from models like SetFit or OpenAI) between training and live data.
 - Significant shifts in embedding clusters indicate data drift.
- **Input Data Logging:** 
 - Log all incoming queries and periodically sample them for manual or automated analysis.
 - Use scripts (like those in UIDS: `misclassified_intents.py`, `similarORduplicate_utterance.py`) to analyze new or unusual patterns.

**Concept Drift Detection:**
- **Performance Metrics Monitoring:** 
 - Track key metrics (accuracy, F1, precision, recall) over time for each intent class.
 - If metrics drop for specific intents but input data distribution remains stable, it suggests concept drift.
- **Misclassification Analysis:** 
 - Use misclassification reports to identify if the same queries are now being mapped to different intents.
 - Analyze confusion matrices to spot new or shifting intent boundaries.
- **Label Consistency Checks:** 
 - Periodically re-label a sample of production data using human annotators or LLMs-as-judges to see if the ground truth mapping has changed.

**Automation & CI/CD Integration:**
- Integrate drift detection scripts into CI/CD pipelines (as in UIDS, using Jenkins or Azure DevOps) for regular automated checks.
- Trigger retraining or model review if significant drift is detected.

**Summary:**
- Data drift: Detected by changes in input/query distributions.
- Concept drift: Detected by changes in model performance or intent mapping, even when input data is stable.

This dual monitoring ensures early detection and targeted remediation, maintaining high intent classification accuracy in production.

---

**Q: How do you prevent data leakage via prompts, and how do you handle audit logs and compliance in your AI system?**

- Preventing data leakage and ensuring compliance are critical in enterprise AI systems, especially when handling confidential data and user prompts.

**Preventing Data Leakage via Prompts:**
- **Input Validation & Filtering:**
 - Use validators (like AWS Comprehend and custom scripts) to detect and block PII or sensitive information in user prompts before they reach the LLM.
 - Enforce strict input validation rules (e.g., max length, allowed roles, content filters) to prevent injection attacks or accidental data exposure.
 - If Azure OpenAI content filter triggers on LLM output, the system returns a localized fallback message and sets `finish_reason="content_filter"` (as per `completions_payload_validator.py`).
- **Prompt Engineering:**
 - Design prompts to avoid echoing back sensitive user input or confidential context.
 - Use role-based access controls to restrict which users can access or query certain data.
- **Secrets Management:**
 - All credentials (OpenAI keys, database passwords, etc.) are securely stored in AWS Secrets Manager and never hardcoded in code or prompts.

**Audit Logs and Compliance Handling:**
- **Structured Logging:**
 - Every request and response is logged with a unique Correlation ID (using `context.py`), enabling full traceability across the system.
 - Logs include metadata but mask sensitive query content if `log_user_input="no"` or `log_input_allowed="no"` (see `log_input_output.py`).
- **Audit Logging:**
 - Separate logs for input payloads and LLM outputs, with masking applied as needed for compliance.
 - All logs are stored securely and can be audited for compliance or incident investigation.
- **Compliance & Localization:**
 - User-facing error messages are localized and managed centrally (using `messages_handler.py`), supporting compliance with language and accessibility requirements.
- **Access Control & Monitoring:**
 - System access is protected by enterprise authentication (OAuth, JWT validation) and web application firewalls (AWS WAF).
 - All API calls are authenticated, and access is logged for audit purposes.
- **Observability:**
 - Use centralized dashboards (Grafana, CloudWatch, Athena) for monitoring, alerting, and compliance reporting.
- **Retention & Governance:**
 - Log retention policies and audit trails are enforced as per enterprise and regulatory requirements.

This approach ensures that confidential data is protected at every stage, all access and actions are auditable, and the system remains compliant with enterprise security standards.

---

**Q: How do you debug and identify the cause when LLM costs suddenly triple over three months?**

- When LLM costs spike unexpectedly, I follow a structured debugging and monitoring approach to quickly identify the root cause and control expenses.

**Step-by-Step Debugging Process:**

- **1. Analyze Usage Metrics:**
 - Check API usage dashboards (OpenAI/Azure/AWS) for trends in request volume, token consumption, and endpoint usage.
 - Compare current metrics with historical data to pinpoint when and where the spike started.

- **2. Correlation ID Tracing:**
 - Use unique Correlation IDs (as implemented in the API architecture) to trace high-cost requests across logs and dashboards.
 - Identify which endpoints, users, or workflows are generating the most traffic or largest payloads.

- **3. Audit Log Review:**
 - Review audit logs for changes in request patterns, such as increased frequency, longer prompts, or larger context windows.
 - Check for new features, integrations, or user behaviors that may have increased LLM calls.

- **4. Prompt and Payload Analysis:**
 - Analyze the average and max prompt/response sizes. Look for cases where chunking or context construction is including too much data, leading to higher token usage.
 - Ensure input validation is still enforcing limits (e.g., max 2000 chars/query, max 10,000 chars/content).

- **5. Endpoint and Model Review:**
 - Check if more expensive models (e.g., GPT-4 instead of GPT-3.5) are being used more frequently.
 - Review if fallback mechanisms are triggering excessive LLM calls due to upstream failures.

- **6. User and Feature Attribution:**
 - Attribute costs to specific users, teams, or features using metadata in logs.
 - Identify any misuse, abuse, or unintentional high-frequency usage.

- **7. Observability Dashboards:**
 - Use centralized dashboards (Grafana, Athena, CloudWatch) to visualize cost drivers, request patterns, and anomalies.

**Cost Control Actions:**
- Set up alerts for abnormal usage or cost thresholds.
- Enforce stricter input validation and chunking strategies.
- Optimize prompt engineering to minimize unnecessary context.
- Limit access to expensive endpoints or models.
- Review and optimize fallback logic to avoid redundant LLM calls.

This systematic approach ensures rapid identification of cost drivers and enables targeted optimizations to bring LLM expenses back under control.

---

**Q: How do you define and justify the accuracy level of your LLM to a customer?**

- I define and justify LLM accuracy using a combination of automated evaluation metrics and human/LLM-as-judge validation, tailored to the use case.

**How I Define LLM Accuracy:**
- For intent classification (UIDS), I use standard classification metrics:
 - **Accuracy**: Percentage of correct predictions over total samples.
 - **F1 Score (micro, macro, weighted)**: Balances precision and recall, especially important for imbalanced intent classes.
 - **Precision & Recall**: Measures correctness and completeness for each intent.
 - These are calculated using scripts (see `calculate_metrics` in UIDS) and validated on held-out test sets and real production data.

- For generative QA or RAG systems (Knowledge GPT), I use a multi-metric evaluation pipeline:
 - **Answer Correctness**: Combines factual accuracy (LLM-as-judge, e.g., GPT-4o), semantic similarity, and reference health.
 - Example: `answer_correctness_score = (factual_accuracy_score × 0.7) + (semantic_similarity_score × 0.2) + (reference_health_score × 0.1)`
 - **Context Relevance**: Measures if the answer is grounded in the retrieved context.
 - **Automated LLM Evaluation**: Prompts LLMs to score answers against ground truth using strict JSON outputs for consistency.
 - **Human Review**: Periodic manual audits or LLM-as-judge reviews for critical use cases.

**How I Justify Accuracy to Customers:**
- I present detailed evaluation reports showing:
 - Test set metrics (accuracy, F1, etc.) for intent classification.
 - LLM evaluation scores for generative answers, with breakdowns by sub-metric (factual, semantic, reference).
 - Examples of correct and incorrect predictions, confusion matrices, and error analysis.
- I explain the evaluation process, including:
 - Use of held-out data, real-world queries, and continuous monitoring.
 - Automated pipelines for regular re-evaluation and drift detection.
- I ensure transparency by sharing the methodology, sample outputs, and how metrics are calculated and validated.

This approach gives customers confidence in the LLM’s performance and a clear understanding of how accuracy is measured and maintained.

---
