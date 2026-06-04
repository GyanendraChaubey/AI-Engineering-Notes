# Generative AI Engineer (Part 1) — Interview 29

**Q: How is MCP (Model Context Protocol) being used in a production RAG system project?**

- In the Knowledge GPT (KGPT) project, MCP (Model Context Protocol) is implemented as a unified, AI-native interface layer that enables seamless integration between LLMs (like GPT) and various enterprise tools, data sources, and workflows.
- The core value of MCP is to move beyond traditional REST APIs, allowing AI agents to autonomously discover, select, and invoke enterprise tools based on user intent, rather than relying on hardcoded logic or manual integration.
- MCP acts as a standardized protocol—similar to a “USB-C port” for AI—enabling LLMs to reason about which tool or data source to use, invoke it dynamically, and synthesize responses for the user.
- In KGPT, MCP servers expose capabilities such as enterprise knowledge search (kgpt_search) and intent detection (UIDS) as self-describing tools. The LLM can read tool schemas, maintain conversation context, and autonomously decide which tool to call to fulfill user queries.
- The MCP implementation in KGPT is containerized (Python-based, running in Docker), with Chainlit as the frontend chat UI and Apigee as the API gateway for authentication and rate limiting.
- Authentication is managed via OAuth 2.0, with credentials stored securely in AWS Secrets Manager.
- This architecture enables rapid integration of new tools (minutes instead of weeks), consistent security and governance, and improved observability of AI usage patterns.
- The result is a more flexible, scalable, and context-aware AI assistant that can reason across multiple enterprise systems, reducing duplicated effort and compliance risks.

**Industry Perspective:**
- MCP enables AI-native workflows where LLMs act as autonomous agents, orchestrating complex tasks across siloed systems—something not possible with traditional stateless APIs.
- This approach is critical for enterprise environments where extensibility, security, and rapid integration are key requirements.

**Example Flow:**
- User asks a question in the chat UI.
- LLM reasons about the query and determines it needs to fetch knowledge from an internal repository.
- LLM uses MCP to discover and invoke the kgpt_search tool.
- Results are retrieved, synthesized, and presented to the user—all within a single, context-aware AI workflow.

**Summary:** 
MCP in KGPT transforms the AI assistant from a simple query interface into an intelligent, autonomous agent capable of dynamic tool selection, contextual reasoning, and seamless integration with enterprise knowledge systems.

---


**1. Model Output Evaluation**

- **Automated Evaluation Pipeline**: 
 - A standalone Python application sends real user questions (from a ground truth dataset) to the KGPT system (via Completions or Search APIs).
 - Responses are collected and scored using a combination of LLM-based and algorithmic evaluators.

- **Key Metrics & Evaluators**:
 - **Answer Correctness**: 
 - Uses GPT-4o to judge factual accuracy and Azure OpenAI Embeddings for semantic similarity to ground truth.
 - Also checks if cited URLs are alive using HTTP HEAD requests.
 - **Answer Relevance**: 
 - Evaluates if the answer directly addresses the user’s question.
 - **Context Utilization**: 
 - Measures what fraction of retrieved context was actually used in the answer (using chunk alignment scoring via GPT-4o).
 - **Mean Reciprocal Rank (MRR)**: 
 - Pure Python computation to rank search results against ground truth document IDs.
 - **Harmful Content Detection**: 
 - Uses GPT-4o to score for harmful or inappropriate content, computing composite scores like TMI and BSR.
 - **Reporting**: 
 - Generates multi-sheet Excel reports, metric aggregation charts, and cross-release comparison reports, all uploaded to S3 for traceability.

- **Pipeline Flow**:
 - Loads datasets from S3 (CSV/JSON).
 - Instantiates KPI evaluators and orchestrates batch evaluations.
 - Aggregates results and generates reports for stakeholders.

---

**2. System Monitoring & Observability**

- **Three Pillars of Observability**:
 - **Logging**: 
 - Centralized log aggregation using CloudWatch Logs and Fluent Bit (EKS), with logs in JSON format including correlation IDs for traceability.
 - Captures application events (gateway requests, adapter calls), infrastructure events (pod restarts, DNS), and security events (auth attempts, denials).
 - **Metrics**: 
 - Real-time metrics collection via CloudWatch Metrics, Athena, and Grafana dashboards.
 - Tracks request rates, latency (p50/p95/p99), error rates, backend invocation counts, and resource utilization.
 - **Distributed Tracing**: 
 - End-to-end request tracing using CloudWatch X-Ray and correlation ID propagation.
 - Every request is tagged with a unique correlation ID from the API gateway (Apigee) through all system layers, enabling precise root cause analysis and performance monitoring.

- **Alerting & Audit**:
 - Automated alerting via SNS, Slack, and PagerDuty for critical failures or anomalies.
 - Audit logging with CloudTrail for compliance and security reviews.

---

**Industry Best Practices & Mathematical Intuition**:

- **Evaluation**: 
 - Combines LLM-based subjective scoring (for nuanced metrics like relevance and harmfulness) with objective algorithmic metrics (like MRR and semantic similarity).
 - Example: Semantic similarity is computed using cosine similarity between embedding vectors, providing a quantitative measure of answer closeness to ground truth.
- **Monitoring**: 
 - Uses distributed tracing and correlation IDs to ensure every user request can be tracked across microservices, which is critical for debugging and SLA compliance in production AI systems.
 - Real-time dashboards and alerting ensure rapid response to incidents, minimizing downtime and maintaining trust in the AI assistant.

---

**Summary**:
- The KGPT project employs a robust, multi-layered evaluation and monitoring framework that combines advanced LLM-based assessment, quantitative metrics, and enterprise-grade observability to ensure high-quality, reliable, and safe AI-driven knowledge retrieval at scale.

---

**Q: What are the limitations of vector-based systems in AI applications?**

- Vector-based systems, especially those used for semantic search and retrieval (like in RAG pipelines), are powerful for capturing conceptual similarity, but they do have several limitations that are important to consider in practical, enterprise-scale AI deployments.

---

**Key Limitations:**

- **Loss of Fine-Grained Context:**
 - Embeddings compress complex documents or queries into fixed-length vectors (e.g., 1536-dim), which can lead to loss of nuanced information, especially for long or multi-topic documents.
 - Example: Two documents about "Python" (the language vs. the snake) may have similar embeddings if context is not well captured.

- **Semantic Drift and Ambiguity:**
 - Vectors may capture general semantic similarity but struggle with domain-specific meanings, polysemy, or context-dependent queries.
 - Example: "Apple" as a fruit vs. "Apple" as a company.

- **Recall vs. Precision Trade-off:**
 - KNN-based retrieval can surface semantically similar but contextually irrelevant chunks, leading to lower precision.
 - Lexical search (BM25) is better for exact matches (e.g., error codes), while vector search may miss these unless hybrid approaches are used.

- **Scalability and Latency:**
 - High-dimensional vector search (KNN) can be computationally expensive, especially as the corpus grows, impacting latency and requiring specialized infrastructure (e.g., OpenSearch, FAISS, or Pinecone).
 - Bulk ingestion and real-time retrieval require careful engineering (parallelization, sharding, caching).

- **Explainability and Debugging:**
 - It is often difficult to interpret why a particular chunk was retrieved, as vector similarity is not inherently explainable to end-users or stakeholders.
 - This can be a challenge for compliance and auditability in regulated industries.

- **Cold Start and Data Drift:**
 - New documents or concepts may not be well-represented until sufficient examples are embedded and indexed.
 - Embedding models may become stale as language or domain knowledge evolves, requiring periodic retraining.

- **Security and Privacy:**
 - Embeddings can sometimes leak sensitive information if not properly anonymized or if the embedding model is not robust to adversarial inputs.

---

**Industry Best Practices to Mitigate Limitations:**

- **Hybrid Search Architectures:**
 - Combine lexical (BM25) and vector (KNN) search using Reciprocal Rank Fusion (RRF) or linear blending to balance precision and recall (as implemented in KGPT).
 - Example: Run both searches in parallel, then fuse results for best overall relevance.

- **Prompt Engineering and Context Windows:**
 - Carefully design chunking and context windows to maximize relevant information in each vector.
 - Use metadata (titles, source, language) to enrich retrieval and ranking.

- **Continuous Evaluation and Monitoring:**
 - Regularly evaluate retrieval quality using metrics like MRR, answer correctness, and context utilization.
 - Monitor for drift and retrain embedding models as needed.

- **Explainability Layers:**
 - Provide traceability by logging which chunks were retrieved and why, and expose this in user-facing explanations.

---

**Summary:** 
While vector-based systems are foundational for modern semantic search and generative AI, they have inherent limitations around context loss, ambiguity, scalability, and explainability. Industry-standard solutions involve hybrid retrieval, robust evaluation, and continuous monitoring to ensure high-quality, reliable AI applications.

---

**Q: How do you handle unclear requirements and decision-making for architecture and implementation in evolving AI/GenAI projects, especially when success criteria and technology choices are not well-defined by the client?**

- In enterprise AI and GenAI projects—especially those involving new paradigms like RAG, agentic workflows, or MCP—the requirements and success criteria are often ambiguous at the outset. Unlike traditional software projects, clients may not have a clear vision of the technical solution or even the evaluation metrics.
- My approach is to adopt a collaborative, iterative, and consultative process that balances client input with technical leadership and rapid prototyping.

---

**Process Overview:**

- **1. Stakeholder Workshops & Requirement Elicitation**
 - Begin with deep-dive workshops with business stakeholders, product owners, and end-users to understand the core business problem, pain points, and desired outcomes.
 - Use techniques like user journey mapping, problem framing, and “jobs to be done” analysis to clarify the real needs behind vague requirements.

- **2. Hypothesis Formulation & Success Criteria Definition**
 - Translate business objectives into technical hypotheses (e.g., “Can we reduce support ticket resolution time by 30% using a knowledge assistant?”).
 - Define measurable KPIs collaboratively—such as answer accuracy, latency, user satisfaction, or cost per query—even if they are initially high-level.
 - Document these in a living requirements spec that evolves as the project matures.

- **3. Solution Architecture & Technology Selection**
 - Present multiple architecture options (e.g., pure vector search, hybrid search, agentic orchestration) with trade-offs in scalability, explainability, and cost.
 - Leverage reference architectures (like those in the MCP design doc) and platform risk assessments to guide technology choices (e.g., AWS vs. Azure vs. custom).
 - Make recommendations based on technical fit, enterprise constraints (security, compliance), and future extensibility.

- **4. Rapid Prototyping & Lean Experimentation**
 - Build quick POCs or MVPs to validate assumptions and gather early feedback.
 - Use modular, configurable pipelines so components (retrievers, LLMs, evaluators) can be swapped or tuned as requirements evolve.
 - Share demo environments or sandbox UIs with clients for hands-on validation.

- **5. Iterative Feedback & Decision-Making**
 - Maintain regular syncs with clients to review progress, demo features, and refine requirements.
 - Push back constructively when client asks are technically infeasible or misaligned with best practices—always providing rationale and alternative solutions.
 - Document all decisions, trade-offs, and open questions in shared artifacts (e.g., Confluence, design docs).

- **6. Evaluation & Success Criteria Refinement**
 - As the system matures, refine KPIs and evaluation pipelines (e.g., add new metrics, automate LLM-based evaluation, or introduce user feedback loops).
 - Use data-driven insights from pilot deployments to adjust priorities and roadmap.

- **7. Governance, Security, and Compliance**
 - Ensure all architectural decisions align with enterprise governance, security, and audit requirements (as highlighted in the MCP design doc).
 - Involve security and infra teams early, especially for cloud, data privacy, and integration concerns.

---

**Decision-Making Balance:**

- **Client-Driven:** 
 - Business goals, high-level priorities, and non-negotiable constraints (e.g., data residency, compliance).
- **AI Architect/Team-Driven:** 
 - Technical architecture, technology stack, evaluation methodology, and implementation details.
 - Final say on best practices, scalability, and maintainability.

---

**Summary:** 
In rapidly evolving AI projects, I drive the architecture and implementation process by combining structured stakeholder engagement, hypothesis-driven development, rapid prototyping, and transparent decision-making. This ensures that even with ambiguous requirements, we deliver solutions that are technically robust, aligned with business goals, and adaptable as the landscape evolves.

---
