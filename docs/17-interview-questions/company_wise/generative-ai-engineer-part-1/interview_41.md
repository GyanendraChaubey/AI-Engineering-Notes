# Generative AI Engineer (Part 1) — Interview 41

**Q: What are 3–4 effective ways to reduce hallucinations in LLM-based RAG systems?**

- **1. Improve Retrieval Quality and Contextual Relevance**
 - Enhance the semantic search pipeline by optimizing chunking strategies (e.g., logical section-based splits, merging small chunks) to ensure only highly relevant, context-rich information is retrieved for the LLM.
 - Use advanced retrieval techniques like Reciprocal Rank Fusion (RRF) to combine lexical (BM25) and vector (embedding-based) search, increasing the likelihood that the most contextually appropriate chunks are surfaced.

- **2. Prompt Engineering and Context Framing**
 - Design prompts that explicitly instruct the LLM to answer only using the provided context and to respond with “I don’t know” or a fallback message if the answer is not present in the retrieved content.
 - Use system-level instructions to reinforce boundaries and discourage the model from generating unsupported information.

- **3. Post-Processing and Output Validation**
 - Implement answer validation pipelines using LLM-as-a-judge frameworks (as in your custom evaluation pipeline) to assess factual accuracy, semantic similarity, and context utilization before returning the answer to the user.
 - Assign higher weights to factual accuracy and context alignment in the scoring formula to filter out hallucinated or unsupported responses.

- **4. Retrieval Augmentation and Negative Sampling**
 - During training or fine-tuning, include negative samples (irrelevant or misleading context) and reinforce correct behavior by penalizing hallucinated outputs.
 - Continuously monitor and retrain the retrieval and generation components using real-world feedback and error analysis to iteratively reduce hallucination rates.

These strategies, when combined, significantly reduce hallucinations by ensuring the LLM operates within well-defined, contextually grounded boundaries and by validating outputs before exposure to end users.

---

**Q: Provide a sample code structure and main components for a graph-based pipeline application (e.g., using LangGraph).**

- For a graph-based pipeline application (such as those built with LangGraph), the architecture is modular, where each processing step is a node in a state machine, and edges define the execution flow.
- This approach is highly extensible, supports conditional logic, and is ideal for orchestrating complex AI/ML workflows (like RAG pipelines, data labeling, or inference pipelines).
- The main components typically include:
 - **Entry Point (main.py):** Parses CLI arguments, sets up environment/config, and triggers the pipeline.
 - **Pipeline Orchestrator (full.py):** Defines the state graph, adds nodes (steps), and manages execution order and branching.
 - **Individual Step Modules:** Each node (e.g., data preparation, translation, indexing, inference, evaluation) is implemented as a separate module/function.
 - **Config & Logging:** Centralized configuration (via .env or environment variables) and logging for monitoring and debugging.
 - **Data Management:** Handles input/output (local or S3), intermediate artifacts, and results.
 - **Extensibility:** Easy to add/replace nodes for new steps or models.

**Sample Code Structure:**

```python
# main.py
# Entry point for the pipeline
import os
from pipelines.full import run_pipeline

if __name__ == "__main__":
 # Parse CLI arguments and set environment variables
 # Call the pipeline orchestrator
 run_pipeline()

# pipelines/full.py
# Orchestrates the pipeline using a state graph (LangGraph)
from langgraph import StateGraph
from pipelines.steps import (
 language_translation_graph,
 data_prep_graph,
 indexing_graph,
 inference_graph,
 evaluation_graph,
 model_prediction_graph
)

def run_pipeline():
 # Define the state schema (shared state across nodes)
 workflow = StateGraph(state_schema=State)
 
 # Add nodes (steps)
 workflow.add_node("language_translation", language_translation_graph)
 workflow.add_node("prepare_dataset", data_prep_graph)
 workflow.add_node("indexing", indexing_graph)
 workflow.add_node("inference", inference_graph)
 workflow.add_node("evaluation", evaluation_graph)
 workflow.add_node("model_prediction", model_prediction_graph)
 
 # Define conditional edges (branching logic)
 workflow.add_conditional_edges("START", lambda state: ..., ["language_translation", "prepare_dataset"])
 workflow.add_edge("language_translation", "prepare_dataset")
 workflow.add_edge("prepare_dataset", "indexing")
 workflow.add_edge("indexing", "inference")
 workflow.add_edge("inference", "evaluation")
 workflow.add_conditional_edges("evaluation", lambda state: ..., ["model_prediction", "END"])
 workflow.add_edge("model_prediction", "END")
 
 # Run the workflow
 workflow.run(initial_state={})

# pipelines/steps.py
# Individual step implementations (each as a function or class)
def language_translation_graph(state):
 # Translate data if required
 return state

def data_prep_graph(state):
 # Load, clean, and prepare data
 return state

def indexing_graph(state):
 # Index data into vector store
 return state

def inference_graph(state):
 # Run inference (LLM or classifier)
 return state

def evaluation_graph(state):
 # Evaluate results, compute metrics
 return state

def model_prediction_graph(state):
 # Optional: run model-based prediction
 return state

# config.py
# Centralized configuration management

# log_manager.py
# Centralized logging for debugging and monitoring
```

- **Data & Artifacts:** Input/output handled via local or S3, with intermediate artifacts (cleaned data, embeddings, metrics) managed for traceability.
- **Extensibility:** New steps can be added as nodes; supports both local and cloud workflows; LLM/vector store integrations are abstracted for flexibility.

This modular, graph-based structure ensures maintainability, scalability, and ease of extension for complex AI pipelines.

---

**Q: Python code to find the first non-repeating character in a string.**

- The optimal approach is to use a dictionary to count character frequencies in a single pass, then iterate again to find the first character with a count of 1.
- This ensures O(n) time complexity and is easy to explain in interviews.

**🔑 Key Steps**:
- Traverse the string and count occurrences of each character.
- Traverse the string again and return the first character with a count of 1.
- If no such character exists, return a suitable message or value.

**💻 Code**:
```python
def first_non_repeating_char(s):
 # Create an empty dictionary to store character counts
 char_count = {}
 # Iterate over each character in the string
 for char in s:
 # Increment the count for each character
 char_count[char] = char_count.get(char, 0) + 1
 # Iterate over the string again to find the first non-repeating character
 for char in s:
 # Check if the character count is 1
 if char_count[char] == 1:
 # Return the first non-repeating character
 return char
 # If no non-repeating character is found, return None or a message
 return None

# Example usage
input_str = "aabbcdeff"
result = first_non_repeating_char(input_str)
print(result) # Output: 'c'
```

**💡 Explanation**:
- The function uses a dictionary to count the frequency of each character in the string.
- The first loop populates the dictionary with counts.
- The second loop checks the original order to find the first character with a count of 1.
- Time complexity is O(n), where n is the length of the string.
- Space complexity is O(1) if the character set is fixed (e.g., ASCII), otherwise O(k) where k is the number of unique characters.
- This approach is efficient and commonly used in interviews for such problems.

---

**Q: What threshold did you set for harmful content scores to allow or block LLM responses?**

- For harmful content evaluation, we used a structured scoring system where each LLM-generated answer was evaluated by GPT-4o and assigned six sub-scores: `harmful_score`, `malicious_score`, `bias_score`, `racism_score`, `vulgarity_score`, and `discrimination_score`, each ranging from 0.0 (safe) to 1.0 (completely harmful).
- Composite indices were also computed:
 - **TMI (Threat & Malicious Index):** Average of `harmful_score` and `malicious_score`
 - **BSR (Bias, Safety & Respect Index):** Average of `bias_score`, `racism_score`, `vulgarity_score`, and `discrimination_score`
 - **Overall Harmful Score:** Maximum of all sub-scores

**Threshold Policy:**
- The **blocking threshold** was set at **0.5** for any individual sub-score or composite score.
 - If any sub-score or the overall harmful score was **≥ 0.5**, the response was considered unsafe and blocked.
 - In such cases, a safe fallback message was returned to the user instead of the original LLM output.
- This threshold was chosen to balance strict safety with minimizing false positives, ensuring that any content with moderate or higher risk was filtered out.

- All evaluation results, including scores and block/pass decisions, were logged and included in detailed reports for compliance and continuous monitoring. If the LLM evaluator failed to return a parseable response, the system defaulted to marking the content as safe (fail-open), but the raw response was logged for further review.

---

**Q: How would you design an MCP-based system to allow multiple tools to integrate together?**

- I would architect the MCP (Model Context Protocol) system using a layered, modular approach to enable seamless, secure, and scalable integration of multiple tools and enterprise systems.
- The design would follow the principles of separation of concerns, loose coupling, unified identity, and observability, as outlined in the MCP design documentation.
- The goal is to provide a standardized, protocol-agnostic interface (like a USB-C port for AI) so that AI models and tools can interact with diverse backend systems in a unified, extensible manner.

**Key Steps:**

- **Layered Architecture:**
 - **Client Layer:** Consumers of MCP (e.g., AI assistants, desktop apps, custom integrations).
 - **Gateway Layer:** Handles traffic management, authentication, and security (using Apigee/API Gateway/MCP Proxy).
 - **Service Layer:** Provides shared services such as registry, identity management, discovery, and observability.
 - **MCP Server Layer:** Each tool or backend system is wrapped by its own MCP server, exposing its capabilities via standardized MCP contracts.
 - **Backend Layer:** Actual enterprise systems, APIs, and databases.

- **Unified Identity & Security:**
 - Implement enterprise SSO using OAuth2/OIDC for unified authentication.
 - Enforce zero-trust security with mTLS and token validation at every layer.
 - Each request carries a unified MCP identity for consistent authorization and audit.

- **Standardized Contracts:**
 - Define OpenAPI and MCP Tool Schemas for all tool integrations, ensuring consistent API contracts and discoverability.
 - Each MCP server exposes its tool’s capabilities via these contracts, making onboarding and integration straightforward.

- **Loose Coupling & Scalability:**
 - Each MCP server is independently deployable and can be updated or scaled without impacting others.
 - Communication is asynchronous where possible, supporting multiple transport protocols (HTTP, WebSocket, SSE, stdio).

- **Observability & Governance:**
 - Built-in distributed tracing and structured logging across all layers for full visibility and auditability.
 - Centralized registry and discovery services to manage available tools and integrations.

- **Integration Flow Example:**
 - A client (e.g., AI assistant) sends a request to the Gateway with a valid token.
 - The Gateway authenticates and routes the request to the appropriate MCP server based on the tool/resource needed.
 - The MCP server translates the request into the backend system’s native format, executes the operation, and returns the result via the standardized MCP response.
 - All interactions are logged and traced for observability.

- **Extensibility:**
 - New tools can be integrated by developing new MCP servers that conform to the standard contracts, without changing the core platform.
 - Protocol abstraction ensures future-proofing for new communication patterns.

This architecture ensures secure, scalable, and maintainable integration of multiple tools, with strong governance and observability, as required for enterprise AI ecosystems.

---

**Q: What type of rate limiting parameters would you use in your MCP-based system design?**

- For an MCP-based system, robust rate limiting is essential to ensure fair usage, protect backend resources, and maintain system stability under load. I would implement rate limiting at multiple layers, primarily at the API Gateway (e.g., APIGEE) and optionally at the MCP Gateway or individual MCP servers for fine-grained control.

**Key Rate Limiting Parameters:**

- **Per-User/Per-Token Limits:** 
 - Limit the number of requests per user or per JWT token within a defined time window (e.g., 1000 requests/hour/user).
 - Enforced using the user ID or token subject claim extracted during authentication.

- **Per-API/Endpoint Limits:** 
 - Set different limits for different API endpoints or tool integrations (e.g., stricter limits for resource-intensive tools).
 - Example: `/knowledge-gpt/search` might allow 500 requests/hour, while `/mcp/tools/execute` allows 200/hour.

- **Burst Limits (Short-Term Spike Control):** 
 - Allow short bursts above the steady-state rate (e.g., 20 requests/second for up to 10 seconds), then enforce the average rate.
 - Protects against sudden spikes while maintaining user experience.

- **Global/System-Wide Limits:** 
 - Cap the total number of requests across all users to prevent system overload (e.g., 10,000 requests/minute/system-wide).

- **IP-Based Limits:** 
 - Optionally limit requests per client IP to mitigate abuse or DDoS attacks.

- **Role/Plan-Based Limits:** 
 - Different rate limits for different user roles or subscription plans (e.g., higher limits for admins or premium users).

- **Custom Headers for Quota Feedback:** 
 - Return headers like `X-RateLimit-Limit`, `X-RateLimit-Remaining`, and `X-RateLimit-Reset` in API responses to inform clients of their usage and reset times.

**Implementation Notes (from MCP Design):**
- APIGEE is used as the primary entry point for traffic management, enforcing rate limits and quotas based on token identity and context claims.
- All rate limiting decisions and denials are logged with correlation IDs for traceability (as per distributed tracing best practices).
- Metrics such as request rate, error rate, and quota denials are tracked in CloudWatch and visualized in Grafana for operational monitoring.

This multi-dimensional rate limiting approach ensures fairness, protects critical resources, and provides transparency to both users and operators.

**interviewee**: So rate limiting means how many request, uh, we are passing to the API means, uh, per user per token limits. We can limit means, uh, I can say the limit, the number of requests per user, uh, or like the JWT token itself defined a kind of window 100, uh, request per hour or maybe per user, I can say we can, uh, we can enforce like that or, uh, uh, what if I have a scenario that I have a centralized system where there are multiple clients that will integrate and every client has a particular rate limit. So, and it's getting crashed, but everybody wants, so what can be possible solutions to, you know, increase rate limiting for that particular second or that particular hit? What can be multiple design will possible solutions.

---

**Q: What are possible solutions to handle rate limiting when multiple clients with different limits cause a centralized system to hit its rate cap or crash, and everyone wants higher throughput?**

- When a centralized system faces rate limiting pressure from multiple clients—each with their own quotas—but the aggregate demand exceeds system capacity, several strategies can be used to increase throughput or manage load more gracefully:

**Possible Solutions:**

- **Dynamic/Burst Rate Limiting:**
 - Implement burstable quotas, allowing temporary spikes above the steady-state rate (e.g., token bucket or leaky bucket algorithms).
 - Clients can exceed their base rate for short periods, as long as the overall average remains within limits.

- **Priority-Based Throttling:**
 - Assign priorities to clients or request types (e.g., premium clients get higher limits).
 - During high load, lower-priority requests are throttled first, ensuring critical clients maintain service.

- **Adaptive Rate Limiting:**
 - Use real-time system metrics (CPU, memory, backend latency) to dynamically adjust rate limits.
 - If the system is underutilized, temporarily increase limits; if overloaded, reduce them.

- **Quota Borrowing/Pooling:**
 - Allow clients to borrow unused quota from others in the same organization or pool.
 - Implement a shared quota pool with per-client soft and hard limits.

- **Queueing and Backoff:**
 - Introduce a request queue with fair scheduling (e.g., weighted round robin).
 - When limits are hit, queue excess requests or return a `429 Too Many Requests` with a `Retry-After` header, encouraging clients to back off and retry later.

- **Horizontal Scaling:**
 - Scale out backend resources (e.g., add more MCP servers, increase container replicas in EKS/Kubernetes) to handle higher aggregate throughput.
 - Use auto-scaling policies based on request rate and latency metrics (monitored via CloudWatch/Grafana).

- **Distributed Rate Limiting:**
 - Use a distributed cache (e.g., Redis, DynamoDB) to coordinate rate limits across multiple API gateway instances, ensuring global consistency.

- **API Gateway Configuration:**
 - Configure APIGEE or API Gateway to enforce both per-client and global system-wide limits.
 - Use correlation IDs and distributed tracing to monitor which clients are causing spikes and adjust policies accordingly.

- **Graceful Degradation:**
 - When limits are exceeded, degrade non-essential features or return cached/stale responses to maintain core functionality.

**Industry Practice (from MCP Design):**
- APIGEE is used as the unified entry point for traffic management, enforcing both per-client and global rate limits.
- Metrics and distributed tracing (CloudWatch, Grafana) are used to monitor and adjust limits in real time.
- Horizontal scaling and burstable quotas are standard for handling temporary spikes.

This combination of adaptive, priority-aware, and scalable rate limiting ensures fair access, protects system stability, and allows for controlled increases in throughput during peak demand.

---

**Q: How do you approach fine-tuning a model, and what techniques do you use?**

- My approach to fine-tuning depends on the model type (e.g., transformer-based LLM, classical ML, or deep learning) and the specific use case (classification, intent detection, RAG, etc.). I follow a structured, modular pipeline to ensure reproducibility, scalability, and optimal performance.

**General Fine-Tuning Workflow:**

- **1. Data Preparation:**
 - Collect and clean labeled data relevant to the target task (e.g., intent classification, domain-specific Q&A).
 - Perform data augmentation if needed (e.g., synthetic data generation, translation).
 - Encode labels using a label encoder for classification tasks.

- **2. Model Selection:**
 - Choose a base model suitable for the task:
 - For intent classification: SetFit (Sentence Transformers), CatBoost, or Falcon-7B (for LLM-based approaches).
 - For RAG or LLM tasks: Start with a pre-trained transformer (e.g., GPT, Falcon, BERT).

- **3. Fine-Tuning Techniques:**
 - **SetFit (Sentence Transformers):**
 - Use contrastive learning with few-shot labeled examples.
 - Fine-tune both the transformer backbone and the classification head.
 - Efficient for small/medium datasets and intent classification.
 - **CatBoost/XGBoost:**
 - Use transformer embeddings as features.
 - Train a gradient boosting classifier on top of embeddings.
 - Useful for tabular or structured data with text features.
 - **LLM Fine-Tuning (e.g., Falcon-7B):**
 - Use parameter-efficient fine-tuning (PEFT) techniques like QLoRA or LoRA to adapt large models with limited compute.
 - Apply supervised fine-tuning with domain-specific prompts and responses.
 - Use bitsandbytes for memory-efficient training.
 - **Prompt Engineering:**
 - For LLMs, iteratively refine prompts and use few-shot examples to guide model behavior without full retraining.

- **4. Training Pipeline:**
 - Use modular, cloud-native pipelines (e.g., AWS SageMaker, LangGraph state machines) for orchestration.
 - Track experiments, hyperparameters, and metrics using MLFlow or similar tools.
 - Save models, encoders, and artifacts to S3 or a model registry for reproducibility.

- **5. Evaluation & Monitoring:**
 - Evaluate on held-out test data using accuracy, F1, confusion matrix, etc.
 - Analyze misclassified samples and iterate on data or model as needed.
 - Log metrics and inference results for monitoring and future audits.

- **6. Deployment:**
 - Package the fine-tuned model as a REST API (e.g., FastAPI, SageMaker endpoint).
 - Integrate with downstream applications and monitor real-world performance.

**Project Examples from My Experience:**
- In the UIDS project, I fine-tuned SetFit for intent classification, used CatBoost with transformer embeddings, and experimented with Falcon-7B using QLoRA for LLM-based tasks.
- The pipeline was modular (each step as a node in a LangGraph state machine), cloud-native (S3 integration), and supported both local and cloud workflows.
- All steps were logged and tracked for reproducibility and auditability.

This approach ensures that fine-tuning is efficient, scalable, and aligned with enterprise best practices.

---

**Q: What are the multiple techniques available for fine-tuning, such as quantization, and how are they used?**

- There are several techniques for fine-tuning models, especially large language models (LLMs) and transformer-based architectures. The choice depends on the use case, data size, and available compute resources.

**Key Fine-Tuning Techniques:**

- **Full Fine-Tuning:**
 - Update all model parameters using your labeled dataset.
 - Requires significant compute and memory, typically used when you have a large, domain-specific dataset.

- **Parameter-Efficient Fine-Tuning (PEFT):**
 - Techniques like LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) update only a small subset of parameters or add trainable adapters.
 - QLoRA combines quantization (reducing model precision to 4-bit/8-bit) with LoRA adapters, making it feasible to fine-tune large models like Falcon-7B on standard GPUs.
 - This approach is memory-efficient and effective for domain adaptation.

- **Quantization:**
 - Reduces the precision of model weights (e.g., from 16-bit/32-bit to 8-bit or 4-bit).
 - Used both for efficient inference and as part of QLoRA for efficient fine-tuning.
 - Tools like `bitsandbytes` enable quantized training and inference.

- **Prompt Tuning / Prompt Engineering:**
 - Instead of updating model weights, optimize the prompts or add trainable prompt tokens.
 - Useful for adapting LLMs to new tasks with minimal compute.

- **Feature Extraction + Classical ML:**
 - Use transformer models to generate embeddings, then train a lightweight classifier (e.g., CatBoost, XGBoost) on top.
 - Example: In the UIDS project, SentenceTransformer embeddings were used with CatBoost for intent classification.

- **Contrastive Learning (SetFit):**
 - Fine-tune sentence transformers using contrastive loss with few-shot labeled pairs.
 - Efficient for intent classification and tasks with limited data.

**Project Examples:**
- In the UIDS project, we used:
 - **SetFit** for few-shot intent classification (contrastive learning).
 - **CatBoost** with transformer embeddings for structured classification.
 - **QLoRA** for experimental fine-tuning of Falcon-7B, leveraging quantization and PEFT for resource efficiency.

- The pipeline is modular and supports switching between these techniques based on the task and resource constraints.

These techniques allow for flexible, scalable, and efficient fine-tuning of models for various enterprise AI applications.

---

**Q: What experience do you have working with Kubernetes?**

- I have hands-on experience deploying, managing, and scaling AI workloads on Kubernetes, particularly in the context of enterprise AI systems and MCP-based architectures.
- My work has involved both infrastructure setup and application-level orchestration for scalable, secure, and observable AI services.

**Key Areas of Kubernetes Experience:**

- **AI/ML Workload Deployment:**
 - Deployed MCP adapters and AI microservices as containerized workloads on Kubernetes clusters (EKS).
 - Used Helm charts and YAML manifests to define deployments, services, and config maps for various AI tools and backend adapters.
 - Configured Horizontal Pod Autoscalers (HPA) to scale MCP adapter pods based on CPU/memory utilization (e.g., 2–10 pods per adapter as per MCP design).
 - Implemented health checks (`/health`, `/ready`) for liveness and readiness probes to ensure robust service orchestration.

- **Resource Management & Security:**
 - Managed resource requests and limits (CPU, memory) for each pod to optimize cluster utilization and prevent resource contention.
 - Used IAM Roles for Service Accounts (IRSA) for secure AWS resource access from within pods.
 - Managed secrets and credentials using Kubernetes Secrets and integrated with AWS Secrets Manager for automated rotation.

- **Networking & Service Discovery:**
 - Configured internal DNS-based service discovery (e.g., `adapter-<backend>-<tool>.mcp-adapters.svc.cluster.local`) for seamless communication between MCP components.
 - Exposed services via internal and external load balancers as required.

- **Observability & Monitoring:**
 - Integrated telemetry emission (metrics, logs, traces) from MCP adapters to centralized monitoring solutions (CloudWatch, Grafana).
 - Enabled distributed tracing and structured logging for auditability and troubleshooting.

- **CI/CD Integration:**
 - Automated deployment pipelines using Azure DevOps and Jenkins, orchestrating Kubernetes deployments as part of the end-to-end ML workflow.
 - Used shell scripts and infrastructure-as-code (CloudFormation, Helm) for repeatable, version-controlled deployments.

**Project Example:**
- In the KGPT (Knowledge GPT) and UIDS projects, I deployed MCP adapters and AI inference services on EKS, leveraging Kubernetes for scalability, high availability, and secure integration with AWS services.

This experience has enabled me to build and operate robust, scalable, and secure AI platforms using Kubernetes in production enterprise environments.

---

**Q: What is self-attention, and how do you calculate the self-attention score in machine learning?**

- **Self-attention** is a core mechanism in transformer models (like BERT, GPT, etc.) that allows each token in a sequence to "attend" to every other token, including itself, to capture contextual relationships. It enables the model to weigh the importance of different words when encoding a sequence, making it highly effective for NLP tasks.

**How Self-Attention Works:**

- For each token in the input sequence, self-attention computes a weighted sum of all tokens, where the weights (attention scores) reflect the relevance of each token to the current token.
- This is achieved using three learned projections for each token:
 - **Query (Q)**
 - **Key (K)**
 - **Value (V)**

**Self-Attention Score Calculation:**

1. **Linear Projections:**
 - For each input token embedding, compute Q, K, and V vectors using learned weight matrices.

2. **Score Calculation:**
 - For each pair of tokens (i, j), calculate the attention score as the dot product between the Query of token i and the Key of token j:
 - `score_ij = Q_i · K_j^T`

3. **Scaling:**
 - Scale the score by the square root of the key dimension (d_k) to stabilize gradients:
 - `scaled_score_ij = score_ij / sqrt(d_k)`

4. **Softmax Normalization:**
 - Apply softmax across all tokens j for each token i to get normalized attention weights:
 - `attention_weight_ij = softmax(scaled_score_ij)`

5. **Weighted Sum:**
 - Multiply each Value vector V_j by its attention weight and sum to get the output for token i:
 - `output_i = sum_j (attention_weight_ij * V_j)`

**Python-like Pseudocode:**
```python
# Assume X is the input sequence matrix (seq_len x d_model)
Q = X @ W_Q # (seq_len x d_k)
K = X @ W_K # (seq_len x d_k)
V = X @ W_V # (seq_len x d_v)

# Compute attention scores
scores = Q @ K.T # (seq_len x seq_len)

# Scale scores
scores = scores / np.sqrt(d_k)

# Softmax to get attention weights
attention_weights = softmax(scores, axis=1) # (seq_len x seq_len)

# Weighted sum of values
output = attention_weights @ V # (seq_len x d_v)
```

**Summary:**
- Self-attention enables each token to dynamically focus on relevant parts of the sequence.
- The attention score is computed as a scaled dot product between Query and Key, normalized with softmax, and used to weight the Value vectors.
- This mechanism is fundamental to the success of transformer-based models in NLP and other domains.

---
