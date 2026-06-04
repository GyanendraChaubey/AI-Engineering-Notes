# Generative AI Engineer (Part 1) — Interview 44

**Q: How would you design a system that answers user questions by retrieving information from both SharePoint documents (various formats) and Snowflake database tables?**

- I would architect a unified Retrieval-Augmented Generation (RAG) system that can intelligently route queries to the correct data source (SharePoint or Snowflake) and return accurate, context-rich answers.
- The solution would involve document ingestion, semantic search, structured data querying, and an orchestration layer to determine the right retrieval path.

**System Design Approach:**

- **1. Data Ingestion & Indexing**
 - For SharePoint:
 - Build a pipeline to extract documents (Word, Excel, PDF, images) from SharePoint.
 - Convert documents to text (using OCR for images, PDF parsers, etc.).
 - Chunk large documents into manageable pieces (e.g., 25KB chunks).
 - Generate embeddings for each chunk using an LLM embedding model (e.g., Azure OpenAI).
 - Store metadata and embeddings in a vector database (e.g., OpenSearch, ChromaDB).
 - For Snowflake:
 - Connect to Snowflake and extract schema metadata (table/column names, sample data).
 - Optionally, create embeddings for table/column descriptions to enable semantic matching.
 - Store metadata in a structured index for fast lookup.

- **2. Query Understanding & Routing**
 - Use an intent classification model or rule-based logic to determine if the user query is about unstructured documents (SharePoint) or structured data (Snowflake).
 - For ambiguous queries, use LLM-based reasoning or prompt the user for clarification.

- **3. Retrieval & Generation**
 - If SharePoint:
 - Embed the user query.
 - Perform hybrid search (BM25 + vector search) over the document index using Reciprocal Rank Fusion (RRF) for best relevance.
 - Retrieve top relevant chunks and pass them as context to the LLM for answer generation.
 - If Snowflake:
 - Use LLM to translate the natural language query into SQL.
 - Execute the SQL on Snowflake, retrieve results.
 - Optionally, use LLM to summarize or explain the results in natural language.

- **4. Orchestration Layer**
 - Implement an API layer (e.g., FastAPI) to handle user queries, manage authentication, and orchestrate the retrieval/generation workflow.
 - Ensure secure access and disclosure level validation for sensitive data.

- **5. Monitoring & Evaluation**
 - Track query routing accuracy, retrieval relevance, and LLM output quality.
 - Implement feedback loops for continuous improvement.

**Key Technologies:**
- Python, FastAPI, Azure OpenAI, OpenSearch/ChromaDB, Snowflake connector, OCR/PDF parsers, LLMs for SQL generation and summarization.

**My Experience:**
- I have built similar enterprise RAG systems (e.g., Knowledge GPT) that combine semantic search over document repositories and structured data sources, using hybrid search, LLM orchestration, and robust data pipelines.
- I have experience integrating with both unstructured (SharePoint, PDFs) and structured (databases like Snowflake) sources, ensuring scalable, secure, and accurate information retrieval for enterprise users.

---


**Q: What do you mean by "structured index" for Snowflake metadata?**

- A "structured index" means a well-organized database or table that stores metadata about the Snowflake schema in a way that is easy to search and query.
- For example, I create a table (in PostgreSQL, MongoDB, or even in Snowflake itself) where each row represents a table or column from Snowflake, with fields like:
 - Table name
 - Column name
 - Data type
 - Description
 - Relationships (foreign keys, etc.)
- This index allows fast lookup and filtering when a user asks a question, so the system can quickly find which tables or columns are relevant.
- For advanced search, I also generate vector embeddings for table/column descriptions and store them in a vector database (like OpenSearch or ChromaDB). This enables semantic search—matching user queries to the right schema elements, even if the wording is different.
- In summary, a structured index is just a searchable, organized catalog of your database schema, stored in a way that supports both keyword and semantic search for efficient query routing.

---

**Q: What is the format of the Snowflake schema metadata when storing it in a structured index like PostgreSQL?**

- The metadata is stored in a tabular (relational) format, where each row represents a schema element (such as a table or a column).
- Typical fields (columns) in this metadata table include:
 - `table_name` (string): Name of the table in Snowflake.
 - `column_name` (string): Name of the column (if storing column-level metadata).
 - `data_type` (string): Data type of the column (e.g., VARCHAR, INT).
 - `description` (string): Description or comment for the table/column.
 - `primary_key` (boolean): Indicates if the column is a primary key.
 - `foreign_key` (string): References to related tables/columns, if any.
 - `last_modified` (datetime): Timestamp of last schema update.
 - `sample_values` (JSON or string): Optional, for storing example values.
- The data is usually stored as rows in a relational database like PostgreSQL, with each field as a column in the table.
- If semantic search is needed, an additional field like `embedding_vector` (array/JSON) can be included to store vector embeddings for descriptions.
- This structured, tabular format makes it easy to search, filter, and join metadata for query understanding and routing.

---

**Q: How do you determine whether a user query should be routed to SharePoint or Snowflake?**

- To decide whether to route a query to SharePoint (unstructured documents) or Snowflake (structured database), I use a combination of intent classification and metadata matching:

**1. Intent Classification:**
 - I use an intent classification model (trained on enterprise queries) to predict if the question is about documents/content (SharePoint) or data/tables (Snowflake).
 - For example, queries like "Show me the latest sales report" likely map to SharePoint, while "What is the total revenue for Q1 2024?" likely maps to Snowflake.

**2. Keyword and Schema Matching:**
 - I extract keywords and entities from the user query.
 - I match these against the metadata index:
 - If the query mentions table names, column names, or data types present in the Snowflake schema, it is routed to Snowflake.
 - If the query matches document titles, content types, or keywords from SharePoint metadata, it is routed to SharePoint.

**3. Hybrid or Ambiguous Queries:**
 - If the query is ambiguous or could relate to both sources, I can:
 - Use an LLM to analyze the query context and suggest the most relevant source.
 - Optionally, retrieve results from both sources and let the user choose or combine the answers.

**4. Practical Implementation:**
 - In my previous projects (like Knowledge GPT), I implemented this logic as part of the API orchestration layer.
 - The system first runs a lightweight intent classifier or rule-based router.
 - If confidence is low, it falls back to LLM-based reasoning or prompts the user for clarification.

- This approach ensures that user queries are routed accurately and efficiently to the correct backend, providing relevant answers from either SharePoint or Snowflake as needed.

---

**Q: How do you technically implement intent classification to route queries between SharePoint and Snowflake, considering you have different metadata sources (chunk-level for SharePoint, schema for Snowflake), and how does your classifier interact with these?**

- At the code level, the intent classification and routing logic is implemented as a pipeline that processes the user query and determines the correct backend (SharePoint or Snowflake) using both metadata sources.

**1. Metadata Storage:**
 - For SharePoint:
 - During document ingestion, I extract metadata such as document title, type, author, creation date, and any custom tags.
 - After OCR and text extraction, I chunk the content and store each chunk with its metadata (e.g., document ID, chunk index, section headers) in a vector database (like OpenSearch or ChromaDB).
 - This metadata is stored as part of the chunk record, often in JSON format, alongside the embedding vector.
 - For Snowflake:
 - I extract schema metadata (table names, column names, descriptions) and store it in a structured relational table (e.g., PostgreSQL or Snowflake itself).

**2. Intent Classification Pipeline:**
 - When a user query arrives, the pipeline works as follows:
 - **Step 1:** The query is passed to an intent classifier (e.g., a SetFit or transformer-based model).
 - **Step 2:** The classifier is trained on labeled examples that represent typical queries for both SharePoint (document/content queries) and Snowflake (data/analytics queries).
 - **Step 3:** The classifier predicts the intent label (e.g., "document_search" or "data_query").
 - **Step 4:** Based on the predicted intent, the system routes the query:
 - If "document_search", it uses the vector database to search chunk embeddings and retrieve relevant SharePoint content, using chunk-level metadata for context.
 - If "data_query", it uses the schema metadata index to identify relevant tables/columns in Snowflake and generates the appropriate SQL.

**3. Metadata Interaction:**
 - The classifier itself does not directly query the metadata stores. Instead:
 - It uses features derived from both metadata sources during training (e.g., sample queries, document types, table names).
 - At inference, after intent is classified, the retrieval/generation modules interact with the respective metadata stores to fetch the relevant data.

**4. Example Code Structure:**
 - The pipeline orchestrator (e.g., `pipeline.py`) manages the flow:
 - Receives the query.
 - Calls the intent classifier.
 - Based on the result, calls either the document retrieval module (which queries the vector DB with chunk metadata) or the data query module (which queries the schema index).
 - All metadata (for both SharePoint and Snowflake) is kept up-to-date and accessible for fast retrieval.

**5. Practical Experience:**
 - In my UIDS and Knowledge GPT projects, I implemented similar pipelines where both unstructured (document chunk) and structured (schema) metadata are used for accurate intent classification and routing.
 - This approach ensures the system can scale, remain modular, and support both types of enterprise data sources efficiently.

---

**Q: How would you build an intent classifier from scratch?**

- To build an intent classifier, I follow a modular, industry-standard pipeline that ensures scalability, robustness, and easy integration with enterprise systems.

**1. Data Preparation**
 - Collect and clean labeled data: Each record should have a user query (text) and its corresponding intent label.
 - Handle missing values, normalize text, and ensure consistent schema.
 - Store datasets locally or on cloud storage (like AWS S3) for scalability.

**2. Feature Engineering**
 - Convert text data into numerical features using embeddings (e.g., OpenAI embeddings, BERT, or custom transformer models).
 - Optionally, augment the dataset with synthetic queries using LLMs to improve coverage for rare intents.

**3. Model Selection**
 - For few-shot or small datasets: Use models like SetFit or sentence transformers.
 - For larger datasets: Use fine-tuned transformer models (e.g., BERT, RoBERTa).
 - For traditional ML: Use TF-IDF with classifiers like Logistic Regression or XGBoost for baseline.

**4. Training**
 - Split data into train/test sets.
 - Train the model on the training set, optimizing for accuracy and generalization.
 - Use cross-validation and hyperparameter tuning for best results.

**5. Indexing for Semantic Search (Optional)**
 - Embed all training examples and store them in a vector database (like ChromaDB or OpenSearch).
 - This enables semantic retrieval for ambiguous or unseen queries.

**6. Inference Pipeline**
 - For each incoming query:
 - Preprocess and embed the query.
 - Use the trained model to predict the intent.
 - Optionally, retrieve similar examples from the vector store for explainability or fallback.

**7. Evaluation & Monitoring**
 - Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.
 - Log inference results and monitor for drift or misclassifications in production.

**8. Deployment**
 - Package the model as a REST API (using FastAPI or Flask) for integration with enterprise applications.
 - Use MLOps tools (like MLFlow, Docker, and CI/CD pipelines) for versioning, deployment, and monitoring.

**9. Extensibility**
 - Modularize each step (data prep, training, inference) for easy updates and scaling.
 - Support both local and cloud environments.

**Practical Example from My Experience:**
 - In my UIDS project, I used SetFit for few-shot intent classification, generated synthetic data with LLMs, indexed examples in a vector store, and deployed the model as a scalable API with monitoring and logging.

This approach ensures the intent classifier is accurate, robust, and production-ready for enterprise use cases.

---


**Q: How do you write an intent classifier?**

- To write an intent classifier, I follow a structured pipeline that is robust and production-ready, as I did in my UIDS project:

**1. Data Preparation**
 - Collect and clean labeled query data, each with an associated intent.
 - Handle missing values and ensure consistent schema.
 - Support both local and cloud (S3) data sources for flexibility.

**2. Synthetic Data Generation (Optional)**
 - Use LLMs (like OpenAI or Azure) to generate synthetic queries for each intent.
 - This helps balance the dataset and improves model generalization, especially for rare intents.

**3. Feature Engineering**
 - Convert text queries into embeddings using models like OpenAI embeddings or sentence transformers.
 - Store these embeddings along with metadata (intent, hierarchy, etc.) in a vector store for semantic search.

**4. Model Training**
 - Use a model like SetFit or a transformer-based classifier.
 - Encode the intent labels numerically.
 - Train the model on the prepared dataset, optimizing for accuracy.
 - Example: In UIDS, I used SetFit with a differentiable head and cosine similarity loss.

**5. Inference Pipeline**
 - For each new query, preprocess and embed the text.
 - Use the trained model to predict the intent.
 - Optionally, retrieve similar examples from the vector store for better explainability.

**6. Evaluation & Monitoring**
 - Evaluate using metrics like accuracy and F1-score.
 - Log results and monitor for drift or misclassifications.

**7. Deployment**
 - Package the model as a REST API (using FastAPI or Flask).
 - Integrate with enterprise systems and automate retraining as needed.

This approach ensures the intent classifier is accurate, scalable, and easy to maintain in a real-world enterprise environment.

---

**Q: How would you convert your current pipeline-based solution into an agentic (agent-based) framework?**

- To convert the current pipeline into an agentic framework, I would redesign the architecture so that autonomous agents handle each major function, enabling dynamic reasoning, tool orchestration, and adaptive workflows.

**1. Identify Agent Roles**
 - Define specialized agents for each core task:
 - **Intent Classification Agent**: Determines user intent (document vs. data query).
 - **Metadata Retrieval Agent**: Fetches schema or chunk metadata as needed.
 - **Document Retrieval Agent**: Handles SharePoint queries using vector search.
 - **SQL Generation Agent**: Maps natural language to SQL for Snowflake.
 - **Orchestration/Coordinator Agent**: Manages workflow, delegates tasks, and maintains context.

**2. Agentic Framework Selection**
 - Use frameworks like LangChain Agents, CrewAI, or AWS Bedrock Agents for orchestration.
 - Each agent is implemented as a modular, callable service (can be serverless, containerized, or microservice-based).

**3. Tool and Memory Integration**
 - Equip agents with access to tools (APIs, vector DBs, SQL engines) and shared memory/state (using Redis, DynamoDB, or custom context stores).
 - Use Model Context Protocol (MCP) for structured communication and tool orchestration, as described in my KGPT MCP design.

**4. Dynamic Task Delegation**
 - The Orchestrator Agent receives the user query and dynamically delegates sub-tasks to specialized agents based on context and intermediate results.
 - Agents can call each other, reason over results, and adapt the workflow as needed (not just fixed steps).

**5. Governance, Guardrails, and Monitoring**
 - Implement built-in guardrails, audit trails, and cost controls (using Bedrock/Azure agentic features or custom logic).
 - Log all agent actions and decisions for traceability.

**6. Extensibility and Cloud-Native Design**
 - Deploy agents as serverless functions (AWS Lambda) or containers (EKS/ECS, Azure Containers) for scalability.
 - Integrate with event-driven architectures (EventBridge, webhooks) for real-time responsiveness.

**7. Example Flow**
 - User query → Orchestrator Agent → Intent Classification Agent → (routes to) Document Retrieval Agent or SQL Generation Agent → Results aggregated and returned.

**8. Practical Reference**
 - In my KGPT MCP PoC, I used a Python-based MCP server with modular agent endpoints, Chainlit for chat UI, and Apigee for API gateway/authentication.
 - Each agent exposes capabilities over HTTP, and the orchestrator manages tool calls and context using MCP.

This agentic approach enables more flexible, autonomous, and maintainable AI systems, supporting complex workflows and adaptive reasoning beyond static pipelines.

---

**Q: When do you use LangChain vs. LangGraph, and how would you architect an agentic solution using these tools?**

- **LangChain** and **LangGraph** serve different but complementary purposes in agentic AI architectures:

---

**When to Use LangChain:**
- Use LangChain for building modular, composable chains of LLM calls, tool integrations, and prompt templates.
- Ideal for linear or branching workflows where each step is deterministic and follows a set order.
- Good for simple agentic tasks, tool calling, and orchestrating LLMs with APIs, databases, or vector stores.
- Example: A chain that takes a user query, classifies intent, retrieves documents, and generates a response.

**When to Use LangGraph:**
- Use LangGraph when you need to model complex workflows as state machines or graphs.
- Suitable for agentic systems where tasks may loop, branch, or require conditional execution based on intermediate results.
- Enables dynamic, multi-step reasoning and coordination between multiple agents.
- Example: An agentic orchestrator that routes tasks to specialized agents (intent, metadata, SQL, document retrieval), manages context, and adapts the workflow based on outcomes.

---

**Agentic Architecture Using LangChain & LangGraph:**

1. **Agent Definition:**
 - Define specialized agents for each core function:
 - Intent Classification Agent
 - Metadata Retrieval Agent
 - Document Retrieval Agent
 - SQL Generation Agent
 - Coordinator/Orchestrator Agent

2. **Workflow Orchestration:**
 - Use **LangGraph** to model the overall workflow as a state machine:
 - Each node represents an agent or a processing step.
 - Edges define the flow, including conditional logic (e.g., if intent is "data_query," route to SQL agent).
 - Supports loops, retries, and dynamic branching.

3. **Agent Implementation:**
 - Implement each agent using **LangChain**:
 - Each agent can be a LangChain chain, combining LLM calls, tool integrations (APIs, vector DBs, SQL engines), and prompt templates.
 - Agents can call external tools (e.g., OpenAI API, AWS Lambda, Snowflake, SharePoint).

4. **Context & State Management:**
 - Use a shared state store (e.g., Redis, DynamoDB) to maintain conversation context, intermediate results, and authentication tokens.
 - The orchestrator agent manages state transitions and delegates tasks.

5. **Deployment & Integration:**
 - Deploy agents and orchestrator as microservices or serverless functions (e.g., AWS Lambda, Docker containers).
 - Expose the orchestrator as a REST API for integration with UIs (e.g., Chainlit) or other enterprise systems.

6. **Governance & Monitoring:**
 - Implement guardrails, audit trails, and cost controls as described in the MCP design.
 - Log all agent actions for traceability and debugging.

---

**Summary Table:**

| Use Case | LangChain | LangGraph |
|----------------------------------|---------------------|--------------------------|
| Linear/branching pipelines | ✅ | |
| Simple tool/LLM orchestration | ✅ | |
| Complex agentic workflows | | ✅ |
| State machine, dynamic routing | | ✅ |
| Multi-agent coordination | | ✅ |

---

**Example (from my KGPT MCP PoC):**
- Used LangChain for individual agent logic (intent, retrieval, SQL).
- Used LangGraph to orchestrate the workflow, manage state, and enable dynamic agent collaboration.
- Centralized orchestrator (MCPManager) handled routing, context, and security, with Redis for state and Apigee for API gateway.

This approach ensures a scalable, maintainable, and truly agentic AI system, leveraging the strengths of both LangChain and LangGraph.

---

**Q: What is the alternative to LangGraph for agentic workflows on AWS?**

- The main alternative to LangGraph for agentic workflows on AWS is **AWS Bedrock Agents**.
 - AWS Bedrock Agents provide a managed platform for building, orchestrating, and deploying agentic AI workflows.
 - They support multi-agent orchestration, tool integration, and built-in governance (guardrails, audit trails, cost controls).
 - Bedrock Agents can coordinate LLM calls, tool usage (like Lambda functions, API calls), and manage state and memory using AWS services (DynamoDB, S3).
 - For custom agentic logic, you can also use AWS Step Functions to model state machines and orchestrate Lambda-based agents, but Bedrock Agents are purpose-built for LLM and agentic AI use cases.
 - In my MCP (Model Context Protocol) design, I have also used AWS Lambda, EventBridge, and DynamoDB for custom agent orchestration and state management, which gives full control for advanced scenarios.

- **Summary Table:**

| Feature | LangGraph | AWS Bedrock Agents / Step Functions |
|------------------------|---------------------|-------------------------------------|
| State machine modeling | ✅ | ✅ (Step Functions) |
| Agentic AI orchestration | ✅ | ✅ (Bedrock Agents) |
| LLM/Tool integration | ✅ | ✅ |
| Built-in governance | ❌ (custom) | ✅ |
| Cloud-native | ❌ | ✅ |

- For AWS-native, scalable, and governed agentic workflows, **AWS Bedrock Agents** are the recommended alternative to LangGraph. For more custom or fine-grained control, combine Step Functions, Lambda, and other AWS services.

---
