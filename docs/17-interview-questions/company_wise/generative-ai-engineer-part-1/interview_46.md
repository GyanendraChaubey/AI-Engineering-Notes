# Generative AI Engineer (Part 1) — Interview 46

**Q: What technique do you use to extract content (including images and tables) from PDF, HTML, and Word documents for a RAG pipeline?**

- For the Knowledge GPT (KGPT) project, we handle multiple document types: PDF, HTML, and Word (doc/docx), each potentially containing images, tables, and complex formatting.
- The extraction process is designed to preserve document structure and semantic meaning, which is critical for effective chunking and embedding in the RAG pipeline.

**Extraction Techniques by Document Type:**

- **PDF Documents:**
 - We use PyMuPDF (pymupdf) with a custom `PdfContentHandler` to convert PDF pages to HTML.
 - The handler maps font sizes to heading levels (H1–H6), preserving the document’s logical structure.
 - After conversion to HTML, we use an HTML-to-Markdown converter (like `html2text`) to further clean and standardize the content.
 - Images and tables are extracted during the HTML conversion phase and represented in Markdown, ensuring they are included in the semantic chunks.
 - This approach allows us to split documents semantically using Markdown headers, which is essential for downstream chunking and embedding.

- **HTML Documents:**
 - For HTML sources (from CMS like Concentra, DITA, or process documents), we parse and merge articles using custom helpers (e.g., `DitaHtmlHelper`).
 - HTML is converted to Markdown to remove unnecessary tags and scripts, making the content cleaner for embedding.
 - Images, tables, and links are preserved in Markdown format, and video links are wrapped in `<iframe>` tags for consistency.

- **Word Documents:**
 - For Word (doc/docx) files, we use libraries like `python-docx` to extract text, images, and tables.
 - The extracted content is converted to Markdown, maintaining the structure (headings, lists, tables, images).

**Why Markdown?**
- Markdown preserves headings, lists, tables, code blocks, and images in a lightweight, structured format.
- It enables reliable semantic chunking using tools like LangChain’s `MarkdownHeaderTextSplitter`.
- Clean Markdown improves embedding quality by removing HTML noise.

**Summary of the Flow:**
- Download document (PDF/HTML/Word) → Convert to HTML (if needed) → Convert to Markdown → Semantic chunking (using headers) → Generate embeddings per chunk → Store in OpenSearch with metadata.

- This robust extraction and conversion pipeline ensures that all relevant content—including images and tables—is available for semantic search and LLM-based answer generation in the RAG system.

---

**Q: What is the main objective of chunking documents in the pipeline?**

- The primary objective of chunking is to break down large documents into smaller, semantically meaningful sections that are optimal for embedding and retrieval in the RAG (Retrieval-Augmented Generation) pipeline.
- Key reasons for chunking:
 - **Token Limit Management:** LLMs and embedding models have strict token limits. Chunking ensures each section fits within these limits, preventing truncation and information loss.
 - **Semantic Relevance:** By splitting at logical boundaries (like headings), each chunk represents a coherent topic or section, improving the relevance of search and retrieval.
 - **Efficient Retrieval:** Smaller, well-structured chunks allow the system to retrieve only the most relevant parts of a document in response to a query, rather than processing entire documents.
 - **Improved Embedding Quality:** Embedding smaller, focused chunks leads to higher-quality vector representations, which enhances semantic search accuracy.
 - **Performance Optimization:** Chunking enables parallel processing for embedding generation and retrieval, improving overall system scalability and response time.
- In our pipeline, we use a Markdown-aware splitting strategy, starting with H1/H2 headers and recursively splitting further if needed, to preserve document structure and semantic meaning while keeping chunk sizes within the optimal threshold (e.g., 25 KB).
- This approach ensures that the RAG system delivers precise, contextually relevant answers while maintaining efficiency and scalability.

---

**Q: What is the length (dimension) of the embedding vector for document chunks, regardless of input token count?**

- The embedding vector length is determined by the embedding model, not by the number of tokens in the input chunk.
- For both OpenAI’s `text-embedding-ada-002` and the newer `text-embedding-3-small` models:
 - **Each input (regardless of whether it’s 100 tokens or 900 tokens) produces a fixed-size embedding vector.**
 - For `text-embedding-ada-002`, the output is always a **1536-dimensional float vector**.
 - For `text-embedding-3-small`, the output is a **1536-dimensional float vector** as well.
- This means that whether your chunk is 100 tokens or 900 tokens, the resulting embedding will always have 1536 dimensions.

- This fixed-size vector representation is essential for efficient similarity search and retrieval in vector databases like OpenSearch.

---

**Q: Why does the embedding vector have a fixed dimension (e.g., 1536) regardless of input length?**

- The fixed embedding dimension (e.g., 1536 for OpenAI’s Ada-002 and text-embedding-3-small) is a property of the embedding model’s architecture, not the input size.
- **Reason for Fixed Dimension:**
 - Embedding models are neural networks trained to map variable-length text inputs (sentences, paragraphs, or documents) into a fixed-size vector space.
 - This fixed-size vector (e.g., 1536 floats) allows for efficient similarity comparison (using cosine similarity or dot product) between any two pieces of text, regardless of their original length.
 - The model processes the input text, aggregates its semantic information, and compresses it into a single vector of predetermined size.
- **Why is this important?**
 - Enables fast and scalable vector search in databases like OpenSearch or Pinecone.
 - Standardizes downstream processing—every chunk, query, or document is represented by the same-size vector, simplifying retrieval and ranking logic.
 - The dimension (e.g., 1536) is chosen by the model designers as a trade-off between expressiveness (capturing enough semantic detail) and computational/storage efficiency.
- **Summary:** The embedding dimension is fixed by the model’s design to ensure all text inputs, regardless of length, are represented in the same vector space for consistent and efficient semantic search and retrieval.

---

**Q: How would you approach diagnosing and identifying the root cause when users report that a chatbot is not providing accurate answers?**

- My approach would be systematic, combining both qualitative and quantitative evaluation to pinpoint the root cause of inaccurate chatbot responses. Here’s how I would proceed:

---

**1. Reproduce and Collect Failing Cases**
 - Gather specific user queries and chatbot responses where accuracy is reported as poor.
 - Collect ground truth or expected answers for these queries, if available.

**2. Categorize the Failure Types**
 - Determine if the issue is due to:
 - Poor retrieval (irrelevant or missing context)
 - LLM hallucination (fabricated or incorrect answers)
 - Incomplete or outdated knowledge base
 - Prompt engineering issues
 - Language/model limitations

**3. Automated Evaluation Pipeline**
 - Use an evaluation pipeline similar to the one in the Knowledge GPT project:
 - **Answer Correctness:** Use LLM-based evaluators (e.g., GPT-4o) to compare chatbot answers with ground truth for factual accuracy and semantic similarity.
 - **Answer Relevance:** Check if the answer addresses the user’s question using LLM scoring and embedding-based similarity.
 - **Context Utilization:** Evaluate if the retrieved context chunks are actually being used in the generated answer (chunk alignment scoring).
 - **MRR (Mean Reciprocal Rank):** For search-based systems, check if the correct document appears in the top results.
 - **Harmful Content:** Ensure the chatbot is not producing biased or inappropriate content.

**4. Analyze Retrieval Pipeline**
 - Inspect the retrieval logs to see if relevant chunks are being fetched for user queries.
 - Check embedding quality and chunking logic—poor chunking or embeddings can lead to irrelevant retrieval.

**5. Prompt and Model Review**
 - Review prompt templates for clarity and completeness.
 - Test with alternative prompts or few-shot examples to improve LLM guidance.

**6. Data and Knowledge Base Audit**
 - Ensure the knowledge base is up-to-date and covers the required topics.
 - Check for gaps or inconsistencies in the ingested documents.

**7. User Feedback Loop**
 - Incorporate user feedback into the evaluation pipeline for continuous improvement.
 - Use reporting tools (Excel, charts, comparison reports) to track metrics and identify trends.

**8. Continuous Monitoring**
 - Set up dashboards (e.g., Grafana, Athena, CloudWatch) to monitor answer quality, latency, and error rates in production.

---

- By following this structured approach—leveraging automated evaluation, retrieval analysis, prompt/model review, and user feedback—I can systematically identify and address the root causes of chatbot inaccuracy, ensuring continuous improvement and higher user satisfaction.

---

**Q: What is your approach to diagnosing and resolving chatbot answer accuracy issues?**

- When users report inaccurate chatbot answers, I follow a structured evaluation and debugging process, leveraging both automated tools and manual analysis:

- **1. Identify Potential Root Causes:**
 - Poor retrieval (irrelevant or missing context provided to the LLM)
 - LLM hallucination (incorrect or fabricated answers)
 - Incomplete, outdated, or low-quality knowledge base
 - Prompt engineering issues (unclear or ineffective prompts)
 - Model limitations (inherent weaknesses in the LLM or embedding model)

- **2. Quick Comparative Evaluation:**
 - Test the LLM’s output with and without retrieved context to see if context is improving answer quality.
 - Evaluate answer faithfulness, completeness, and correctness.
 - Assess context utilization—whether the LLM is actually using the provided context in its response.

- **3. Automated Evaluation Pipeline (as implemented in Knowledge GPT):**
 - Use LLM-based evaluators (e.g., GPT-4o) to score:
 - **Answer Correctness:** Compare chatbot answers to ground truth for factual accuracy and semantic similarity.
 - **Answer Relevance:** Check if the answer addresses the user’s question.
 - **Context Utilization:** Score how well the retrieved chunks are used in the answer.
 - **MRR (Mean Reciprocal Rank):** For search, check if the correct document appears in the top results.
 - **Harmful Content:** Ensure no biased or inappropriate content is generated.
 - Generate detailed Excel reports and metric charts for analysis and cross-release comparison.

- **4. Debug and Iterate:**
 - Based on evaluation results, pinpoint whether the issue is in retrieval, prompt design, knowledge base coverage, or model performance.
 - Adjust chunking, improve retrieval logic, refine prompts, or update the knowledge base as needed.
 - Continuously monitor and re-evaluate after changes.

- This systematic approach ensures that I can quickly identify the source of chatbot inaccuracies and implement targeted improvements, leveraging both automated evaluation pipelines and hands-on debugging.

---

**Q: How would you design a secure system so developers can build an application without direct access to sensitive API keys or credentials?**

- To ensure developers never have direct access to sensitive API keys or credentials, I would implement a secure, centralized secrets management and access control architecture, following industry best practices:

---

**1. Use a Centralized Secrets Manager**
 - Store all sensitive credentials (API keys, OAuth client secrets, etc.) in a secure secrets management service such as AWS Secrets Manager or Azure Key Vault.
 - Only allow access to these secrets from trusted backend services, not from developer workstations or client-side code.

**2. Role-Based Access Control (RBAC)**
 - Assign IAM roles or service principals to backend services (e.g., Docker containers, Lambda functions, ECS tasks) that need to access secrets.
 - Developers only get access to non-sensitive configuration and code, not the secrets themselves.

**3. Environment Variable Injection at Runtime**
 - Inject secrets into the application environment at runtime (e.g., via ECS task definitions, Kubernetes secrets, or CI/CD pipelines).
 - The application reads secrets from environment variables or secure mounts, never hardcoding them in source code or config files.

**4. API Gateway or Proxy Pattern**
 - Expose APIs through a secure API Gateway (e.g., AWS API Gateway, Apigee) that handles authentication and forwards requests to backend services.
 - Backend services use the stored credentials to interact with third-party APIs (e.g., OpenAI, MCP) on behalf of the application.

**5. Token Broker or Service Account Pattern**
 - Implement a token broker service (as in the KGPT/MCP architecture) that securely fetches and rotates tokens/credentials as needed.
 - The broker authenticates using its own service identity and never exposes raw secrets to developers or client applications.

**6. Audit and Monitoring**
 - Enable logging and monitoring (e.g., CloudWatch, Azure Monitor) to track access to secrets and detect any unauthorized attempts.

**7. Example from KGPT/MCP Project:**
 - OAuth client credentials are stored in AWS Secrets Manager.
 - Only backend containers (e.g., MCPManager, kgpt-mcp) have permission to retrieve these secrets at runtime.
 - Developers work on application logic and integration, but never see or handle the actual credentials.
 - All sensitive API calls (e.g., to OpenAI) are made server-side, with secrets injected securely at runtime.

---

- This approach ensures strong security, prevents accidental leaks, and aligns with zero-trust and least-privilege principles, while allowing the development team to build and test the application efficiently.

---

**Q: How do you enable and use the debugger in VS Code for Python development?**

- To enable and use the debugger in VS Code for Python, follow these practical steps:

- **1. Install the Python Extension:**
 - Make sure the official Python extension by Microsoft is installed in VS Code. This provides debugging support and other Python features.

- **2. Open Your Python Project:**
 - Open the folder or file you want to debug in VS Code.

- **3. Set Breakpoints:**
 - Click in the gutter (left margin) next to the line numbers in your Python file to set breakpoints where you want the debugger to pause.

- **4. Start Debugging:**
 - Press `F5` or click the green "Run and Debug" button in the Run panel (left sidebar).
 - If prompted, select "Python File" or configure a launch.json for more advanced setups.

- **5. Debug Controls:**
 - Use the debug toolbar at the top to step over, step into, continue, or stop execution.
 - Inspect variables, call stack, and watch expressions in the Debug panel.

- **6. Additional Features:**
 - You can add conditional breakpoints, logpoints, and use the interactive debug console for evaluating expressions at runtime.

- **Summary:** 
 - Install the Python extension, set breakpoints, and use `F5` or the Run panel to start debugging. VS Code provides a user-friendly interface for stepping through code, inspecting variables, and troubleshooting Python applications efficiently.

---

**Q: Can you describe any API framework you have worked on?**

- Yes, I have extensive experience designing and implementing enterprise-grade API frameworks, particularly in the context of Generative AI and Retrieval-Augmented Generation (RAG) systems. One of my key projects is the Knowledge GPT (KGPT) API framework, which I architected and developed for enterprise knowledge retrieval and conversational AI.

- **Key Features and Components of the KGPT API Framework:**
 - **Modular API Design:** The framework is organized into distinct modules for search, completions, feedback, and shared utilities, ensuring maintainability and scalability.
 - **RAG Orchestration:** The `completion.py` module handles the core RAG workflow—auto-selecting search type, orchestrating context retrieval, and handing off to the LLM for response generation.
 - **Hybrid Search:** The `hybrid_search.py` module supports multiple search strategies (lexical, semantic, hybrid) and parallel query execution, with robust ranking and scoring logic.
 - **Secure Authentication:** JWT-based authentication and token validation are implemented (`auth_token_validator.py`, `jwt_access_token_data.py`), with secrets managed centrally via AWS Secrets Manager (`secrets_retriever.py`).
 - **Audit Logging:** All input and output payloads are logged for traceability, with masking controls for sensitive data (`log_input_output.py`).
 - **Internationalization and Error Handling:** User-facing error messages are localized and managed centrally (`messages_handler.py`), with custom exception handling for robust API behavior.
 - **Feedback API:** Designed to collect user feedback on responses, associating ratings and comments with specific chat interactions for continuous improvement.
 - **Evaluation Pipeline Integration:** The API framework is tightly integrated with an automated evaluation pipeline, enabling batch testing, KPI aggregation, and quality monitoring across different search types and languages.
 - **Cloud-Native and Secure:** All secrets and credentials are injected securely at runtime, and the framework is designed for deployment on AWS or Azure, following best practices for security and scalability.

- **Summary:** 
 - My experience with the KGPT API framework demonstrates my ability to design, build, and operate robust, secure, and scalable API systems for enterprise AI applications, with a strong focus on modularity, security, and continuous quality improvement.

---

**Q: What is a router in FastAPI and how does it work?**

- In FastAPI, a **router** is a modular way to organize and manage API endpoints, making your codebase more maintainable and scalable—especially for large applications like enterprise AI systems.

- **How Routers Work in FastAPI:**
 - A router is created using `APIRouter()` from FastAPI.
 - You define endpoints (routes) on the router just like you would on the main FastAPI app.
 - Routers can be grouped by functionality (e.g., search, completions, feedback) and imported into the main application.
 - This allows you to keep related endpoints together, apply specific dependencies, tags, or middleware, and avoid a monolithic API file.

- **Example Usage:**
 ```python
 from fastapi import APIRouter

 # Create a router instance
 router = APIRouter()

 # Define an endpoint on the router
 @router.post("/search")
 async def search_endpoint(request: SearchRequest):
 # Logic for search
 return {"result": "search results"}

 # In your main app file
 from fastapi import FastAPI
 from .search_router import router as search_router

 app = FastAPI()
 app.include_router(search_router, prefix="/api")
 ```

- **In Practice (from Knowledge GPT API):**
 - The Knowledge GPT API framework uses routers to separate the Search API, Completions API, and Feedback API.
 - Each router handles its own endpoints, validation, and business logic, then is included in the main FastAPI app.
 - This modular approach supports clean architecture, easier testing, and better team collaboration.

- **Summary:** 
 - Routers in FastAPI help you organize endpoints into logical groups, making large API projects easier to manage and extend. They are essential for scalable, production-grade API frameworks.

---
