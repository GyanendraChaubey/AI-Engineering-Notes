# GenAI NLP Engineer (Part 1) — Interview 2

**Q: How do you evaluate your ranking model in a production RAG system project?**

- For the Knowledge GPT (KGPT) project, we use a dedicated evaluation pipeline to measure the quality of our ranking and retrieval system.
- The main metric we use is Mean Reciprocal Rank (MRR), which checks if the expected (ground truth) document appears in the search results and at what position.
- For each query, we:
 - Send the query to the Search API.
 - Check if the correct document is present in the returned results and note its rank (position).
 - Calculate the reciprocal rank (1/rank) for each query. If the document is not found, the score is 0.
- We then average these reciprocal ranks across all queries to get the MRR score.
- We also track Precision@1, which is the fraction of queries where the correct document is ranked first.
- The evaluation pipeline generates detailed reports, including:
 - Per-query rank and reciprocal rank.
 - Aggregate MRR and Precision@1 for different search types (like lexical, semantic, hybrid).
 - Breakdown of found, not found, and empty results.
- All evaluation results are stored in S3 for release comparison and tracking improvements over time.
- This process helps us monitor and improve the ranking quality of our RAG system in a systematic way.

---

**Q: Why use LLM-based evaluation if you already have ground truth data?**

- Even though we have ground truth data, we use LLM-based evaluation because not all aspects of answer quality can be measured by simple string matching or exact document ID checks.
- The LLM (like GPT-4o) acts as a judge to assess deeper qualities of the generated answer, such as:
 - **Factual correctness**: The LLM compares the generated answer to the ground truth and scores how factually accurate it is, not just if it matches word-for-word.
 - **Semantic similarity**: Sometimes, the answer can be correct but phrased differently. The LLM can judge if the meaning matches, even if the wording is different.
 - **Reference health**: The LLM checks if the answer properly cites or uses the right references from the context.
 - **Answer relevance**: The LLM evaluates if the answer actually addresses the question, covering all required parts.
 - **Context utilization**: The LLM checks if the answer uses the provided context effectively, not just copying or ignoring it.
- These metrics are combined using weighted scores to give a more complete and nuanced evaluation of the model’s performance.
- This approach helps us catch issues like hallucinations, partial answers, or irrelevant content, which simple ground truth checks might miss.
- In summary, LLM-based evaluation gives us a more human-like, comprehensive assessment of answer quality, beyond what ground truth alone can provide.

---

**Q: How can you compare LLM-generated answers to ground truth using traditional NLP methods, without using LLM-as-a-judge?**

- Before LLM-based evaluation, we used several traditional NLP techniques to compare generated answers with ground truth:
 - **String Matching**: Directly compare the generated answer with the ground truth using exact or partial string match. This is simple but often too strict, as it does not handle paraphrasing.
 - **Token Overlap Metrics**: Use metrics like BLEU, ROUGE, or F1-score to measure the overlap of words or n-grams between the generated answer and the ground truth. These metrics are common in text summarization and translation tasks.
 - **Semantic Similarity**: Compute the cosine similarity between embeddings of the generated answer and the ground truth. We can use models like Sentence Transformers (e.g., BERT, SBERT) to get sentence embeddings and then calculate similarity. This helps capture meaning even if the wording is different.
 - **Jaccard Similarity**: Measure the intersection over union of unique words or tokens in both answers.
 - **Levenshtein Distance**: Calculate the minimum number of edits needed to change one answer into the other, which gives a sense of how close the answers are.
- In practice, semantic similarity using embeddings is the most robust, as it can handle paraphrased or reworded answers better than pure string-based methods.
- For our projects, we often use Sentence Transformers to encode both the generated answer and the ground truth, then compute cosine similarity. If the similarity score is above a certain threshold (e.g., 0.8), we consider the answer correct.
- These methods are fast, interpretable, and do not require LLM inference, but they may miss subtle errors or hallucinations that only a human or LLM judge can catch.
- Overall, combining these traditional metrics gives a good baseline for automated answer evaluation when ground truth is available.

---

**Q: Which evaluation metric is best for comparing LLM-generated answers to ground truth, and why?**

- In my experience, the best option for comparing LLM-generated answers to ground truth is **semantic similarity using embeddings**.
- This approach uses models like Sentence Transformers or OpenAI embeddings to convert both the generated answer and the ground truth into high-dimensional vectors, then calculates the cosine similarity between them.
- The main reasons why semantic similarity is best:
 - It captures the actual meaning of the text, not just exact word matches, so it works well even if the answer is paraphrased.
 - It is more robust than string matching or n-gram overlap metrics like BLEU or ROUGE, which can miss correct answers if the wording is different.
 - It is fast, scalable, and easy to automate for large datasets.
- For example, in our Knowledge GPT evaluation pipeline, we use Azure OpenAI embeddings (text-embedding-ada-002) to compute cosine similarity between answers and ground truth, and this gives a score between 0 and 1.
- We also combine this with other metrics (like factual accuracy from LLM judge and reference health), but for pure automated comparison, semantic similarity is the most reliable and practical.
- This method helps us identify correct answers even when the language is different, making it ideal for real-world LLM evaluation.

---

**Q: What are the risks or limitations of using BLEU score for evaluating LLM-generated answers?**

- BLEU score mainly measures n-gram overlap between the generated answer and the ground truth, focusing on exact word or phrase matches.
- The main risks or limitations are:
 - It does not capture the actual meaning or semantics of the answer, so if the answer is correct but paraphrased, BLEU may give a low score.
 - BLEU is sensitive to word order and exact wording, which is not ideal for open-ended or generative tasks where multiple correct phrasings are possible.
 - It can penalize valid answers that use synonyms or different sentence structures.
 - BLEU was originally designed for machine translation, so it may not reflect answer quality well in QA or summarization tasks.
 - It may miss issues like hallucinations or factual errors if the n-gram overlap is high but the answer is still wrong.
- Because of these reasons, BLEU is not reliable for evaluating LLM-generated answers in most real-world scenarios, especially when semantic correctness is more important than exact wording. That’s why semantic similarity or LLM-based evaluation is preferred.

---


**Q: How do you handle large prompt sizes in prompt engineering to manage latency and performance?**

- Yes, handling large prompts is a common challenge, especially when working with LLMs like GPT-4 or GPT-4o, as longer prompts can increase both latency and cost.
- In our production projects, we use several strategies to manage prompt size and improve performance:
 - **Chunk Summarization**: Before including context in the prompt, we summarize large document chunks using Azure OpenAI. This reduces the amount of text passed to the LLM while preserving key information.
 - **Chunk Consolidation**: We merge small adjacent chunks up to a certain size limit (e.g., 25 KB) to avoid sending too many small pieces, which helps keep the prompt concise and relevant.
 - **Context Window Management**: We carefully select only the top-ranked, most relevant chunks (using BM25 and vector search with RRF) to include in the prompt, instead of adding all possible context.
 - **Prompt Template Optimization**: We design prompt templates to be as compact as possible, removing unnecessary instructions or metadata.
 - **Token Counting**: We monitor the total token count before sending the prompt to the LLM, and if it exceeds the model’s limit, we trim less relevant context first.
 - **Fallback Mechanism**: If summarization fails (e.g., due to token limits), we fall back to using the original chunk text, but still enforce a maximum size.
- These steps help us keep prompts within the LLM’s token limits, reduce latency, and control costs, while still providing enough context for accurate answers.
- This approach is part of our Knowledge GPT pipeline, and it has proven effective in production for both latency and answer quality.

---

**Q: Any latest MLOps technique you tried recently that worked well?**

- Yes, recently I worked on implementing the Model Context Protocol (MCP) as part of our GenAI platform at [Company].
- MCP is designed to enable structured communication between LLMs and enterprise tools, supporting advanced agentic workflows and tool integration.
- From an MLOps perspective, some of the latest techniques we applied include:
 - **Declarative Infrastructure**: We used Terraform and Kubernetes manifests stored in Git for all infrastructure as code, making deployments reproducible and auditable.
 - **CI/CD Integration**: Our pipelines (using Azure DevOps and GitHub Actions) automate Docker image builds, push to ECR, and trigger deployments to EKS clusters, ensuring fast and reliable rollouts.
 - **Secrets & Config Management**: We centralized secrets in AWS Secrets Manager and used DynamoDB for low-latency config state, improving security and operational flexibility.
 - **Auto-scaling & Observability**: We enabled auto-scaling on EKS and Lambda, and set up unified logging and distributed tracing with AWS CloudWatch and OpenTelemetry for real-time monitoring and debugging.
 - **Modular Pipeline Orchestration**: For our RAG and agentic workflows, we used LangGraph to define modular, state-machine-based pipelines, making it easy to add or update steps without breaking the system.
 - **Backup & Disaster Recovery**: We automated regular snapshots using AWS Backup, ensuring data safety and fast recovery.
- These techniques helped us deliver a robust, scalable, and maintainable GenAI platform, and the modular orchestration with MCP has made it much easier to integrate new tools and workflows as business needs evolve.

---

**Q: What techniques are you using for user authentication in your production architecture (with Redis, Kubernetes, and cloud deployment)?**

- In our production architecture, we use a robust, enterprise-grade authentication system to ensure secure access across all services and tools.
- We integrate with [Company]’s OAuth 2.0 and OpenID Connect protocols for user authentication and authorization.
 - Users authenticate via [Company]’s Identity Provider (IdP), which supports multi-factor authentication (MFA) and issues JWT tokens after successful login.
 - These JWT tokens are used for all API requests and are validated by our backend services before granting access.
- For API security and traffic management, we use an API Gateway (like Apigee), which:
 - Terminates TLS, validates JWT tokens, applies rate limiting, and enforces security policies at the edge.
- Role-based access control (RBAC) is implemented using custom scopes in the JWT tokens, so users only access tools and data they are permitted to.
- Redis (Amazon ElastiCache) is used to cache access tokens and session state for fast validation and to reduce repeated authentication overhead.
- All secrets (like OAuth client credentials and API keys) are securely managed in AWS Secrets Manager, never hardcoded or exposed in code.
- The entire authentication and authorization flow is integrated with our Kubernetes-deployed services, ensuring secure, scalable, and centralized user management across the platform.
- This approach provides strong security, easy integration with enterprise SSO, and efficient performance for both internal and external users.

---

**Q: What is the benefit of using JWT tokens for authentication?**

- JWT (JSON Web Token) tokens are very useful for secure, scalable authentication in modern cloud and microservices architectures.
- The main benefits of using JWT tokens are:
 - **Stateless Authentication**: JWTs are self-contained and carry all the user information and permissions inside the token, so the backend does not need to store session data. This makes it easy to scale services horizontally (like in Kubernetes).
 - **Security**: JWTs are signed (e.g., with RS256), so the backend can verify the token’s authenticity and integrity. This prevents tampering and ensures only valid users can access protected APIs.
 - **Role-Based Access Control**: Custom claims and scopes can be added to the JWT, so you can control what resources or tools a user can access (for example, mcp:tools:read or knowledge-gpt:search).
 - **Single Sign-On (SSO)**: JWTs work well with enterprise identity providers (like [Company]’s OpenID Connect), allowing users to log in once and access multiple services securely.
 - **Performance**: Since JWTs are validated locally (no database lookup needed), authentication is fast and efficient, which is important for high-throughput APIs.
 - **Interoperability**: JWT is an open standard and widely supported, making it easy to integrate with API gateways (like Apigee), cloud services, and third-party tools.
- In our architecture, JWT tokens help us enforce strong security, flexible access control, and efficient authentication across all APIs and services.

---

**Q: What is the drawback of using identity-based tokens (session tokens) compared to JWT tokens?**

- Identity-based tokens (like traditional session tokens) usually require the backend to store session state in a database or cache.
 - This means every API call needs to check the session store to validate the token and fetch user info.
 - It creates a dependency on a central session store, which can become a bottleneck or single point of failure, especially in distributed or cloud-native environments.
- With JWT tokens:
 - All user info and permissions are embedded and cryptographically signed inside the token.
 - The backend can validate the token and extract user info without any database lookup, making it stateless and highly scalable.
 - This is especially useful in Kubernetes or microservices, where services can scale up or down without worrying about session synchronization.
- Drawbacks of identity/session tokens:
 - Harder to scale horizontally (need sticky sessions or shared session store).
 - More complex to manage in multi-region or multi-cloud setups.
 - Session store outages can block all authentication.
 - More overhead for every request due to session lookups.
- JWT tokens solve these problems by being self-contained, stateless, and easy to validate, which is why they are preferred for modern cloud architectures.

---

**Q: How will you evaluate and ensure response consistency when migrating from one LLM (e.g., GPT-4) to another (e.g., GPT-5), so that client-approved responses remain consistent in the new model?**

- When migrating from one LLM to another (like GPT-4 to GPT-5), my strategy focuses on both automated evaluation and client feedback to ensure response quality and consistency.
- **Automated Evaluation Pipeline**:
 - I use a dedicated evaluation pipeline (like the one built for Knowledge-GPT) to compare responses from both models using a ground truth dataset of real client questions and previously approved answers.
 - The pipeline sends the same set of questions to both the old (GPT-4) and new (GPT-5) models and collects their responses.
 - For each response, I use multiple metrics to evaluate and compare:
 - **Answer Correctness**: Checks if the answer is factually and semantically correct compared to the ground truth, using LLM-based scoring (e.g., GPT-4o as a judge) and embeddings for semantic similarity.
 - **Answer Relevance**: Measures if the answer actually addresses the question, using a combination of LLM scoring and embeddings-based similarity.
 - **Format Consistency**: Ensures the new model’s responses match the expected structure and style (using prompt engineering and output validation).
 - **Harmful Content Detection**: Scans for any harmful, biased, or inappropriate content in the new model’s responses.
 - **MRR (Mean Reciprocal Rank)**: For search-based systems, checks if the correct document appears at the top, as before.
 - **Context Utilization**: Verifies that the model is using the retrieved context properly in its answers.
 - All results are compiled into detailed reports (Excel, charts) for easy comparison and shared with stakeholders.
- **Client Feedback Loop**:
 - I involve the client in reviewing a sample of responses from the new model, especially for critical or high-impact queries.
 - Any gaps or inconsistencies are addressed by refining prompts, adjusting model parameters, or post-processing outputs.
- **Release Management**:
 - I maintain versioned releases and cross-release comparison reports, so we can track improvements or regressions for each model upgrade.
 - If needed, I can roll back to the previous model or selectively use the old model for specific queries until the new model is fully validated.
- **Summary**:
 - This approach ensures that the new model (GPT-5) meets or exceeds the quality and consistency of the previous model (GPT-4), and that client-approved responses remain reliable after migration.

---

**Q: What will you do if, even after prompt engineering and post-processing, the client still does not like the model’s response?**

- If the client is still not satisfied with the model’s response after prompt engineering and post-processing, I would take these steps:
 - **Collect Specific Feedback**: Ask the client to provide clear examples and details about what is missing or wrong in the response. This helps identify the exact gap between the model output and client expectations.
 - **Manual Response Override**: For critical or repeated queries, I can implement a manual override system. This means storing client-approved responses in a database or knowledge base, so when the same query comes, the system returns the exact approved answer instead of generating a new one.
 - **Custom Fine-Tuning**: If the issue is common across many queries, I would consider fine-tuning the model using the client’s preferred responses and feedback data. This helps the model learn the client’s style and requirements.
 - **Hybrid Approach**: Use a hybrid system where the model first checks for a matching answer in the approved database, and only generates a new response if no match is found. This ensures consistency for important queries.
 - **Continuous Feedback Loop**: Set up a feedback API or interface so the client can flag unsatisfactory answers, and use this data to improve both the model and the override database over time.
 - **Transparent Communication**: Keep the client informed about the steps being taken and set realistic expectations about what the model can and cannot do.
- This approach ensures that the client always gets the responses they want for their most important queries, even if the model alone cannot deliver them perfectly.

---

**Q: How can you identify the rank of a matrix?**

- The rank of a matrix is the maximum number of linearly independent rows or columns in the matrix.
- To find the rank, you can use these steps:
 - Convert the matrix to its row echelon form (REF) or reduced row echelon form (RREF) using Gaussian elimination.
 - The number of non-zero rows in the row echelon form is the rank of the matrix.
- In Python, you can use NumPy to find the rank easily.

**Example in Python:**
```python
import numpy as np # Import the numpy library

# Define a matrix as a 2D numpy array
matrix = np.array([[1, 2, 3],
 [2, 4, 6],
 [1, 1, 1]]) # Create a 3x3 matrix

# Use numpy's linalg.matrix_rank function to find the rank
rank = np.linalg.matrix_rank(matrix) # Calculate the rank

print("Rank of the matrix:", rank) # Print the rank
```

- In this example, `np.linalg.matrix_rank()` automatically computes the rank by checking for linear independence.
- You can also do it manually by performing row operations to count the number of non-zero rows after reducing the matrix.

- The rank tells you about the dimension of the vector space spanned by the rows or columns of the matrix. If the rank is less than the number of rows or columns, it means some rows or columns are linearly dependent.

---

**Q: What is the importance of the rank of a matrix in fine-tuning?**

- The rank of a matrix is important in fine-tuning, especially in deep learning and NLP, because it tells us about the amount of useful information or diversity in the data or in the model’s parameters.
- In the context of fine-tuning large language models (LLMs), "low-rank adaptation" (like LoRA) is a popular technique:
 - Instead of updating all the weights in a large model, we add small trainable matrices (with low rank) to the original weights.
 - These low-rank matrices capture the most important changes needed for the new task, making fine-tuning much more efficient and requiring less memory.
- If the rank is too low, the model might not learn enough new information; if it’s too high, it can overfit or become inefficient.
- In general, checking the rank helps us:
 - Detect if our data or parameter updates are redundant (linearly dependent).
 - Ensure that the model is actually learning new, independent features during fine-tuning.
- In summary, the rank helps us balance between efficiency and expressiveness when adapting models to new tasks, especially with techniques like LoRA or parameter-efficient fine-tuning.

---
