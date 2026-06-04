# Generative AI Engineer (Part 1) — Interview 13

**Q: Clarification on expected output for the coding task (extract employees earning above department average salary).**

- Understood, you are looking for a solution that, given a dataset (dictionary) with employee ID, salary, and department ID, extracts all employees whose salary is greater than the average salary of their respective department.
- I will use the pandas library to:
 - Convert the dictionary to a DataFrame.
 - Calculate the average salary per department.
 - Filter and return employees whose salary exceeds their department's average.
- Please share the data when ready, and I will proceed with the code implementation.

---

**Q: Find employees earning more than the average salary of their department using pandas.**

- The task is to filter employees whose salary is greater than the average salary of their respective department.
- We'll use pandas to:
 - Convert the dictionary to a DataFrame.
 - Calculate the average salary per department.
 - Compare each employee's salary to their department's average and filter accordingly.

**🔑 Key Steps**:
- Create a DataFrame from the dictionary.
- Group by department and calculate the mean salary.
- Merge the average salary back to the original DataFrame.
- Filter employees whose salary is greater than their department's average.

**💻 Code**:
```python
import pandas as pd # Import pandas library for data manipulation

# Provided data dictionary
data = {
 'empid': [101, 102, 103, 104], # Employee IDs
 'salary': [50000, 60000, 55000, 65000], # Salaries
 'deptid': [10, 20, 10, 30] # Department IDs
}

df = pd.DataFrame(data) # Convert dictionary to DataFrame

dept_avg = df.groupby('deptid')['salary'].mean().reset_index() # Calculate average salary per department

dept_avg = dept_avg.rename(columns={'salary': 'avg_salary'}) # Rename column for clarity

df = df.merge(dept_avg, on='deptid') # Merge average salary into original DataFrame

result = df[df['salary'] > df['avg_salary']] # Filter employees earning more than department average

print(result[['empid', 'salary', 'deptid']]) # Print the required columns
```

**💡 Explanation**:
- **Time Complexity**: O(n), where n is the number of employees (since groupby and merge are linear for small datasets).
- **Space Complexity**: O(n), as we create new DataFrames for intermediate steps.
- **Steps**:
 - The code first creates a DataFrame from the input dictionary.
 - It then computes the average salary for each department using `groupby`.
 - The average salary is merged back to the original DataFrame so each employee row has their department's average.
 - Finally, it filters and prints employees whose salary is greater than their department's average.
- **Result**: The output will list employees (with their IDs, salaries, and department IDs) who earn more than the average salary in their department.

---

**Q: What are the architectural differences between the encoder and decoder in a transformer?**

- The transformer architecture consists of two main components: the encoder and the decoder, each with distinct roles and internal structures.
- **Encoder**:
 - Processes the input sequence and generates contextualized representations.
 - Each encoder layer contains:
 - Multi-head self-attention mechanism (attends to all positions in the input).
 - Add & Norm (residual connection + layer normalization).
 - Position-wise feed-forward network.
 - No access to output tokens or future information—only attends to the input sequence.
- **Decoder**:
 - Generates the output sequence, one token at a time, using both the encoder’s output and previously generated tokens.
 - Each decoder layer contains:
 - Masked multi-head self-attention (prevents attending to future tokens in the output sequence).
 - Add & Norm.
 - Multi-head cross-attention (attends to encoder outputs, allowing the decoder to use encoded input information).
 - Add & Norm.
 - Position-wise feed-forward network.
 - Add & Norm.
- **Key Differences**:
 - The decoder has an extra cross-attention layer to incorporate encoder outputs.
 - The decoder’s self-attention is masked to prevent information leakage from future tokens, while the encoder’s self-attention is unmasked.
 - The encoder processes the entire input in parallel, while the decoder generates outputs sequentially (during inference).

---

- In summary, the encoder focuses on understanding the input, while the decoder generates the output by leveraging both its own history and the encoder’s representations. The cross-attention and masking mechanisms are the primary architectural differences.

---

**Q: Why do two sentences with the same set of words in different orders (e.g., "man bites dog" vs. "dog bites man") produce different encodings in a transformer model?**

- The main reason is that transformer models use **positional encoding** to capture the order of words in a sequence.
- Although both sentences contain the same words, their **order** changes the meaning, and the model must distinguish between them.
- In transformers, each word embedding is combined with a positional encoding that represents its position in the sequence.
- This allows the model to understand not just which words are present, but also their order and context within the sentence.
- As a result, "man bites dog" and "dog bites man" will have different positional encodings for each word, leading to different overall sentence representations (encodings).
- This is crucial for tasks like translation, summarization, and question answering, where word order significantly affects meaning.

---

- In summary, positional encoding ensures that the transformer model is sensitive to word order, so sentences with the same words in different orders produce different encodings and, therefore, different contextual meanings.

---

**Q: How do you handle API throttling (rate limit exceeded) exceptions in production applications?**

- To handle throttling (rate limit exceeded, e.g., HTTP 429) in production, I use a combination of client-side and server-side strategies to ensure reliability and a good user experience:
 - **Exponential Backoff & Retry**: Implement automatic retries with exponential backoff when a 429 error is received. This means waiting progressively longer between each retry attempt to avoid overwhelming the API.
 - **Rate Limiting on Client Side**: Proactively limit the number of requests sent from the application, using token buckets or leaky bucket algorithms, to stay within the allowed quota.
 - **Graceful Error Handling**: Show user-friendly error messages or fallback responses when throttling occurs, so users are aware of the temporary issue.
 - **Queueing Requests**: Buffer or queue requests on the client side and release them gradually to avoid bursts that can trigger throttling.
 - **Monitoring & Alerting**: Set up monitoring (e.g., using CloudWatch, Grafana) to track request rates, error rates, and latency. Trigger alerts if throttling errors spike, so the team can investigate and adjust limits or usage patterns.
 - **API Key Management**: If possible, distribute load across multiple API keys or service accounts, while respecting provider policies.
 - **Quota Increase Requests**: For critical workloads, coordinate with the API provider to request higher rate limits or dedicated quota.
 - **Circuit Breaker Pattern**: Temporarily halt requests to the API if repeated throttling is detected, and resume only after a cooldown period, to prevent cascading failures.
 - **Caching**: Cache frequent or repeated responses to reduce unnecessary API calls.

- In my recent projects (like Knowledge GPT and MCP-based adapters), I also emit telemetry and trace spans for all API calls, so we can analyze throttling patterns and optimize request flows. This helps in both proactive prevention and rapid troubleshooting of rate limit issues in production.

---

**Q: Why do users encounter throttling exceptions when using an API like Gemini in production?**

- Throttling exceptions occur when the number of API requests exceeds the allowed rate or quota set by the API provider.
- Common reasons for throttling:
 - **Rate Limits**: The API enforces a maximum number of requests per second/minute/hour to prevent abuse and ensure fair usage.
 - **Concurrent Connections**: There may be a cap on the number of simultaneous requests or connections.
 - **Quota Exhaustion**: Daily or monthly usage quotas may be exceeded, especially in production with many users.
 - **Burst Traffic**: Sudden spikes in traffic (e.g., batch jobs, peak user activity) can trigger throttling if they exceed burst limits.
- In production, increased user activity or lack of proper request pacing can easily lead to hitting these limits.
- API providers (like Gemini, OpenAI, etc.) typically return specific error codes (e.g., HTTP 429) when throttling occurs.
- To mitigate:
 - Implement exponential backoff and retry logic in your client.
 - Monitor and optimize request rates.
 - Consider requesting higher quotas from the API provider if needed.
 - Use batching or caching strategies to reduce redundant calls.

---

- In summary, throttling exceptions are a protective mechanism by the API provider to manage load and ensure service stability. They are triggered when your application exceeds the allowed request rate or quota. Proper rate limiting, retries, and quota management are essential for robust production deployments.

---

**Q: How do you debug and address hallucinations in a RAG (Retrieval-Augmented Generation) system? (Crisp, concise points)**

- Check retrieval quality: Ensure top-k retrieved documents are relevant to the query.
- Analyze chunking & embeddings: Validate chunk size, overlap, and embedding model accuracy.
- Prompt engineering: Refine prompts to enforce stricter context usage and reduce open-endedness.
- Evaluate context utilization: Use automated evaluators to check if answers actually use retrieved context (e.g., chunk alignment scoring).
- Monitor LLM output: Log and review hallucinated responses for patterns.
- Ground truth comparison: Benchmark against known answers to quantify hallucination rate.
- Adjust retrieval parameters: Tune similarity thresholds and retrieval strategies.
- Add guardrails: Use post-processing or LLM-based evaluators to flag or filter hallucinated content.

---

**Q: When do you use hybrid search in a retrieval system?**

- Use hybrid search when you want to combine the strengths of both lexical (keyword-based) and semantic (embedding-based) retrieval to maximize relevance and coverage.
- It is especially effective when:
 - Queries may contain both specific keywords (e.g., product names, error codes) and natural language questions.
 - You need to handle cases where either lexical or semantic search alone might miss relevant results.
 - The content base includes both structured (exact match) and unstructured (conceptual) information.
 - You want to improve recall and precision by blending BM25 (lexical) and KNN (semantic) results, as in ranked_hybrid or linear_hybrid modes.
- In production (e.g., Knowledge GPT), hybrid search is the default for best overall relevance, using Reciprocal Rank Fusion or weighted scoring to merge results from both methods.

---

**Q: When do you use TF-IDF and BM25 in hybrid search?**

- Use **TF-IDF** or **BM25** as the lexical (keyword-based) component in hybrid search when you need:
 - Precise keyword matching, especially for queries with product names, error codes, or technical terms.
 - To complement semantic search by capturing exact matches that embeddings might miss.
 - To improve recall and precision by ensuring both conceptual (semantic) and literal (lexical) matches are considered.
- **BM25** is generally preferred over TF-IDF in modern search systems because:
 - BM25 provides better ranking by considering term frequency saturation and document length normalization.
 - It is more robust for longer documents and varied query types.
- In hybrid search (as in Knowledge GPT), BM25 is typically used in parallel with vector (semantic) search, and results are fused (e.g., using Reciprocal Rank Fusion or weighted scoring) to maximize relevance and coverage.
- Use TF-IDF if you need a simpler, faster, or more interpretable baseline, but BM25 is the standard for production-grade hybrid search.

---

**Q: Why do LLMs sometimes fail at complex reasoning, and what are the possible fixes?**

- **Reasons for Failure:**
 - LLMs are primarily trained on next-token prediction, not explicit logical or multi-step reasoning.
 - Lack of high-quality, reasoning-focused data in the training set.
 - Limited context window may cause loss of relevant information for multi-step reasoning.
 - Model may hallucinate or make unsupported inferences when context is ambiguous or insufficient.
 - Absence of explicit reasoning chains or intermediate steps in prompt or training data.

- **Fixes:**
 - Fine-tune the LLM on datasets specifically designed for reasoning (e.g., chain-of-thought, step-by-step solutions).
 - Use prompt engineering to explicitly request reasoning steps (e.g., "Let's think step by step").
 - Incorporate retrieval-augmented generation (RAG) to provide supporting facts or context for complex queries.
 - Use tool-augmented LLMs or agentic frameworks (e.g., LangChain, CrewAI) to break down and solve reasoning tasks in steps.
 - Increase model size or use more advanced architectures if resource permits.
 - Evaluate and monitor reasoning performance using automated evaluators and ground truth comparisons, as done in Knowledge GPT's evaluation pipeline.

---
