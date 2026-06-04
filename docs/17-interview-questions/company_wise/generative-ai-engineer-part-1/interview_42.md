# Generative AI Engineer (Part 1) — Interview 42

**Q: Write a simple Python class for a calculator with addition and subtraction, taking input via the constructor.**

- The task is to create a Python class named `Calculator`.
- The class should accept two numbers as input through its constructor.
- It should have two methods: one for addition and one for subtraction.
- Both methods should operate on the numbers provided during initialization.

---

**💻 Code**:
```python
class Calculator:
 def __init__(self, num1, num2):
 # Store the two numbers as instance variables
 self.num1 = num1
 # Store the first number
 self.num2 = num2
 # Store the second number

 def add(self):
 # Return the sum of the two numbers
 return self.num1 + self.num2
 # Perform addition and return the result

 def subtract(self):
 # Return the difference between the two numbers
 return self.num1 - self.num2
 # Perform subtraction and return the result

# Example usage:
# calc = Calculator(10, 5)
# print(calc.add()) # Output: 15
# print(calc.subtract()) # Output: 5
```

---

⭐️ **Explanation**:

- The `__init__` method initializes the class with two numbers.
- The `add` method returns the sum of the two numbers.
- The `subtract` method returns the difference (first minus second).
- This structure is simple, readable, and follows Python best practices for class design.
- Time complexity for both operations is O(1) since they are basic arithmetic operations.
- This approach is practical and aligns with industry standards for writing clean, maintainable code.

---

**Q: Flatten a nested list in Python (e.g., x = [1, 2, 3, [4, 6, [7]]] → output = [1, 2, 3, 4, 5, 6, 7])**

- The problem is to flatten a nested list of arbitrary depth into a single list of integers.
- We need a recursive approach to handle any level of nesting.
- We'll define a function that checks if an element is a list; if so, it recursively flattens it, otherwise, it adds the element to the result.

**🔑 Key Steps**:
- Define a function `flatten_list` that takes a list as input.
- Iterate through each element in the list.
- If the element is a list, recursively call `flatten_list` on it and extend the result.
- If the element is not a list, append it to the result.
- Return the flattened list.

**💻 Code**:
```python
def flatten_list(nested_list):
 # Initialize an empty list to store the flattened elements
 flat = []
 # Iterate through each element in the input list
 for item in nested_list:
 # Check if the current item is a list
 if isinstance(item, list):
 # If it is a list, recursively flatten it and extend the result
 flat.extend(flatten_list(item))
 # Add all elements from the flattened sublist to the main list
 else:
 # If it is not a list, append the item directly to the result
 flat.append(item)
 # Add the non-list item to the result
 # Return the fully flattened list
 return flat

# Example usage:
x = [1, 2, 3, [4, 5, [6, 7]]]
# Call the flatten_list function with the nested list
output = flatten_list(x)
# Print the flattened output
print(output) # Output: [1, 2, 3, 4, 5, 6, 7]
```

**💡 Explanation**:
- The function uses recursion to handle any level of nesting.
- For each element, if it's a list, it flattens it further; otherwise, it adds the element to the result.
- This approach ensures all nested elements are extracted in order.
- **Time Complexity**: O(N), where N is the total number of elements (including all nested elements).
- **Space Complexity**: O(N), due to the storage of the flattened list and recursion stack.
- This is a standard, industry-accepted way to flatten arbitrarily nested lists in Python.

---

**Q: What is "faithfulness" in the context of RAG/LLM evaluation, especially when given components like query and context?**

- In the context of Retrieval-Augmented Generation (RAG) and LLM evaluation, "faithfulness" refers to how accurately the generated answer reflects the facts and information present in the provided context or source documents.
- A faithful answer does not introduce hallucinations, fabrications, or unsupported claims; it strictly adheres to the evidence in the retrieved context.
- Faithfulness is a critical metric for evaluating RAG systems because it ensures that the LLM's output is grounded in the retrieved knowledge and not just plausible-sounding but incorrect information.

**Industry Practice & Project Experience:**
- In my recent work on the Knowledge GPT (KGPT) evaluation pipeline, we measured faithfulness using an "Answer Correctness Evaluator."
 - This evaluator checks if the generated answer is factually and semantically correct compared to the ground truth.
 - We use a combination of LLM-based (e.g., GPT-4o as a judge) and embedding-based metrics to assess factual accuracy and semantic similarity.
 - The evaluation pipeline sends the answer and ground truth to the LLM, which returns a factual accuracy score, ensuring the answer is faithful to the context.
- Faithfulness is often weighted heavily in the overall evaluation score, as unfaithful answers can lead to misinformation and erode user trust.

**Summary:**
- Faithfulness = factual alignment of the answer with the provided context.
- It is measured by checking if the answer is supported by the retrieved context and does not introduce unsupported information.
- In production RAG systems, automated evaluators (often LLM-based) are used to score faithfulness as part of the quality assurance process.

---

**Q: For measuring faithfulness in a RAG/LLM system, which components among query, context, output, and ground truth are required?**

- To accurately measure faithfulness in a RAG or LLM system, the key components required are:
 - **Output (Generated Answer)**
 - **Context (Retrieved Evidence)**
- Optionally, **Ground Truth** is used for more robust evaluation, but for strict faithfulness (i.e., is the answer grounded in the provided context), the primary focus is on the Output and Context.

**Detailed Reasoning:**
- **Faithfulness** evaluates whether the generated answer is factually supported by the retrieved context, not just whether it matches the ground truth.
- In the KGPT Evaluation Pipeline, faithfulness (termed as "Answer Correctness") is primarily measured by comparing the output to the context:
 - The system uses an LLM (e.g., GPT-4o) to judge if the answer is factually accurate and semantically aligned with the context.
 - The ground truth is used for additional correctness and benchmarking, but the core faithfulness metric is about grounding in the context.
- **Query** is important for relevance and context retrieval, but not directly for faithfulness scoring.

**Summary Table:**

| Component | Faithfulness Role |
|--------------|----------------------------------|
| Query | Not directly required |
| Context | Required (retrieved evidence) |
| Output | Required (generated answer) |
| Ground Truth | Optional (for benchmarking) |

- In practice, for faithfulness: **Context + Output** are essential; **Ground Truth** is used for additional validation but not strictly necessary for faithfulness itself.

---

**Q: For context precision in RAG/LLM evaluation, which two components are required to measure it?**

> ⚠️ **Important distinction:** This question conflates two different metrics. The KGPT project uses a custom **Context Utilization** metric, which is NOT the same as the standard RAGAS **Context Precision** metric. Both are described below.

**Standard RAGAS Definition — Context Precision:**
- Components required: **Retrieved Context** + **Ground Truth** (reference answer)
- Measures the signal-to-noise ratio in retrieval: what fraction of retrieved chunks are actually relevant to answering the question?
- A chunk is "relevant" if it contains information needed to produce the ground truth answer.
- Formula: weighted precision across retrieved chunks = relevant_chunks / total_retrieved_chunks

**KGPT Project's Custom Metric — Context Utilization (≠ RAGAS Context Precision):**
- Components required: **Context (Retrieved Chunks)** + **Output (Generated Answer)**
- Measures how much of the retrieved context is actually reflected in the generated answer (utilization rate).
- According to the KGPT Evaluation Pipeline: the **ContextUtilizationEvaluator** uses `CHUNK_ALIGNMENT_PROMPT` to analyze which sentences from retrieved chunks appear in the answer.
- This is closer to **faithfulness from the retrieval side**, not standard context precision.

**Summary Table:**

| Metric | Component 1 | Component 2 | Measures |
|--------------------------|-------------------|-------------------|---------------------------------------------|
| Context Precision (RAGAS)| Retrieved Context | Ground Truth | Are retrieved chunks relevant to GT answer? |
| Context Utilization (KGPT)| Retrieved Context | Generated Answer | How much of context was used in answer? |

- In an interview, if asked about **RAGAS context precision**, answer with: **Retrieved Context + Ground Truth**.

---

**Q: For context precision, what exactly are the two things that need to be compared?**

> ⚠️ **Clarification:** The answer depends on whether you are using the standard **RAGAS framework** or the **KGPT project's custom evaluation**.

**Standard RAGAS — Context Precision:**
- Compare: **Retrieved Context Chunks** ↔ **Ground Truth (reference answer)**
- For each retrieved chunk, ask: "Does this chunk contain information needed to produce the ground truth answer?"
- This tells you whether the retrieval is fetching relevant content (signal-to-noise ratio).

**KGPT Custom — Context Utilization (what KGPT calls "context precision"):**
- Compare: **Retrieved Context Chunks** ↔ **Generated Answer**
- The **ContextUtilizationEvaluator** uses `CHUNK_ALIGNMENT_PROMPT` to check which sentences from retrieved chunks were reflected in the generated answer.
- This measures utilization rate — how much of what was retrieved was actually used.

**Summary Table:**

| Metric | Compare Component 1 | With Component 2 | Framework |
|---------------------------|---------------------------|---------------------|-------------|
| Context Precision (RAGAS) | Retrieved Context Chunks | Ground Truth | RAGAS |
| Context Utilization (KGPT)| Retrieved Context Chunks | Generated Answer | KGPT Custom |

- **For interviews:** Always use the RAGAS definition unless asked specifically about the KGPT project.

---

**Q: What exactly is context precision in RAG/LLM evaluation, and how is it measured?**

> ⚠️ **Two definitions exist — make sure to use the right one in the right context.**

**Definition 1: RAGAS Context Precision (Industry Standard)**
- Measures the **signal-to-noise ratio** in the retrieved context: what fraction of retrieved chunks are actually relevant to the question?
- A chunk is "relevant" if it contains information that supports the ground truth answer.
- **Components:** Retrieved Context + Ground Truth
- **Formula:** weighted precision = `Σ (precision@k × relevance_k) / total_relevant_chunks`
- High context precision = most retrieved chunks are useful, not noise.
- Low context precision = many irrelevant chunks retrieved, diluting the signal.

**Definition 2: KGPT Context Utilization (Custom Project Metric)**
- In the KGPT project, the `ContextUtilizationEvaluator` uses `CHUNK_ALIGNMENT_PROMPT` to measure how much of the retrieved context was reflected in the generated answer.
- **Components:** Retrieved Context + Generated Answer
- Returns per-chunk utilization scores and a weighted `context_utilization_score`.
- This metric is internally named "context precision" in the KGPT pipeline but is better described as **context utilization** — it is NOT the same as RAGAS context precision.

**Summary Table:**

| Metric | What’s Compared | Framework | Measures |
|---------------------------|-----------------------------------|--------------|---------------------------------------|
| Context Precision (RAGAS) | Retrieved Context vs. Ground Truth| RAGAS | Signal-to-noise in retrieval |
| Context Utilization (KGPT)| Retrieved Context vs. Generated Answer | KGPT | How much retrieved context was used |

- **Interview default:** Use the RAGAS definition (context vs. ground truth = signal-to-noise).

---

**Q: For context precision, should the retrieved context be compared with the query or the ground truth?**

- For **standard RAGAS context precision**, the retrieved context is compared with the **ground truth** (reference answer).

**Why ground truth, not query alone?**
- The query is used to *retrieve* chunks, but it does not tell you if what was retrieved is actually *correct and sufficient*.
- Ground truth lets you check whether the retrieved chunks contain the information needed to produce the correct answer.
- A chunk may match the query semantically but still not contain the right answer — ground truth catches this.

**The RAGAS approach:**
- For each retrieved chunk, ask: "Does this chunk contain information that supports or is required by the ground truth answer?"
- If yes → relevant chunk (contributes to precision).
- If no → irrelevant chunk (noise, lowers precision).

**KGPT Note:**
- The KGPT pipeline also computes a `context_relevance_score` using both the **question** and **context** (via `CONTEXT_RELEVANCE_PROMPT`) — this is a relevance score, not the same as context precision.
- Separately, the `ContextUtilizationEvaluator` compares **retrieved context** with **generated answer** — this is context utilization.

**Summary Table:**

| Metric | Compare | With |
|------------------------------|---------------------|-----------------|
| Context Precision (RAGAS) | Retrieved Context | Ground Truth ✅ |
| Context Relevance (KGPT) | Retrieved Context | Query |
| Context Utilization (KGPT) | Retrieved Context | Generated Answer|

- **Interview answer:** Context precision = retrieved context vs. **ground truth** (RAGAS standard).

---

**Q: What is context recall in RAG/LLM evaluation, and how is it measured?**

- **Context recall** measures how much of the ground truth information (i.e., the reference answer or evidence) is actually present in the retrieved context.
- It evaluates whether the retrieved context contains all the necessary information required to answer the question as per the ground truth.

**How It’s Measured (Based on Knowledge GPT Evaluation Pipeline):**
- The process involves comparing the **ground truth** (reference answer or expected evidence) with the **retrieved context** (chunks).
- The metric checks what proportion of the ground truth content is covered or matched by the retrieved context.
- High context recall means the retrieved context includes most or all of the information needed to answer the question correctly (as per the ground truth).
- Low context recall indicates that important parts of the ground truth are missing from the retrieved context.

**Key Components Compared:**
- **Ground Truth** (reference answer/evidence)
- **Retrieved Context Chunks** (evidence retrieved by the system)

**Summary Table:**

| Metric | Compare Component 1 | With Component 2 | Purpose |
|----------------|---------------------|-----------------------|-------------------------------------------------------|
| Context Recall | Ground Truth | Retrieved Context | Measures how much of the ground truth is present in the retrieved context |

- In summary, **context recall** is about ensuring the retrieved context is comprehensive and covers all the key information from the ground truth, which is critical for accurate and complete answers in RAG/LLM systems.

---

**Q: What are the two components compared for context precision?**

- For **context precision**, the two components you compare are:
 - **Retrieved Context Chunks** (the evidence or context retrieved by the system)
 - **Ground Truth** (the reference answer or evidence)

**Industry Practice & Documentation Reference:**
- Context precision measures how much of the retrieved context is actually relevant to the ground truth.
- According to the Knowledge GPT Evaluation Pipeline, context precision is about aligning the retrieved context with the ground truth, ensuring that what you retrieve is not just related to the query, but is actually correct and necessary for the answer.
- This is distinct from context recall, which checks if all necessary ground truth information is present in the retrieved context.

**Summary Table:**

| Metric | Compare Component 1 | With Component 2 |
|--------------------|-------------------------|-----------------------|
| Context Precision | Retrieved Context | Ground Truth |

- In summary, for context precision, you compare the **retrieved context** with the **ground truth** to determine the proportion of retrieved information that is actually correct and relevant.

---

**Q: What does context precision mean, and is it measured by comparing the retrieved context with the query?**

- **Context precision** is typically not measured by comparing the retrieved context with the query.
- In standard RAG/LLM evaluation pipelines (including Knowledge GPT), context precision refers to how much of the retrieved context is actually relevant to the **ground truth** (i.e., the reference answer or evidence), not just the query.
- Comparing context with the query is more related to **context relevance** (i.e., does the retrieved context relate to the user's question), but not precision in the strict evaluation sense.

**Industry Standard Mapping:**
- **Context Recall:** Retrieved context vs. ground truth (measures coverage of ground truth in context)
- **Faithfulness:** Generated answer vs. context (measures if the answer is grounded in the retrieved context)
- **Context Precision:** Retrieved context vs. ground truth (measures how much of the retrieved context is actually correct/relevant)
- **Context Relevance:** Retrieved context vs. query (measures if the context is related to the question, but not precision)

**Summary Table:**

| Metric | Compare Component 1 | With Component 2 |
|--------------------|-------------------------|-----------------------|
| Context Precision | Retrieved Context | Ground Truth |
| Context Recall | Ground Truth | Retrieved Context |
| Faithfulness | Generated Answer | Retrieved Context |
| Context Relevance | Retrieved Context | Query |

- So, **context precision** is not about context vs. query, but about context vs. ground truth. Context vs. query is context relevance.

**interviewee**: Versus query will be the final presence. Exactly. That is the key. You have give it. If you do not know the answer. Also, the answer should be no no, no. That is because there are two things you have already said. Okay. Good. Uh, can you tell me what is called the answer relevancy.

---

**Q: What is answer relevance in LLM/RAG evaluation, and how is it measured?**

- **Answer relevance** measures whether the generated answer actually addresses the question asked, regardless of whether it is factually correct.
- It focuses on the alignment between the answer and the user's query, ensuring the response is on-topic and covers the intent of the question.

**How It’s Measured (Knowledge GPT Evaluation Pipeline):**
- The **Answer Relevance Evaluator** uses two sub-metrics:
 1. **Answer Conformity** (LLM-based): Checks if the answer covers all parts of the question.
 - Uses GPT-4o with a specific prompt, scoring from 0.0 (does not address the question) to 1.0 (fully answers all parts).
 2. **Question Conformity** (Embeddings-based): Checks if the answer, when used as a query, would retrieve a similar question.
 - Uses embeddings to compute cosine similarity between the answer and the question.

- The final **answer_relevance_score** is a weighted sum:
 - `answer_relevance_score = (answer_conformity_score × 0.7) + (question_conformity_score × 0.3)`

**Summary Table:**

| Metric | What’s Compared | Purpose |
|-------------------|------------------------|----------------------------------------------|
| Answer Relevance | Answer vs. Question | Measures if the answer addresses the question|

- In summary, answer relevance ensures that the generated response is contextually appropriate and directly responds to the user's query, even before checking for factual correctness. This is critical for user satisfaction and system usability in real-world applications.

---

**Q: In a RAG system, if the generated answer is correct but the supporting evidence is only available on the internet (not in your internal knowledge base), how do you evaluate answer correctness?**

- In a RAG (Retrieval-Augmented Generation) system, the evaluation of answer correctness typically relies on comparing the generated answer to a ground truth reference, not just the internal knowledge base.
- If the answer is factually correct but the supporting evidence is not present in your internal corpus, you can still evaluate correctness using external ground truth data.

**Industry-Standard Approach (as per Knowledge GPT Evaluation Pipeline):**
- **Answer Correctness** is measured by comparing the generated answer to the ground truth, regardless of where the evidence is stored.
 - Uses LLM-based factual accuracy scoring (e.g., GPT-4o as a judge).
 - Uses semantic similarity (embeddings) between the answer and the ground truth.
 - Optionally, checks reference health if URLs are cited.
- The evaluation does not require the answer to be grounded in your internal knowledge base for correctness scoring.
 - If the answer matches the ground truth (factually and semantically), it is considered correct.
 - If you want to enforce that answers must be grounded in your internal data, you would use a separate metric like **faithfulness** or **context utilization**.

**Practical Steps:**
- Use your evaluation pipeline to compare the generated answer with the ground truth (which may be sourced from the internet or a curated dataset).
- Score factual accuracy and semantic similarity.
- If the answer is correct but not grounded in your internal context, it will score high on correctness but may score low on faithfulness/context utilization.

**Summary Table:**

| Metric | What’s Compared | Purpose |
|---------------------|-------------------------------|----------------------------------------------|
| Answer Correctness | Answer vs. Ground Truth | Is the answer factually/semantically correct?|
| Faithfulness | Answer vs. Retrieved Context | Is the answer grounded in internal evidence? |

- In summary, you do not need to retrain the model. You evaluate correctness by comparing the answer to the ground truth, even if the evidence is not in your internal knowledge base. If you want to ensure answers are only based on internal data, use faithfulness/context utilization metrics in addition to correctness.

---

**Q: How do you obtain or define ground truth for answer correctness evaluation when the answer is not in the internal knowledge base?**

- In practical RAG/LLM evaluation pipelines, the **ground truth** is a curated reference answer or evidence set, typically prepared in advance and stored as part of your evaluation dataset.
- The ground truth is **not dynamically generated from your RAG system**; instead, it is manually or semi-automatically created by subject matter experts, data annotators, or by leveraging trusted external sources (such as the internet, documentation, or authoritative datasets).
- During evaluation, the generated answer is compared against this pre-existing ground truth, regardless of whether the RAG system’s internal corpus contains the supporting evidence.
- This approach ensures that answer correctness can be objectively measured, even if the RAG system’s knowledge base is incomplete or missing relevant information.

**Industry Practice (as implemented in Knowledge GPT Evaluation Pipeline):**
- The evaluation pipeline loads a ground truth dataset (often in CSV/JSON format) where each question is paired with its reference answer(s).
- When a generated answer is produced, it is compared to the ground truth using:
 - LLM-based factual accuracy scoring (e.g., GPT-4o as judge)
 - Semantic similarity (embeddings)
 - Reference health (if URLs are cited)
- The ground truth is essential for objective, repeatable evaluation and is **independent of the RAG’s current knowledge base**.

**Summary Table:**

| Component | Source/How Obtained |
|-------------------|-----------------------------------------------------|
| Ground Truth | Curated reference answers (manual/expert/external) |
| Generated Answer | Output from RAG/LLM system |
| Evaluation | Compare generated answer to ground truth |

- In summary, to measure answer correctness, you must maintain a separate, high-quality ground truth dataset. This allows you to evaluate the factual and semantic accuracy of generated answers, even if the RAG system itself does not contain the supporting evidence. This is a standard practice in industry-grade AI evaluation pipelines.

---

**Q: If a RAG system cannot retrieve the answer from its knowledge base, how should the code handle such cases?**

- In a RAG (Retrieval-Augmented Generation) system, if the retrieval step does not return relevant context (i.e., the answer is not found in the knowledge base), the system should handle this gracefully to avoid generating hallucinated or unsupported answers.
- Industry-standard approaches for handling such cases include:
 - Returning a fallback response indicating insufficient information.
 - Logging the query for future knowledge base expansion.
 - Optionally, allowing the LLM to answer based on its pre-trained knowledge, but clearly flagging that the answer is not grounded in retrieved context.

**Practical Implementation Steps:**
- **Step 1:** After retrieval, check if the retrieved context is empty or below a relevance threshold.
- **Step 2:** If no relevant context is found:
 - Return a response like: "Sorry, I do not have enough information to answer this question based on the available knowledge base."
 - Optionally, log the query for review and future data ingestion.
- **Step 3:** If you allow the LLM to answer anyway, clearly indicate in the response or metadata that the answer is not grounded in the internal knowledge base.

**Example Python Pseudocode:**
```python
# Retrieve context for the query
retrieved_context = retrieve_context(query) # Retrieves relevant documents/chunks

# Check if context is sufficient
if not retrieved_context or relevance_score(retrieved_context) < threshold:
 # Not enough information found in the knowledge base
 return {
 "answer": "Sorry, I do not have enough information to answer this question based on the available knowledge base.",
 "grounded": False
 }
else:
 # Proceed with RAG pipeline (context + LLM generation)
 answer = generate_answer(query, retrieved_context)
 return {
 "answer": answer,
 "grounded": True
 }
```
# Each line is commented for clarity.

**Industry Best Practices:**
- This approach prevents hallucinations and maintains user trust.
- Logging unanswered queries helps improve the knowledge base over time.
- Clearly distinguishing between grounded and ungrounded answers is critical for enterprise applications.

- In summary, always check retrieval results before generating an answer, and handle missing context with a clear, user-friendly fallback response. This is a standard, robust practice in production RAG systems.

---

**Q: What is a context window in the context of LLMs and RAG systems?**

- The **context window** refers to the maximum number of tokens (words, subwords, or characters) that a language model (like GPT-4, Gemini, etc.) can process in a single input sequence.
- It defines how much information (including the user query, retrieved context, system prompts, and conversation history) can be provided to the model at once.
- In RAG systems, the context window limits how much retrieved knowledge (document chunks, passages) can be included alongside the user query for answer generation.
- If the combined input exceeds the context window, you must truncate, summarize, or select the most relevant context to fit within the limit.

**Practical Details:**
- For example, GPT-4o supports up to 128k tokens, while older models like GPT-3.5 support 4k or 16k tokens.
- The context window impacts retrieval strategies, chunk sizing, and prompt engineering in production RAG pipelines.
- Efficient use of the context window is critical for maximizing answer quality and minimizing hallucinations.

**Industry Practice:**
- Always monitor and manage the total token count (query + context + system instructions) to avoid truncation or errors.
- Use chunking, ranking, and summarization to fit the most relevant information within the context window.

- In summary, the context window is the model’s memory span for a single request, directly affecting how much information can be used for reasoning and answer generation.

---

**Q: What are the input components that make up the context window in a RAG (Retrieval-Augmented Generation) pipeline?**

- In a RAG pipeline, the context window is filled with several key input components that together form the prompt for the LLM. These components must all fit within the model’s maximum token limit.

**Typical Input Components in a RAG Context Window:**
- **User Query:** The original question or prompt from the user.
- **Retrieved Context:** Top-ranked document chunks or passages fetched from the knowledge base (using vector/semantic and/or lexical search).
 - Each chunk may include: content, source, title, and sometimes metadata.
- **System Instructions/Prompt Template:** System-level instructions or persona settings that guide the LLM’s behavior (e.g., “You are an enterprise assistant…”).
- **Conversation History (if applicable):** Previous turns in the conversation, if maintaining context across multiple exchanges.
- **Additional Metadata (optional):** Such as language, product, disclosure level, or other parameters relevant to the query.

**Example Breakdown (as per Knowledge GPT API Architecture):**
- System prompt (from PROMPT_TEMPLATE)
- User input (query)
- Retrieved context (concatenated top chunks with source/title)
- (Optional) Conversation history

**Practical Considerations:**
- All these components are concatenated and must not exceed the model’s context window (max tokens).
- If the combined input is too large, you must truncate or prioritize the most relevant context chunks.

**Summary Table:**

| Component | Description |
|------------------------|---------------------------------------------------------|
| System Instructions | Prompt template/system message |
| User Query | The user’s question or command |
| Retrieved Context | Top document chunks/passages from search |
| Conversation History | Previous Q&A turns (if multi-turn) |
| Metadata (optional) | Language, product, disclosure, etc. |

- In summary, the context window in a RAG pipeline is composed of the system prompt, user query, retrieved context, and optionally conversation history and metadata—all of which must fit within the model’s maximum token limit for effective answer generation.

---

**Q: In hybrid search (semantic + lexical), if you retrieve five results from each method, how do you select the best five overall?**

- In a hybrid search setup, both semantic (vector/KNN) and lexical (BM25) searches are run in parallel, each returning their top N results (e.g., 5 each).
- To select the best five overall, you need to combine and rank these results in a way that leverages the strengths of both methods.

**Industry-Standard Approach (as per Knowledge GPT API Architecture):**
- Use a **Reciprocal Rank Fusion (RRF)** algorithm to merge and rank results from both searchers.
 - RRF assigns a score to each document based on its rank in each result set.
 - The formula is: 
 `rrf_score = 1 / (rank_in_lexical + k) + 1 / (rank_in_semantic + k)`
 - `k` is a constant (e.g., 60), used to smooth the scores.
 - If a document appears in both lists, it gets a higher combined score.
 - If a document is missing from one list, its rank defaults to the window size (e.g., 5 or 20).
- After calculating RRF scores for all unique documents, sort them in descending order of score.
- Select the top five documents as the final best results.

**Key Steps:**
- Run both searches in parallel.
- Build rank maps for each result set.
- Calculate RRF scores for all unique documents.
- Sort by RRF score and select the top five.

**Summary Table:**

| Step | Description |
|---------------------|---------------------------------------------------------|
| 1. Retrieve | Get top 5 from semantic, top 5 from lexical |
| 2. Rank Mapping | Map document IDs to their ranks in each list |
| 3. RRF Scoring | Compute RRF score for each unique document |
| 4. Sort & Select | Sort by RRF score, pick top 5 overall |

- This approach ensures a balanced, relevance-optimized selection that combines both semantic meaning and exact keyword matching, as implemented in enterprise RAG systems like Knowledge GPT.

---

**Q: How does a Decision Tree algorithm work?**

- A Decision Tree is a supervised machine learning algorithm used for both classification and regression tasks.
- It works by recursively splitting the dataset into subsets based on feature values, aiming to maximize the separation of classes (for classification) or minimize prediction error (for regression).
- The tree is built from a root node, with each internal node representing a decision based on a feature, and each leaf node representing a final prediction or class label.

**Key Steps in Decision Tree Construction:**
- **1. Select the Best Feature to Split:** 
 - At each node, evaluate all features and choose the one that best separates the data. 
 - For classification, common criteria are Gini impurity or Information Gain (entropy). 
 - For regression, criteria like Mean Squared Error (MSE) are used.
- **2. Split the Data:** 
 - Partition the dataset into subsets based on the selected feature’s values (for categorical) or threshold (for numerical).
- **3. Repeat Recursively:** 
 - For each subset, repeat the process to create child nodes, until a stopping condition is met (e.g., all samples in a node belong to the same class, or a maximum tree depth is reached).
- **4. Assign Leaf Nodes:** 
 - When a node can’t be split further, it becomes a leaf node and is assigned a class label (classification) or a value (regression).

**Example (Classification):**
- At the root, the algorithm checks all features and splits on the one that best separates the classes.
- This process continues recursively, creating a tree structure where each path from root to leaf represents a sequence of decisions.

**Advantages:**
- Easy to interpret and visualize.
- Handles both numerical and categorical data.
- No need for feature scaling.

**Limitations:**
- Prone to overfitting (can be mitigated by pruning or setting max depth).
- Can be unstable with small data changes.

- In summary, a Decision Tree splits data based on feature values to create a flowchart-like structure, making decisions at each node to reach a final prediction at the leaves. This approach is widely used for its interpretability and flexibility in both classification and regression tasks.

---

**Q: What are the key criteria (keywords) used for splitting nodes in a Decision Tree?**

- The main "keywords" or criteria used for splitting nodes in a Decision Tree are mathematical measures that evaluate the quality of a split. These are:

 - **Gini Impurity:** 
 - Measures the probability of incorrectly classifying a randomly chosen element if it was randomly labeled according to the distribution of labels in the subset.
 - Used by default in algorithms like CART (Classification and Regression Trees).

 - **Information Gain (Entropy):** 
 - Entropy measures the impurity or randomness in the data.
 - Information Gain calculates the reduction in entropy after a dataset is split on a feature.
 - Used in algorithms like ID3 and C4.5.

 - **Gain Ratio:** 
 - Adjusts Information Gain by taking into account the intrinsic information of a split (used in C4.5).

 - **Mean Squared Error (MSE):** 
 - Used for regression trees to measure the variance of the target variable within the subsets.

- **Summary Table:**

| Criterion | Used For | Description |
|---------------------|------------------|--------------------------------------------------|
| Gini Impurity | Classification | Measures node impurity (lower is better) |
| Entropy/Info Gain | Classification | Measures reduction in randomness after split |
| Gain Ratio | Classification | Normalizes Info Gain for feature selection |
| Mean Squared Error | Regression | Measures variance within splits |

- In practice, the Decision Tree algorithm evaluates all possible splits for each feature using these criteria and selects the split that results in the highest "purity" (lowest impurity or highest information gain) at each node.

---

**Q: How do you detect and handle issues when a model is deployed in production?**

- In production, robust observability and monitoring are critical to detect, diagnose, and handle issues with deployed AI models and systems.
- My approach is to implement a comprehensive observability stack covering logging, metrics, distributed tracing, and alerting, as outlined in the MCP (Model Context Protocol) architecture for enterprise AI deployments.

**Key Steps for Detection and Handling:**

- **1. Logging:** 
 - Capture structured logs (e.g., JSON format) for all key events: API requests, model inferences, errors, and system actions.
 - Use centralized log aggregation (e.g., AWS CloudWatch Logs, Fluent Bit, Grafana) for real-time visibility.
 - Include correlation IDs for tracing requests end-to-end across microservices.

- **2. Metrics Collection:** 
 - Track key metrics such as request rate, latency (p50/p95/p99), error rates, and model-specific metrics (accuracy, drift, etc.).
 - Use CloudWatch Metrics and Grafana dashboards for visualization and trend analysis.

- **3. Distributed Tracing:** 
 - Implement distributed tracing (AWS X-Ray, OpenTelemetry) to follow requests across all components, enabling root cause analysis for failures or performance bottlenecks.

- **4. Alerting:** 
 - Set up automated alerts (SNS, Slack, PagerDuty) for anomalies such as high error rates, latency spikes, or degraded model performance.
 - Alerts trigger incident response workflows for rapid investigation and mitigation.

- **5. Audit Logging & Governance:** 
 - Maintain audit trails (CloudTrail) for all critical actions, supporting compliance and post-incident analysis.

- **6. Automated Rollback & Safe Deployment:** 
 - Use CI/CD pipelines (Azure DevOps, GitHub Actions) with automated testing and canary deployments to minimize risk.
 - If an issue is detected, roll back to a previous stable version or switch traffic to a healthy endpoint.

- **7. Incident Handling:** 
 - Investigate logs, traces, and metrics to identify the root cause.
 - Apply hotfixes, retrain or redeploy models as needed.
 - Document incidents and update monitoring rules to prevent recurrence.

**Industry Example (from MCP/Knowledge GPT):**
- All production traffic is monitored via CloudWatch and Grafana.
- Distributed tracing with correlation IDs enables quick diagnosis.
- Alerts are configured for error spikes and latency issues.
- Audit logs ensure traceability for all actions.

- In summary, I ensure production reliability by combining real-time monitoring, structured logging, distributed tracing, automated alerting, and robust incident response, leveraging cloud-native observability tools and best practices for enterprise AI systems.

---

**Q: How do you detect and handle issues in production when the data distribution changes (data drift), not just code failures?**

- In production, data drift—where the input data distribution changes over time—can degrade model performance even if the code is stable.
- Proactively detecting and handling data drift is critical for maintaining reliable AI systems.

**Detection Strategies:**
- **Monitor Input Data Statistics:** 
 - Continuously track key statistics (mean, variance, value distributions) of incoming data and compare them to training data.
 - Use tools like AWS SageMaker Model Monitor or custom scripts to automate this process.
- **Drift Detection Metrics:** 
 - Implement statistical tests (e.g., Kolmogorov-Smirnov, Population Stability Index) to quantify drift between current and baseline data.
- **Monitor Model Output Metrics:** 
 - Track prediction distributions, confidence scores, and output class frequencies for unexpected shifts.
- **Performance Monitoring:** 
 - Continuously evaluate model performance using ground truth labels (if available) and metrics like accuracy, F1-score, or MAPE.
 - Use automated evaluation pipelines (as in Knowledge GPT and UIDS projects) to generate regular reports on model quality.

**Handling Data Drift:**
- **Alerting:** 
 - Set up automated alerts when drift metrics exceed predefined thresholds.
- **Retraining Pipelines:** 
 - Trigger model retraining workflows using the latest data to adapt to new patterns.
 - Use MLOps tools (e.g., SageMaker Pipelines, MLFlow) for automated retraining, versioning, and deployment.
- **Data Quality Checks:** 
 - Implement validation steps to catch anomalies, missing values, or schema changes before data reaches the model.
- **A/B Testing and Canary Releases:** 
 - Deploy updated models to a subset of traffic to validate improvements before full rollout.
- **Audit and Governance:** 
 - Maintain audit trails of data changes, model versions, and retraining events for traceability and compliance.

**Industry Example (from my experience):**
- In the UIDS and Knowledge GPT projects, we implemented automated monitoring pipelines that:
 - Track input data distributions and model outputs.
 - Generate evaluation reports (accuracy, F1, MRR) and upload them to S3.
 - Alert the team if significant drift or performance drop is detected.
 - Support rapid retraining and redeployment using CI/CD and cloud-native MLOps.

- In summary, I ensure production model reliability by combining automated data drift detection, continuous performance monitoring, alerting, and robust retraining pipelines—enabling the system to adapt quickly to changing data and maintain high-quality predictions.

---

**Q: What threshold value of Population Stability Index (PSI) indicates that model retraining is needed due to data drift?**

- The Population Stability Index (PSI) is a widely used metric to quantify data drift by comparing the distribution of a feature (or model input) in current (production) data versus baseline (training) data.
- PSI values help determine when the drift is significant enough to warrant model retraining.

**Industry-Standard PSI Thresholds:**
- **PSI < 0.1:** 
 - No significant drift. 
 - Model is stable; no action needed.
- **0.1 ≤ PSI < 0.25:** 
 - Moderate drift detected. 
 - Monitor closely; consider retraining if performance metrics also degrade.
- **PSI ≥ 0.25:** 
 - Significant drift. 
 - Immediate investigation and likely model retraining required.

**Practical Approach (as used in enterprise pipelines):**
- Set automated alerts when PSI exceeds 0.25 for any critical feature or overall input distribution.
- Trigger retraining pipelines if PSI crosses this threshold, or if moderate drift (0.1–0.25) is accompanied by a drop in key model metrics (e.g., accuracy, F1-score, balanced accuracy).
- Regularly monitor PSI alongside model performance metrics (as seen in the UIDS project, where metrics like balanced accuracy, F1-score, and recall are tracked).

- In summary, a PSI value of **0.25 or higher** is a strong indicator that retraining is needed due to significant data drift. For values between 0.1 and 0.25, monitor closely and consider retraining if other metrics also show degradation.

---

**Q: How do you handle significant data drift (PSI > 0.25) in production, and what type of retraining should be performed?**

- When Population Stability Index (PSI) exceeds 0.25, it indicates significant data drift, which can lead to model performance degradation.
- The best practice is to initiate a robust retraining workflow to adapt the model to the new data distribution.

**Steps to Handle Significant Data Drift:**

- **1. Trigger Automated Retraining Pipeline:**
 - Use MLOps tools (e.g., AWS SageMaker Pipelines, MLFlow) to automate the retraining process.
 - In the UIDS project, retraining is triggered when drift or performance thresholds are breached.

- **2. Data Preparation:**
 - Aggregate recent production data reflecting the new distribution.
 - Combine with a portion of historical data if needed to avoid overfitting to recent anomalies.
 - Apply preprocessing steps: data cleaning, balancing, and feature engineering as per pipeline configuration.

- **3. Model Retraining:**
 - Retrain the model from scratch or use incremental learning if supported by the algorithm.
 - Use the same architecture and hyperparameters unless there is evidence that changes are needed.
 - In UIDS, retraining uses the latest labeled data and tracks metrics like balanced accuracy, F1-score, and recall.

- **4. Model Evaluation:**
 - Evaluate the retrained model on a hold-out validation set and compare key metrics (e.g., balanced accuracy, F1-score, recall) to previous versions.
 - Only promote the new model if it meets or exceeds predefined accuracy thresholds (as configured in the pipeline).

- **5. Deployment and Monitoring:**
 - Deploy the new model using CI/CD pipelines with safe rollout strategies (canary or blue-green deployment).
 - Continue monitoring for further drift or performance issues.

- **6. Documentation and Governance:**
 - Log all retraining events, data versions, and model metrics for auditability and compliance.

**Types of Retraining:**
- **Full Retraining:** 
 - Retrain the model from scratch using the latest and historical data. Preferred when drift is significant.
- **Incremental/Online Learning:** 
 - Update the model incrementally if the algorithm supports it (e.g., some tree-based or neural network models).
- **Transfer Learning:** 
 - Fine-tune a pre-trained model on the new data if using deep learning architectures.

- In summary, when PSI > 0.25, I initiate a full retraining pipeline using the latest production data, validate the new model against key metrics, and deploy it only if it meets accuracy and stability requirements—ensuring the system remains robust against data drift.

---

**Q: In production, after detecting significant data drift, how do you select and prepare the data for retraining—what data do you include or exclude?**

- When retraining a model due to data drift, careful data selection and preparation are crucial to ensure the new model adapts to recent trends without overfitting or introducing bias.

**Practical Steps for Data Selection and Preparation:**

- **1. Aggregate Recent Production Data:**
 - Collect a window of the most recent production data that reflects the new distribution causing the drift.
 - The window size (e.g., last 2–4 weeks) depends on data volume and business context.

- **2. Combine with Historical Data (if needed):**
 - To avoid overfitting to short-term anomalies, blend recent data with a representative sample of historical data.
 - This helps maintain model generalization and robustness.

- **3. Data Cleaning and Validation:**
 - Use automated scripts (as in `prepare_dataset.py` and `data_preparation.py` from the UIDS pipeline) to:
 - Remove duplicates, handle missing values, and ensure schema consistency.
 - Validate data quality and filter out corrupted or irrelevant records.

- **4. Data Balancing and Bias Mitigation:**
 - Apply balancing techniques if class imbalance is detected (see `apply_data_balance` and `remove_biased_intents` in the pipeline).
 - Exclude or downsample overrepresented or biased classes using configuration lists (e.g., `biased_intents_list`).

- **5. Data Augmentation (if needed):**
 - Generate synthetic examples for rare or underrepresented classes using LLM-based augmentation (as in `synthetic_data_gen.py`).
 - This improves model robustness and generalization.

- **6. Feature Engineering:**
 - Recompute features as per the latest schema and business logic.
 - Ensure all preprocessing steps are consistent with the original pipeline.

- **7. Store and Version Artifacts:**
 - Save processed datasets and metadata as versioned artifacts (locally or to S3) for reproducibility and auditability.

**Industry Example (UIDS Project):**
- The pipeline orchestrated by `full.py` automates these steps:
 - Loads and cleans both local and S3 data sources.
 - Applies balancing, bias removal, and augmentation as configured.
 - Indexes the final dataset for downstream training and evaluation.

- In summary, I select recent production data reflecting the drift, blend with historical data if needed, clean and balance the dataset, augment rare classes, and ensure all artifacts are versioned—using automated, configurable pipelines to guarantee data quality and reproducibility for retraining.

---

**Q: What are the concrete steps to retrain a model using current production data after detecting data drift?**

- When significant data drift is detected, the retraining process should be systematic, automated, and reproducible to ensure the new model adapts to the latest data patterns while maintaining quality.

**Step-by-Step Retraining Workflow (as implemented in UIDS and similar enterprise pipelines):**

- **1. Data Aggregation:**
 - Collect a window of recent production data that reflects the drift (e.g., last few weeks).
 - Optionally, blend with a sample of historical data to maintain generalization or apply exponential weighting to give more importance to recent data.

- **2. Data Preprocessing:**
 - Clean the aggregated data: remove duplicates, handle missing values, and ensure schema consistency.
 - Use automated preprocessing scripts (e.g., `preprocessing.py` in the pipeline) to standardize and validate the dataset.
 - Store cleaned data in designated S3 locations for traceability.

- **3. Data Splitting:**
 - Split the cleaned data into training and test sets (as shown in the pipeline: outputs for "train" and "test").
 - Ensure the split maintains class balance and represents the current data distribution.

- **4. Model Training:**
 - Launch a training job using the prepared training data (e.g., via SageMaker TrainingStep).
 - Use the same or updated model architecture, depending on performance needs.
 - Configure training parameters (e.g., use spot instances, set max run time) for cost and efficiency.

- **5. Model Evaluation:**
 - Evaluate the trained model on the test set using automated evaluation scripts (e.g., SKLearnProcessor).
 - Calculate key metrics (accuracy, F1-score, recall, etc.).
 - Compare metrics against predefined thresholds (e.g., `accuracy_threshold` in the pipeline).

- **6. Conditional Promotion:**
 - Only proceed to model packaging and deployment if evaluation metrics meet or exceed thresholds (handled by `ConditionStep` in the pipeline).
 - If not, trigger alerts for manual review or further data investigation.

- **7. Model Deployment:**
 - Package and deploy the validated model to production endpoints.
 - Use CI/CD and safe deployment strategies (canary, blue-green) to minimize risk.

- **8. Monitoring and Logging:**
 - Continuously monitor model performance and data drift post-deployment.
 - Log all steps, metrics, and artifacts for auditability and compliance.

**Summary:** 
I follow a robust, automated pipeline: aggregate and clean recent production data, split into train/test, retrain and evaluate the model, promote only if metrics are met, and deploy with full traceability—ensuring the model stays aligned with evolving data in production.

---

**Q: If you have perfectly balanced, clean data with no outliers, and one model gives 70% accuracy while another gives 10% accuracy, which model should you choose?**

- In this scenario, with 100% balanced data, no outliers, and no data leakage or feature issues, the expected baseline accuracy for a random classifier would be around 1/number_of_classes (e.g., 10% for 10 classes).
- A model achieving 70% accuracy is significantly above random chance, while a model with 10% accuracy is at the level of random guessing.
- However, the question is described as "tricky," so let's consider possible interpretations:

 - **If the problem is a standard classification task:** 
 - The 70% accuracy model is clearly better and should be chosen, as it demonstrates meaningful learning and predictive power.
 - The 10% accuracy model is performing at random chance, indicating it has not learned any useful patterns.

 - **If the question is testing for overfitting or label inversion:** 
 - Sometimes, a model with very low accuracy (close to 0% or 10% in a 10-class problem) could indicate systematic label inversion or a bug, but the question explicitly rules out such issues.

 - **If the question is about model reliability:** 
 - With all else being equal and no data or feature issues, the higher accuracy model is always preferred.

- **Conclusion:** 
 - I would select the model with 70% accuracy, as it is performing significantly better than random and there are no confounding issues in the data or features.
 - The 10% accuracy model is not useful in this context.

- In practical industry scenarios, I always validate that the model's performance is above the random baseline and investigate any model performing at or below chance, but in this clean, balanced scenario, the higher accuracy model is the clear choice.

---

**Q: In a binary classification scenario with a perfectly balanced dataset and no data issues, would you still choose the model with 70% accuracy over one with 50% or 10%?**

- Yes, in a binary classification problem with a perfectly balanced dataset (i.e., equal number of positive and negative samples), the baseline accuracy for random guessing is 50%.
- A model achieving 70% accuracy is performing significantly better than random, indicating it has learned meaningful patterns from the data.
- A model with 10% accuracy is performing far below random chance, which could indicate systematic misclassification, but as per the question, there are no data or feature issues.
- In practical industry scenarios, we always compare model performance against the random baseline (50% for binary, 1/N for N-class).
- Therefore, I would confidently select the 70% accuracy model, as it demonstrates strong predictive power and adds value over random guessing.
- This approach aligns with standard model evaluation practices, as also reflected in the UIDS project, where accuracy and balanced accuracy are key metrics for model selection and promotion in the pipeline.

---

**Q: In a gambling scenario where one person can predict outcomes with 70% accuracy and another with 1% accuracy, whom would you choose to rely on?**

- In a gambling context, the goal is to maximize the probability of winning or making correct predictions.
- A person with 70% accuracy is significantly better than random chance (which is 50% in binary outcomes), providing a clear statistical advantage.
- A person with 1% accuracy is far worse than random and would almost always be wrong.
- Logically and mathematically, I would always choose the person with 70% accuracy, as this gives me a consistent edge in the game.
- This is analogous to model selection in machine learning, where a model performing well above the baseline is preferred for deployment and decision-making.
- In both AI systems and real-world scenarios like gambling, leveraging the most accurate predictor maximizes success and minimizes risk.

---

**Q: In a binary gambling game, would you choose the person with 70% accuracy or the one with 1% accuracy to maximize your chances of winning?**

- In a binary classification or gambling scenario, the objective is to maximize the probability of making correct predictions and winning.
- A person with 70% accuracy consistently predicts the correct outcome much better than random chance (which is 50% in binary).
- A person with 1% accuracy is almost always wrong, offering no practical value for winning.
- Logically and mathematically, I would always choose the person with 70% accuracy, as this gives a clear statistical advantage and increases my expected winnings over time.
- This is the same principle used in AI model selection: in the UIDS project, for example, models are only promoted if they meet or exceed a minimum accuracy threshold, ensuring only high-performing predictors are used in production.
- In summary, to maximize success in any predictive or decision-making scenario, I always select the most accurate and reliable predictor available.

---

**Q: In a coin toss game, if one person can predict 70 out of 100 outcomes correctly and another can only predict 1 out of 100, which person would you choose?**

- In a binary outcome scenario like a coin toss, the probability of random guessing is 50%.
- A person who can predict 70 out of 100 tosses correctly (70% accuracy) is performing significantly better than chance, indicating a strong predictive advantage.
- A person who can only predict 1 out of 100 correctly (1% accuracy) is performing far below random chance, offering no practical value.
- To maximize my probability of winning in a gambling or prediction scenario, I would always choose the person with 70% accuracy.
- This is consistent with industry best practices in AI model selection, where models are evaluated using metrics like accuracy, balanced accuracy, and F1-score (as implemented in the UIDS project), and only those performing significantly above baseline are promoted to production.
- In summary, I would always rely on the predictor with the highest demonstrated accuracy to maximize my chances of success.

---

**Q: If one person is 70% accurate and another is 0% accurate (always wrong), which person would you choose for a binary guessing game?**

- In a binary guessing scenario, a person with 70% accuracy is clearly better than random (50%) and provides a strong advantage.
- A person with 0% accuracy is always wrong—meaning, if you simply invert their prediction, you would get 100% accuracy.
- If you know with certainty that someone is always wrong (0% accurate), you can use their prediction and always choose the opposite, effectively achieving perfect (100%) accuracy.
- In practical terms:
 - If you can invert the 0% accurate person's prediction, they become the optimal choice.
 - If you cannot invert (for example, if you must use their answer as-is), then the 70% accurate person is better.
- In AI model evaluation (as in the UIDS project), we always look for models that perform above random, but if a model is systematically wrong, inverting its output can be a valid strategy.
- So, if allowed to invert, I would choose the 0% accurate person and always do the opposite. If not, I would choose the 70% accurate person.

- This logic is also reflected in model pipelines, where systematic errors can sometimes be leveraged if detected, but generally, models with higher direct accuracy are preferred for reliability and simplicity.

---

**Q: Is accuracy a good metric for evaluating models on imbalanced datasets (e.g., 70/30 split), and why or why not?**

- Accuracy is generally **not a good metric** for imbalanced datasets because it can be misleading.
 - In a 70/30 split, a model predicting only the majority class can achieve 70% accuracy without learning anything meaningful about the minority class.
 - This means high accuracy does not necessarily indicate good performance on the minority class, which is often the class of interest in real-world scenarios (e.g., fraud detection, rare disease diagnosis).
- Instead, **metrics that account for class imbalance** should be used:
 - **Balanced Accuracy**: Averages recall obtained on each class, giving equal weight to both classes regardless of their frequency.
 - **Precision, Recall, and F1-Score**: Especially the macro or weighted versions, which provide a better sense of performance across all classes.
 - **ROC-AUC or PR-AUC**: Useful for binary classification, especially when the positive class is rare.
- In my recent projects (e.g., UIDS – Universal Intent Determination System), I always report balanced accuracy, macro/micro/weighted F1-scores, precision, and recall to ensure robust evaluation on imbalanced datasets.
- This approach ensures the model is evaluated fairly on both majority and minority classes, leading to more reliable and actionable results in production.

---
