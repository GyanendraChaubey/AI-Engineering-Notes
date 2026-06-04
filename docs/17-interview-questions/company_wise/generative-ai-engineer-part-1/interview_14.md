# Generative AI Engineer (Part 1) — Interview 14

**Q: What happens to precision and recall if the classification threshold is increased from 0.5 to 0.7 in a binary classification problem?**

- Increasing the classification threshold from 0.5 to 0.7 means the model will only predict the positive class if it is at least 70% confident, instead of 50%.
- This adjustment directly impacts the trade-off between precision and recall:

 - **Precision** (True Positives / (True Positives + False Positives)):
 - Precision will generally **increase**.
 - The model becomes more conservative in predicting positives, so fewer false positives are likely.
 - Only the most confident positive predictions are labeled as positive, reducing the chance of incorrectly labeling negatives as positives.

 - **Recall** (True Positives / (True Positives + False Negatives)):
 - Recall will generally **decrease**.
 - Some true positives that had scores between 0.5 and 0.7 will now be classified as negative, increasing false negatives.
 - The model misses more actual positives because the bar for a positive prediction is higher.

- In summary:
 - **Higher threshold → Higher precision, Lower recall**
 - **Lower threshold → Lower precision, Higher recall**

- The choice of threshold should be based on the business requirement—whether it’s more important to avoid false positives (favoring precision) or to catch as many positives as possible (favoring recall).

---

**Q: What is gradient descent in the context of machine learning models?**

- Gradient descent is an optimization algorithm used to minimize the loss (or cost) function in machine learning models, especially in supervised learning and deep learning.
- The core idea is to iteratively adjust the model parameters (weights and biases) in the direction that reduces the loss, based on the gradient (partial derivatives) of the loss function with respect to those parameters.
- At each iteration:
 - The algorithm computes the gradient of the loss function with respect to each parameter.
 - Parameters are updated by moving them a small step (learning rate) in the opposite direction of the gradient.
 - This process is repeated until the loss converges to a minimum (ideally the global minimum, but often a local minimum in complex models).
- Gradient descent comes in several variants:
 - **Batch Gradient Descent**: Uses the entire dataset to compute gradients at each step.
 - **Stochastic Gradient Descent (SGD)**: Uses a single data point per update, leading to faster but noisier convergence.
 - **Mini-batch Gradient Descent**: Uses small batches of data, balancing speed and stability.
- It is fundamental for training neural networks, logistic regression, and many other ML models, enabling them to learn from data by minimizing prediction errors.

---

**Q: How do you identify if your model is stuck in a local minima during training, and how do you address it?**

- **Identifying Local Minima:**
 - If the training loss plateaus and stops decreasing significantly, even after many epochs, it may indicate the optimizer is stuck in a local minima or a saddle point.
 - The validation loss may also stagnate or not improve, despite changes in learning rate or other hyperparameters.
 - Visualization of the loss curve: A flat or oscillating loss curve (without further descent) can be a sign.
 - In deep learning, true local minima are rare due to high-dimensional parameter spaces, but saddle points and flat regions are common.

- **How to Address Local Minima or Stagnation:**
 - **Optimizer Choice:** Use advanced optimizers like Adam, RMSprop, or Adagrad, which adapt learning rates and help escape shallow minima or saddle points.
 - **Learning Rate Scheduling:** Adjust the learning rate dynamically (e.g., learning rate decay, ReduceLROnPlateau) to help the optimizer jump out of flat regions.
 - **Random Initialization:** Re-initialize weights and retrain; different initializations may lead to better minima.
 - **Batch Normalization:** Helps smooth the loss landscape and can aid in escaping poor local minima.
 - **Gradient Noise/Regularization:** Adding noise to gradients or using dropout can help the optimizer explore more of the loss surface.
 - **Early Stopping and Restarts:** Use early stopping to avoid overfitting and restart training from different points.
 - **Increase Batch Size:** Sometimes, larger batch sizes can help the optimizer move past small local minima.

- **Practical Steps in Industry:**
 - Monitor training and validation loss curves closely.
 - Experiment with different optimizers and learning rate schedules.
 - Use callbacks in frameworks (like PyTorch or TensorFlow) to automate learning rate adjustments and early stopping.
 - For large-scale models (like in the UIDS project), leverage SageMaker’s hyperparameter tuning and experiment tracking to systematically explore different configurations and avoid suboptimal convergence.

- In summary, while it’s challenging to definitively “detect” a local minima, careful monitoring and use of modern optimization techniques can help mitigate the risk and improve model convergence.

---

**Q: What do the individual numbers in a word embedding vector (e.g., 768-dimensional vector for "cat") represent at a high level?**

- Each number in the embedding vector represents a learned feature or latent dimension that captures some aspect of the word’s meaning or context, as determined by the embedding model during training.
- The embedding model (such as BERT, text-embedding-ada-002, etc.) is trained on large corpora to map semantically similar words or phrases to vectors that are close together in high-dimensional space.
- At a high level:
 - The vector as a whole encodes the semantic meaning of the word "cat" in a way that is useful for downstream tasks (e.g., similarity search, classification).
 - Each individual dimension does not have a direct human-interpretable meaning; rather, it is a component of the overall semantic representation.
 - The values are floating-point numbers that, together, position "cat" in the embedding space relative to other words or chunks.
 - Similar words (like "dog" or "kitten") will have vectors that are close to "cat" in this space, while unrelated words (like "car" or "banana") will be farther away.
- In practical terms, these vectors enable efficient semantic search and retrieval in vector databases, as used in RAG pipelines and systems like Knowledge-GPT.
- The high-dimensional representation allows the model to capture complex relationships, analogies, and contextual nuances that are not possible with simple keyword matching.

---


**Q: Why are embedding vector dimensions typically multiples of 32 (e.g., 512, 768, 1536) instead of arbitrary numbers like 100 or 500?**

- **Hardware Efficiency**:
 - Modern CPUs and GPUs are optimized for parallel computation using SIMD (Single Instruction, Multiple Data) instructions.
 - These instructions operate most efficiently on data blocks that are aligned to certain sizes—commonly multiples of 8, 16, or 32.
 - Using vector dimensions that are multiples of 32 ensures optimal memory alignment and maximizes throughput for matrix operations, which are fundamental in deep learning and embedding computations.

- **Deep Learning Frameworks**:
 - Libraries like PyTorch and TensorFlow are designed to leverage hardware acceleration (e.g., CUDA, cuDNN) that expects data in these aligned sizes.
 - Batch processing and tensor operations are faster and more memory-efficient when dimensions are powers or multiples of 2, 8, 16, or 32.

- **Model Architecture Design**:
 - Transformer-based models (BERT, GPT, etc.) often use hidden sizes and embedding dimensions that are multiples of 32 for architectural consistency and compatibility with hardware.
 - This also simplifies the design of multi-head attention, where the embedding dimension is split evenly among attention heads (e.g., 768/12 = 64 per head).

- **Scalability and Standardization**:
 - Standardizing on these dimensions makes it easier to scale models, share weights, and interoperate between different components or models.

- **Why not arbitrary numbers?**
 - Using non-standard sizes (like 100 or 500) can lead to inefficient memory usage, slower computation, and incompatibility with optimized hardware routines.
 - It may also complicate model architecture, especially for components like attention mechanisms that require even splits.

- **Summary**:
 - Multiples of 32 are chosen for a combination of hardware efficiency, deep learning framework optimization, and architectural convenience, leading to faster, more scalable, and more compatible AI systems.

---

**Q: What do "groundedness" and "answer relevance" mean in the context of RAG evaluation metrics?**

- **Groundedness**:
 - Groundedness measures how well the generated answer is supported by the retrieved context or source documents.
 - A highly grounded answer means that all factual claims or statements in the answer can be directly traced back to the provided context, minimizing hallucinations or unsupported information.
 - In RAG pipelines, groundedness is critical for trustworthiness, especially in enterprise settings where answers must be verifiable against internal knowledge bases.
 - For example, if the context retrieved contains a specific policy statement and the answer repeats or paraphrases it accurately, the answer is considered well-grounded.

- **Answer Relevance**:
 - Answer relevance evaluates how well the generated answer addresses the user’s original query or intent.
 - A relevant answer is one that is directly responsive to the question, providing useful and contextually appropriate information.
 - High answer relevance means the answer is not only correct but also focused and pertinent to what was asked, avoiding unnecessary or off-topic details.
 - In practice, relevance is often measured by comparing the answer to reference answers or by using LLM-based scoring to assess semantic similarity to the query.

- **Summary**:
 - Groundedness ensures factual accuracy and traceability to source context.
 - Answer relevance ensures the response is useful and directly answers the user’s question.
 - Both metrics are essential for evaluating and improving the quality of RAG-based AI systems, and are commonly used in tools like RAGAS, Trulens, and custom LLM-as-a-judge pipelines.

---

**Q: What do precision and recall mean in the context of RAG (Retrieval-Augmented Generation) pipelines?**

- In the context of RAG pipelines, precision and recall are adapted to evaluate how well the system retrieves and/or predicts relevant information, especially for tasks like intent classification or answer retrieval.
- **Precision**:
 - Measures the proportion of relevant (correct) results among all results returned by the RAG system.
 - In RAG intent classification, it’s the fraction of predicted intents that are actually correct (i.e., how many of the predicted answers/labels are true positives).
 - High precision means that when the system predicts a certain intent or answer, it is usually correct.
- **Recall**:
 - Measures the proportion of relevant (correct) results that were successfully retrieved or predicted by the system out of all possible relevant results.
 - In RAG workflows, it’s the fraction of ground truth intents or answers that the system correctly identifies (i.e., how many true positives are captured out of all actual positives).
 - High recall means the system is able to find most of the correct answers/intents, even if it sometimes includes incorrect ones.
- **Application in UIDS RAG Labelling**:
 - As implemented in our UIDS RAG labelling and inference workflows, after generating predictions (via LLM or hybrid model), we compare them to ground truth labels.
 - We calculate precision and recall using standard metrics functions (e.g., scikit-learn), as shown in our project code and documentation.
 - These metrics help us understand both the accuracy and completeness of our RAG pipeline’s predictions, supporting continuous improvement and robust evaluation.

- In summary, precision and recall in RAG measure the correctness and coverage of the system’s predictions or retrievals, similar to their use in traditional ML, but applied to the outputs of the RAG workflow.

---

**Q: Given 100 documents, 10 retrieved for a query, and 3 used in the answer, what are recall and precision in this RAG example?**

- **Recall**:
 - Recall measures how many of the truly relevant documents were retrieved out of all possible relevant documents.
 - In this example, if the 10 retrieved documents include all the documents that are actually relevant to the user’s question, recall is high.
 - If, for instance, there are 5 truly relevant documents in the corpus and you retrieved 3 of them among your 10, recall = 3/5 = 0.6 (60%).
 - If you retrieved all 5 relevant documents, recall = 5/5 = 1.0 (100%).

- **Precision**:
 - Precision measures how many of the retrieved documents are actually relevant (i.e., used in the answer) out of all retrieved documents.
 - In your example, you retrieved 10 documents, but only 3 were actually used to answer the question (assuming these 3 are relevant).
 - Precision = 3/10 = 0.3 (30%).

- **Summary**:
 - **Recall** = (Number of relevant documents retrieved) / (Total number of relevant documents in the corpus)
 - **Precision** = (Number of relevant documents retrieved) / (Total number of documents retrieved)
 - In your scenario, if 3 out of 10 retrieved documents are relevant and there are 5 relevant documents in total:
 - Recall = 3/5 = 0.6 (60%)
 - Precision = 3/10 = 0.3 (30%)

- These metrics help you evaluate both the coverage (recall) and the accuracy (precision) of your RAG retrieval process.

---

**Q: What does "state" mean in LangGraph?**

- In LangGraph, "state" refers to the structured data object that represents the current context, variables, and intermediate results as the workflow progresses through its nodes (steps).
- The state is passed and updated between nodes, allowing each step in the pipeline to access and modify relevant information.
- It acts as a shared memory or context for the entire workflow, enabling conditional branching, data transformation, and orchestration logic.
- For example, in our UIDS RAG labelling pipeline, the state schema might include fields like:
 - Input data (e.g., user queries, documents)
 - Intermediate results (e.g., translated text, indexed vectors)
 - Model predictions or inference outputs
 - Metrics and logs
 - Flags for conditional execution (e.g., whether translation is needed)
- The state is defined using a schema (often a Python dataclass or dictionary) and is validated as it moves through the workflow.
- This design enables modular, extensible, and debuggable pipelines, as each node operates on and updates the state in a controlled manner.
- In summary, "state" in LangGraph is the central data structure that carries all relevant information through the pipeline, ensuring each step has the context it needs to execute correctly.

---

**Q: Why is the state immutable in LangGraph?**

- **Immutability in LangGraph** is a core design principle to ensure reliability, predictability, and debuggability of complex AI workflows.
- **Key reasons for immutability:**
 - **Deterministic Execution:** Each node receives a copy of the state, processes it, and returns a new state without altering the original. This guarantees that the same input always produces the same output, making workflows reproducible.
 - **Debugging & Traceability:** Since previous states are never mutated, you can easily trace the evolution of the state through each node. This helps in debugging, auditing, and understanding how data changes at every step.
 - **Parallelism & Branching:** Immutability allows safe parallel execution and branching. Multiple nodes can operate on the same input state without risk of side effects or race conditions, which is crucial for conditional logic and concurrent processing.
 - **Error Isolation:** If a node introduces an error, it only affects the new state returned by that node, not the original state. This isolates failures and makes error handling more robust.
 - **Functional Programming Paradigm:** Immutability aligns with functional programming best practices, reducing bugs caused by unintended side effects and making the workflow logic easier to reason about.
- **Industry Practice:** In our UIDS RAG labelling pipeline, we leverage this principle to ensure that each step (node) in the LangGraph workflow operates independently, making the pipeline modular, extensible, and maintainable.
- **Summary:** Immutability in LangGraph ensures that workflows are safe, predictable, and easy to maintain, especially as pipelines grow in complexity and scale.

---
