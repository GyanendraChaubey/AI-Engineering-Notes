# Generative AI Engineer (Part 1) — Interview 48

**Q: Explain the RAG pipeline architecture from user query to LLM response, detailing each component and technical process.**

- The RAG (Retrieval-Augmented Generation) pipeline in the Knowledge GPT system is designed to enable natural language access to enterprise knowledge bases, leveraging both retrieval and generative AI components for accurate, context-rich responses.
- The architecture integrates multiple AWS services, OpenAI/Azure LLMs, and vector search technologies to ensure scalability, security, and high performance.

**Pipeline Components & Technical Flow:**

---

1. **Document Ingestion & Preprocessing**
 - **Source:** Internal knowledge base (KAAS – Knowledge-as-a-Service) exposes APIs and regularly exports document metadata to AWS S3.
 - **AWS Glue Job:** Scheduled ETL job scans S3 for new/updated metadata JSONs.
 - **Document Retrieval:** For new metadata, the pipeline calls KAAS APIs to generate secure download links and fetches the actual document content (HTML/PDF).
 - **Conversion:** Documents are converted to Markdown for uniform processing.
 - **Chunking:** Markdown content is split into semantic chunks (using custom chunking strategies), ensuring each chunk is manageable (e.g., up to 25 KB).
 - **Embedding Generation:** Each chunk is summarized (if needed) and passed to Azure OpenAI to generate a 1536-dimensional embedding vector.
 - **Indexing:** Chunks, along with metadata and embeddings, are bulk-ingested into Amazon OpenSearch for fast semantic and lexical retrieval.

---

2. **Query Processing & Retrieval**
 - **User Query:** User submits a natural language query via the API (FastAPI endpoint).
 - **Embedding:** The query is embedded using the same embedding model (OpenAI/Azure).
 - **Retrieval:** 
 - **Parallel Search:** Both lexical (BM25) and vector (KNN) searches are performed in OpenSearch.
 - **RRF (Reciprocal Rank Fusion):** Results from both searches are combined and ranked to select the most relevant document chunks.
 - **Context Construction:** Top-ranked chunks are consolidated, and context data is built (including chunk text, source, title, etc.).

---

3. **Prompt Construction & LLM Generation**
 - **Prompt Template:** A prompt is dynamically constructed using a template, injecting the retrieved context, user query, persona, and other parameters.
 - **LLM Call:** The prompt, along with conversation history, is sent to the LLM (Azure OpenAI GPT or similar) for response generation.
 - **Response:** The LLM generates a context-aware answer, which is returned to the user.

---

4. **Monitoring & Evaluation**
 - **Logging:** All interactions, retrievals, and LLM outputs are logged for monitoring and evaluation.
 - **Evaluation Module:** Automated frameworks assess response accuracy, contextual relevance, and hallucination rates for continuous improvement.

---

**Summary of Key Technologies:**
- **AWS S3, Glue:** Data storage and ETL orchestration.
- **KAAS APIs:** Secure document access.
- **OpenSearch:** Hybrid semantic and lexical search.
- **Azure OpenAI:** Embedding and LLM generation.
- **FastAPI:** API layer for user interaction.
- **Custom Python Pipelines:** For chunking, embedding, and orchestration.

This architecture ensures that user queries are answered with high relevance and accuracy by combining the strengths of retrieval-based and generative AI approaches, while maintaining scalability and enterprise-grade security.

---

**Q: What do you mean by embeddings?**

- Embeddings are dense, fixed-size numerical vector representations of text (words, sentences, or documents) that capture the semantic meaning and context of the input.
- In the context of our RAG pipeline, embeddings are generated using models like OpenAI’s text-embedding-ada-002, which produces a 1536-dimensional float vector for each input chunk or query.
- These vectors encode semantic similarity—meaning that texts with similar meanings will have embeddings that are close together in the high-dimensional space.
- We use embeddings for two main purposes:
 - **Semantic Search:** By embedding both document chunks and user queries, we can perform efficient vector similarity searches (e.g., KNN search in OpenSearch) to retrieve the most relevant content based on meaning, not just keywords.
 - **Context Construction:** The retrieved chunks (based on embedding similarity) are used to build the context that is injected into the LLM prompt, ensuring the generated response is grounded in relevant enterprise knowledge.
- Technically, the embedding process involves:
 - Summarizing or processing the text chunk (if needed).
 - Passing the text to the embedding model/API, which returns the vector.
 - Storing the embedding vector along with metadata in the vector database (OpenSearch).
- Embeddings are crucial for enabling accurate, context-aware retrieval and generation in modern AI systems, especially in large-scale enterprise applications.

---

**Q: Can you give examples of algorithms that can be used to train an embedding model?**

- There are several well-known algorithms and neural network architectures used to train embedding models, each suited for different types of data and tasks. Here are some practical examples:

---

- **Word2Vec** 
 - Uses shallow neural networks to learn word embeddings based on context in large text corpora.
 - Two main approaches: Skip-gram (predicts context words from a target word) and CBOW (predicts target word from context words).

- **GloVe (Global Vectors for Word Representation)** 
 - Learns word embeddings by factorizing word co-occurrence matrices, capturing global statistical information about word usage.

- **FastText** 
 - Extension of Word2Vec that represents words as bags of character n-grams, allowing it to generate embeddings for out-of-vocabulary words.

- **Sentence Transformers (e.g., SBERT)** 
 - Uses transformer-based models (like BERT, RoBERTa) fine-tuned for producing sentence or document embeddings.
 - Widely used for semantic search, clustering, and retrieval tasks.

- **Doc2Vec** 
 - Extends Word2Vec to learn embeddings for entire documents or paragraphs, not just individual words.

- **Transformer-based Embedding Models (e.g., BERT, RoBERTa, GPT, OpenAI Embeddings)** 
 - Pre-trained transformer models can be fine-tuned or used directly to generate contextual embeddings for words, sentences, or documents.
 - Example: OpenAI’s text-embedding-ada-002, which outputs 1536-dimensional vectors.

- **Deep Metric Learning (e.g., Siamese Networks, Triplet Networks)** 
 - Trains models to map similar items close together and dissimilar items far apart in the embedding space, often used for tasks like face recognition or intent classification.

---

- In industry, transformer-based models (like Sentence Transformers or OpenAI embeddings) are most commonly used for high-quality, context-aware embeddings, especially for semantic search and retrieval-augmented generation pipelines. 
- For custom use cases, you can train your own embedding models using frameworks like PyTorch or TensorFlow, leveraging architectures such as BERT, RoBERTa, or even custom Siamese/Triplet networks depending on the task.

---

**Q: Can you explain a bit more about CBOW (Continuous Bag of Words) in the context of embedding models?**

- CBOW (Continuous Bag of Words) is one of the two main architectures used in the Word2Vec algorithm for learning word embeddings.
- In CBOW, the model predicts a target word based on its surrounding context words within a fixed window size.
- **How it works:**
 - For a given sentence, select a window of context words around a target word.
 - The context words are input to the model, which tries to predict the target (center) word.
 - For example, in the sentence “The cat sat on the mat”, if the window size is 2, and the target word is “sat”, the context words are [“The”, “cat”, “on”, “the”].
- The model learns to represent each word as a dense vector (embedding) such that words appearing in similar contexts have similar embeddings.
- CBOW is efficient and works well for large datasets, capturing semantic and syntactic relationships between words.
- The learned embeddings can then be used for downstream NLP tasks or as input features for other models.

**Summary:**
- CBOW predicts a word from its context.
- It is used to train word embeddings that capture semantic similarity.
- It is foundational in traditional NLP and inspired many modern embedding techniques.

---

**Q: What is temperature in language models (LLMs)?**

- Temperature is a hyperparameter used during the text generation (inference) phase of language models (LLMs) to control the randomness or creativity of the generated output.
- It affects the probability distribution from which the next token (word or subword) is sampled:
 - **Low temperature (e.g., 0.1–0.3):** Makes the model more deterministic and conservative. The model is more likely to pick the highest probability token, resulting in repetitive or predictable outputs.
 - **High temperature (e.g., 0.7–1.0+):** Increases randomness and diversity. The model samples from a wider range of possible tokens, leading to more creative or varied responses, but potentially less coherent or relevant.
- **How it works:** 
 - The logits (raw output scores) from the model are divided by the temperature value before applying the softmax function to get probabilities.
 - Lower temperature sharpens the probability distribution (peaks), higher temperature flattens it (more uniform).
- **Practical use:** 
 - For factual, precise, or business-critical tasks, use a lower temperature.
 - For creative writing, brainstorming, or open-ended tasks, use a higher temperature.

**Summary:** 
Temperature tunes the balance between determinism and creativity in LLM outputs, allowing you to control how predictable or diverse the generated text will be.

---

**Q: What is the theoretical range of the temperature parameter in LLMs?**

- Theoretically, the temperature parameter in language models is a positive real number and is most commonly set in the range **0 to 1** for practical applications.
- **0** means fully deterministic output (always picks the highest probability token).
- **1** is the standard baseline for normal randomness.
- Technically, temperature can be set above 1 (e.g., 1.5 or 2), which increases randomness even further, but in most industry and research settings, values between **0 and 1** are preferred for balancing determinism and diversity.
- Setting temperature below 0 is not meaningful, as it would invert the probability distribution and is not supported by standard implementations.
- In summary: **Temperature is typically set between 0 and 1, but can theoretically be any positive value greater than 0.** However, values above 1 are rarely used in production due to loss of coherence in generated text.

---

**Q: What is the theoretical minimum and maximum value for the temperature parameter in LLMs?**

- Theoretically, the temperature parameter can take **any positive real value greater than zero** (0, ∞).
- **Minimum:** Approaching zero (but not exactly zero), which makes the model output almost always the highest probability token (fully deterministic).
- **Maximum:** There is no strict upper bound; as temperature increases towards infinity, the output distribution becomes more uniform and random, but in practice, very high values make the output incoherent.
- **Zero is not valid** (as it would cause division by zero in the softmax calculation).
- In summary: **Temperature ∈ (0, ∞)** theoretically, but practical values are usually between 0 and 1.

---

**Q: What happens if we set the temperature parameter to a very high value (e.g., 3000) in an LLM?**

- Setting the temperature to a very high value (like 3000) makes the probability distribution for the next token almost completely uniform.
- This means every possible token has nearly the same chance of being selected, regardless of its actual likelihood according to the model.
- As a result, the generated output becomes extremely random, incoherent, and loses any meaningful structure or relevance to the input prompt.
- In practice, such high temperatures are never used because they destroy the model’s ability to generate contextually appropriate or useful responses.
- For most applications, temperature is kept between 0 and 1 to balance determinism and creativity, ensuring outputs remain relevant and coherent.

---

**Q: How does the probability distribution look when temperature is set to a very high value (e.g., 30,000) in an LLM?**

- When the temperature is set to a very high value, the logits (raw scores) output by the model are divided by this large number before applying the softmax function.
- This causes all logits to become very close to zero, making their exponentials (e^0 = 1) nearly equal.
- As a result, the softmax function produces a **uniform probability distribution** across all possible tokens.
- In this scenario, every token has almost the same probability of being selected, regardless of its actual relevance or likelihood according to the model.
- This uniformity leads to highly random and incoherent outputs, as the model loses its ability to prioritize contextually appropriate tokens.

**Summary:** 
With a very high temperature, the probability distribution becomes almost perfectly uniform, meaning every token is equally likely to be chosen.

---

**Q: How does LangGraph work, and what are its main components for creating agents?**

- **LangGraph** is a framework designed for building complex, multi-step, and stateful agent workflows on top of language models (LLMs).
- It extends the capabilities of LangChain by allowing you to define agent behaviors as directed graphs, where each node represents a step (or tool/action), and edges define the flow of execution based on state or output.
- This approach enables the creation of sophisticated agentic systems that can handle branching logic, loops, and conditional execution, making it suitable for enterprise-grade AI automation.

**Main Components of LangGraph:**
- **Nodes:** 
 - Each node represents a discrete action, tool, or function (e.g., calling an LLM, querying a database, invoking an API).
 - Nodes encapsulate the logic for a single step in the agent workflow.
- **Edges:** 
 - Edges define the transitions between nodes, specifying how the workflow progresses based on the output or state from the previous node.
 - They can encode conditional logic, allowing for dynamic branching and looping.
- **State Management:** 
 - LangGraph maintains a state object that is passed and updated as the workflow progresses through nodes.
 - This state can include intermediate results, context, or any variables needed for decision-making.
- **Graph Definition:** 
 - The overall workflow is defined as a directed graph, where you specify nodes, edges, and the rules for transitions.
 - This structure allows for flexible and reusable agent architectures.
- **Integration with LLMs and Tools:** 
 - Nodes can be configured to interact with LLMs (e.g., OpenAI, Azure OpenAI), external APIs, databases, or custom business logic.
 - This enables agents to perform complex, multi-modal tasks.

**Industry Use Case Example (from your experience):**
- In a typical production implementation, LangGraph is used to orchestrate multi-step LLM-driven annotation workflows for data labeling and intent validation.
- Each node handled a specific task (e.g., semantic retrieval, label suggestion, validation), and the graph structure allowed for conditional relabeling and iterative improvement of dataset quality.

**Summary:** 
LangGraph enables the design of modular, stateful, and dynamic agent workflows by representing agent logic as a directed graph of nodes and edges, supporting advanced automation and decision-making in enterprise AI systems.

---

**Q: What is the main difference between Flask and FastAPI, and why is FastAPI preferred?**

- **FastAPI** and **Flask** are both popular Python web frameworks, but FastAPI is designed for modern, high-performance APIs, while Flask is a lightweight, general-purpose web framework.
- **Key Differences:**
 - **Performance:** 
 - FastAPI is built on ASGI (Asynchronous Server Gateway Interface) and supports asynchronous programming out of the box, making it significantly faster and more scalable than Flask, which is WSGI-based and synchronous by default.
 - **Type Hints & Data Validation:** 
 - FastAPI leverages Python type hints and Pydantic models for automatic request validation, serialization, and documentation. Flask requires manual validation and lacks built-in type enforcement.
 - **Automatic Documentation:** 
 - FastAPI auto-generates interactive OpenAPI (Swagger) and ReDoc documentation for all endpoints, which is extremely useful for API development and client integration. Flask does not provide this natively.
 - **Developer Experience:** 
 - FastAPI’s use of type hints improves code readability, maintainability, and editor support (auto-completion, linting).
 - **Async Support:** 
 - FastAPI natively supports async/await, enabling non-blocking I/O and better performance for concurrent requests. Flask requires additional libraries for async support and is less efficient for high-concurrency workloads.
- **Why FastAPI is Preferred:**
 - For modern AI and microservices architectures (like RAG pipelines, LLM orchestration, and high-throughput APIs), FastAPI’s speed, async capabilities, and automatic validation/documentation make it the preferred choice.
 - In my recent projects (e.g., Knowledge GPT API), FastAPI enabled rapid development of scalable, production-grade REST APIs with robust validation, JWT authentication, and seamless integration with AI models and cloud services.

**Summary:** 
FastAPI is preferred over Flask for building high-performance, type-safe, and well-documented APIs, especially in AI and data-driven applications where scalability and developer productivity are critical.

---

**Q: What are the main fine-tuning methods for machine learning and LLM models?**

- There are several fine-tuning methods used for adapting pre-trained models (including LLMs and other deep learning models) to specific tasks or domains. The main approaches include:

- **Full Fine-Tuning:**
 - All model parameters are updated during training on the new dataset.
 - Provides maximum flexibility and adaptation but is resource-intensive and requires more data.
 - Common for smaller models or when significant domain adaptation is needed.

- **Feature-Based Fine-Tuning (Feature Extraction):**
 - The pre-trained model is used as a fixed feature extractor; only the final classification/regression layers are trained.
 - Useful when labeled data is limited or computational resources are constrained.

- **Partial Fine-Tuning / Layer Freezing:**
 - Only certain layers (often the last few) are unfrozen and updated, while the rest remain fixed.
 - Balances adaptation and efficiency, reducing overfitting and training time.

- **Adapter-Based Fine-Tuning:**
 - Small trainable modules (adapters) are inserted into the model layers; only these adapters are trained.
 - The main model weights remain frozen, making this approach parameter-efficient and suitable for multi-task or multi-domain scenarios.

- **Parameter-Efficient Fine-Tuning (PEFT):**
 - Includes methods like LoRA (Low-Rank Adaptation), QLoRA, and Prefix Tuning.
 - Only a small subset of parameters are updated, significantly reducing memory and compute requirements.
 - Widely used for large LLMs (e.g., Falcon, GPT, Llama) in industry.

- **Prompt Tuning / Prompt Engineering:**
 - Instead of updating model weights, learnable prompts or embeddings are optimized to steer the model’s behavior.
 - Useful for LLMs where full fine-tuning is impractical.

**Industry Example (from my experience):**
- In the UIDS project, we used SetFit for few-shot fine-tuning of sentence transformers, and also experimented with QLoRA-like fine-tuning for Falcon-7B using PEFT and bitsandbytes for efficient adaptation.
- For CatBoost models, we used feature-based approaches with sentence embeddings as input features.

**Summary:** 
Fine-tuning methods range from full model updates to parameter-efficient techniques like adapters and LoRA, with the choice depending on data availability, compute resources, and the specific use case. In modern LLM workflows, PEFT methods are highly preferred for scalability and efficiency.

---

**Q: What are the main fine-tuning methods, specifically parameter-efficient fine-tuning (PEFT) like QLoRA, and how are they used in LLM projects?**

- Fine-tuning methods for LLMs and ML models can be broadly categorized as:
 - **Full Fine-Tuning:** Updates all model parameters; resource-intensive, used when large adaptation is needed.
 - **Feature-Based/Partial Fine-Tuning:** Only updates specific layers or uses the model as a feature extractor.
 - **Adapter-Based & Parameter-Efficient Fine-Tuning (PEFT):** Updates a small subset of parameters, making it efficient for large models.

- **Parameter-Efficient Fine-Tuning (PEFT):**
 - PEFT methods like **LoRA** and **QLoRA** introduce small trainable modules or low-rank adapters into the model, keeping most of the original weights frozen.
 - **QLoRA** (Quantized LoRA) further reduces memory usage by quantizing model weights, enabling fine-tuning of very large models (e.g., Falcon-7B) on limited hardware.
 - These methods are highly efficient, require less compute, and are ideal for domain adaptation or task-specific tuning in enterprise settings.

- **Industry Application (UIDS Project):**
 - In the UIDS project, we implemented QLoRA-like fine-tuning for Falcon-7B using the `peft` and `bitsandbytes` libraries, allowing us to adapt large LLMs for intent classification without retraining the entire model.
 - For sentence transformer models, we used SetFit for few-shot fine-tuning, and for CatBoost, we used sentence embeddings as features with only the classifier trained.

- **Benefits:**
 - PEFT methods enable scalable, cost-effective adaptation of large models.
 - They are especially useful in production environments where compute resources are limited and rapid iteration is required.

**Summary:** 
Parameter-efficient fine-tuning methods like QLoRA allow you to efficiently adapt large LLMs by training only a small subset of parameters, making them practical for enterprise AI projects with limited resources. In my recent work, I have used QLoRA for Falcon-7B and SetFit for sentence transformers to achieve efficient and scalable fine-tuning.

---

**Q: What does "low rank" mean in the context of LoRA (Low-Rank Adaptation)?**

- In LoRA (Low-Rank Adaptation), "low rank" refers to the mathematical technique of approximating large weight matrices in neural networks with the product of two much smaller matrices.
- Instead of updating the full (high-dimensional) weight matrix during fine-tuning, LoRA introduces two small trainable matrices (of lower rank) that capture the necessary adaptations.
- This approach significantly reduces the number of trainable parameters, making fine-tuning more memory- and compute-efficient, especially for large language models (LLMs).
- The original large weight matrix remains frozen, and only the low-rank matrices are updated during training, allowing efficient adaptation with minimal resource usage.
- In practice, this means LoRA can achieve similar performance to full fine-tuning but with a fraction of the computational cost.

**Summary:** 
"Low rank" in LoRA means using smaller matrices to efficiently adapt large models, enabling scalable and resource-efficient fine-tuning by only updating a small subset of parameters.

---

**Q: If the original weight matrix is of size m × n, what are the dimensions of the two low-rank matrices used in LoRA?**

- In LoRA, the original weight matrix **W** has dimensions **m × n** (output_dim × input_dim).
- LoRA approximates updates to **W** by introducing two smaller trainable matrices whose product is m × n.
- The two matrices have sizes **m × r** and **r × n**, where **r** is the rank (a small integer, r << min(m, n)).

**Naming Convention — Beware of Confusion:**

The original **LoRA paper (Hu et al. 2022)** and the **HuggingFace PEFT library** use this convention:
- **A** (lora_A): shape **r × n** — the "down-projection" (initialized with random Gaussian)
- **B** (lora_B): shape **m × r** — the "up-projection" (initialized to **zero** so ΔW = 0 at training start)
- Update rule: **ΔW = B · A** → (m×r) · (r×n) = m×n ✅

Some textbooks and older sources swap the names, calling the first matrix A (m×r) and the second B (r×n) with ΔW = A·B. The result is identical — only the labeling differs.

**Key facts to remember for interviews:**
1. The two matrices have dimensions **r × n** and **m × r** (or equivalently m×r and r×n)
2. One is initialized to **zero**, so the net weight change is zero at the start of training
3. Only these two small matrices are trained; W stays frozen
4. Trainable parameters: r×n + m×r = r(m+n) vs full m×n — massive saving when r is small

**Summary:** 
LoRA uses two matrices of size **r × n** (down) and **m × r** (up) for an original W of size m × n. Their product gives the weight update ΔW of shape m × n. r is the rank (typically 4–32).

---
