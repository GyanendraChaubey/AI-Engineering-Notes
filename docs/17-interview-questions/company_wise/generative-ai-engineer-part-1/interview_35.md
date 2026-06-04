# Generative AI Engineer (Part 1) — Interview 35

**Q: Which base model was used for the intent classification system intent classifier, and what fine-tuning methodology was applied?**

- For the UIDS (Universal Intent Determination System) intent classifier, we primarily used the **SetFit** model as the base.
- **SetFit** is a sentence transformer-based model designed for efficient few-shot learning, making it suitable for scenarios with limited labeled data across many intent classes.
- The fine-tuning process involved:
 - Starting with a pre-trained sentence transformer model (such as `sentence-transformers/paraphrase-mpnet-base-v2`).
 - Using the **SetFit** framework to fine-tune the model on [Company]’s internal, multilingual intent dataset, which covers 639 global intent classes.
 - The fine-tuning was performed using **Cosine Similarity Loss** to optimize the embedding space for intent classification.
 - The training pipeline was orchestrated on **AWS SageMaker**, enabling scalable, distributed training and experiment tracking.
 - We leveraged **differentiable classification heads** in SetFit to adapt the model for multi-class classification.
 - Hyperparameter tuning was managed using SageMaker’s HyperparameterTuner to optimize for accuracy and generalization.
- The model was then deployed as a real-time inference endpoint on SageMaker, integrated with [Company]’s support tools for production use.

- In addition to SetFit, we also experimented with alternative paths:
 - **CatBoost**: Used sentence transformer embeddings as features for a CatBoostClassifier, mainly as a benchmark.
 - **Falcon-7B**: Explored QLoRA-style fine-tuning for large language model-based intent classification, though this was more experimental.

- The main production path remains SetFit due to its efficiency, scalability, and strong performance on multilingual, multi-class intent classification tasks.

---


**Q: In the context of cosine similarity loss, what are the positive and negative pairs used for optimization during training?**

- In the SetFit training process, we optimize the embedding space by constructing **positive and negative pairs** from the labeled intent data.
- For each training example (e.g., a user query like "My printer is losing Wi-Fi connection"), we:
 - **Positive Pair**: Pair it with another example from the same intent class (e.g., "Printer keeps disconnecting from wireless network"). The goal is to make their embeddings close together (high cosine similarity).
 - **Negative Pair**: Pair it with an example from a different intent class (e.g., "How do I replace my printer ink cartridge?"). The goal is to push their embeddings further apart (low cosine similarity).
- During training, the model is optimized so that:
 - Embeddings of queries with the **same intent** are close together in the vector space.
 - Embeddings of queries with **different intents** are far apart.
- This is achieved by minimizing the cosine similarity loss, which encourages the model to learn a semantically meaningful embedding space where intent classes are well-separated.
- In practice, the SetFit framework automatically samples these positive and negative pairs from the labeled dataset during each training iteration, ensuring robust semantic alignment for intent classification.

- **Summary**: The second set of vectors are other queries from the dataset—either from the same intent (positive) or different intents (negative)—and the loss function optimizes the model to cluster similar intents and separate dissimilar ones in the embedding space.

---


**Q: How do you evaluate and ensure that different chunking or retrieval strategies actually improve output relevance and answer quality in a RAG pipeline?**

- We implemented a comprehensive, automated evaluation pipeline specifically for the Knowledge GPT (KGPT) RAG system to rigorously measure the impact of different chunking and retrieval strategies.
- **Evaluation Pipeline Overview:**
 - The pipeline is a standalone Python application that sends real user questions from a ground truth dataset to the KGPT APIs (Completions or Search).
 - It collects the generated answers and uses a combination of LLM-based and programmatic evaluators to score the responses across multiple dimensions.
- **Key Evaluation Metrics:**
 - **Answer Correctness:** Assessed by comparing the generated answer to the ground truth using both factual accuracy (via GPT-4o as a judge) and semantic similarity (using Azure OpenAI embeddings). Also checks if cited URLs are alive.
 - **Answer Relevance:** Evaluates whether the answer actually addresses the user’s question.
 - **Context Utilization:** Measures how well the retrieved context chunks are used in the generated answer. This is done by prompting the LLM to identify which sentences from each chunk were actually utilized, and scoring chunk and sentence-level utilization.
 - **Chunk Alignment:** Assesses per-chunk utilization and alignment with the answer, ensuring that the most relevant chunks are being leveraged.
 - **Mean Reciprocal Rank (MRR):** Used to evaluate retrieval effectiveness across different search types (lexical, semantic, hybrid).
 - **Negative Test Handling:** Ensures that for queries with no ground truth, the system correctly returns no document.
- **Automated Reporting:**
 - The pipeline generates detailed Excel reports, charts, and comparison tables for each experiment, making it easy to track improvements or regressions when changing chunking or retrieval strategies.
 - All LLM evaluation prompts enforce strict JSON output for reliable parsing and error tracking.
- **Iterative Experimentation:**
 - By running this pipeline after each change (e.g., switching from fixed-size to recursive chunking, or tuning retrieval parameters), we can quantitatively measure improvements in answer quality, relevance, and context utilization.
 - This data-driven approach ensures that any modifications to chunking or retrieval are validated with real-world metrics, not just intuition.

- **Summary:** 
 - We use an automated, LLM-augmented evaluation pipeline to objectively measure and compare the impact of different chunking and retrieval strategies on answer quality, relevance, and context utilization, ensuring continuous improvement of the KGPT system.

---
