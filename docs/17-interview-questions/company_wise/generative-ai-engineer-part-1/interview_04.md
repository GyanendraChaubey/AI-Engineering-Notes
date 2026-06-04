# Generative AI Engineer (Part 1) — Interview 4

**Q: Explain the end goal of the intent classification system and RAG data labeling projects, and how the RAG pipeline supports the main intent classification system.**

- The end goal of the UIDS (Universal Intent Determination System) project is to build an enterprise-grade intent classification platform that can automatically detect and route user intents from various support channels (like IVR, chatbots, and virtual agents) across multiple enterprise applications.
 - The core ML model (SetFit) is trained on [Company]’s internal support data and deployed as a real-time inference endpoint on AWS SageMaker, handling high throughput and supporting multilingual use cases.
 - The main business objective is to automate intent detection, improve routing efficiency, and enhance customer support experiences.

- The RAG (Retrieval-Augmented Generation) data labeling pipeline is a supporting system designed to address data quality and labeling challenges that were limiting the main model’s accuracy.
 - The RAG pipeline uses LLMs (OpenAI/Azure) and vector search to automate and improve the data labeling process, especially for misclassified or ambiguous queries.
 - It orchestrates a modular, state-machine-based workflow (using LangGraph) that includes:
 - Data ingestion and cleaning (from local or S3 sources)
 - Optional language translation for multilingual support
 - Synthetic data generation to augment the dataset
 - Indexing of labeled data into a vector store for semantic retrieval
 - Batch inference and intent prediction using LLMs
 - Calculation and storage of evaluation metrics and logs

- The RAG pipeline is triggered when the main model’s accuracy plateaus (e.g., stuck at 73% for English), often due to noisy or inconsistent labels from third-party vendors.
 - It helps generate high-quality ground truth datasets by:
 - Translating non-English queries to English for unified processing
 - Using LLMs to suggest or validate intent labels
 - Automating the relabeling of ambiguous or misclassified samples
 - Supporting iterative improvement by feeding corrected labels back into the training pipeline

- This approach ensures continuous improvement of the intent classification model by maintaining high-quality, consistent, and multilingual training data, ultimately driving better model performance and business outcomes.

**In summary:** 
UIDS is the main intent classification system, while the RAG data labeling pipeline is a critical support system that automates and enhances data labeling, enabling higher model accuracy and scalability across languages and use cases.

---
