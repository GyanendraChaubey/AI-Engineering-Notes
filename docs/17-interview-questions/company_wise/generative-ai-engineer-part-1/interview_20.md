# Generative AI Engineer (Part 1) — Interview 20

**Q: What systems or tools have you integrated your AI solution with?**

- For the Universal Intent Determination System (UIDS) project, I integrated the AI solution with several key enterprise systems and tools to ensure robust deployment, scalability, and seamless workflow integration:
 - **AWS SageMaker**: Deployed the intent classification models as real-time inference endpoints, enabling scalable and low-latency predictions for production workloads.
 - **CDAC Support Management System**: Integrated the SageMaker inference API directly into [Company]’s CDAC tool, allowing real-time intent detection and automated routing of customer queries from IVR, chatbots, and email channels.
 - **AWS S3**: Used for storing training datasets, model artifacts, and evaluation reports, supporting both local and cloud-based workflows.
 - **LangChain & LangGraph**: Leveraged for building agentic RAG pipelines and orchestrating multi-step data labeling and annotation workflows.
 - **OpenAI APIs**: Integrated for embeddings, prompt-based data augmentation, and LLM-driven annotation tasks.
 - **Falcon LLM**: Experimented with Falcon-7B and Falcon-40B models for multilingual intent classification, using QLoRA-style fine-tuning.
 - **CI/CD Tools**: Utilized Azure DevOps and Jenkins for automated deployment pipelines, including model training, endpoint deployment, and infrastructure provisioning.
 - **Vector Databases (ChromaDB)**: Used for semantic search and retrieval in the RAG-based data labeling pipeline.
 - **Monitoring & Evaluation Frameworks**: Implemented automated pipelines for model evaluation, reporting, and continuous monitoring of production endpoints.

- These integrations ensured the AI solution was production-ready, scalable, and could be easily embedded into [Company]’s enterprise support ecosystem, supporting multilingual and multi-channel customer interactions.

---
