# Generative AI Engineer (Part 1) — Interview 22

**Q: How do you validate whether the LLM is providing the correct intent labels during automated data annotation?**

- We validate LLM-generated intent labels by comparing them against a manually curated ground truth dataset.
- For each query to be annotated, we retrieve the top-k most semantically similar examples (with known intents) from our vector store (ChromaDB).
- These examples are included as few-shot prompts to the LLM, which then predicts the intent for the new query.
- The predicted intent from the LLM is compared to the ground truth label for that query.
- We compute standard evaluation metrics such as accuracy and success rate by aggregating the number of correct predictions over the total queries.
- All predictions, along with their ground truth and LLM outputs, are saved as artifacts for further analysis and error tracking.
- This process ensures that the LLM’s automated annotations are systematically validated, and any discrepancies can be reviewed and corrected, maintaining high data quality for downstream model training.

**Supporting Document Reference:** 
As described in the UIDS-RAG-Labelling_Process documentation, the pipeline retrieves similar examples, constructs prompts, uses the LLM for prediction, and then compares the LLM’s output with ground truth to compute metrics and save results.

---

**Q: How do you implement enterprise security for your GenAI (JNI) applications?**

- For all GenAI applications, enterprise security is a top priority and is implemented across multiple layers, following best practices and compliance requirements.
- **Authentication & Authorization**:
 - We integrate with [Company]’s OAuth system for user authentication, ensuring only authorized users can access the APIs and services.
 - JWT (JSON Web Token) validation is enforced for every request, with token caching for performance.
- **Secrets & Credential Management**:
 - All sensitive credentials (API keys, certificates) are securely stored and managed using AWS Secrets Manager, ensuring no hardcoded secrets in code or configuration.
- **Network Security & Firewalls**:
 - AWS Web Application Firewall (WAF) is deployed at the edge (via CloudFront) to filter malicious traffic and protect against common web exploits.
 - Security groups and subnets are configured to restrict network access to only necessary resources.
- **Data Security**:
 - All data at rest is encrypted using AWS KMS-managed keys.
 - Data in transit is protected using HTTPS/TLS across all endpoints.
- **Access Control & Least Privilege**:
 - IAM roles and policies are strictly defined to follow the principle of least privilege for all services and users.
- **Caching & Session Security**:
 - Redis (Amazon ElastiCache) is used for session and token caching, with access restricted to internal services only.
- **Audit & Observability**:
 - All access and actions are logged using Amazon CloudWatch and OpenTelemetry, providing a full audit trail for compliance and incident response.
- **Vulnerability Management**:
 - Regular vulnerability scans and patching are performed on all infrastructure components.
- **Compliance**:
 - The architecture is designed to meet enterprise compliance standards, including GDPR and internal [Company] security guidelines.

These measures ensure robust, multi-layered security for all GenAI applications, protecting both data and infrastructure in enterprise environments.

---

**Q: What measures do you implement for prompt security, network security, and data security in your AWS-based GenAI applications?**

- **Prompt Security:**
 - All user inputs are pre-processed and scanned for sensitive information using AWS Comprehend’s PII detection before being passed to LLMs, ensuring no PII or confidential data is exposed in prompts.
 - Guardrails are implemented at the application layer to filter out harmful or malicious content, and prompt injection attacks are mitigated by sanitizing and validating all incoming queries.
 - LLM outputs are also post-processed to detect and redact any sensitive or inappropriate content before returning responses to users.

- **Network Security:**
 - AWS WAF (Web Application Firewall) is deployed at the CloudFront edge to filter malicious traffic and protect against common web exploits.
 - All north-south traffic (external ingress/egress) is routed through AWS ALB/NLB, while east-west traffic (internal service-to-service) is restricted within the VPC using Kubernetes NetworkPolicies.
 - IAM Roles for Service Accounts (IRSA) are used for secure service-to-service AWS API access, and security groups/NACLs are tightly configured to restrict network access.
 - Optional mTLS (mutual TLS) is considered for service-to-service encryption if compliance requires.

- **Data Security:**
 - All data at rest is encrypted using AWS KMS-managed keys.
 - Data in transit is protected using HTTPS/TLS across all endpoints.
 - Secrets, API keys, and certificates are managed securely using AWS Secrets Manager—no hardcoded credentials.
 - Redis (Amazon ElastiCache) is used for secure, internal session and token caching.
 - PII detection and redaction are enforced using AWS Comprehend before any data is stored or processed further.
 - IAM roles and policies enforce least-privilege access to all data and services.

- **Additional Controls:**
 - Audit trails and observability are maintained using Amazon CloudWatch and OpenTelemetry for logging, metrics, and distributed tracing.
 - Regular vulnerability scans and patching are performed on all infrastructure components.
 - Compliance with enterprise and regulatory standards (such as GDPR) is ensured throughout the architecture.

These combined measures provide robust, multi-layered security for prompt handling, network communication, and data management in enterprise GenAI applications on AWS.

---

**Q: How do you handle conversation memory and session management in your GenAI applications?**

- Conversation memory is managed using Redis as a centralized conversation state store.
- For each user session, a unique `chat_session_id` is generated and used to track the conversation context.
- The MCPManager Orchestrator loads and updates conversation history for every user message, ensuring multi-turn context is preserved.
- Conversation history (such as the last 10 turns) is stored in Redis, allowing the system to maintain and retrieve context efficiently across requests.
- Redis is configured with a 2GB memory limit and LRU (Least Recently Used) eviction policy to optimize performance and resource usage.
- This setup enables seamless multi-turn interactions, supports long-term memory (with plans to extend to persistent DB storage in future sprints), and ensures that each user’s session context is isolated and efficiently managed.
- All session and token data are kept internal to the VPC, ensuring security and privacy of conversation data.

**Supporting Document Reference:** 
- As per the KGPT_MCP_Design_Document, Redis is used for both token caching and conversation state, with session IDs and conversation history managed by the MCPManager and Conversation Store components.

---

**Q: How would you approach building a unified dataset to match and compare product descriptions and prices across multiple e-commerce sites with differing product descriptions?**

- **Problem Summary**: 
 The challenge is to match the same products across different e-commerce platforms (Amazon, Flipkart, eBay, etc.) where product descriptions and metadata may differ, and then create a unified dataset for price comparison and analytics.

- **Industry-Standard Approach**:
 - **1. Data Ingestion & Normalization**:
 - Collect product data (title, description, price, attributes) from each platform via APIs or web scraping.
 - Normalize fields (e.g., convert all prices to a common currency, standardize attribute names like "color", "storage", etc.).
 - **2. Product Matching (Entity Resolution)**:
 - **Text Preprocessing**: Clean and tokenize product titles/descriptions (remove stopwords, lowercase, etc.).
 - **Feature Extraction**: Extract structured attributes (brand, model, color, storage) using NLP techniques or regular expressions.
 - **Embedding-Based Similarity**:
 - Use pre-trained sentence transformers (e.g., SBERT, OpenAI embeddings) to generate vector representations of product titles/descriptions.
 - Compute cosine similarity between products across platforms to identify likely matches.
 - **Hybrid Matching**:
 - Combine attribute-based matching (exact/partial match on brand, model, etc.) with embedding similarity for robust matching.
 - Optionally, use fuzzy string matching (Levenshtein distance) for additional validation.
 - **Clustering**:
 - Cluster similar products using unsupervised methods (e.g., DBSCAN on embedding space) to group variants of the same product.
 - **3. Unified Dataset Construction**:
 - For each matched product group, aggregate prices from all platforms into a single row (Amazon price, Flipkart price, etc.).
 - Store additional metadata (URLs, product IDs, etc.) for traceability.
 - **4. Validation & Human-in-the-Loop**:
 - For ambiguous matches, flag for manual review or use active learning to improve matching accuracy.
 - **5. Automation & Scaling**:
 - Build the pipeline using Python (Pandas, scikit-learn, sentence-transformers), and orchestrate with Airflow or AWS Glue for scalability.
 - Store results in a relational database or data warehouse for analytics.

- **GenAI/NLP Application**:
 - Use LLMs or custom prompt engineering to extract structured attributes from unstructured descriptions.
 - Use embeddings for semantic similarity, which is more robust than simple keyword or rule-based matching.
 - Optionally, use RAG pipelines to enrich product data with additional context or resolve ambiguities.

- **Why Not Only Rule-Based?**
 - Rule-based or fuzzy matching alone is insufficient due to variations in descriptions, typos, and attribute order.
 - Embedding-based and hybrid approaches provide higher accuracy and scalability.

- **Example Workflow**:
 1. Ingest product data from Amazon, Flipkart, eBay.
 2. Extract and normalize attributes (brand, model, color, storage).
 3. Generate embeddings for each product.
 4. For each product on Flipkart, find the most similar product on Amazon/eBay using cosine similarity and attribute checks.
 5. Aggregate matched products into a unified table with prices from each platform.

- **Tools & Libraries**:
 - Python, Pandas, sentence-transformers, scikit-learn, fuzzywuzzy, AWS Glue/Athena, Airflow.

- **Scalability & Maintenance**:
 - Use cloud-based orchestration and storage for large-scale data.
 - Implement monitoring and periodic retraining for the matching models.

This approach ensures accurate, scalable, and maintainable product matching and price comparison across multiple e-commerce platforms using a combination of NLP, embeddings, and traditional data engineering techniques.

---

**Q: How would you implement a RAG (Retrieval-Augmented Generation) pipeline for product matching and enrichment in this e-commerce use case?**

- **RAG Pipeline Implementation for Product Matching:**
 - **1. Data Indexing:**
 - Index all product descriptions, titles, and structured attributes from each e-commerce platform into a vector database (e.g., OpenSearch with KNN, ChromaDB, or FAISS).
 - Generate embeddings for each product using a pre-trained model (e.g., OpenAI, SBERT, or BERT-based sentence transformers).
 - Optionally, also index metadata fields (brand, model, color, storage) for hybrid search.
 - **2. Query Construction:**
 - For a given product (e.g., from Flipkart), construct a query using its title, description, and extracted attributes.
 - Use both the raw text and structured fields to maximize retrieval accuracy.
 - **3. Retrieval Phase:**
 - Perform a hybrid search:
 - **Lexical Search**: Use BM25 or keyword-based search on product titles and descriptions for exact or near-exact matches.
 - **Semantic Search**: Use KNN vector search to find semantically similar products based on embeddings.
 - **Hybrid Ranking**: Combine results using Reciprocal Rank Fusion (RRF) or a weighted linear blend (as described in the Knowledge GPT architecture) to get the most relevant candidates.
 - **4. Context Building:**
 - Aggregate the top-k retrieved product candidates from other platforms as context.
 - Extract and align structured attributes (brand, model, color, storage) from these candidates.
 - **5. Generation/Enrichment Phase:**
 - Use an LLM (e.g., OpenAI GPT) with a prompt template that includes the original product and the retrieved context.
 - Ask the LLM to:
 - Confirm if the products are the same (entity resolution).
 - Extract or standardize attributes (e.g., normalize color names, storage sizes).
 - Suggest unified product records or flag ambiguous cases for manual review.
 - **6. Output Construction:**
 - For each matched product group, create a unified row with prices and attributes from all platforms.
 - Store the results in a unified dataset for downstream analytics or price comparison.

- **Why RAG?**
 - RAG enables leveraging both structured retrieval (vector/keyword search) and generative reasoning (LLM) to handle noisy, inconsistent, or incomplete product data.
 - It is robust to variations in product descriptions, typos, and attribute order, outperforming pure rule-based or fuzzy matching.

- **Reference to Documented Architecture:**
 - This approach mirrors the hybrid and ranked_hybrid search strategies described in the Knowledge GPT API Architecture, combining BM25 and KNN vector search with RRF or linear blending for optimal relevance.
 - The context-building and prompt-injection steps follow the same principles as the RAG flows used for knowledge retrieval and LLM prompt construction.

- **Tools & Stack:**
 - OpenSearch (BM25 + KNN), ChromaDB, or FAISS for vector search.
 - Sentence-transformers or OpenAI embeddings for vectorization.
 - Python for orchestration, with FastAPI for API endpoints if needed.
 - LLMs (OpenAI, Azure OpenAI) for attribute extraction and entity resolution.

This RAG-based pipeline ensures accurate, scalable, and explainable product matching and enrichment across multiple e-commerce platforms.

---

**Q: If product image URLs are also available, would your approach to product matching and unification change?**

- **Core Approach Remains the Same**:
 - The main workflow—data ingestion, normalization, hybrid retrieval (BM25 + KNN), context building, and LLM-based entity resolution—remains unchanged.
 - Textual and structured data matching continues as before.

- **Image Data as an Additional Signal**:
 - Images provide a valuable, orthogonal signal for product matching, especially when textual descriptions are ambiguous or inconsistent.
 - I would enhance the pipeline by incorporating image similarity as an additional feature in the matching process.

- **How to Integrate Images**:
 - **Image Embedding Generation**:
 - Use a pre-trained vision model (e.g., CLIP, OpenAI’s image embeddings, or AWS Rekognition) to generate vector embeddings for each product image.
 - **Multi-Modal Retrieval**:
 - For each product, combine text embeddings (from titles/descriptions) and image embeddings.
 - During retrieval, compute both text-based and image-based similarities between products across platforms.
 - Use a weighted or hybrid scoring mechanism (e.g., linear blend or RRF) to aggregate text and image similarity scores for final candidate ranking.
 - **LLM Context Enrichment**:
 - When building the prompt for the LLM, include references to the most similar images (e.g., image URLs or generated captions).
 - Optionally, use LLMs to generate or validate image captions for further context (as done in the Knowledge GPT pipeline for missing alt text).

- **Benefits**:
 - Improves matching accuracy for products with visually distinctive features (e.g., color, design).
 - Reduces false positives/negatives in cases where text alone is insufficient.

- **Reference to Documented Practice**:
 - The Knowledge GPT pipeline already processes image URLs, generates captions using LLMs, and stores image metadata for downstream use—this can be adapted for product matching.

- **Summary**:
 - The approach becomes multi-modal, leveraging both text and image data for robust, scalable, and accurate product matching and unification across platforms.

This multi-modal strategy ensures higher confidence in product entity resolution, especially in challenging or ambiguous cases.

---

**Q: If you have product images in addition to descriptions, how would you use them in your product matching and unification approach?**

- If product images are available, I would enhance the matching pipeline by making it multi-modal—leveraging both text and image data for more robust product matching.
- **Image Embedding Generation**: For each product image, I would use a pre-trained vision model like CLIP or AWS Rekognition to generate image embeddings (vector representations).
- **Multi-Modal Similarity**: During the matching phase, I would compute both text-based similarity (using embeddings from product titles/descriptions) and image-based similarity (using image embeddings). These similarity scores can be combined—using a weighted or hybrid approach—to improve matching accuracy, especially when textual descriptions are ambiguous or inconsistent.
- **Hybrid Ranking**: The final candidate ranking would blend text and image similarity scores, similar to how the Knowledge GPT pipeline uses reciprocal rank fusion (RRF) or linear blending for BM25 and KNN results.
- **LLM Context Enrichment**: When constructing the prompt for the LLM, I could include image URLs or even generate image captions using an LLM (as done in the Knowledge GPT pipeline for missing alt text), providing richer context for entity resolution.
- **Benefits**: This approach increases confidence in product matching, reduces false positives/negatives, and is especially useful for visually distinctive products or when text data is insufficient.
- **Reference to Practice**: In the Knowledge GPT pipeline, image URLs are reconstructed using CloudFront CDN, and missing alt text is generated using Azure OpenAI LLMs—demonstrating how image data can be integrated and enriched for downstream tasks.

By combining both text and image modalities, the product matching process becomes more accurate, scalable, and resilient to inconsistencies in either data source.

---

**Q: Will having product images along with descriptions help improve product matching accuracy?**

- Yes, having product images in addition to descriptions will significantly help improve product matching accuracy.
- Images provide a strong, independent signal that can resolve ambiguities where text descriptions differ, are incomplete, or contain typos.
- By generating image embeddings using models like CLIP or AWS Rekognition, we can compare visual similarity between products across platforms.
- Combining image-based similarity with text-based similarity (multi-modal matching) leads to more robust and reliable entity resolution, especially for visually distinctive products (e.g., color, design, packaging).
- This approach is already reflected in industry best practices and aligns with enhancements described in the Knowledge GPT pipeline, where image URLs are processed, validated, and even enriched with LLM-generated captions for better downstream matching and analytics.
- In summary, integrating images into the matching pipeline increases confidence, reduces false positives/negatives, and is highly recommended for comprehensive product unification across e-commerce platforms.

---
