# Generative AI Engineer (Part 1) — Interview 16

**Q: How do you select the best model when different models perform better on different evaluation metrics?**

- In real-world multi-class classification projects like UIDS, it’s common for models to perform differently across various metrics (accuracy, balanced accuracy, F1 scores, precision, recall).
- To make a robust and fair model selection, I follow a structured approach:

 - **Primary Metric Selection**:
 - First, I align with stakeholders and business requirements to identify the most critical metric for the use case (e.g., balanced accuracy for imbalanced classes, or weighted F1 for multilingual intent detection).
 - For UIDS, due to class imbalance and multilingual support, we prioritized **weighted F1 score** and **balanced accuracy** as primary metrics.

 - **Thresholding and Minimum Acceptable Values**:
 - Set minimum acceptable thresholds for all key metrics to ensure no critical aspect is compromised (e.g., minimum recall to avoid missing intents).

 - **Ranking and Trade-off Analysis**:
 - Rank models based on the primary metric.
 - If two models are close on the primary metric, use secondary metrics (e.g., precision, recall) to break ties or assess trade-offs.
 - Analyze confusion matrices and error patterns to ensure the model is not overfitting to dominant classes or languages.

 - **Business and Operational Considerations**:
 - Consider model inference speed, resource consumption, and language coverage.
 - For example, even if Falcon-7B had higher accuracy, we chose SetFit due to its broader language support and lower latency.

 - **Final Decision**:
 - Select the model that best balances the primary metric, meets all minimum thresholds, and aligns with operational/business constraints.
 - Document the rationale for selection and validate with stakeholders before production deployment.

- This approach ensures the chosen model is not just statistically optimal but also practical and aligned with business goals.

---


**Q: How do you control and optimize costs in multi-agent LLM systems where agents interact iteratively and each interaction incurs API costs?**

- In multi-agent LLM systems (e.g., using LangChain, CrewAI, or custom agentic frameworks), uncontrolled agent-to-agent interactions can quickly escalate API usage and costs, especially with iterative or recursive workflows.
- To ensure cost efficiency and prevent runaway expenses, I implement the following strategies:

 - **Iteration Limits**:
 - Set a strict maximum number of iterations or conversational turns per agent or per workflow session.
 - Enforce these limits programmatically in the orchestration logic (e.g., using counters or state graphs in LangGraph).

 - **Token and API Call Budgets**:
 - Define a maximum token budget or API call quota per session, agent, or user request.
 - Track cumulative tokens/calls and terminate or gracefully degrade the workflow when limits are reached.

 - **Early Stopping Criteria**:
 - Implement convergence checks or success criteria (e.g., if a goal is achieved, or a certain confidence threshold is met) to exit loops early and avoid unnecessary calls.

 - **Cost-Aware Orchestration**:
 - Use cost estimation utilities (many LLM SDKs provide token/cost calculators) to estimate and monitor real-time costs.
 - Log and alert if projected costs approach predefined thresholds.

 - **Batching and Caching**:
 - Where possible, batch agent requests or cache intermediate results to reduce redundant API calls.

 - **Role-Based Access and Throttling**:
 - Apply stricter limits for non-critical or experimental workflows, and use role-based controls to restrict high-cost operations.

 - **Monitoring and Reporting**:
 - Integrate detailed logging and monitoring (e.g., via centralized log managers or cloud monitoring tools) to track API usage, token consumption, and cost per workflow.
 - Generate regular cost reports for transparency and proactive management.

 - **Fail-Safe Mechanisms**:
 - Implement hard fail-safe cutoffs (e.g., abort workflow and return a fallback message) if any cost or iteration threshold is breached.

- By combining these controls, I ensure that multi-agent LLM systems remain cost-effective, predictable, and production-safe, even as agent complexity and interactions scale. This approach is critical for enterprise deployments where cost governance is as important as technical performance.

---

**Q: How do you ensure that an agent in a multi-agent system is performing well?**

- Ensuring agent performance in a multi-agent LLM system requires both quantitative and qualitative evaluation across several dimensions:

 - **Task Success Rate**:
 - Measure the percentage of tasks or goals successfully completed by the agent within the allowed iterations or token budget.
 - Use automated test suites with ground truth datasets to validate agent outputs.

 - **Response Quality Metrics**:
 - Evaluate factual correctness, semantic relevance, and context utilization of agent responses.
 - For example, in the Knowledge-GPT RAG system, we use an evaluation pipeline that:
 - Sends real questions to the agent and collects responses.
 - Uses LLM-based evaluators (e.g., GPT-4o) to score answers for factual accuracy, semantic similarity, and reference health.
 - Aggregates these into a composite answer correctness score (e.g., 70% factual accuracy, 20% semantic similarity, 10% reference health).

 - **Efficiency Metrics**:
 - Monitor inference latency, number of API calls, and token usage per task.
 - Ensure agents operate within defined cost and performance budgets.

 - **Context Utilization**:
 - Assess how effectively the agent uses retrieved context or supporting information in its answers.
 - Use chunk alignment and context relevance scoring (as implemented in the KGPT evaluation pipeline).

 - **Error Rate and Robustness**:
 - Track the frequency of failed, incomplete, or harmful responses.
 - Use harmful content evaluators and error logging to identify and mitigate issues.

 - **Continuous Monitoring and Reporting**:
 - Implement dashboards and automated reports (e.g., Excel, charts, comparison reports) to visualize agent performance over time.
 - Compare current performance against previous releases to detect regressions or improvements.

 - **Human-in-the-Loop Validation**:
 - Periodically review agent outputs with human evaluators for edge cases or ambiguous scenarios.

- By combining automated evaluation pipelines, real-time monitoring, and periodic human review, I ensure that each agent consistently meets quality, efficiency, and reliability standards in production environments.

---
