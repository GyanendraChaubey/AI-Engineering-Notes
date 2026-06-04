# ai — Interview 2

**Q: What is prompt hijacking and in what context is it used?**

- Prompt hijacking, also known as prompt injection, is a security vulnerability specific to systems that use Large Language Models (LLMs) or Generative AI, especially when user input is incorporated into prompts sent to the model.
- It occurs when a malicious user crafts input designed to manipulate, override, or "hijack" the intended instructions or context provided to the LLM, potentially causing the model to behave in unintended or unsafe ways.
- Common contexts where prompt hijacking is a concern:
 - AI assistants and chatbots that dynamically build prompts from user input.
 - Retrieval-Augmented Generation (RAG) systems where user queries are combined with retrieved context and sent to the LLM.
 - Any application exposing LLMs to end-users, especially in enterprise environments where sensitive data or actions could be exposed or manipulated.
- Industry best practices to mitigate prompt hijacking include:
 - Implementing prompt injection detection models or validators that scan user input and conversation history for suspicious patterns before sending to the LLM.
 - Blocking or sanitizing requests with high injection risk, as seen in the Knowledge GPT architecture where requests with a high injection score are blocked and a safe fallback message is returned.
 - Prioritizing system availability by allowing requests to proceed if the detection service is down, but always logging and monitoring such events.
- Prompt hijacking is a critical security consideration in production LLM deployments, as it can lead to information leakage, policy bypass, or even execution of unintended actions by the AI system.

---

**Q: How can prompt hijacking (prompt injection) be detected and avoided in LLM-based systems?**

- In production LLM systems, prompt hijacking is detected and mitigated using a combination of automated validators, external ML models, and robust fallback mechanisms.
- A typical approach involves:
 - **Prompt Injection Detection Validator**: Aggregate all user messages (including conversation history and current input) and split them into manageable windows (e.g., 256 words each) to fit the detection model’s input constraints.
 - **External Detection Model**: Send these windows to an external prompt injection detection API, which returns a score indicating the likelihood of injection for each window.
 - **Threshold-Based Blocking**: If any window’s injection score exceeds a defined threshold (e.g., 0.5), the request is blocked. The system returns a safe fallback message, does not perform any search or LLM call, and ensures no information leakage.
 - **Fail-Open Policy**: If the detection service is unavailable, the system allows the request to proceed (fail-open) to maintain API availability, but logs the event for monitoring.
 - **Parallel Validators**: Run prompt injection detection alongside other validators (like PII detection) to ensure comprehensive input sanitization before invoking the LLM.
 - **Fallback Routing**: If injection is detected, route the response to a fallback handler that provides a generic or safe message, preventing the LLM from processing malicious input.
- These controls are implemented as part of the API architecture, ensuring that every user request is validated before reaching the LLM, significantly reducing the risk of prompt hijacking in enterprise environments.

---

**Q: How do you resolve deadlock situations when multiple AI agents are involved?**

- In multi-agent AI systems, deadlocks can occur when agents wait indefinitely for each other’s actions or resources, leading to stalled workflows.
- Practical strategies to resolve or prevent deadlocks in agentic AI architectures include:
 - **Timeouts and Retries**: Implement timeouts for agent actions or tool calls. If an agent does not receive a response within a defined period, it should retry, escalate, or abort the operation.
 - **Centralized Orchestration**: Use an orchestration layer (like MCP or a workflow manager) to monitor agent states and dependencies, detect circular waits, and intervene when deadlocks are detected.
 - **Resource Locking Protocols**: Apply resource allocation protocols (e.g., ordering resource acquisition, using lock hierarchies) to prevent circular dependencies among agents.
 - **Deadlock Detection Algorithms**: Periodically analyze agent state graphs for cycles (using algorithms like wait-for graphs) and trigger recovery actions if a deadlock is detected.
 - **Fallback and Escalation Mechanisms**: If deadlock is detected, agents can escalate to a human operator, trigger a fallback workflow, or release held resources to break the cycle.
 - **Stateless Design Where Possible**: Design agents to minimize persistent locks or dependencies, favoring stateless or loosely coupled interactions.
- In enterprise AI platforms (like those using MCP), these controls are often built into the orchestration logic, leveraging event-driven architectures (e.g., AWS EventBridge, Lambda) to monitor and resolve agent interactions dynamically.
- Regular audits, logging, and monitoring are essential to quickly identify and address deadlock scenarios in production.

---
