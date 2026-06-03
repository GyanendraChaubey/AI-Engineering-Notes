# Hallucination Control Techniques

> Hallucination is the #1 reliability problem in production LLM systems. Here's how to fight it at every layer.

---

## 1. What are the different forms of hallucination in LLMs?

**Hallucination** is when an LLM produces confident, fluent content that is factually incorrect, unsupported, or fabricated.

### Types by Origin

| Type | Description | Example |
|---|---|---|
| **Factual hallucination** | Wrong facts presented as true | "The Eiffel Tower was built in 1901" (it was 1889) |
| **Source hallucination** | Fabricated citations, papers, URLs | "According to Smith et al. (2019)..." — paper doesn't exist |
| **Entity hallucination** | Made-up people, companies, events | "As CEO Jane Doe said in the 2022 earnings call..." |
| **Numerical hallucination** | Wrong numbers, dates, statistics | Revenue of $2.3B (actual: $1.1B) |
| **Instruction following hallucination** | Appears to follow instructions but subtly doesn't | Returns JSON with wrong field names |
| **Context hallucination** | Answer contradicts provided context | RAG answer that ignores retrieved document |

### Types by Cause

- **Knowledge gap:** Model doesn't know → fills in plausibly
- **Conflicting training data:** Multiple facts conflict → picks one incorrectly
- **Sycophancy:** User's premise is false → model agrees rather than corrects
- **Overthinking:** Chain-of-thought reasons itself into a wrong conclusion

---

## 2. How do you control hallucinations at each layer of the system?

### Layer 1: Prompt Level

```python
system_prompt = """
You are a factual assistant with the following rules:
1. ONLY answer from the provided context below.
2. If the context doesn't contain the answer, say: "I don't have enough information."
3. Never make up facts, dates, names, or numbers.
4. If uncertain, explicitly say "I'm not certain, but..."
5. After your answer, cite the specific sentence from the context you used.

Context:
{context}
"""
```

- Ground responses in retrieved context (RAG)
- Add uncertainty quantification instructions
- Use low temperature for factual tasks (0.0–0.2)
- Add self-check instructions ("verify your answer before responding")

### Layer 2: Retrieval Level

Better retrieval = less hallucination because the model has more relevant context to work from.

```python
# Multi-source verification
def retrieve_with_verification(query, retrievers):
    """Retrieve from multiple sources and check consistency."""
    all_chunks = []
    for retriever in retrievers:
        chunks = retriever.get(query, k=3)
        all_chunks.extend(chunks)
    
    # Check if retrieved chunks agree with each other
    consistency_check = llm(f"""
    Do these passages agree or contradict each other?
    {[c.text for c in all_chunks]}
    Return: {{"consistent": true/false, "contradictions": [...]}}
    """)
    return all_chunks, consistency_check
```

### Layer 3: Generation Level

- **Chain of Verification (CoVe):** Generate → verify → revise
- **Self-consistency:** Sample 5 answers at temperature=0.7, take majority vote
- **Citation enforcement:** Force model to cite source sentences
- **Constrained decoding:** Restrict output to validated formats (especially for structured data)

```python
# Self-consistency for factual questions
def self_consistent_answer(question, n=5):
    answers = [llm(question, temperature=0.7) for _ in range(n)]
    
    # Find most common answer
    vote_prompt = f"""
    These are {n} independent answers to the same question.
    Identify the most common answer and return it.
    
    Answers: {answers}
    Question: {question}
    """
    return llm(vote_prompt, temperature=0)
```

### Layer 4: Post-Generation Verification

```python
def verify_factual_claims(answer, source_context):
    """Use an LLM to check if every claim in the answer is supported by context."""
    verification = llm(f"""
    Check each factual claim in the answer against the source context.
    
    Answer: {answer}
    Source context: {source_context}
    
    For each claim, determine: SUPPORTED, UNSUPPORTED, or CONTRADICTED.
    Return JSON: {{"claims": [{{"claim": "...", "status": "...", "evidence": "..."}}]}}
    """)
    
    claims = json.loads(verification)
    unsupported = [c for c in claims["claims"] if c["status"] != "SUPPORTED"]
    
    if unsupported:
        # Trigger regeneration or flag for human review
        return revise_answer(answer, unsupported)
    return answer
```

### Layer 5: System Design Level

- **Human-in-the-loop:** Flag low-confidence answers for human review
- **Confidence scoring:** Train a classifier to detect likely hallucinations
- **Knowledge graph grounding:** Validate entities against a knowledge graph
- **Output monitoring:** Track hallucination rate in production, alert on spikes

```python
# Hallucination detection classifier
from transformers import pipeline

hallucination_detector = pipeline(
    "text-classification",
    model="vectara/hallucination_evaluation_model"
)

score = hallucination_detector(f"premise: {context} hypothesis: {answer}")
# {"label": "HALLUCINATION", "score": 0.87} → flag for review
```

---

*Next: [LLM Deployment →](../12-deployment/README.md)*
