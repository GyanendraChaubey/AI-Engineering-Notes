# Evaluation of LLM Systems

> You can't improve what you can't measure. Rigorous evaluation is what separates production-grade systems from demos.

---

## 1. How do you select the best LLM for your specific use case?

Never rely solely on public benchmarks — they may not reflect your task. Build a **task-specific evaluation harness**:

```python
# Evaluation pipeline
eval_dataset = [
    {
        "input": "Summarize this legal clause in plain English: ...",
        "reference": "The expected plain-English summary...",
        "metadata": {"domain": "legal", "complexity": "high"}
    },
    # ... 200-500 examples
]

def evaluate_model(model_name, eval_dataset):
    scores = {"latency": [], "quality": [], "cost": []}
    
    for example in eval_dataset:
        start = time.time()
        output = call_model(model_name, example["input"])
        
        scores["latency"].append(time.time() - start)
        scores["quality"].append(llm_judge_score(output, example["reference"]))
        scores["cost"].append(estimate_cost(example["input"], output, model_name))
    
    return {
        "avg_quality": np.mean(scores["quality"]),
        "p95_latency": np.percentile(scores["latency"], 95),
        "cost_per_1k": np.mean(scores["cost"]) * 1000
    }

# Compare candidates
for model in ["gpt-4o", "claude-sonnet-4-5", "llama-3-70b"]:
    print(model, evaluate_model(model, eval_dataset))
```

**Evaluation dimensions:** Quality, Latency (p50/p95/p99), Cost, Reliability (error rate), Context length support, Safety/refusal behavior.

---

## 2. How do you evaluate a RAG-based system end to end?

RAG has two components to evaluate: **retrieval** and **generation**. Both must be assessed.

### Retrieval Metrics
```python
# Recall@K: Did the correct chunk appear in top-K retrieved?
def recall_at_k(retrieved_ids, relevant_ids, k):
    return int(any(r in retrieved_ids[:k] for r in relevant_ids))

# Context Precision: Of retrieved chunks, what fraction were relevant?
def context_precision(retrieved_ids, relevant_ids):
    relevant_retrieved = [1 if r in relevant_ids else 0 for r in retrieved_ids]
    return sum(relevant_retrieved) / len(retrieved_ids)
```

### Generation Metrics (using RAGAS framework)
```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,        # Is the answer grounded in context?
    answer_relevancy,   # Is the answer relevant to the question?
    context_recall,     # Does context contain answer information?
    context_precision,  # Is context free of irrelevant chunks?
)
from datasets import Dataset

test_data = Dataset.from_list([
    {
        "question": "What is the refund window?",
        "answer": "30 days",                          # RAG system output
        "contexts": ["Returns accepted within 30 days"], # Retrieved chunks
        "ground_truth": "Items can be returned in 30 days"
    }
])

results = evaluate(test_data, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
print(results)
```

### End-to-End: LLM-as-Judge
```python
judge_prompt = """
Rate this answer on a scale of 1-5 for:
- Correctness (is it factually accurate?)
- Completeness (does it fully answer the question?)
- Groundedness (is it supported by the context?)

Question: {question}
Context: {context}
Answer: {answer}
Ground truth: {ground_truth}

Return JSON: {{"correctness": N, "completeness": N, "groundedness": N, "explanation": "..."}}
"""
```

---

## 3. What are the standard metrics for evaluating LLM outputs?

### Automated Text Quality Metrics

| Metric | What it measures | Use when |
|---|---|---|
| **BLEU** | N-gram overlap with reference | Machine translation |
| **ROUGE-L** | Longest common subsequence | Summarization |
| **BERTScore** | Semantic similarity via embeddings | When wording can vary |
| **METEOR** | BLEU + stemming + synonyms | Translation |

```python
from evaluate import load

rouge = load("rouge")
scores = rouge.compute(predictions=["The cat sat."], references=["The cat was sitting."])
# {'rouge1': 0.8, 'rouge2': 0.5, 'rougeL': 0.8}
```

### LLM-as-Judge (most practical for open-ended tasks)

```python
def llm_judge(question, response, criteria=["helpfulness", "accuracy", "clarity"]):
    prompt = f"""
    Evaluate this AI response on each criterion. Score 1-10.
    
    Question: {question}
    Response: {response}
    
    Criteria: {criteria}
    Return: {{"scores": {{}}, "rationale": ""}}
    """
    return call_llm(prompt)
```

### Human Evaluation (gold standard)
- Preference rate (A vs B)
- Absolute quality (1–5 Likert scale)
- Task completion rate
- Error rate per 100 responses

### Safety Metrics
- Refusal rate on harmful prompts
- Factual accuracy rate
- Bias detection (stereotypical outputs)

---

## 4. What is Chain of Verification (CoVe) and how does it reduce hallucination?

**CoVe (Chain of Verification)** is a technique where the model verifies its own answer through targeted fact-checking questions.

**Steps:**
1. **Draft:** Generate initial answer
2. **Plan:** Identify verifiable claims in the answer and generate verification questions
3. **Execute:** Answer each verification question independently (without seeing the draft answer)
4. **Refine:** If verification answers contradict the draft, revise the final answer

```python
# Step 1: Draft answer
draft = llm(f"Answer: {user_question}")
# "Marie Curie was born in Warsaw in 1867 and won two Nobel Prizes."

# Step 2: Generate verification questions
verification_questions = llm(f"""
List factual claims in this answer that should be verified, as questions:
Answer: {draft}
""")
# Q1: What year was Marie Curie born?
# Q2: Where was Marie Curie born?
# Q3: How many Nobel Prizes did Marie Curie win?

# Step 3: Answer verification questions independently (no peeking at draft)
verifications = [llm(f"Answer independently: {q}") for q in verification_questions]
# A1: 1867 ✓  A2: Warsaw ✓  A3: Two (Physics 1903, Chemistry 1911) ✓

# Step 4: Revise if contradictions found
final = llm(f"""
Original answer: {draft}
Verification Q&As: {verifications}
If any facts are wrong, provide a corrected final answer. Otherwise, confirm the answer.
""")
```

CoVe significantly reduces factual errors, especially for answers involving numbers, dates, names, and statistics.

---

*Next: [Hallucination Control →](../11-hallucination/README.md)*
