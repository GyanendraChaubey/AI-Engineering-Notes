# Prompt Engineering & LLM Fundamentals

> Core concepts every LLM engineer must know — from how models generate text to advanced reasoning techniques.

---

## 1. How does Generative AI differ from traditional Predictive/Discriminative AI?

**Predictive/Discriminative AI** learns a boundary between classes. Given input X, it outputs a label or probability — e.g., "is this email spam?" It models P(Y|X).

**Generative AI** learns the underlying data distribution itself and can *synthesize* new data. LLMs model P(X) — the probability of a sequence of tokens — and generate new sequences by sampling from that distribution.

| Aspect | Discriminative AI | Generative AI |
|---|---|---|
| Goal | Classify / predict | Create new content |
| Models | SVM, Logistic Regression, BERT (classifier head) | GPT, Gemini, Llama |
| Output | Label / score | Text, image, audio |
| Training signal | Labeled pairs (X, Y) | Raw unlabeled data |

---

## 2. What is a Large Language Model and how is it trained?

An LLM is a deep neural network (transformer-based) trained on massive text corpora to predict the next token in a sequence. Once trained, the model captures grammar, facts, reasoning patterns, and world knowledge in its parameters.

**Training stages:**

1. **Pre-training** — Self-supervised next-token prediction on trillions of tokens (books, web, code). Loss = cross-entropy over the vocabulary.
2. **Supervised Fine-Tuning (SFT)** — Train on high-quality instruction-response pairs so the model follows instructions.
3. **Preference Alignment (RLHF / DPO)** — Use human feedback or preference data to steer the model toward helpful, safe, and honest outputs.

```python
# Conceptual training loop (simplified)
for batch in dataloader:
    input_ids = batch["input_ids"]          # [B, T]
    labels    = input_ids[:, 1:]            # predict next token
    logits    = model(input_ids[:, :-1])    # [B, T-1, V]
    loss      = cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()
```

---

## 3. What exactly is a "token" in the context of language models?

A token is the atomic unit the model processes — not necessarily a word. Tokenization splits text into subword pieces using algorithms like **Byte-Pair Encoding (BPE)** or **WordPiece**.

Examples (GPT-4 tokenizer):
- `"hello"` → 1 token
- `"unbelievable"` → 3 tokens: `["un", "believ", "able"]`
- `"AI"` → 1 token, `" AI"` (with space) → 1 different token

**Why it matters:**
- Cost is billed per token (API usage)
- Context window limits are in tokens, not words
- Rare words / names / code fragments consume more tokens

```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")
tokens = enc.encode("Large Language Models are fascinating.")
print(len(tokens))  # 7 tokens
```

---

## 4. How do you estimate the cost of running LLMs — both API-based and self-hosted?

**API-based (e.g., OpenAI, Anthropic):**

```
Cost = (input_tokens × price_per_input_token) + (output_tokens × price_per_output_token)
```

Example: GPT-4o at $5/1M input, $15/1M output  
1000 requests × 500 input + 200 output tokens = $5.50

**Self-hosted (e.g., Llama 3):**

```
Cost = GPU hours × GPU hourly rate
GPU hours = (tokens_per_day / throughput_tokens_per_sec) / 3600
```

Key factors: model size (7B vs 70B), quantization, batch size, GPU type (A100 vs H100).

A rule of thumb: a 7B model at 4-bit quantization runs on a single A10G GPU (~$1.50/hr on AWS), handling ~50 req/s at 200 tokens each.

---

## 5. What is the Temperature parameter and how should it be configured?

Temperature controls randomness in token sampling by scaling the logits before applying softmax.

```python
# Without temperature (greedy)
probs = softmax(logits)

# With temperature
probs = softmax(logits / temperature)
```

- **Temperature → 0**: Near-deterministic, always picks highest-probability token. Best for factual Q&A, SQL generation, code.
- **Temperature = 1.0**: Samples from the raw model distribution. Good for chat.
- **Temperature > 1.0**: More random/creative. Useful for brainstorming, storytelling.

**Practical guidance:**

| Task | Recommended Temperature |
|---|---|
| Code generation | 0.0 – 0.2 |
| Factual Q&A | 0.0 – 0.3 |
| Chat assistant | 0.7 – 1.0 |
| Creative writing | 1.0 – 1.3 |

---

## 6. What are the different strategies for selecting output tokens (decoding strategies)?

| Strategy | How it works | Best for |
|---|---|---|
| **Greedy** | Always pick the highest-probability token | Deterministic tasks |
| **Beam Search** | Maintain top-K candidate sequences | Translation, summarization |
| **Top-K Sampling** | Sample from the top K tokens | Balanced creativity |
| **Top-P (Nucleus)** | Sample from the smallest set summing to probability P | Creative tasks |
| **Temperature Sampling** | Scale logits before sampling | General use |

```python
# Top-P nucleus sampling (pseudocode)
sorted_probs = sort(probs, descending=True)
cumulative   = cumsum(sorted_probs)
nucleus      = sorted_probs[cumulative <= top_p]
token        = sample(nucleus)
```

In practice, Top-P + Temperature together give the best results for most applications.

---

## 7. What are the ways to define stopping criteria for LLM generation?

1. **Max tokens** — hard cap on output length
2. **Stop sequences** — predefined strings that halt generation (e.g., `"\n\nHuman:"`)
3. **EOS token** — model generates the special end-of-sequence token
4. **Logit bias** — suppress certain tokens (e.g., force model not to generate harmful keywords)

```python
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "List 3 fruits"}],
    max_tokens=100,
    stop=["4.", "\n\n"]   # stop after 3 items
)
```

---

## 8. How do stop sequences work and when should you use them?

Stop sequences are strings that, when generated, immediately terminate output — the sequence itself is not included in the response. They're useful for:

- **Structured generation**: Stop after JSON closes `}`
- **Multi-turn chat**: Stop when model would generate the next human turn
- **Templated outputs**: Stop at a delimiter like `---`

```python
# Extract only the SQL query
prompt = "Convert to SQL:\nQuestion: Show all users\nSQL:"
response = client.complete(prompt, stop=["\n\n", "Question:"])
# Model stops before generating the next question
```

---

## 9. What is the foundational structure of a well-designed prompt?

A well-structured prompt has these components:

```
[System Context / Persona]
You are an expert data analyst...

[Task Description]
Your job is to analyze the sales data and identify trends.

[Input Data / Context]
Sales data: {data}

[Constraints / Format]
- Respond in bullet points
- Limit to 5 key insights
- Do not make predictions beyond the data

[Examples (optional)]
Input: Q1 revenue dropped 10%
Output: • Q1 revenue declined 10%, potentially due to...

[Final Instruction]
Analyze the following: {user_query}
```

The order matters: context before task, constraints before output, examples before the actual query.

---

## 10. What is in-context learning and why is it powerful?

In-context learning (ICL) is the ability of LLMs to adapt their behavior purely from examples provided in the prompt — **without any weight updates**. The model learns the pattern from demonstrations at inference time.

```python
prompt = """
Classify sentiment as POSITIVE or NEGATIVE.

Review: "The food was amazing!" → POSITIVE
Review: "Terrible service, never coming back." → NEGATIVE
Review: "Worst experience of my life." → NEGATIVE
Review: "Absolutely loved the atmosphere!" → POSITIVE

Review: "The wait time was ridiculous but food was decent." → """
```

Why it works: During pre-training, the model has seen millions of examples of "pattern completion," so it recognizes the structure and continues it.

---

## 11. What are the main types/techniques of prompt engineering?

1. **Zero-shot** — No examples, just a direct instruction
2. **Few-shot** — 3–10 input/output examples in the prompt
3. **Chain-of-Thought (CoT)** — Ask the model to reason step-by-step before answering
4. **Self-consistency** — Generate multiple CoT paths, take majority vote
5. **Tree-of-Thought (ToT)** — Explore multiple reasoning branches like a search tree
6. **ReAct** — Interleave reasoning and tool-use actions
7. **Role prompting** — Assign a persona ("You are a senior software engineer...")
8. **Generated Knowledge** — Ask model to generate relevant facts first, then answer

---

## 12. What are the key considerations when using few-shot prompting?

- **Example quality > quantity** — 3 great examples beat 10 mediocre ones
- **Diversity** — Examples should cover edge cases and different input types
- **Label balance** — For classification, include balanced positive/negative examples
- **Ordering effect** — Models can be biased by the last few examples (recency bias); randomize order
- **Format consistency** — Input/output format must exactly match what you want in production
- **Context length budget** — Each example consumes tokens; balance richness vs. window usage

---

## 13. What strategies lead to consistently better prompt outputs?

1. **Be explicit about format** — "Respond as a JSON object with keys: `title`, `summary`, `tags`"
2. **Use positive instructions** — Say what to DO, not just what to avoid
3. **Add constraints** — "In under 100 words", "Use only information from the provided context"
4. **Assign a role** — "You are a board-certified physician..."
5. **Separate sections with delimiters** — Use `###`, `---`, XML tags
6. **Iterate with examples** — Test edge cases systematically
7. **Ask for reasoning first** — "Think step by step before giving your final answer"

```python
prompt = """
### Role
You are a senior Python developer reviewing code for production readiness.

### Task
Review the following code and identify issues.

### Output Format
Return a JSON array of issues, each with:
- "severity": "critical" | "warning" | "info"
- "line": line number
- "issue": description
- "fix": suggested fix

### Code
```python
{code}
```
"""
```

---

## 14. What is hallucination in LLMs, and how can it be reduced through prompting?

**Hallucination** is when an LLM generates confident, fluent, but factually incorrect or fabricated information. It occurs because the model is trained to produce plausible continuations, not verified facts.

**Prompt-level mitigation strategies:**

1. **Ground the model in context** — "Answer only using the provided document. If the answer isn't in the document, say 'I don't know.'"
2. **Ask for citations** — "After each claim, cite the source sentence from the context."
3. **Add uncertainty acknowledgment** — "If you're unsure, explicitly say so."
4. **Chain-of-Thought** — Forces model to reason explicitly, exposing faulty logic
5. **Self-verification** — "Now review your answer and identify any claims you're not certain about."

```python
system_prompt = """
You are a factual assistant. You ONLY answer based on the context provided below.
If the context does not contain sufficient information to answer, respond with:
"I don't have enough information to answer this accurately."

Context:
{context}
"""
```

---

## 15. How can prompt engineering enhance LLM reasoning capabilities?

**Chain-of-Thought (CoT):** Add "Let's think step by step" or show reasoning in examples.

```
Q: A train travels 60mph for 2.5 hours. How far does it travel?
A: Let me work through this step by step.
   Speed = 60 mph, Time = 2.5 hours
   Distance = Speed × Time = 60 × 2.5 = 150 miles
   Answer: 150 miles
```

**Zero-shot CoT:** Simply append "Think step by step." to any question — shown to improve accuracy significantly on math/logic tasks.

**Self-consistency:** Sample N reasoning paths (e.g., temperature=0.7), then take the majority answer. This is especially powerful for math problems.

---

## 16. What do you do when Chain-of-Thought prompting still fails?

If standard CoT isn't working:

1. **Self-consistency sampling** — Generate 5–10 answers, take the majority
2. **Decompose the problem** — Break complex tasks into sub-questions, solve sequentially
3. **Tree of Thought** — Have the model explore multiple solution branches and evaluate each
4. **Program-aided reasoning** — Ask the model to write Python code and execute it

```python
# Program-aided reasoning
prompt = """
Solve this math problem by writing Python code.
Problem: If 5 workers can build a wall in 8 days, how many days will 10 workers take?

Write executable Python code, then show the result.
"""
# Model writes: workers=5; days=8; new_workers=10; print(workers*days/new_workers)
# Output: 4.0
```

5. **Fine-tune** — If prompting consistently fails on a task type, SFT with CoT examples is the most reliable fix.

---

*Next: [Retrieval Augmented Generation →](../02-rag/README.md)*
