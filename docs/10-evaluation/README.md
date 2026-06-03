# Evaluation of LLM Systems

> You cannot improve what you cannot measure. Moving an LLM application from a "cool demo" to a production system requires a rigorous, automated evaluation harness. Answers are calibrated for a **Google L5 Senior AI/ML Engineer** interview bar.

---

## Q1. What is the fundamental difference between evaluating traditional ML models vs LLMs?

### Core Answer

In traditional Machine Learning, evaluation is deterministic and mathematically absolute. You calculate the F1-Score, RMSE, or Accuracy against a static ground-truth label. 

Large Language Models generate open-ended, non-deterministic text. If you ask an LLM to "Summarize the French Revolution," there are millions of valid linguistic permutations. There is no single "correct" string to compare against, rendering traditional exact-match math entirely obsolete. You must evaluate **Semantic Quality** rather than **Lexical Overlap**.

### Related Questions

!!! question "Follow-up Interview Questions"
    1. Why did traditional NLP metrics like BLEU and ROUGE fail for LLMs?
    2. What is BERTScore and how did it bridge the semantic gap?
    3. How do you measure production metrics beyond just text quality?
    4. What is the "Vibe Check" problem in LLM engineering?

??? success "View Answers"
    **1. BLEU and ROUGE failures?**
    BLEU (for translation) and ROUGE (for summarization) measure n-gram overlap (how many exact words/phrases from the generated text appear in the reference text). If the reference is *"The feline rested on the rug"* and the LLM outputs *"The cat sat on the mat"*, BLEU gives a score of 0.0 because there are zero matching words. It severely penalizes models for using valid synonyms or paraphrasing.

    **2. BERTScore?**
    BERTScore was the first step away from exact-match metrics. Instead of comparing strings, it uses a pre-trained BERT model to embed both the generated text and the reference text into dense vectors, and then calculates their Cosine Similarity. In the cat/feline example above, BERTScore would output a 0.95 similarity, successfully recognizing the semantic equivalence. 

    **3. Production Metrics?**
    A production harness must track 3 non-linguistic metrics: **Latency** (Time-To-First-Token and Tokens-Per-Second at p50/p90/p99), **Cost** (Calculated exactly by tracking input/output tokens multiplied by API pricing), and **Error Rate** (Timeouts, JSON parsing failures, Context Length exceeded). A model might have a perfect quality score but be rejected because its p99 latency is 12 seconds.

    **4. The Vibe Check Problem?**
    In early AI development, engineers tweak a prompt, run 5 examples manually, read the outputs, and declare "the new prompt feels better." This "vibe check" is catastrophic in production. A prompt change might improve formatting for those 5 edge cases, but silently regress the accuracy of 5,000 other cases. Only an automated, comprehensive evaluation harness prevents regression.

---

## Q2. How do you construct a deterministic Evaluation Harness for an open-ended GenAI system?

### Core Answer

The modern industry standard for evaluating open-ended text is **LLM-as-a-Judge**. You use a vastly superior, highly capable model (like GPT-4) to grade the outputs of your cheaper production model (like Llama-3-8B).

You must build a pipeline that iterates over a "Golden Dataset" of test cases, executes the prompt against your system, passes the output to the Judge, and aggregates the scores.

```mermaid
flowchart TD
    A["Golden Dataset (1,000 Edge Cases)"] --> B["Production LLM Pipeline"]
    
    B --> C["Output Generated"]
    
    C --> D["LLM-as-a-Judge (GPT-4)"]
    A -.->|"Pass Ground Truth"| D
    
    D --> E["Score Output 1-5"]
    D --> F["Generate Rationale"]
    
    E --> G["Aggregation / Analytics Dashboard"]
    F --> G
```

### Related Questions

!!! question "Follow-up Interview Questions"
    1. How do you write a robust prompt for an LLM Judge?
    2. What are the common biases of LLM Judges?
    3. How do you measure the reliability of an LLM Judge?
    4. What is Pairwise Evaluation vs Single-Point Evaluation?

??? success "View Answers"
    **1. LLM Judge Prompting?**
    A Judge prompt must be incredibly explicit. You cannot just ask, "Is this good?" You must define a rubric: *"Score from 1 to 5 based strictly on Faithfulness. Score 1: Hallucinates facts not in the context. Score 3: Partially grounded but includes minor unverified details. Score 5: 100% grounded. You must output a `<rationale>` block explaining your reasoning BEFORE outputting the `<score>`."* Forcing the LLM to write the rationale first uses Chain-of-Thought to mathematically improve the accuracy of the final score.

    **2. LLM Judge Biases?**
    Judges suffer from:
    - **Position Bias:** In Pairwise evaluation, they arbitrarily prefer the first answer (Option A) over the second (Option B).
    - **Verbosity Bias:** They inherently equate "longer" with "better," penalizing concise, accurate answers.
    - **Self-Enhancement Bias:** A GPT-4 Judge will inherently rate outputs generated by GPT-4 higher than outputs generated by Claude, simply because it recognizes its own linguistic patterns.

    **3. Cohen's Kappa (Inter-Rater Reliability)?**
    To trust an LLM Judge, you must test it against humans. You have Human Experts rate 100 outputs, and the LLM Judge rate the same 100 outputs. You calculate **Cohen's Kappa**, a statistical measure of agreement that accounts for random chance. A Kappa $> 0.8$ proves your LLM Judge is statistically identical to your Human Experts, allowing you to fully automate the pipeline.

    **4. Pairwise vs Single-Point?**
    Single-Point asks the Judge to rate Response A on a scale of 1-5. This is notoriously uncalibrated. Pairwise Evaluation asks the Judge: *"Here is the output from the Old Prompt, and the output from the New Prompt. Which is better, and why?"* Pairwise is vastly more accurate and acts as an immediate A/B test for system regressions.

---

## Q3. How do you evaluate a Retrieval-Augmented Generation (RAG) system end-to-end?

### Core Answer

You cannot evaluate a RAG system as a single "Black Box". If the final answer is wrong, you need to know *why* it's wrong. Did the vector database fail to find the document, or did the LLM fail to read it? 

You must evaluate RAG using the **RAGAS (RAG Assessment) Triad**, which breaks the system into distinct, measurable vectors:

```mermaid
flowchart TD
    A["User Query"] -->|"Context Precision / Context Recall"| B["Retrieved Context"]
    B -->|"Faithfulness (Groundedness)"| C["LLM Answer"]
    C -->|"Answer Relevancy"| A
```

### Related Questions

!!! question "Follow-up Interview Questions"
    1. What is Context Precision vs Context Recall?
    2. What is Faithfulness (Groundedness) and how does it catch hallucinations?
    3. What is Answer Relevancy and how does it detect evasive answers?
    4. How do you build a Golden Dataset for RAG evaluation?

??? success "View Answers"
    **1. Context Precision vs Recall?**
    **Context Recall** measures: *Did we find the needle in the haystack?* Out of all the facts needed to answer the query, how many were present in the retrieved chunks?
    **Context Precision** measures: *How much hay did we bring with the needle?* If we retrieved 10 chunks, and only 1 was relevant, precision is 10%. High recall with terrible precision means you are polluting the LLM's context window and wasting massive amounts of money on token costs.

    **2. Faithfulness (Groundedness)?**
    Faithfulness evaluates the Generation step. The LLM Judge looks *only* at the Retrieved Context and the Final Answer. It asks: *"Can every single claim in the Final Answer be traced back to a sentence in the Context?"* If the Answer contains a date that isn't in the Context, the Faithfulness score drops. This explicitly detects LLM Hallucinations.

    **3. Answer Relevancy?**
    An LLM might generate a 100% Faithful summary of the retrieved document, but the document had nothing to do with the user's question. Answer Relevancy compares the Final Answer directly back to the User Query. It detects situations where the RAG system says, *"I don't have the answer to that, but here are some fun facts about penguins."* (Faithful, but irrelevant).

    **4. RAG Golden Datasets?**
    Writing 1,000 test queries manually takes weeks. Instead, you use Synthetic Generation. You pass your company's proprietary documents to GPT-4 and run a script: *"Read this document. Generate 5 realistic, difficult questions that can be answered by this document, and provide the exact answer."* This instantly gives you a massive dataset of (Query, Ground_Truth_Context, Ground_Truth_Answer) pairs.

---

## Q4. What is Chain of Verification (CoVe) and Self-Consistency?

### Core Answer

While standard evaluation happens *offline* during development, **Inference-Time Evaluation** techniques force the model to evaluate and correct its own work *live in production* before showing the answer to the user.

**Self-Consistency (Majority Voting):**
Instead of generating one answer, the system generates 10 answers in parallel using a non-zero temperature. It then extracts the final conclusion from all 10 answers and returns the most frequent one (Majority Vote).

**Chain of Verification (CoVe):**
1. **Draft:** The LLM generates a baseline answer.
2. **Plan:** The LLM reads its own answer and lists verifiable claims as questions.
3. **Execute:** The LLM answers the verification questions *independently* (without looking at the draft).
4. **Refine:** The LLM compares the verification answers to the draft and revises any contradictions.

### Related Questions

!!! question "Follow-up Interview Questions"
    1. How does Self-Consistency mathematically improve LLM accuracy on logic tasks?
    2. Why does CoVe require independent verification questions?
    3. What is the latency impact of using CoVe in production?
    4. How does Self-Reflection differ from CoVe?

??? success "View Answers"
    **1. Math of Self-Consistency?**
    LLMs are probabilistic. On complex math or logic puzzles, an LLM might take a wrong turn early in the reasoning chain 30% of the time. However, there are usually many different correct reasoning paths that lead to the single right answer. By sampling 10 times, the "wrong" paths will yield 3 random, scattered answers, while the "correct" paths will converge heavily on the exact same final answer. The majority vote mathematically marginalizes out the random hallucinated errors.

    **2. CoVe Independent Verification?**
    If the LLM generates a hallucination in the Draft step, and you ask it to verify that hallucination *while it is looking at the Draft*, its self-attention mechanism will lock onto the hallucinated tokens and confidently confirm the lie. CoVe forces the model to answer the verification questions in a completely isolated context window, forcing it to retrieve the facts fresh from its pre-trained weights without bias.

    **3. CoVe Latency Impact?**
    CoVe is devastating to production latency. It requires 4 sequential LLM calls. If a standard query takes 2 seconds, CoVe will take 8 to 12 seconds. It also quadruples token costs. It should only be used on async background tasks, high-stakes financial/medical queries, or agentic systems where correctness is paramount and latency is secondary.

    **4. Self-Reflection vs CoVe?**
    Self-Reflection is a simpler, naive approach where you feed the LLM's output back to it and prompt: *"Is this correct? Are you sure? Fix any mistakes."* It often fails because the LLM is sycophantic; it will apologize profusely and change the answer even if the original answer was completely correct, degrading performance. CoVe provides a rigid, structural framework to prevent this.

---

*Next: [Hallucination Control →](../11-hallucination/README.md)*
