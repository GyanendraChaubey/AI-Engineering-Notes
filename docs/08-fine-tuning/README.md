# Supervised Fine-Tuning of LLMs

> Fine-tuning adapts a general-purpose model to your specific task, style, or domain — without training from scratch.

---

## 1. What is fine-tuning and why is it necessary?

**Fine-tuning** is the process of continuing to train a pre-trained LLM on a smaller, task-specific dataset to adapt its behavior.

**Why it's needed:**
- Pre-trained models are generalists — they don't know your company's tone, format requirements, or domain terminology
- Some tasks (e.g., always respond in JSON, follow a specific clinical report format) can't be reliably achieved through prompting alone
- Fine-tuning can reduce prompt length needed → lower inference cost
- It bakes in behaviors that would otherwise require complex system prompts

---

## 2. When should you consider fine-tuning an LLM?

**Fine-tune when:**
- The base model consistently fails at a specific task despite good prompt engineering
- You need a specific output format reliably (JSON schema, structured reports)
- You want to inject a consistent tone/persona
- Latency matters and you want to reduce system prompt length
- You have 1K+ high-quality labeled examples

**Don't fine-tune when:**
- You just need to add knowledge (use RAG instead)
- You have fewer than ~500 examples
- The task works well with prompting
- Your data changes frequently

**Decision flowchart:**
```
Does prompting solve it? → YES → Use prompting
         ↓ NO
Is it a knowledge problem? → YES → Use RAG
         ↓ NO
Is it a behavior/style/format problem? → YES → Fine-tune
         ↓
Do you have labeled data? → NO → Generate synthetic data first
```

---

## 3. How do you make the final decision to fine-tune?

Run this checklist before committing to fine-tuning:

1. **Baseline prompt engineering:** Have you tried chain-of-thought, few-shot, and system prompt variations?
2. **Failure analysis:** Are failures systematic (always bad at format) or random (hallucinations)? Systematic failures → good candidate for fine-tuning.
3. **Data availability:** Do you have 500–10K quality examples? If not, can you generate synthetic data?
4. **ROI calculation:** Fine-tuning cost (compute + data prep) vs. ongoing inference savings (shorter prompts) + quality improvement
5. **Evaluation setup:** Do you have a clear, measurable success metric before starting?

---

## 4. How do you train a model to respond only when it has sufficient context?

This is a **calibration / abstention** problem. Train the model to output a special "I don't know" response when evidence is insufficient.

```python
# Training data format
training_examples = [
    # Sufficient context
    {
        "context": "Our return policy allows 30-day returns with receipt.",
        "question": "Can I return an item after 25 days?",
        "answer": "Yes, you can return items within 30 days with a valid receipt."
    },
    # Insufficient context
    {
        "context": "Our return policy allows 30-day returns with receipt.",
        "question": "Can I return items bought online to a physical store?",
        "answer": "I don't have enough information to answer this. The provided context doesn't specify online vs in-store return rules."
    }
]

# System prompt during fine-tuning
system = """You are a customer service assistant. 
Answer ONLY based on the provided context. 
If the context doesn't contain enough information, say exactly: 
"I don't have enough information to answer this from the provided context."
Never guess or make up information."""
```

Include ~20–30% "insufficient context" examples in your training set.

---

## 5. How do you create a fine-tuning dataset for question answering?

**Sources of training data:**
1. **Human-curated:** SMEs write question + answer pairs (highest quality, expensive)
2. **Existing data transformation:** Convert FAQs, support tickets, documentation into Q&A format
3. **Synthetic generation:** Use a strong LLM (GPT-4) to generate Q&A pairs from your documents

```python
# Synthetic Q&A generation
generate_qa_prompt = """
Given this document passage, generate 5 diverse question-answer pairs.
The questions should be realistic queries a user might ask.
The answers should be based solely on the passage.

Passage:
{passage}

Return as a JSON array:
[{"question": "...", "answer": "..."}, ...]
"""

# Format for fine-tuning (ChatML format)
def format_for_finetuning(question, answer, context=""):
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
            {"role": "assistant", "content": answer}
        ]
    }
```

**Quality control:**
- Filter out answers that are too short (<10 words) or too long (>500 words)
- Check answer faithfulness to source context
- Deduplicate similar questions

---

## 6. How do you set hyperparameters for fine-tuning?

| Hyperparameter | Typical Range | Notes |
|---|---|---|
| **Learning rate** | 1e-5 to 5e-5 | Lower than pre-training; use cosine schedule |
| **Batch size** | 8–128 | Larger = more stable; limited by GPU memory |
| **Epochs** | 1–5 | More epochs → overfitting on small datasets |
| **Warmup steps** | 3–10% of training | Prevents early unstable updates |
| **Max sequence length** | 512–4096 | Based on your data distribution |
| **LoRA rank (r)** | 8–64 | Higher = more capacity, more parameters |
| **LoRA alpha** | 2×r | Standard rule of thumb |

**Learning rate rule of thumb:** For instruction fine-tuning, 2e-5 is a safe default. Watch validation loss — if it increases while training loss decreases, you're overfitting.

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="./fine-tuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,   # Effective batch = 32
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    fp16=True,
    evaluation_strategy="steps",
    eval_steps=200,
    save_strategy="best",
    load_best_model_at_end=True,
)
```

---

## 7. How do you estimate infrastructure requirements for fine-tuning an LLM?

**Memory formula (full fine-tuning):**
```
Total GPU Memory = Model params × bytes_per_param × multiplier

bytes_per_param:
  FP32 training: 4 bytes → weights (4) + gradients (4) + Adam states (8) = 16 bytes/param
  BF16 + Adam:   weights (2) + master copy (4) + gradients (4) + Adam (8) = ~18 bytes/param

Multiplier: ~2–4× for activations

Example: 7B model, BF16
  = 7B × 18 = 126GB + activations → needs 4× A100 80GB minimum
```

**With LoRA (parameter-efficient):**
```
Only train LoRA params (typically 0.1–1% of model size)
7B model with LoRA r=16: ~17M trainable params
Memory ≈ 7B×2 (frozen weights in BF16) + 17M×16 = ~16GB → fits on 1× A100 40GB
```

**Training time estimate:**
```
Training tokens = num_examples × avg_sequence_length
Throughput (tokens/sec) ≈ GPU_TFLOPS / (6 × model_params)  [rough estimate]
Training time = Training tokens / Throughput / 3600  [hours]
```

---

## 8. How do you fine-tune a large model on consumer hardware?

**Key techniques for consumer GPU (RTX 3090/4090, 24GB VRAM):**

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

# Step 1: Load in 4-bit quantization (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True  # Nested quantization for extra savings
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    quantization_config=bnb_config,
    device_map="auto"
)

# Step 2: Apply LoRA adapters (only these weights are trained)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Trainable params: 6,815,744 (0.08% of 7B) — fits in 24GB VRAM
```

**Additional memory tricks:**
- Gradient checkpointing: `model.gradient_checkpointing_enable()`
- Gradient accumulation: simulate larger batch sizes
- Flash Attention 2: reduces attention memory usage

---

## 9. What are the different categories of Parameter-Efficient Fine-Tuning (PEFT)?

| Category | Method | How it works |
|---|---|---|
| **Additive** | Adapters | Insert small trainable FFN modules between layers |
| **Additive** | Prefix Tuning | Prepend trainable virtual tokens to input |
| **Additive** | Prompt Tuning | Learn soft prompt embeddings only |
| **Re-parameterized** | LoRA | Low-rank decomposition of weight updates |
| **Re-parameterized** | DoRA | Decompose into magnitude + direction components |
| **Selective** | BitFit | Train only bias terms |
| **Selective** | Sparse fine-tuning | Train a small % of selected parameters |

**LoRA** (Low-Rank Adaptation) is the dominant approach:
```
Instead of updating W (d×k), learn W + ΔW where ΔW = A×B
A: (d×r), B: (r×k), r << min(d,k)

Parameters: d×k → d×r + r×k = r×(d+k) << d×k
```

---

## 10. What is catastrophic forgetting in LLMs and how is it mitigated?

**Catastrophic forgetting** occurs when fine-tuning on task-specific data causes the model to "forget" its general capabilities (reasoning, language understanding, other task knowledge).

```
Before fine-tuning:  Model answers math, code, and general questions well
After fine-tuning on medical Q&A:  Model may fail basic math or code tasks
```

**Mitigation strategies:**

1. **PEFT / LoRA:** Only update a tiny fraction of weights — base model capabilities preserved in frozen weights
2. **Lower learning rate:** Less aggressive weight update = less forgetting
3. **Fewer epochs:** Stop before the model over-specializes
4. **Data mixing:** Include a small % of general instruction data with your domain data (e.g., 80% domain, 20% general)
5. **EWC (Elastic Weight Consolidation):** Penalize changes to parameters that were important for previous tasks
6. **Replay:** Include examples from previous tasks in the training mix

---

## 11. What are the different re-parameterization methods for fine-tuning?

### LoRA (Low-Rank Adaptation)
```python
# W_new = W_frozen + A @ B
# A: (d, r), B: (r, k) initialized A~N(0,σ), B=0
# At inference: merge W_new = W_frozen + (alpha/r) * A @ B
# Zero extra latency after merging!
```

### DoRA (Weight-Decomposed Low-Rank Adaptation)
Decomposes weight into magnitude (scalar) and direction (unit vector), then applies LoRA to the direction:
```
W = m × (V / ||V||) where m=magnitude, V=direction
Updates: learn Δm (scalar) and ΔV via LoRA
```
More expressive than standard LoRA; used in newer models.

### LoRA-FA (Frozen A)
Freeze matrix A, only train B → further reduces compute/memory while maintaining quality.

### GaLore (Gradient Low-Rank Projection)
Project gradients into a low-rank subspace during training — allows full-parameter fine-tuning with LoRA-level memory. Good for pre-training or continued pre-training.

```python
from galore_torch import GaLoreAdamW

optimizer = GaLoreAdamW(
    model.parameters(),
    lr=1e-4,
    rank=128,
    update_proj_gap=200,  # Re-project every 200 steps
    scale=0.25
)
```

---

*Next: [Preference Alignment →](../09-preference-alignment/README.md)*
