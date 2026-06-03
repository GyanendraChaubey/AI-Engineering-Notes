# Language Models — Internal Working

> Understanding transformers from the ground up — attention, positional encoding, architecture variants, and scaling challenges.

---

## 1. Can you give a detailed explanation of self-attention?

Self-attention allows each token in a sequence to attend to (gather information from) every other token. It computes three projections per token: **Query (Q)**, **Key (K)**, and **Value (V)**.

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) × V
```

**Step-by-step:**

```python
import torch
import torch.nn.functional as F

def self_attention(X, W_Q, W_K, W_V):
    # X: (batch, seq_len, d_model)
    Q = X @ W_Q   # (batch, seq, d_k)
    K = X @ W_K
    V = X @ W_V

    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)  # (batch, seq, seq)
    
    # Causal mask for decoder (optional)
    # mask upper triangle to -inf so future tokens are invisible
    
    weights = F.softmax(scores, dim=-1)  # Attention weights
    output = weights @ V                  # (batch, seq, d_v)
    return output, weights
```

**Intuition:** For each token, the query asks "what do I need?", keys answer "what do I have?", and values provide the actual content. The dot product Q·K measures relevance; softmax normalizes it into a probability distribution; V is then weighted-summed to produce the output.

---

## 2. What are the limitations of self-attention and how are they addressed?

| Limitation | Problem | Solution |
|---|---|---|
| **Quadratic complexity** | O(n²) memory/compute in sequence length | Sparse attention, Flash Attention, linear attention |
| **Fixed context window** | Can't attend beyond the window size | RoPE/ALiBi positional encoding, sliding window attention |
| **No inherent order** | Attention is permutation-invariant | Positional encoding |
| **Uniform attention** | All positions treated equally before softmax | Relative positional biases |
| **Memory during training** | Storing attention matrices is expensive | Flash Attention (recomputes from SRAM, no HBM storage) |

**Flash Attention** (the most impactful fix):
```
Standard: Store full (n×n) attention matrix in GPU HBM → O(n²) memory
Flash Attention: Compute in tiles within SRAM → O(n) memory, 3-8× speedup
```

---

## 3. What is positional encoding and why is it needed?

Transformers process all tokens in parallel — unlike RNNs, they have no built-in sense of order. Without positional encoding, "The cat chased the dog" and "The dog chased the cat" would produce identical representations.

### Sinusoidal Positional Encoding (original Transformer)
```python
import numpy as np

def sinusoidal_encoding(seq_len, d_model):
    PE = np.zeros((seq_len, d_model))
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            PE[pos, i]   = np.sin(pos / 10000 ** (i / d_model))
            PE[pos, i+1] = np.cos(pos / 10000 ** (i / d_model))
    return PE  # Added to token embeddings
```

### Rotary Position Embedding (RoPE) — used in Llama, GPT-NeoX
RoPE encodes position by rotating the Q and K vectors in the complex plane. Benefits:
- Relative positions are captured naturally
- Can generalize to sequences longer than seen during training (with extensions like YaRN)

### ALiBi — linear position bias added to attention scores
```
scores[i][j] -= slope × |i - j|
```
Simple, extrapolates well to longer contexts at inference.

---

## 4. Describe the Transformer architecture in detail.

```
Input Text
    │
[Token Embedding] + [Positional Encoding]
    │
    ┌──────────────────────────────────┐
    │     Transformer Block × N       │
    │                                  │
    │  [Multi-Head Self-Attention]     │
    │  [Add & Layer Norm]              │
    │  [Feed-Forward Network]          │
    │  [Add & Layer Norm]             │
    └──────────────────────────────────┘
    │
[Linear Projection → Vocabulary size]
[Softmax → Token probabilities]
```

**Key components:**

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm (modern LLMs use pre-norm, not post-norm)
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
```

**Multi-Head Attention:** Run H independent attention heads, each with reduced dimension d_k = d_model/H. Concatenate outputs → project to d_model. Each head can learn different relationships.

**FFN:** Two linear layers with a non-linearity. d_ff is typically 4× d_model. This is where most factual knowledge is believed to be stored.

---

## 5. Why are transformers better than LSTMs for most NLP tasks?

| Aspect | LSTM | Transformer |
|---|---|---|
| **Parallelization** | Sequential — can't parallelize training | Fully parallel — all tokens processed simultaneously |
| **Long-range dependencies** | Vanishing gradients degrade over distance | Direct attention between any two positions |
| **Training speed** | Slow (sequential) | Fast (GPUs fully utilized) |
| **Scalability** | Hard to scale | Scales to billions of parameters |
| **Context** | Limited by hidden state size | Explicit via context window |

The key advantage: transformers can train 100× faster on modern GPU hardware because every position can be processed in parallel, while LSTMs must process token-by-token.

---

## 6. What is the difference between local attention and global attention?

**Local attention:** Each token attends to only a fixed window of neighboring tokens (e.g., ±256 tokens). O(n × window_size) complexity — much cheaper for long sequences.

**Global attention:** Every token attends to every other token. O(n²) complexity. Standard transformer behavior.

**Longformer** combines both:
- Most tokens: local sliding window attention
- Special `[CLS]` tokens or question tokens: global attention

```
Local:  Token 500 attends to tokens 400–600 only
Global: Token [CLS] attends to ALL tokens
```

This allows efficient processing of documents with 4K–16K tokens on standard hardware.

---

## 7. What makes transformers computationally and memory intensive?

**The quadratic bottleneck:**

```
Attention matrix: (batch × heads × seq_len × seq_len) floats
At seq_len=4096: 4096² × 4 bytes = 64MB per head per batch item
GPT-3 (96 heads): ~6GB just for attention matrices at seq_len=4096
```

**Memory breakdown for a large LLM:**
1. Model weights: 2 bytes/param for float16
2. Activations: ~4× model size during training
3. Optimizer states: 8 bytes/param for Adam (3× model size)
4. KV Cache at inference: grows linearly with sequence length and batch size

**Solutions:**
- **Flash Attention** — compute attention in tiles without materializing the full matrix
- **Gradient checkpointing** — recompute activations on backward pass instead of storing them
- **Mixed precision (BF16/FP8)** — halve memory vs FP32
- **KV Cache compression** — Multi-Query Attention (MQA), Grouped Query Attention (GQA)

---

## 8. How do you increase the context length of a pre-trained LLM?

The challenge: models trained on 4K context have learned positional encodings only up to position 4K. Longer sequences → out-of-distribution positions → degraded performance.

**Techniques:**

1. **Position Interpolation (PI):** Compress positional indices to fit within the trained range
   ```
   Original: pos = [0, 1, 2, ..., 32768]
   PI to 2×:  pos = [0, 0.5, 1, ..., 16384]  → scale by 1/2
   ```

2. **YaRN (Yet another RoPE extension):** Combines NTK-aware scaling with attention temperature adjustment — state-of-the-art for extending RoPE models

3. **LongLoRA:** Fine-tune with shift short attention + LoRA — enables 100K+ context with minimal compute

4. **Sliding window attention (Mistral):** Never attend beyond a fixed window, but cache KV from old tokens

5. **Full fine-tuning:** Fine-tune on long documents — expensive but most reliable

---

## 9. You have a vocabulary of 100K words. How do you optimize the transformer for this?

Large vocabularies create two bottlenecks:
1. **Embedding layer:** 100K × d_model = 100K × 4096 = 409M parameters just for embeddings
2. **Output projection:** Same size + softmax over 100K classes is expensive

**Optimizations:**

1. **Adaptive Embedding / Adaptive Softmax:**
   - Frequent tokens: full dimension (d_model)
   - Rare tokens: reduced dimension (d_model/4)
   - 50% of vocabulary → rare; sparse access patterns

2. **Tied embeddings:** Share input embedding weights with output projection matrix (reduces parameters by ~50%)

3. **Hierarchical softmax:** Two-level prediction (cluster → word)

4. **Sampled softmax during training:** Only compute loss over a random subset of negative vocabulary items

```python
# Tied embeddings in PyTorch
self.embedding = nn.Embedding(vocab_size, d_model)
self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
self.output_proj.weight = self.embedding.weight  # Tie weights
```

---

## 10. What is the best tokenization strategy to balance vocabulary size vs. coverage?

**Too large vocabulary:**
- Expensive embedding/output layers
- Rare tokens are undertrained

**Too small vocabulary:**
- Out-of-vocabulary (OOV) issues
- Long tokenization → more tokens per sentence → shorter effective context

**Solution: Byte-Pair Encoding (BPE) or SentencePiece** at 32K–100K tokens:

```
BPE algorithm:
1. Start with character-level vocabulary
2. Repeatedly merge the most frequent adjacent pair
3. Stop at desired vocabulary size

"unbelievable" → ["un", "believ", "able"] (3 tokens, not 13 chars)
"API" → ["API"] (1 token, seen frequently)
"gyanendra" → ["gy", "an", "end", "ra"] (4 tokens, rare name)
```

BPE naturally handles:
- Common words → single token
- Rare words → broken into subword pieces (never OOV)
- Code → operators and keywords as single tokens

**Practical choice:** 32K–64K BPE vocabulary is the sweet spot for English + multilingual + code.

---

## 11. What are the different LLM architecture types and which is best for which task?

### Encoder-Only (e.g., BERT, RoBERTa)
- Bidirectional attention — sees full context
- No generation capability
- Best for: Classification, NER, sentence similarity, embedding

### Decoder-Only (e.g., GPT, Llama, Mistral)
- Causal (left-to-right) attention
- Generative — produces text autoregressively
- Best for: Chat, code generation, summarization, completion
- **Dominant architecture today for LLMs**

### Encoder-Decoder (e.g., T5, BART, mT5)
- Encoder processes input; decoder generates output
- Cross-attention between encoder and decoder
- Best for: Translation, structured summarization, question answering with structured output

### Mixture of Experts (MoE) (e.g., Mixtral, GPT-4)
- Only K of N expert FFN layers activated per token
- Scales model capacity without proportional compute increase
- Best for: Large-scale models where parameter count matters more than FLOPs

| Task | Best Architecture |
|---|---|
| Text classification | Encoder-only (BERT) |
| Semantic search / embeddings | Encoder-only |
| Open-ended generation | Decoder-only |
| Translation | Encoder-decoder |
| Large-scale general assistant | Decoder-only or MoE |

---

*Next: [Supervised Fine-Tuning →](../08-fine-tuning/README.md)*
