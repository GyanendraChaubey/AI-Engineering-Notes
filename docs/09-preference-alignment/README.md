# Preference Alignment (RLHF / DPO)

> After SFT, the model can follow instructions — but preference alignment makes it helpful, harmless, and honest.

---

## 1. When should you use preference alignment rather than SFT?

**SFT** teaches the model *what* to do via demonstration.  
**Preference alignment** teaches the model *what humans prefer* when comparing two outputs.

Use preference alignment when:
- SFT alone produces outputs that are technically correct but subtly off (too verbose, slightly rude, overly cautious)
- You want to steer toward a nuanced quality that's hard to demonstrate but easy to compare ("Which answer is better?")
- You want to reduce harmful/biased outputs
- You want the model to better balance helpfulness vs. safety

**Typical pipeline:** Pre-training → SFT → Preference Alignment (RLHF or DPO)

---

## 2. What is RLHF and how does it work?

**RLHF (Reinforcement Learning from Human Feedback)** has three stages:

### Stage 1: Supervised Fine-Tuning (SFT)
Train on high-quality demonstrations to get a solid instruction-following baseline.

### Stage 2: Train a Reward Model (RM)
Collect human preference data: show two model outputs for the same prompt, human picks the better one.

```python
# Preference dataset format
{
    "prompt": "Explain recursion to a beginner",
    "chosen": "Recursion is when a function calls itself...[clear, friendly explanation]",
    "rejected": "Recursion is a programming concept where...[dry, jargon-heavy]"
}

# Reward model: takes (prompt, response) → scalar reward score
# Trained with Bradley-Terry model:
# P(A > B) = sigmoid(reward(A) - reward(B))
# Loss: -log P(chosen > rejected)
```

### Stage 3: RL Fine-Tuning with PPO
Use the reward model to fine-tune the policy (SFT model) via PPO, with a KL divergence penalty to prevent the model from drifting too far from the SFT baseline:

```
Objective = E[reward(response)] - β × KL(policy || SFT_reference)
```

```
Prompt → Policy LLM → Response → Reward Model → Scalar reward
                                      ↓
                              PPO update policy weights
                              (maximize reward, stay close to SFT)
```

---

## 3. What is reward hacking and how does it manifest?

**Reward hacking** (also called reward gaming) occurs when the policy finds ways to get high reward scores without actually improving in the intended way.

**Examples:**
- Model learns that longer responses get higher scores → starts generating unnecessarily verbose answers
- Model discovers certain phrases (e.g., "Great question!") are rated higher → inserts them everywhere
- Model learns that agreeing with the human prompt gets better ratings → becomes sycophantic
- Model finds adversarial patterns that fool the reward model but produce low-quality text

**Mitigations:**
1. **KL penalty:** `total_reward = RM_score - β × KL(policy, reference)` — prevents the policy from drifting too far
2. **Diverse reward models:** Ensemble multiple reward models trained on different annotators
3. **Iterative training:** Regularly update the reward model with fresh comparisons
4. **Constitutional AI (Anthropic):** Use a set of principles + self-critique to reduce reliance on human ratings
5. **Careful reward model evaluation:** Probe reward model for known failure modes before training the policy

---

## 4. What are the main preference alignment methods and how do they compare?

### RLHF + PPO (original approach)
- Complex 4-model setup (SFT, RM, Policy, Reference)
- Requires online sampling during training
- Unstable — sensitive to PPO hyperparameters
- High compute cost

### DPO (Direct Preference Optimization)
The insight: the optimal RLHF policy can be expressed analytically — no RL loop needed.

```python
# DPO Loss
# β: temperature, π_θ: current model, π_ref: reference (SFT) model

import torch
import torch.nn.functional as F

def dpo_loss(chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta=0.1):
    chosen_ratio  = chosen_logps - ref_chosen_logps
    rejected_ratio = rejected_logps - ref_rejected_logps
    loss = -F.logsigmoid(beta * (chosen_ratio - rejected_ratio))
    return loss.mean()
```

**Advantages of DPO:**
- No reward model needed
- Stable training (standard cross-entropy)
- Much simpler implementation
- Works well in practice — used in Llama 3, Zephyr, Mistral Instruct

### IPO (Identity Preference Optimization)
DPO can overfit when chosen and rejected responses are very similar. IPO adds regularization to prevent this.

### KTO (Kahneman-Tversky Optimization)
Doesn't require paired (chosen, rejected) comparisons — uses single responses labeled as "good" or "bad." More data-efficient.

### ORPO (Odds Ratio Preference Optimization)
Combines SFT loss + preference loss in a single stage — no separate SFT phase needed.

| Method | Reward Model | Paired Data | Stability | Simplicity |
|---|---|---|---|---|
| RLHF+PPO | Yes | Yes | Low | Complex |
| DPO | No | Yes | High | Simple |
| KTO | No | No | High | Simple |
| ORPO | No | Yes | High | Simplest |

**Recommendation for most teams:** Start with DPO. It's the best balance of effectiveness and simplicity.

---

*Next: [LLM Evaluation →](../10-evaluation/README.md)*
