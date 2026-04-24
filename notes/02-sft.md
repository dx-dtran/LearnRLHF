# 02 — Supervised fine-tuning (SFT)

## Purpose

Theory for Module 2 (Problems 2.1 through 2.5). By the end of this you should be able
to:

1. Write the SFT loss in equations (and explain every symbol).
2. Derive the gradient of softmax + cross-entropy with respect to the logits by hand.
3. Explain *exactly* what the `loss_mask` is masking out and why.

SFT is the easiest of the three phases, but the masking logic is the source of the
bug in the old code in `old/`. Get this right or everything downstream is wrong.

---

## 1. What SFT actually does

We have a pretrained GPT-2 that models $P(x_t \mid x_{<t})$ on raw web text. We want
a model that generates **assistant turns** that imitate the `chosen` responses from
HH-RLHF. SFT = continue training with the same next-token objective, but on formatted
dialogue data, and **only count loss on assistant tokens**.

Think of it as next-token-prediction with a mask: the model sees the full dialogue,
but the loss only "asks" it to predict the assistant's tokens. The prompt tokens are
there as context, not as targets.

---

## 2. The loss

### 2.1 Tokenized example

One example: a full formatted dialogue tokenized to a sequence $x = (x_0, x_1, \dots, x_{T-1})$.
Alongside, a binary mask $m = (m_0, m_1, \dots, m_{T-1}) \in \{0, 1\}^T$ where
$m_t = 1$ iff **the target at position $t$ is an assistant token**.

"Target at position $t$" is $x_{t+1}$ under the next-token convention — position $t$'s
logits predict position $t+1$'s token. So really the mask is on the *predictee*. The
cleanest way to write it is to shift the sequence and mask in the shifted frame:

```
input_ids   = x[:-1]     # positions 0..T-2
labels      = x[1:]      # positions 1..T-1  (what we predict)
loss_mask   = m[1:]      # 1 iff labels[t] is an assistant token
```

From here on, "position $t$" refers to the shifted frame — so $\text{logits}_t$ predicts
$y_t = \text{labels}_t$.

### 2.2 Per-token cross-entropy

At position $t$, the model outputs a vector of logits $z_t \in \mathbb{R}^V$. Convert to
probabilities via softmax:

$$
p_{t, v} = \frac{\exp(z_{t, v})}{\sum_{v'} \exp(z_{t, v'})}.
$$

Cross-entropy with the true token $y_t$:

$$
\ell_t = -\log p_{t, y_t} = -z_{t, y_t} + \log \sum_{v'} \exp(z_{t, v'}).
$$

The second form (logits minus log-sum-exp) is what you implement for numerical
stability — never compute the softmax explicitly and then log it.

### 2.3 Masked mean over the batch

Summed over a batch of examples indexed by $(b, t)$:

$$
\mathcal{L}_\text{SFT} = \frac{1}{N_\text{resp}} \sum_{b, t} m_{b, t} \cdot \ell_{b, t},
\qquad N_\text{resp} = \sum_{b, t} m_{b, t}.
$$

$N_\text{resp}$ is the total number of **assistant tokens** in the batch. Divide by that,
not by $B \cdot T$ — otherwise the loss magnitude depends on how much prompt padding
you happened to include, which is noise.

---

## 3. Why mask the prompt?

Two reasons.

### 3.1 It's the wrong distribution

User prompts are human input; we do not want the model to learn $P(\text{user} \mid \text{history})$.
If you train on prompt tokens, the model wastes capacity memorizing prompt phrasings
that, at inference, *we* provide — not the model. At best it's wasted capacity; at
worst the model starts hallucinating user turns into its own output (the classic
"Human: ... Assistant:" leakage failure).

### 3.2 The loss scale is dominated by prompts

In HH-RLHF, prompts are often longer than responses. If you don't mask, roughly 60–80%
of the loss comes from prompt tokens and the training signal for the assistant tokens
is drowned out. You'd see the loss decrease fast but the instruction-following behavior
would barely improve.

### 3.3 What the mask includes

Include assistant **content tokens** and the assistant turn's `<|im_end|>`. Exclude:

- The `<|im_start|>assistant\n` header tokens (those are scaffolding the model doesn't
  need to generate — they come from the template).
- All user content and user's `<|im_start|>user\n` and `<|im_end|>`.
- Any padding tokens.

The `<|im_end|>` at the end of an assistant turn **is** masked in — we want the model
to learn *when to stop*.

---

## 4. Gradient of softmax cross-entropy (derive this)

You must be able to produce this from a blank page. It's the one gradient every ML
engineer memorizes, and it's foundational for the PPO surrogate later.

### 4.1 Setup

Single position for now. Let $z \in \mathbb{R}^V$, $p = \text{softmax}(z)$, $y$ the
target class. Loss $\ell = -\log p_y$.

### 4.2 Softmax derivative

For any $v, u$:

$$
\frac{\partial p_v}{\partial z_u}
= p_v \left(\delta_{v, u} - p_u\right),
$$

where $\delta_{v, u} = 1$ if $v = u$ else $0$. Derive this by differentiating
$p_v = e^{z_v} / \sum_{v'} e^{z_{v'}}$ and grouping.

### 4.3 Chain rule for $\ell$

$$
\frac{\partial \ell}{\partial z_u}
= -\frac{1}{p_y} \cdot \frac{\partial p_y}{\partial z_u}
= -\frac{1}{p_y} \cdot p_y(\delta_{y, u} - p_u)
= p_u - \delta_{y, u}.
$$

So in vector form:

$$
\boxed{\; \frac{\partial \ell}{\partial z} = p - \mathbf{1}_{y} \;}
$$

where $\mathbf{1}_y$ is the one-hot vector at class $y$.

Interpretation: the gradient is "predicted distribution minus the delta on the true
class". It *pushes mass onto the true class and away from everything else* in
proportion to what the model currently predicts.

### 4.4 With the mask

Across positions $t$ and with the mask:

$$
\frac{\partial \mathcal{L}_\text{SFT}}{\partial z_t}
= \frac{m_t}{N_\text{resp}} \left( p_t - \mathbf{1}_{y_t} \right).
$$

Three things to notice:

- The gradient at position $t$ is **exactly zero** when $m_t = 0$. Flipping the value
  of a masked label anywhere in the sequence changes nothing — no loss contribution,
  no gradient. This is what the "flip a masked token" unit test checks.
- The denominator is $N_\text{resp}$, not $N_\text{resp} \cdot V$. The factor of $V$
  doesn't appear because $p$ is a distribution (sums to 1), not $V$ independent
  variables.
- This gradient is what gets backpropagated through the `lm_head` into the transformer.

---

## 5. Implementation checklist

### 5.1 `sft_loss(logits, labels, loss_mask)` (Problem 2.2)

```
logits:    (B, T, V)
labels:    (B, T)        int64, already shifted
loss_mask: (B, T)        float or bool
```

Reference implementation (concepts, not code):

1. `logp = log_softmax(logits, dim=-1)` — fused, stable.
2. `nll = -logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)`  # shape (B, T)
3. `loss = (nll * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)`

Do **not** use `F.cross_entropy(..., ignore_index=-100)` and stuff `-100` into masked
labels. The whole point of this course is that you handle the mask explicitly. The
`ignore_index` trick hides the mask from you and is where subtle bugs live.

### 5.2 Gradient check (Problem 2.2)

Two tests:

1. **Analytic vs. centered-difference** at fp64 on a tiny tensor (e.g. $B=2, T=4, V=8$).
   Compute the gradient w.r.t. logits both ways; relative error should be $< 10^{-5}$.
2. **Mask flip invariance**: construct inputs, compute loss. Now change `labels[b, t]`
   at any masked-out position to a different token id and recompute. The two losses
   must be exactly equal. The gradient must also be identical element-wise.

### 5.3 DataLoader (Problem 2.3)

- Pad to the longest in the batch, not the block_size. Returning $(B, 1024)$ tensors
  when the longest example is 300 tokens wastes 70% of compute.
- Return `input_ids`, `labels`, `loss_mask`, `attention_mask` all at the same shape
  $(B, T_\text{batch})$.
- `attention_mask` is the position mask used by the transformer (1 on real tokens, 0
  on padding). `loss_mask` is the assistant-token mask. They are different tensors and
  neither implies the other.

### 5.4 Training loop (Problem 2.4)

The boring-but-important parts:

- AdamW with $\beta = (0.9, 0.95)$, $\text{wd} = 0.1$, **no weight decay on 1D
  params** (LayerNorm gains/biases and all biases — partition them into a decay
  group and a no-decay group). This is the karpathy-style param grouping.
- Cosine LR to 10% of peak over the run, linear warmup of 200 steps from 0 to peak.
- Gradient clipping at norm 1.0.
- bf16 autocast for the forward; fp32 master copy and optimizer state.
- Gradient accumulation: effective batch = `batch_size * accum_steps` — scale loss
  by `1/accum_steps` before `.backward()` and only step the optimizer every
  `accum_steps` micro-steps.

Run for 2 epochs on HH's `chosen` stream. Log train loss every step, eval loss every
250 optimizer steps on a held-out split. Save `sft.pt`.

### 5.5 What good looks like

- Training loss drops from ~3.5 (base GPT-2 on chat-formatted HH) to ~1.5–2.0 by end
  of epoch 2. Exact number depends on your tokenization and mask.
- Qualitative eval (Problem 2.5): base GPT-2 rambles and leaks "Human:"; SFT model
  produces a coherent single assistant turn that ends with `<|im_end|>`.

If train loss plateaus above 3.0 after the first few hundred steps, your mask is
probably wrong (you're training on prompt tokens and the loss is dominated by noise).

---

## 6. What to commit to `notes/02-sft.md`

After finishing Module 2, append:

- Your own re-derivation of the softmax-CE gradient (paper photo or typed).
- The training/eval loss curves (or a description if no plots).
- Observations from the qualitative side-by-side in 2.5 — what does base say on your
  held-out prompts, what does SFT say? Where does SFT still fail? These observations
  motivate why we need RLHF.
