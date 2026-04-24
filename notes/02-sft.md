# 02 — Supervised fine-tuning (SFT)

## Purpose

This is the theory packet for Module 2 (Problems 2.1 through 2.5). By the end you
should be able to:

1. Write down the SFT loss and explain what every piece of it means in words.
2. Derive the gradient of softmax + cross-entropy with respect to the logits on a
   blank page.
3. Explain exactly which tokens the `loss_mask` zeros out and why.

SFT is the easiest of the three training phases. But the masking logic is *critical*.
A single off-by-one or inverted mask here will quietly corrupt every downstream phase
(RM and PPO both assume SFT was done correctly).

---

## 1. What SFT actually does

We start with a pretrained GPT-2. GPT-2 models the probability of the next token
given the previous tokens, trained on web text. What we want is a GPT-2 that, when
given a user's message in our ChatML format, produces an assistant reply that looks
like the `chosen` responses from HH-RLHF.

SFT = keep the same next-token prediction objective, but train on formatted dialogues
instead of raw web text, and **only count the loss on assistant tokens**.

The mental model: the model sees the entire dialogue, but during training we only
"grade" it on its predictions of assistant tokens. It can look at user messages for
context, but we never ask it "predict the next user token" — user text is input,
assistant text is the target.

---

## 2. The loss

### 2.1 Tokenized example

One example is one full formatted dialogue, tokenized to a sequence
`x = [x_0, x_1, ..., x_{T-1}]`.

Alongside `x`, we have a binary mask `m` of the same length. `m[t] = 1` means
"position `t` is part of an assistant turn's content", and `m[t] = 0` means
"position `t` is user text, scaffolding, or padding".

Because GPT-2 predicts the *next* token at each position, it's cleaner to work with
the shifted version:

```
input_ids   = x[:-1]      # positions 0 .. T-2, what we feed in
labels      = x[1:]       # positions 1 .. T-1, what we try to predict
loss_mask   = m[1:]       # 1 iff the token we're trying to predict is assistant
```

From now on when we say "position `t`" we mean the shifted frame — so the model's
logits at position `t` predict `labels[t]`, and `loss_mask[t]` tells us whether that
prediction counts toward the loss.

### 2.2 Per-token cross-entropy (LaTeX + code view)

At each position `t`, the model outputs a logit vector of length `V` (vocab size).
Softmax converts logits to a probability distribution over the vocabulary:

\[
p_{t,v}=\frac{e^{z_{t,v}}}{\sum_{j=1}^{V} e^{z_{t,j}}}
\]

where `z` is just the logits tensor (`z == logits` in code).

The cross-entropy loss at position `t` is the negative log-probability the model
assigns to the true next token `y_t = labels[t]`:

\[
\ell_t=-\log p_{t,y_t}
=-z_{t,y_t}+\log\!\left(\sum_{j=1}^{V} e^{z_{t,j}}\right)
\]

Quick intuition: logits are like raw "scores" for every token, softmax converts
scores into odds, and cross-entropy measures how surprised we are by the correct
token. Low surprise means good prediction.

The second form is what you actually compute — subtract the correct token's logit
from the log-sum-exp of all logits. Never compute the softmax first and then take
its log; that's numerically unstable.

### 2.3 Masked mean over the batch

Summing across a batch of examples:

\[
L_{\text{SFT}}=
\frac{\sum_{b,t} m_{b,t}\,\ell_{b,t}}
{\sum_{b,t} m_{b,t}}
=
\frac{\sum_{b,t} m_{b,t}\,\ell_{b,t}}
{N_{\text{resp}}}
\]

where \(N_{\text{resp}}=\sum_{b,t} m_{b,t}\) is the total number of assistant tokens
in the batch.

Dividing by `N_resp` (instead of by `B * T` or anything else) means the loss scale
doesn't depend on how much padding happened to be in the batch. Padding tokens
contribute zero to the numerator *and* zero to the denominator, so they don't pollute
either side of the average.

---

## 3. Why mask the prompt?

Two big reasons.

### 3.1 User text is not something we want the model to learn to generate

At inference time, **we** write the user's message — the model never has to
generate user text. If we train on user tokens, the model wastes capacity memorizing
human prompt phrasings, and worse, sometimes starts hallucinating "Human: ..." turns
into its own output. This is the classic SFT failure mode where the model keeps
inventing the other side of the conversation. Masking out user tokens prevents it.

### 3.2 The loss scale is dominated by the prompt

In HH-RLHF, prompts are often *longer* than responses. Without masking, maybe 60–80%
of every example's loss is coming from the user's prompt tokens. The assistant's
response — the part we actually care about — is a small fraction of the signal. You'd
see training loss drop fast, but qualitatively the model barely gets better at
responding in the ChatML format.

### 3.3 Exactly which tokens the mask includes

Include:

- The **content tokens of each assistant turn**.
- The assistant turn's closing `<|im_end|>` — we want the model to learn when to
  stop.

Exclude:

- The `<|im_start|>assistant\n` header. That's scaffolding that comes from our
  template, not something the model needs to emit.
- All user tokens: the content, and the user's `<|im_start|>user\n` and `<|im_end|>`.
- Any padding.

---

## 4. Gradient of softmax cross-entropy

You must be able to derive this from a blank page. It's the one gradient every ML
engineer has memorized, and the PPO surrogate we build in Module 4 uses the same
mechanics.

### 4.1 Setup

Fix one position for now. Let `z` be the length-`V` vector of logits and let `p` be
the softmax of `z`. Let `y` be the index of the correct class. The loss at this
position is:

    ell = -log p[y]

We want `d ell / d z` — the gradient of the loss with respect to the logits.

### 4.2 Softmax derivative

For any two indices `v` and `u`:

    d p[v] / d z[u] = p[v] * (delta(v, u) - p[u])

where `delta(v, u)` is 1 if `v == u` and 0 otherwise. You get this by differentiating
`p[v] = exp(z[v]) / sum(exp(z))` and applying the quotient rule — worth doing once
by hand.

### 4.3 Chain rule (step-by-step, lighter notation)

\[
\frac{\partial \ell}{\partial z_u}
=
-\frac{1}{p_y}\frac{\partial p_y}{\partial z_u}
=
-\frac{1}{p_y}\,p_y\left(\mathbf{1}[u=y]-p_u\right)
=
p_u-\mathbf{1}[u=y]
\]

So in vector form:

\[
\nabla_z \ell = p-\mathrm{onehot}(y)
\]

Read this out loud: **"the gradient is the model's predicted distribution minus a
delta spike on the correct class."**

Interpretation: the gradient subtracts probability mass from the true class and adds
it to every other class, proportional to what the model currently predicts. A
gradient step (minus this direction) pushes mass *toward* the correct class and *away
from* every wrong class.

Analogy: this is like rebalancing a budget where the "correct class" account should
hold all the money. Classes with too much probability get debited, and the true
class gets credited.

### 4.4 With the mask

Across positions `t`, with the mask factored in:

\[
\frac{\partial L_{\text{SFT}}}{\partial z_t}
=
\frac{m_t}{N_{\text{resp}}}\left(p_t-\mathrm{onehot}(y_t)\right)
\]

Three things worth noticing:

- When `m[t] = 0`, the gradient at that position is exactly zero. Changing the label
  at a masked-out position doesn't change the loss. This is exactly what the
  "flip a masked token, loss unchanged" unit test checks.
- The denominator is `N_resp`, not `N_resp * V`. There's no factor of `V` because
  `p` is a proper probability distribution (sums to 1) — it's not `V` independent
  scalars.
- This is the gradient that flows from the loss into the `lm_head` and then
  backpropagates through the rest of the transformer.

---

## 5. Implementation checklist

### 5.1 `sft_loss(logits, labels, loss_mask)` (Problem 2.2)

Shapes:

```
logits:    (B, T, V)     float
labels:    (B, T)        int64, already shifted
loss_mask: (B, T)        float or bool (0 or 1)
```

The implementation in three lines:

1. `logp = log_softmax(logits, dim=-1)` — numerically stable, computes log-softmax
   in one pass.
2. `nll = -logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)` — pick the
   log-probability of the true label at each position. Shape `(B, T)`.
3. `loss = (nll * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)` — masked mean.

**Do not** use `F.cross_entropy(..., ignore_index=-100)` and stuff `-100` into masked
labels. The point of this course is that you manage the mask explicitly. The
`ignore_index` trick hides it from you, which is exactly where subtle bugs live.

### 5.2 Gradient check (Problem 2.2)

Two tests:

1. **Analytic vs. finite difference.** Use fp64 and a tiny tensor (say `B=2, T=4,
   V=8`). Compute the gradient with respect to `logits` both ways: once by calling
   `.backward()` on the loss, once by perturbing each logit component by `+eps` and
   `-eps` and averaging. Relative error should be under `1e-5`.

2. **Mask flip invariance.** Construct inputs, compute the loss, then change
   `labels[b, t]` at any masked-out position to a different token id. The loss
   must be exactly equal to the original, and the gradient must be elementwise
   identical.

### 5.3 DataLoader (Problem 2.3)

- Pad to the length of the longest example in the batch, not to the full
  `block_size`. If your batch's longest example is 300 tokens but you pad to 1024,
  you're wasting 70% of compute on padding.
- Return four tensors, all with the same shape `(B, T_batch)`:
  - `input_ids`
  - `labels` (shifted input_ids)
  - `loss_mask` (the assistant-token mask, shifted)
  - `attention_mask` (the usual 1-on-real-tokens, 0-on-padding mask)
- `attention_mask` and `loss_mask` are different things. `attention_mask` tells the
  transformer where the real tokens are. `loss_mask` tells the loss which tokens
  count. Neither implies the other.

### 5.4 Training loop (Problem 2.4)

The boring-but-important parts:

- **AdamW** with `betas = (0.9, 0.95)`, `weight_decay = 0.1`, **no weight decay on
  1D parameters**. In practice this means: LayerNorm scales and biases, plus all
  biases, go into a "no-decay" group; everything else goes into a "decay" group.
  This is the param grouping Karpathy used in nanoGPT and InstructGPT follows the
  same pattern.
- **Cosine learning rate schedule** from peak LR down to 10% of peak, with a linear
  warmup of 200 steps from 0 up to peak.
- **Gradient clipping** at max norm 1.0.
- **bf16 autocast** for the forward pass, with fp32 master weights and optimizer
  state.
- **Gradient accumulation** to hit a larger effective batch. Scale the loss by
  `1 / accum_steps` before `.backward()`, and only call `opt.step()` every
  `accum_steps` micro-steps.

Run for 2 epochs on HH's `chosen` stream. Log training loss every step, eval loss
every 250 optimizer steps on a held-out split. Save `sft.pt` at the end.

### 5.5 What good looks like

- Training loss typically drops from around 3.5 (base GPT-2 seeing chat-formatted
  text for the first time) down to roughly 1.5–2.0 by the end of epoch 2. Exact
  numbers depend on your tokenization and mask — don't obsess over matching
  someone else's loss.
- In the qualitative eval (Problem 2.5), the base model will ramble, sometimes
  invent "Human:" turns, or drift off-topic. The SFT model should produce a single
  coherent assistant turn that ends cleanly with `<|im_end|>`.

If your training loss plateaus above 3.0 after a few hundred steps, your mask is
almost certainly wrong — you're either training on prompt tokens or masking out
assistant tokens you shouldn't be.

---

## 6. What to commit to `notes/02-sft.md`

After finishing Module 2, add:

- Your own re-derivation of the softmax cross-entropy gradient (photo of paper or
  typed). You must be able to do this cold in a month.
- Training and eval loss curves, or at least a short description ("loss dropped
  from 3.6 to 1.8 over two epochs, flattening in the last 500 steps").
- Observations from the side-by-side in 2.5. What does base GPT-2 say on your
  held-out prompts? What does SFT say? Where does SFT still fail? Those failure
  modes are exactly the motivation for RLHF in the next modules.
