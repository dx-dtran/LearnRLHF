# 02 — Supervised fine-tuning (SFT)

## Purpose

Theory packet for Module 2 (Problems 2.1 through 2.5). The goal of Module 2 is
to take a pretrained GPT-2 and fine-tune it so that, given a user message in
the chat format from Module 0, it produces an assistant reply that resembles
the "chosen" replies from HH-RLHF.

The training objective is the same next-token cross-entropy loss GPT-2 was
pretrained with. Two things change relative to pretraining:

1. The data is formatted dialogues rather than raw web text.
2. The loss is only counted on tokens that come from the assistant. Tokens
   that come from the user (and the chat-template scaffolding around them)
   are excluded from the loss via a per-position **loss mask**.

By the end of this note the objectives are:

1. Write down the SFT loss and explain what every piece of it means in words.
2. Derive the gradient of softmax + cross-entropy with respect to the logits.
3. Explain which tokens the `loss_mask` zeros out and why.

SFT is the simplest of the three training phases, but the masking logic is
critical. A single off-by-one or inverted mask quietly corrupts every
downstream phase, since RM and PPO both assume SFT was done correctly.

Before writing the code in each problem, state the **tensor contract**:
which tensors come in, what their shapes are, and what each one means. For
the SFT loss the contract is "logits predict the next token, labels are
shifted by one position relative to the input, and the mask is attached to
the token being predicted (not to the input token at the same position)."
Almost every bug in this module is a violation of one of those three
clauses.

---

## 1. What SFT actually does

The starting point is a pretrained GPT-2, which models the probability of the
next token given the previous tokens, trained on web text. The target is a
GPT-2 that, given a user's message in the ChatML format, produces an assistant
reply that resembles the `chosen` responses from HH-RLHF.

SFT keeps the next-token prediction objective, but trains on formatted
dialogues instead of raw web text and **only counts the loss on assistant
tokens**.

The model sees the entire dialogue, but during training it is graded only on
its predictions of assistant tokens. It can use user messages as context, but
it is never asked to predict the next user token. User text is input;
assistant text is the target.

SFT changes the *data distribution* and the *mask*. The language-modeling
machinery stays the same: the transformer still produces one vocabulary
distribution per position, and the loss still asks for high probability on
the next token. The mask decides which predictions teach the desired
behavior.

---

## 2. The loss

### 2.1 Tokenized example

One example is one full formatted dialogue, tokenized to a sequence
`x = [x_0, x_1, ..., x_{T-1}]`.

Alongside `x`, there is a binary mask `m` of the same length. `m[t] = 1` means
position `t` is part of an assistant turn's content; `m[t] = 0` means
position `t` is user text, scaffolding, or padding.

Because GPT-2 predicts the *next* token at each position, it is cleaner to
work with the shifted version:

```
input_ids   = x[:-1]      # positions 0 .. T-2, what we feed in
labels      = x[1:]       # positions 1 .. T-1, what we try to predict
loss_mask   = m[1:]       # 1 iff the token we're trying to predict is assistant
```

From this point on, "position `t`" refers to the shifted frame: the model's
logits at position `t` predict `labels[t]`, and `loss_mask[t]` indicates
whether that prediction counts toward the loss.

The shift is worth drawing on paper. If the original sequence is
`[A, B, C, D]`, the model sees `[A, B, C]` and predicts `[B, C, D]`. A mask
value attached to original token `D` must move to the label position where
`D` is predicted. Using `m[:-1]` instead asks whether the *input* token was
assistant text, not whether the *target* token was assistant text.

### 2.2 Per-token cross-entropy

At each position `t`, the model outputs a logit vector $z_t$ of length `V`
(the vocab size). Softmax converts logits into a probability distribution
over the vocabulary:

$$
p_{t,v} = \frac{\exp(z_{t,v})}{\sum_{v'} \exp(z_{t,v'})}
$$

Or in code-style:

    p[t, v] = exp(z[t, v]) / sum over v' of exp(z[t, v'])

The exponential makes the entries positive; the division by their sum
normalizes them to a valid probability distribution.

The cross-entropy loss at position `t` is the negative log-probability the
model assigns to the true next token $y_t = \text{labels}[t]$:

$$
\ell_t = -\log p_{t,y_t} = -z_{t,y_t} + \log \sum_{v'} \exp(z_{t,v'})
$$

Or in code-style:

    ell[t] = -log p[t, y_t]
           = -z[t, y_t] + log(sum over v' of exp(z[t, v']))

The second form (correct-class logit minus log-sum-exp) is what the code
should compute. Computing the softmax first and then taking its log lets the
softmax underflow to zero, after which `log(0)` blows up. PyTorch's
`log_softmax` uses the log-sum-exp trick internally for numerical stability.

`-log p` is close to 0 when the model is confident and correct, and grows
without bound as the assigned probability shrinks. SFT minimizes this
negative log-likelihood on assistant tokens.

The logarithm makes the penalty additive over tokens. A sequence probability
is a product of per-token probabilities; the negative log turns that product
into a sum. Language modeling losses are usually reported as average
negative log-likelihood per token, and perplexity is the exponentiated
version of that average.

### 2.3 Masked mean over the batch

Summing across a batch of examples, indexed by `(b, t)`:

$$
L_{\mathrm{SFT}} = \frac{1}{N_{\mathrm{resp}}} \sum_{b, t} m_{b,t} \cdot \ell_{b,t},
\qquad
N_{\mathrm{resp}} = \sum_{b, t} m_{b,t}
$$

Or in code-style:

    L_SFT  =  sum over (b, t) of { m[b, t] * ell[b, t] }  /  N_resp
    N_resp =  sum over (b, t) of m[b, t]

$N_{\mathrm{resp}}$ is the total number of assistant tokens in the batch.

Dividing by `N_resp` rather than by `B * T` makes the loss scale independent
of how much padding happened to be in the batch. Padding tokens contribute
zero to both the numerator and the denominator, so they do not affect the
average.

This denominator also makes batches comparable. For a batch with short
prompts and long answers, and another batch with long prompts and short
answers, dividing by valid assistant tokens means one unit of loss
corresponds to "average surprise per supervised assistant token" in both
cases.

---

## 3. Why mask the prompt?

There are two independent reasons to set `loss_mask = 0` on every token
that is not assistant content. The first reason is about what the model
learns to *generate* (3.1); the second is about what the loss number
actually measures during training (3.2). Section 3.3 then states the
exact rule for which tokens get a 1 and which get a 0.

### 3.1 User text is not something we want the model to learn to generate

At inference time the user's message is supplied externally; the model never
generates user text. Training on user tokens wastes capacity memorizing
human prompt phrasings, and in some cases the model starts hallucinating
"Human: ..." turns into its own output, continuing the dialogue rather than
stopping after one assistant turn. Masking out user tokens prevents this.

This is especially important with chat templates. The model should learn
that after an assistant header it emits assistant content. It should not
learn to continue by creating a new user message unless the application is
explicitly a dialogue simulator. For an assistant, user text is
conditioning information rather than a target behavior.

### 3.2 The loss scale is dominated by the prompt

In HH-RLHF, prompts are often longer than responses. Without masking, 60–80%
of every example's loss comes from the user's prompt tokens. The assistant's
response, which is the part of the example that the training objective is
about, is a small fraction of the signal. Training loss may drop quickly
while the model barely improves at responding in the ChatML format.

### 3.3 Exactly which tokens the mask includes

Include:

- The **content tokens of each assistant turn**.
- The assistant turn's closing `<|im_end|>`. The model needs to learn when
  to stop.

Exclude:

- The `<|im_start|>assistant\n` header. That is scaffolding supplied by the
  template, not something the model needs to emit.
- All user tokens: the content, the user's `<|im_start|>user\n`, and the
  user's `<|im_end|>`.
- Any padding.

The boundary rule is: at inference time, who is responsible for producing
this token? If the answer is "the caller" or "the formatting code", mask it
out. If the answer is "the assistant model", include it.

---

## 4. Gradient of softmax cross-entropy

The PPO surrogate in Module 4 uses the same mechanics, so the derivation is
worth working through cleanly.

### 4.1 Setup

Fix one position. Let $z$ be the length-$V$ vector of logits and let $p$ be
the softmax of $z$. Let $y$ be the index of the correct class. The loss at
this position is:

$$
\ell = -\log p_y
$$

The goal is $\partial \ell / \partial z$, the gradient of the loss with
respect to the logits. The plan is the chain rule, broken into two pieces:
how the loss changes with the probability, and how the probability changes
with the logit.

### 4.2 Softmax derivative

For any two indices $v$ and $u$:

$$
\frac{\partial p_v}{\partial z_u} = p_v \cdot (\delta_{v,u} - p_u)
$$

where $\delta_{v,u}$ is the Kronecker delta (1 if the subscripts are equal,
0 otherwise). In code-style:

    d p[v] / d z[u] = p[v] * (delta(v, u) - p[u])

To derive this, start from $p_v = \exp(z_v) / S$ where $S = \sum_{v'}
\exp(z_{v'})$, and apply the quotient rule:

$$
\frac{\partial p_v}{\partial z_u}
 = \frac{(\partial \exp(z_v)/\partial z_u) \cdot S - \exp(z_v) \cdot (\partial S/\partial z_u)}{S^2}
$$

The first derivative in the numerator is $\exp(z_v) \cdot \delta_{v,u}$
(since $\exp(z_v)$ depends on $z_u$ only when $v = u$). The second is
$\partial S/\partial z_u = \exp(z_u)$. Substituting and collecting terms:

$$
\frac{\partial p_v}{\partial z_u}
 = \frac{\exp(z_v)}{S} \delta_{v,u} - \frac{\exp(z_v)}{S} \cdot \frac{\exp(z_u)}{S}
 = p_v \delta_{v,u} - p_v p_u
 = p_v (\delta_{v,u} - p_u)
$$

This identity recurs whenever a softmax is differentiated.

### 4.3 Chain rule

Combine the two pieces:

$$
\frac{\partial \ell}{\partial z_u}
 = \frac{\partial \ell}{\partial p_y} \cdot \frac{\partial p_y}{\partial z_u}
$$

The first factor comes from $\ell = -\log p_y$:

$$
\frac{\partial \ell}{\partial p_y} = -\frac{1}{p_y}
$$

The second factor was just computed, with $v = y$:

$$
\frac{\partial p_y}{\partial z_u} = p_y (\delta_{y,u} - p_u)
$$

Multiplying:

$$
\frac{\partial \ell}{\partial z_u}
 = -\frac{1}{p_y} \cdot p_y (\delta_{y,u} - p_u)
 = -(\delta_{y,u} - p_u)
 = p_u - \delta_{y,u}
$$

The $p_y$ cancels. In vector form:

$$
\frac{\partial \ell}{\partial z} = p - e_y
$$

where $e_y$ is the one-hot vector with a 1 at index $y$ and 0 elsewhere. Or
in code-style:

    d ell / d z  =  p - onehot(y)

The gradient is the model's predicted distribution minus a spike on the
correct class. If the model assigns too much probability to a wrong token,
that wrong token has a positive gradient, so gradient descent reduces its
logit. If the model assigns too little probability to the true token, the
true token has a negative gradient, so gradient descent increases its
logit. The update is local to the vocabulary distribution at that position;
the chain rule then carries the gradient backward through the transformer.

### 4.4 A worked example with numbers

Pretend the vocabulary has three tokens: "cat", "dog", "fish" (indices 0, 1,
2). The true next token is "dog" (so $y = 1$). Suppose the model's raw
logits are $z = (1.0,\, 2.0,\, 0.5)$.

Step 1: softmax. Exponentiate each entry and divide by the sum:

- $\exp(1.0) \approx 2.72$
- $\exp(2.0) \approx 7.39$
- $\exp(0.5) \approx 1.65$
- Sum $\approx 11.76$
- $p \approx (0.231,\, 0.628,\, 0.140)$

The model already thinks "dog" is most likely (62.8%) but is not fully
confident.

Step 2: the loss. $\ell = -\log p_y = -\log(0.628) \approx 0.465$.

Step 3: the gradient. Use the formula $\partial \ell / \partial z = p -
e_y$, where $e_y = (0, 1, 0)$ is the one-hot vector at index 1:

- $\partial \ell / \partial z_0 = 0.231 - 0 = +0.231$
- $\partial \ell / \partial z_1 = 0.628 - 1 = -0.372$
- $\partial \ell / \partial z_2 = 0.140 - 0 = +0.140$

A gradient descent step moves $z$ in the *opposite* direction of the
gradient. With learning rate $\eta = 1.0$:

- $z_0 \leftarrow 1.0 - 0.231 = 0.769$ (cat logit went down)
- $z_1 \leftarrow 2.0 - (-0.372) = 2.372$ (dog logit went up)
- $z_2 \leftarrow 0.5 - 0.140 = 0.360$ (fish logit went down)

Recomputing softmax with the new logits gives $p \approx (0.154,\, 0.751,\,
0.095)$. The model is now 75% confident in "dog", up from 63%. One gradient
step moved the prediction in the right direction by an amount determined by
the gradient itself.

The same arithmetic happens independently at every unmasked position in the
batch. The transformer adds parameter sharing: a change to one weight
affects logits at many future examples and positions. Cross-entropy
supplies the local signal at each position; backpropagation distributes
that signal to every parameter.

### 4.5 With the mask

Across positions `t`, with the mask factored in:

$$
\frac{\partial L_{\mathrm{SFT}}}{\partial z_t}
 = \frac{m_t}{N_{\mathrm{resp}}} \cdot (p_t - e_{y_t})
$$

Or in code-style:

    d L_SFT / d z[t]  =  (m[t] / N_resp) * (p[t] - onehot(y[t]))

Three things to notice:

- When `m[t] = 0`, the gradient at that position is exactly zero. Changing
  the label at a masked-out position does not change the loss, which is
  what the "flip a masked token, loss unchanged" unit test checks.
- The denominator is `N_resp`, not `N_resp * V`. There is no factor of `V`
  because `p` is a proper probability distribution (sums to 1) rather than
  `V` independent scalars.
- This is the gradient that flows from the loss into the `lm_head` and
  then backpropagates through the rest of the transformer.

### 4.6 Worked example: end-to-end masked CE on a chat template

Pick up the toy vocab from `00-data.md` and walk one example through. The
original (unshifted) tokens for the chosen dialogue were:

```
pos:        0   1   2   3   4   5   6   7   8   9
input_id:  10  11  12  13  14  15  16  17  18  15
piece:      U   W   I   2   ?  /U   A   4   .  /A
mask m:     0   0   0   0   0   0   0   1   1   1
```

Shift to the predict-next frame:

```
shifted pos:  0   1   2   3   4   5   6   7   8
input_id:    10  11  12  13  14  15  16  17  18      # x[:-1]
labels:      11  12  13  14  15  16  17  18  15      # x[1:]
loss_mask:    0   0   0   0   0   0   1   1   1      # m[1:]
```

Three loss-mask positions: 6 predicts `17` (the assistant content `4`), 7
predicts `18` (the assistant content `.`), 8 predicts `15` (the closing
`<|im_end|>`).

Suppose the model produces these logits at each position over a vocab of
size 4 restricted to the relevant ids `{15, 16, 17, 18}`:

```
shifted pos:  6           7           8
              for `4`:     for `.`:    for `</A>`:
target:      17          18          15
logits[15]:  -1.0        -1.0         2.0
logits[16]:   0.0         0.0         0.0
logits[17]:   2.0         0.0        -1.0
logits[18]:   0.0         2.0         0.0
```

Per-position softmax probabilities (rounding to 3 figures):

```
pos 6:  p[15]=0.034  p[16]=0.094  p[17]=0.690  p[18]=0.094
pos 7:  p[15]=0.034  p[16]=0.094  p[17]=0.094  p[18]=0.690
pos 8:  p[15]=0.690  p[16]=0.094  p[17]=0.034  p[18]=0.094
```

Per-position cross-entropy at the supervised positions:

```
ell[6] = -log(0.690) ≈ 0.371      # target 17
ell[7] = -log(0.690) ≈ 0.371      # target 18
ell[8] = -log(0.690) ≈ 0.371      # target 15
```

`N_resp = sum(loss_mask) = 3`, so:

```
L_SFT = (0 + 0 + 0 + 0 + 0 + 0 + 0.371 + 0.371 + 0.371) / 3 ≈ 0.371
```

The unsupervised positions (0..5) contribute zero to the numerator and
zero to the denominator. The model's predictions on user tokens do not
affect this loss.

### 4.7 Worked example: flipping a masked label

Same setup. Change `labels[2]` from `13` (toy id for `2+2`) to some random
other token id, say `99`. That position has `loss_mask[2] = 0`, so:

- `ell[2]` is computed as `-log p[2, 99]` instead of `-log p[2, 13]`.
- The product `loss_mask[2] * ell[2]` is `0 * (anything) = 0`.
- The numerator sum is unchanged. The denominator sum is unchanged.

Result: `L_SFT ≈ 0.371`, identical. This is what the unit test in Problem
2.2 asserts. Performing the flip in code, `(loss - loss_orig).abs().max()`
should be machine-zero.

If the test fails, the most common cause is that the implementation
divides by the unmasked count (`B * T`) instead of the mask sum, in which
case the denominator changes when the labels do. Another cause is using
`ignore_index=-100` elsewhere in the code path, which lets framework code
reinterpret labels and drift away from the explicit mask.

### 4.8 Worked example: per-batch versus per-sequence denominator

Two sequences in one batch. Lengths after shifting: 4 and 6. Loss masks
(the part that supervises assistant content):

```
row 0:  loss_mask = [0, 0, 1, 1]                 # 2 supervised tokens
row 1:  loss_mask = [0, 0, 0, 0, 1, 1]           # 2 supervised tokens

per-token ell (already computed):
row 0:  [_, _, 1.0, 0.5]
row 1:  [_, _, _, _, 2.0, 1.0]
```

(Underscores are positions where `loss_mask = 0`. Their ell values do not
matter.)

Two ways to average:

**Per-sequence then mean over rows.**

    row 0 mean = (1.0 + 0.5) / 2 = 0.75
    row 1 mean = (2.0 + 1.0) / 2 = 1.50
    batch loss = (0.75 + 1.50) / 2 = 1.125

**Per-batch (the InstructGPT convention).**

    numerator   = 1.0 + 0.5 + 2.0 + 1.0 = 4.5
    denominator = 2 + 2 = 4
    batch loss  = 4.5 / 4 = 1.125

The two methods agree here because both rows happen to have the same
supervised count. If row 0's supervised count rises to 8 with most ells
near zero while row 1 keeps its 2 ells near 1.5, per-sequence averaging
gives equal weight to both rows even though one has 4× more tokens.
Per-batch averaging weights each supervised token equally. The model's
representations are updated per token, so per-batch is the right choice.

Padding contributes 0 to the numerator (because `loss_mask = 0`) and 0 to
the denominator. It is invisible to either calculation. The per-batch
form is therefore robust to the shape of the batch.

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

1. `logp = log_softmax(logits, dim=-1)` for a numerically stable log-softmax.
2. `nll = -logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)` to pick the
   log-probability of the true label at each position. Shape `(B, T)`.
3. `loss = (nll * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)` for the
   masked mean.

Do **not** use `F.cross_entropy(..., ignore_index=-100)` and stuff `-100`
into masked labels. The point of this course is explicit mask management.
The `ignore_index` shortcut hides the masking decision and is exactly where
subtle bugs accumulate.

Explicit masking also makes diagnostics easier. `loss_mask.sum()` can be
printed, masked NLL values inspected, assistant-token loss compared to
prompt-token loss, and tests written that flip masked labels. With
`ignore_index`, that logic is implicit inside a library call.

### 5.2 Gradient check (Problem 2.2)

Two tests:

1. **Analytic vs. finite difference.** Use fp64 and a tiny tensor (say
   `B=2, T=4, V=8`). Compute the gradient with respect to `logits` both
   ways: once by calling `.backward()` on the loss, once by perturbing each
   logit component by `+eps` and `-eps` and averaging. Relative error
   should be under `1e-5`.

2. **Mask flip invariance.** Construct inputs, compute the loss, then
   change `labels[b, t]` at any masked-out position to a different token
   id. The loss must equal the original exactly, and the gradient must be
   elementwise identical.

### 5.3 DataLoader (Problem 2.3)

- Pad to the length of the longest example in the batch, not to the full
  `block_size`. If the batch's longest example is 300 tokens but padding
  goes to 1024, 70% of compute is spent on padding.
- Return four tensors, all with the same shape `(B, T_batch)`:
  - `input_ids`
  - `labels` (shifted input_ids)
  - `loss_mask` (the assistant-token mask, shifted)
  - `attention_mask` (the usual 1-on-real-tokens, 0-on-padding mask)
- `attention_mask` and `loss_mask` are different. `attention_mask` tells
  the transformer where the real tokens are. `loss_mask` tells the loss
  which tokens count. Neither implies the other.

A real assistant token has `attention_mask = 1` and possibly `loss_mask =
1`. A real user token has `attention_mask = 1` and `loss_mask = 0`.
Padding has `attention_mask = 0` and `loss_mask = 0`. Treating "masked
out of the loss" as equivalent to "invisible to the model" leads to bugs.

### 5.4 Training loop (Problem 2.4)

Operational details:

- **AdamW** with `betas = (0.9, 0.95)`, `weight_decay = 0.1`, **no weight
  decay on 1D parameters**. LayerNorm scales and biases, plus all biases,
  go into a "no-decay" group; everything else goes into a "decay" group.
  This grouping comes from nanoGPT and matches InstructGPT.
- **Cosine learning rate schedule** from peak LR down to 10% of peak,
  with a linear warmup of 200 steps from 0 up to peak.
- **Gradient clipping** at max norm 1.0.
- **bf16 autocast** for the forward pass, with fp32 master weights and
  optimizer state.
- **Gradient accumulation** to reach a larger effective batch. Scale the
  loss by `1 / accum_steps` before `.backward()`, and call `opt.step()`
  every `accum_steps` micro-steps.

Run for 2 epochs on HH's `chosen` stream. Log training loss every step,
eval loss every 250 optimizer steps on a held-out split. Save `sft.pt` at
the end.

### 5.5 What good looks like

- Training loss typically drops from around 3.5 (base GPT-2 seeing
  chat-formatted text for the first time) down to roughly 1.5–2.0 by the
  end of epoch 2. Exact numbers depend on the tokenization and mask.
- In the qualitative eval (Problem 2.5), the base model rambles, sometimes
  invents "Human:" turns, or drifts off-topic. The SFT model should
  produce a single coherent assistant turn that ends cleanly with
  `<|im_end|>`.

A training loss that plateaus above 3.0 after a few hundred steps almost
always indicates a wrong mask: either training on prompt tokens, or
masking out assistant tokens that should be supervised.

A model that generates plausible raw text but ignores the assistant role
suggests a data format problem. A model that emits `<|im_end|>`
immediately suggests only closing markers are being supervised. Eval loss
much lower than train loss suggests the eval examples have fewer
supervised tokens. The loss curve only becomes meaningful after the mask
is trustworthy.

---

## 6. What to commit to `notes/02-sft.md`

After finishing Module 2, add:

- Your own re-derivation of the softmax cross-entropy gradient (photo of
  paper or typed).
- Training and eval loss curves, or a short description ("loss dropped
  from 3.6 to 1.8 over two epochs, flattening in the last 500 steps").
- Observations from the side-by-side in 2.5: what base GPT-2 says on the
  held-out prompts, what SFT says, where SFT still fails. Those failure
  modes motivate RLHF in the next modules.
