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

This module is also where you build the habit that carries through the rest of the repo:
state the tensor contract before writing the code. For SFT, the contract is "logits predict
the next token, labels are shifted by one, and the mask is attached to the token being
predicted." Almost every bug in this module is a violation of one of those three clauses.

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

Another way to say this: SFT changes the *data distribution* and the *mask*, not the basic
language-modeling machinery. The transformer still produces one vocabulary distribution per
position. The loss still asks for high probability on the next token. The only question is
which next-token predictions are educational for the behavior we want.

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

This shifted frame is worth drawing on paper. If the original sequence is `[A, B, C, D]`,
the model sees `[A, B, C]` and predicts `[B, C, D]`. A mask value attached to original token
`D` must move to the label position where `D` is predicted. If you instead use `m[:-1]`, you
are asking whether the *input* token was assistant text, not whether the *target* token was
assistant text.

### 2.2 Per-token cross-entropy

At each position `t`, the model outputs a logit vector $z_t$ of length `V` (vocab
size). Softmax converts logits to a probability distribution over the vocabulary:

$$
p_{t,v} = \frac{\exp(z_{t,v})}{\sum_{v'} \exp(z_{t,v'})}
$$

Or in code-style:

    p[t, v] = exp(z[t, v]) / sum over v' of exp(z[t, v'])

Intuition: the logits are unconstrained real numbers; softmax is the function that
turns an arbitrary vector of real numbers into a legal probability distribution (all
entries non-negative and summing to 1). The exponential makes things positive; the
division by the sum normalizes them. Same recipe you'd use to turn any list of
"scores" into a distribution.

The cross-entropy loss at position `t` is the negative log-probability the model
assigns to the true next token $y_t = \text{labels}[t]$:

$$
\ell_t = -\log p_{t,y_t} = -z_{t,y_t} + \log \sum_{v'} \exp(z_{t,v'})
$$

Or in code-style:

    ell[t] = -log p[t, y_t]
           = -z[t, y_t] + log(sum over v' of exp(z[t, v']))

The second form — "correct-class logit, minus the log-sum-exp of all logits" — is
what you actually compute. Never compute the softmax first and then take its log;
the softmax can underflow to zero and `log(0)` blows up. The `log_softmax` op in
PyTorch uses the log-sum-exp trick internally to stay numerically stable.

Intuition for the loss: `-log p` is tiny (close to 0) when the model is confident and
correct, and huge (goes to infinity) when the model is confident and wrong. It's a
surprise. We're training the model to minimize surprise on the assistant tokens we
want it to produce.

The logarithm makes the penalty additive over tokens. A sequence probability is a product of
per-token probabilities; the negative log turns that product into a sum. That is why language
modeling losses are usually reported as average negative log-likelihood per token and why
perplexity is just the exponentiated version of that average.

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

Dividing by `N_resp` (instead of by `B * T` or anything else) means the loss scale
doesn't depend on how much padding happened to be in the batch. Padding tokens
contribute zero to the numerator *and* zero to the denominator, so they don't pollute
either side of the average.

This denominator also makes batches comparable. Suppose one batch has short prompts and long
answers, while another has long prompts and short answers. Dividing by valid assistant tokens
means one unit of loss always means "average surprise per supervised assistant token." That
is the number you actually care about.

---

## 3. Why mask the prompt?

Two big reasons.

### 3.1 User text is not something we want the model to learn to generate

At inference time, **we** write the user's message — the model never has to
generate user text. If we train on user tokens, the model wastes capacity memorizing
human prompt phrasings, and worse, sometimes starts hallucinating "Human: ..." turns
into its own output. This is the classic SFT failure mode where the model keeps
inventing the other side of the conversation. Masking out user tokens prevents it.

This is especially important with chat templates. The model should learn that after the
assistant header it emits assistant content. It should not learn to continue by creating a
new user message unless your application explicitly asks for a dialogue simulator. For an
assistant, user text is conditioning information, not a target behavior.

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

When unsure, ask: "At inference time, who is responsible for producing this token?" If the
answer is "the caller" or "the formatting code", mask it out. If the answer is "the assistant
model", include it. This rule resolves nearly every boundary case.

---

## 4. Gradient of softmax cross-entropy

You must be able to derive this from a blank page. It's the one gradient every ML
engineer has memorized, and the PPO surrogate we build in Module 4 uses the same
mechanics.

### 4.1 Setup

Fix one position for now. Let $z$ be the length-$V$ vector of logits and let $p$ be
the softmax of $z$. Let $y$ be the index of the correct class. The loss at this
position is:

$$
\ell = -\log p_y
$$

We want $\partial \ell / \partial z$ — the gradient of the loss with respect to the
logits. Our plan is the usual one: compute it by the chain rule, breaking the
derivative into two pieces — how the loss changes with the probability, and how the
probability changes with the logit.

### 4.2 Softmax derivative

For any two indices $v$ and $u$ we have:

$$
\frac{\partial p_v}{\partial z_u} = p_v \cdot (\delta_{v,u} - p_u)
$$

where $\delta_{v,u}$ is the Kronecker delta (reads: "1 if the subscripts are equal,
0 otherwise"). In code-style:

    d p[v] / d z[u] = p[v] * (delta(v, u) - p[u])

Derive this once yourself. Start from $p_v = \exp(z_v) / S$ where
$S = \sum_{v'} \exp(z_{v'})$, and use the quotient rule:

$$
\frac{\partial p_v}{\partial z_u}
 = \frac{(\partial \exp(z_v)/\partial z_u) \cdot S - \exp(z_v) \cdot (\partial S/\partial z_u)}{S^2}
$$

The first derivative in the numerator is $\exp(z_v) \cdot \delta_{v,u}$ (since
$\exp(z_v)$ depends on $z_u$ only when $v = u$). The second is
$\partial S/\partial z_u = \exp(z_u)$. Plug in and collect terms:

$$
\frac{\partial p_v}{\partial z_u}
 = \frac{\exp(z_v)}{S} \delta_{v,u} - \frac{\exp(z_v)}{S} \cdot \frac{\exp(z_u)}{S}
 = p_v \delta_{v,u} - p_v p_u
 = p_v (\delta_{v,u} - p_u)
$$

Done. This result is worth memorizing — it shows up every time you differentiate a
softmax.

### 4.3 Chain rule

Now stitch together "how does $\ell$ depend on $p_y$" with "how does $p_y$ depend on
$z_u$":

$$
\frac{\partial \ell}{\partial z_u}
 = \frac{\partial \ell}{\partial p_y} \cdot \frac{\partial p_y}{\partial z_u}
$$

The first factor comes from $\ell = -\log p_y$:

$$
\frac{\partial \ell}{\partial p_y} = -\frac{1}{p_y}
$$

The second factor we just computed (plug $v = y$):

$$
\frac{\partial p_y}{\partial z_u} = p_y (\delta_{y,u} - p_u)
$$

Multiply them:

$$
\frac{\partial \ell}{\partial z_u}
 = -\frac{1}{p_y} \cdot p_y (\delta_{y,u} - p_u)
 = -(\delta_{y,u} - p_u)
 = p_u - \delta_{y,u}
$$

The $p_y$ cancels beautifully. In vector form:

$$
\frac{\partial \ell}{\partial z} = p - e_y
$$

where $e_y$ is the one-hot vector with a 1 at index $y$ and 0 elsewhere. Or in
code-style:

    d ell / d z  =  p - onehot(y)

Read this out loud: **"the gradient is the model's predicted distribution minus a
spike on the correct class."**

That sentence is the core intuition. If the model assigns too much probability to a wrong
token, that wrong token has a positive gradient, so gradient descent pushes its logit down.
If the model assigns too little probability to the true token, the true token has a negative
gradient, so gradient descent pushes its logit up. The update is local to the vocabulary
distribution at that position, then the chain rule carries it backward through the transformer.

Interpretation: the gradient subtracts probability mass from the true class and adds
it to every other class, proportional to what the model currently predicts. A
gradient *step* (minus this direction) pushes mass *toward* the correct class and
*away from* every wrong class. Every update is a small rebalancing.

An analogy that helps: imagine each class as a bucket, and $p$ as how full each
bucket is with water. The true bucket is $y$. The gradient says "drain a little
water from every bucket in proportion to how full it is, then dump an equal total
amount into bucket $y$." After many steps, bucket $y$ is full and everything else
is nearly empty — which is exactly what confident, correct prediction looks like.

### 4.4 A worked example with numbers

Let's make this concrete. Pretend our vocabulary has only 3 tokens, call them
"cat", "dog", "fish" (indices 0, 1, 2). The true next token is "dog" (so $y = 1$).
Suppose the model's raw logits are $z = (1.0,\, 2.0,\, 0.5)$.

Step 1: softmax. Exponentiate each entry and divide by the sum:

- $\exp(1.0) \approx 2.72$
- $\exp(2.0) \approx 7.39$
- $\exp(0.5) \approx 1.65$
- Sum $\approx 11.76$
- $p \approx (0.231,\, 0.628,\, 0.140)$

So the model already thinks "dog" is most likely (62.8%), but it's not fully
confident.

Step 2: the loss. $\ell = -\log p_y = -\log(0.628) \approx 0.465$.

Step 3: the gradient. Use the formula $\partial \ell / \partial z = p - e_y$,
where $e_y = (0, 1, 0)$ is the one-hot vector at index 1:

- $\partial \ell / \partial z_0 = 0.231 - 0 = +0.231$
- $\partial \ell / \partial z_1 = 0.628 - 1 = -0.372$
- $\partial \ell / \partial z_2 = 0.140 - 0 = +0.140$

A gradient descent step moves $z$ in the *opposite* direction of the gradient (with
some learning rate $\eta$), so after one step with $\eta = 1.0$:

- $z_0 \leftarrow 1.0 - 0.231 = 0.769$ (cat logit went down — good, cat is wrong)
- $z_1 \leftarrow 2.0 - (-0.372) = 2.372$ (dog logit went up — good, dog is right)
- $z_2 \leftarrow 0.5 - 0.140 = 0.360$ (fish logit went down — good, fish is wrong)

Recompute softmax with the new logits: $p \approx (0.154,\, 0.751,\, 0.095)$. The
model is now 75% sure about "dog", up from 63%. One gradient step moved us in the
right direction, and the gradient told us *exactly how much* to move each logit.
Do this for 100 million tokens and you've got a trained language model.

The same arithmetic happens independently at every unmasked position in the batch. The only
thing the transformer adds is parameter sharing: a change to one weight affects logits at many
future examples and positions. Cross-entropy supplies the simple local signal; backpropagation
decides how every parameter contributed to that signal.

### 4.5 With the mask

Across positions `t`, with the mask factored in:

$$
\frac{\partial L_{\mathrm{SFT}}}{\partial z_t}
 = \frac{m_t}{N_{\mathrm{resp}}} \cdot (p_t - e_{y_t})
$$

Or in code-style:

    d L_SFT / d z[t]  =  (m[t] / N_resp) * (p[t] - onehot(y[t]))

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

Explicit masking also makes diagnostics easier. You can print `loss_mask.sum()`, inspect
masked NLL values, compare assistant-token loss to prompt-token loss, and write tests that
flip masked labels. With `ignore_index`, much of that logic is implicit inside a library
call.

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

A real assistant token has both `attention_mask = 1` and possibly `loss_mask = 1`. A real
user token has `attention_mask = 1` and `loss_mask = 0`. Padding has `attention_mask = 0` and
`loss_mask = 0`. Keeping those three cases separate prevents a lot of accidental reasoning
like "masked out of the loss" means "invisible to the model"; it does not.

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

If the model generates plausible raw text but ignores the assistant role, suspect the data
format. If the model learns to emit `<|im_end|>` immediately, inspect whether only closing
markers are being supervised. If eval loss is much lower than train loss, check whether eval
examples accidentally have fewer supervised tokens. The loss curve is only meaningful after
the mask is trustworthy.

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
