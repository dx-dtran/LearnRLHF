# 00 — Data: Anthropic HH-RLHF

## Purpose

Read this *before* you write `data_hh.py` (Problem 0.2). It's a data contract, not a
derivation. You're not doing math here — you're figuring out what the dataset looks like,
what you need to produce from it, and how to format everything consistently.

### Quick quantitative lens (light math, plain English)

Even though this module is mostly data plumbing, it helps to formalize the summary
stats you will compute:

\[
\text{len}_{\text{tok}}(x)=\text{number of GPT-2 BPE tokens in string }x
\]

\[
p_q=\text{$q$-th percentile of }\{\text{len}_{\text{tok}}(x_i)\}_{i=1}^{N}
\]

In plain language: `p50` is your "typical length", `p95` is "long but common",
and `p99` is "rare long tail". Those three numbers tell you where batching and
padding will hurt most.

---

## 1. What is HH-RLHF?

HH-RLHF stands for "Helpful and Harmless, with RLHF". Anthropic released it alongside
Bai et al. 2022 ("Training a Helpful and Harmless Assistant with Reinforcement Learning
from Human Feedback").

Each row in the dataset is a pair of conversations. Both conversations start the same
way (same user questions, same earlier assistant replies), but they end differently —
one ending was picked by a human labeler as "better", and the other was not. The picked
one is called **chosen** and the other is called **rejected**.

Here is what a raw row looks like:

```json
{
  "chosen":   "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: <chosen>",
  "rejected": "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: <rejected>"
}
```

Both strings are the full dialogue, including all earlier turns. Usually only the final
assistant turn differs between chosen and rejected, but not always — in principle the
conversations could diverge earlier. Treat them as two full trajectories, not as
"one prompt with two possible endings".

The dataset is split into two subsets, `helpful-base` and `harmless-base`. For this
course just use the whole `Anthropic/hh-rlhf` mix — don't bother separating them.

When you run Problem 0.2, you'll fill in the following statistics for yourself:

- Number of rows in train and test.
- How many turns the typical conversation has (most are 1–4 human turns).
- Token length percentiles (p50, p95, p99) for `chosen` and `rejected`, measured with
  the GPT-2 BPE tokenizer.
- Three random samples printed verbatim, just so you've eyeballed real data.

---

## 2. Chat template

GPT-2 was pretrained on raw web text. It has no concept of "this bit is the user
talking, this bit is the assistant talking". We have to teach it that by wrapping every
turn in special role tags. Here is the template we use:

```
<|im_start|>user
<turn text><|im_end|>
<|im_start|>assistant
<turn text><|im_end|>
```

This format is called ChatML. OpenAI popularized it with their chat models. We'll use
it everywhere — SFT examples, RM preference pairs, PPO rollouts — so the model sees the
same structure during every phase of training.

### Special tokens

`<|im_start|>` and `<|im_end|>` are not real tokens in GPT-2's vocabulary. We have two
reasonable options:

1. Pick two unused slots in the vocab and treat them as new special tokens.
2. Just type the literal text `<|im_start|>` and let the BPE tokenizer split it into a
   handful of ordinary subword tokens (`<`, `|`, `im`, `_start`, `|`, `>`, and so on).

The `train.jsonl` / `test.jsonl` files already in this repo use option (2), so we'll
stick with option (2) too. It wastes 6–8 tokens per tag instead of 1, but it means we
don't have to resize the embedding table or do anything custom at tokenization time.

### Role labels

We write the roles as `user` and `assistant`. The raw HH data uses `Human:` and
`Assistant:` instead, so part of your formatter's job is to translate `Human:` → `user`
and `Assistant:` → `assistant` while wrapping everything in the ChatML tags.

---

## 3. Three derived datasets

One source, three views. All three live in `data_hh.py`.

### 3.1 SFT dataset

**Purpose.** Teach the base model (a) to produce text in our chat format, and (b) to
imitate the `chosen` assistant responses.

Each example is one full formatted dialogue — the `chosen` string, rewritten into our
ChatML template. The dataloader returns two tensors:

- `input_ids`: the whole dialogue tokenized into a 1D sequence.
- `loss_mask`: a 0/1 sequence of the same length. `loss_mask[t] = 1` only if the token
  at position `t` is inside an **assistant turn's content**. By "content" we mean the
  text between the `<|im_start|>assistant\n` header and the following `<|im_end|>`,
  and we include the `<|im_end|>` itself.

This mask is the thing that breaks most often in an SFT implementation. We go deeper
on it in `02-sft.md`. Downstream, RM and PPO both assume this mask is correct, so
getting it right here matters a lot.

### 3.2 Preference dataset (for the reward model)

**Purpose.** Train a reward model to assign a scalar "quality score" to a full
`(prompt, response)` conversation. The loss asks that the score for the chosen response
be higher than the score for the rejected response, on average.

Each example holds `(prompt, chosen_response, rejected_response)`, where `prompt` is
the shared prefix up through the last `<|im_start|>assistant\n` header (just before the
response that differs).

The simplest implementation tokenizes `prompt + chosen_response` and
`prompt + rejected_response` as two separate sequences. You *could* be clever and
share work on the prompt prefix, but don't — the extra compute isn't worth the loss of
clarity for a teaching impl.

For each of the two sequences you also return an attention mask and the index of the
last real (non-pad) token. The RM reads off its scalar score at that index.

### 3.3 Prompt-only dataset (for PPO)

**Purpose.** Hand the policy a batch of prompts it hasn't seen yet, let it generate
completions, then score those completions with the RM and update the policy. For this
we need just prompts — no assistant answers attached.

Each example is a prompt ending with `<|im_start|>assistant\n` and nothing after. That
header is the signal "your turn to talk, model".

We want the full context (prompt + generated response) to fit inside GPT-2's 1024-token
window. We split the budget: keep prompts to at most 512 tokens, and allow the model to
generate up to 256 new tokens.

**Left-pad** the prompts to the max length in the batch. Why left-pad instead of
right-pad? Because generation starts at the *last* real position of the prompt. If we
right-padded, the "last real position" would be at a different index for every row,
and we'd have to index into each row separately. Left-padding puts every row's "next
token to predict" at the same column, which is much simpler to handle in batched code.

---

## 4. Token budget

GPT-2 small has a context window of 1024 tokens. So across all phases we need:

    prompt_tokens + response_tokens  <=  1024

Our default split is:

    prompt_tokens     <=  512
    response_tokens   <=  256

That leaves some headroom and matches the PPO config.

During SFT, some full dialogues exceed 1024 tokens — these are rare but real. When you
hit one, either drop the example or truncate from the **left** (keep the tail). The
assistant's final response is what we actually want to train on, so never throw that
away.

---

## 5. What to commit to `notes/00-data.md`

Fill this file in while you're doing Problem 0.2:

- Number of train and test rows.
- A quick histogram of turn counts ("most examples have 1–2 human turns").
- p50 / p95 / p99 token lengths for chosen, rejected, and prompt-only.
- Three verbatim samples (truncate if very long).
- Any oddities you notice (malformed rows, duplicate pairs, weird formatting, etc.).

This is your reference file. A month from now, when you're wondering "wait, how long is
a typical response?", you'll come back to this file. Write it like you'll need to reload
the dataset's shape into your head quickly.
