# 00 — Data: Anthropic HH-RLHF

## Purpose

Read this *before* implementing `data_hh.py` (Problem 0.2). It tells you what the dataset
looks like, what three derived datasets you need to produce, and what the chat template
is. You are not deriving anything on paper here — this is a data contract.

---

## 1. What is HH-RLHF?

`Anthropic/hh-rlhf` is a human-preference dataset released with Bai et al. 2022
("Training a Helpful and Harmless Assistant with Reinforcement Learning from Human
Feedback"). Each example is a pair of multi-turn dialogues that share the same prompt
prefix but end with two different candidate assistant responses. Humans chose which one
was better; `chosen` is the preferred one, `rejected` is the other.

Raw schema (per row):

```json
{
  "chosen":   "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: <chosen>",
  "rejected": "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: <rejected>"
}
```

Both strings are the full dialogue including all prior turns. The difference between
`chosen` and `rejected` is usually just the final assistant turn, but *in principle* can
diverge earlier. Treat them as two full trajectories, not "prompt + two endings".

The dataset has two major subsets — helpful-base and harmless-base. You can use the full
`Anthropic/hh-rlhf` mix without splitting them for this course.

Key statistics to verify yourself in Problem 0.2 (fill in the actual numbers):

- Number of train/test rows
- Distribution of turn counts (most are 1–4 human turns)
- Token-length percentiles (p50, p95, p99) of `chosen` and `rejected` under the GPT-2
  BPE tokenizer
- 3 random samples printed verbatim

---

## 2. Chat template

GPT-2's pretraining corpus is raw web text. It has no notion of "user" vs "assistant".
To teach it turn structure, we wrap each turn in role tags:

```
<|im_start|>user
<turn text><|im_end|>
<|im_start|>assistant
<turn text><|im_end|>
```

This is the ChatML template originally popularized by OpenAI's chat models. We enforce it
end-to-end — SFT examples, RM preference pairs, and PPO rollouts all use exactly this
format.

### Special tokens

`<|im_start|>` and `<|im_end|>` are **not** in the base GPT-2 BPE vocabulary. You have
two legitimate options:

1. **Reserve two unused BPE ids** and treat them as new special tokens.
2. **Encode them as their literal UTF-8 bytes** and let BPE split them into several
   subwords (`<`, `|`, `im`, `_start`, `|`, `>`, ...).

The existing `train.jsonl` / `test.jsonl` in this repo use option (2). Stick with (2)
for minimal churn. It's slightly wasteful (6–8 tokens per tag instead of 1) but does
not require touching the embedding table.

### Role labels

We use `user` and `assistant`. HH uses `Human:` / `Assistant:` in the raw string. Your
formatter's job is to convert from raw HH format to the ChatML template.

---

## 3. Three derived datasets

One source, three views. All three live in `data_hh.py`.

### 3.1 SFT dataset

**Purpose:** teach the base model to produce text in the chat format and to imitate the
distribution of `chosen` assistant responses.

- One example = one full formatted dialogue (the `chosen` string, reformatted into
  ChatML).
- Returns `input_ids` (the whole dialogue tokenized) and a `loss_mask` of the same shape.
- `loss_mask[t] = 1` iff token $t$ lies inside an **assistant turn's content** (the
  tokens between `<|im_start|>assistant\n` and the following `<|im_end|>`, **including**
  the `<|im_end|>`), else 0.

This last point is the single most common SFT bug. See `02-sft.md` for why masking the
prompt matters — this is critical to get right, as downstream RM and PPO depend on it.

### 3.2 Preference dataset (for RM)

**Purpose:** train a reward model to score full `(prompt, response)` trajectories such
that `r(chosen) > r(rejected)` in expectation.

- One example = `(prompt, chosen_response, rejected_response)` where `prompt` is the
  shared prefix up to and including the last `<|im_start|>assistant\n` header before the
  final turn.
- Tokenize `prompt + chosen_response` and `prompt + rejected_response` **separately**.
  You could be clever and share the prompt prefix in the forward pass, but don't — the
  clarity is worth more than the throughput for a teaching impl.
- Return attention masks and, for each of the two sequences, the index of the last real
  (non-pad) token. The RM reads out its scalar reward at that position.

### 3.3 Prompt-only dataset (for PPO)

**Purpose:** produce prompts that the current policy generates completions from, so we
can score them with the RM and do policy updates.

- One example = a prompt ending with `<|im_start|>assistant\n` (no assistant content yet).
- Truncate / select prompts so that `len(prompt_tokens) <= 512` and we have budget for
  `<= 256` generated tokens within the model's `block_size = 1024`.
- **Left-pad** to max prompt length in the batch so that the "next token to predict"
  position is aligned across the batch. (Right-padding would require per-sample offsets
  during rollout.)

---

## 4. Token budget

GPT-2 small has a context window of 1024 tokens. Your budget across all three phases:

$$T_\text{prompt} + T_\text{response} \le 1024.$$

Default split: $T_\text{prompt} \le 512$, $T_\text{response} \le 256$. This leaves
headroom and matches the PPO config. For SFT, full dialogues occasionally exceed 1024 —
drop those examples or truncate from the left (keep the tail; assistant responses are
what we care about).

---

## 5. What to commit to `notes/00-data.md`

Fill this file in as you run Problem 0.2:

- Train/test sizes
- Turn-count histogram (e.g. "most examples have 1–2 human turns")
- p50 / p95 / p99 token lengths of chosen, rejected, prompt-only
- 3 verbatim samples (truncated if huge)
- Any oddities you notice (malformed examples, duplicate pairs, etc.)

This is your reference every time you later wonder "wait, how long is a typical
response?" — don't skip it.
