# 00 — Data: Anthropic HH-RLHF

## Purpose

Read this *before* you write `data_hh.py` (Problem 0.2). It's a data contract, not a
derivation. You're not doing math here — you're figuring out what the dataset looks like,
what you need to produce from it, and how to format everything consistently.

Think of this note as the map between raw conversation strings and tensors the training code
can trust. Every later module depends on this map. SFT needs assistant tokens marked
correctly. The reward model needs chosen and rejected responses paired with the right prompt.
PPO needs prompt-only examples to end exactly where generation should begin. If those
boundaries are wrong, the model can train for hours while solving the wrong problem.

Module 0 is about understanding what each example means before tokenization. A row contains
two possible assistant endings, one preferred by a human. Your code has to preserve that
comparison while also producing the three views needed by SFT, RM, and PPO. When a tensor
looks suspicious three modules later, you should be able to trace it back to the raw HH row
and know what it was supposed to represent.

---

## 1. What is HH-RLHF?

HH-RLHF stands for "Helpful and Harmless, with RLHF". Anthropic released it alongside
Bai et al. 2022 ("Training a Helpful and Harmless Assistant with Reinforcement Learning
from Human Feedback").

Each row in the dataset is a pair of conversations. Both conversations start the same
way (same user questions, same earlier assistant replies), but they end differently —
one ending was picked by a human labeler as "better", and the other was not. The picked
one is called **chosen** and the other is called **rejected**.

That word "better" is deliberately vague. It can mean more helpful, more harmless, more
truthful, less evasive, better formatted, or more satisfying to the labeler. The reward model
receives no clean decomposition of those reasons. It sees the pairwise outcome: chosen beat
rejected. This is why the reward model is a preference model rather than a supervised
classifier with an absolute target score.

Here is what a raw row looks like:

```json
{
  "chosen":   "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: <chosen>",
  "rejected": "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: <rejected>"
}
```

Both strings are the full dialogue, including all earlier turns. Usually only the final
assistant turn differs between chosen and rejected, but in principle the conversations
could diverge earlier. The safe mental model is two full trajectories that happen to share
a prefix.

For the teaching implementation, we still usually *extract* a shared prompt and two candidate
responses. But while doing that extraction, keep the raw fact in mind: the dataset does not
promise a perfect `(prompt, chosen_response, rejected_response)` schema. Your parsing code
should be simple, but your debugging mindset should be humble. If a row looks malformed or
surprising, print it and inspect it rather than forcing it through silently.

The dataset is split into two subsets, `helpful-base` and `harmless-base`. For this
course just use the whole `Anthropic/hh-rlhf` mix — don't bother separating them.

When you run Problem 0.2, you'll fill in the following statistics for yourself:

- Number of rows in train and test.
- How many turns the typical conversation has (most are 1–4 human turns).
- Token length percentiles (p50, p95, p99) for `chosen` and `rejected`, measured with
  the GPT-2 BPE tokenizer.
- Three random samples printed verbatim, just so you've eyeballed real data.

The random samples are not decoration. They are how you catch assumptions that a percentile
table hides: extra blank lines, assistant turns that start with refusals, multi-turn contexts
where the final answer depends on earlier dialogue, and examples whose "rejected" answer is
not obviously terrible. Preference data is noisy. Looking at actual rows prepares you for
reward-model accuracy numbers that are good but nowhere near 100%.

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

Consistency matters more than the exact template. A model can learn many reasonable chat
formats, but it struggles when SFT uses one format, RM uses another, and PPO prompts use a
third. The template is the contract between phases: the SFT model learns to answer after an
assistant header, the RM learns to score responses in that same frame, and PPO samples from
prompts that end at exactly that same assistant header.

### Special tokens

`<|im_start|>` and `<|im_end|>` are not real tokens in GPT-2's vocabulary. We have two
reasonable options:

1. Pick two unused slots in the vocab and treat them as new special tokens.
2. Just type the literal text `<|im_start|>` and let the BPE tokenizer split it into a
   handful of ordinary subword tokens (`<`, `|`, `im`, `_start`, `|`, `>`, and so on).

The `train.jsonl` / `test.jsonl` files already in this repo use option (2), so we'll
stick with option (2) too. It wastes 6–8 tokens per tag instead of 1, but it means we
don't have to resize the embedding table or do anything custom at tokenization time.

This is a tradeoff between elegance and surface area. Adding real special tokens would make
the sequences shorter and the boundaries cleaner, but it would also require resizing
embeddings, deciding how to initialize the new rows, and making sure tied output embeddings
still behave correctly. For a from-scratch course, those extra mechanics distract from the
main line. Literal tags are verbose but transparent.

### Role labels

We write the roles as `user` and `assistant`. The raw HH data uses `Human:` and
`Assistant:` instead, so part of your formatter's job is to translate `Human:` → `user`
and `Assistant:` → `assistant` while wrapping everything in the ChatML tags.

### Worked example: raw HH row to chat-formatted string

Pretend the raw HH row is short:

```json
{
  "chosen":   "\n\nHuman: What is 2+2?\n\nAssistant: 4.",
  "rejected": "\n\nHuman: What is 2+2?\n\nAssistant: I'm not sure."
}
```

After your formatter applies the ChatML template, the `chosen` side becomes:

```
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
4.<|im_end|>
```

The `rejected` side becomes:

```
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
I'm not sure.<|im_end|>
```

Both share the prompt prefix up through the second `<|im_start|>assistant\n`. From this
point on, the SFT dataset will use the chosen string, the RM will use both, and the PPO
prompt-only dataset will keep just the prefix.

### Worked example: tokens and loss_mask alignment

The full BPE expansion of the chat template is verbose (each `<|im_start|>` literal
splits into about half a dozen subwords). To make the structure visible, this note uses
a tiny printed vocab with one token per logical piece. The *real* code uses the actual
GPT-2 BPE; the alignment rule is the same.

```
toy vocab:
   0: <pad>
  10: <|im_start|>user\n
  11: What
  12: is
  13: 2+2
  14: ?
  15: <|im_end|>
  16: <|im_start|>assistant\n
  17: 4
  18: .
```

The chat-formatted `chosen` string tokenizes to:

```
pos:        0   1   2   3   4   5   6   7   8
input_id:  10  11  12  13  14  15  16  17  18  15
                                                ^
                                                pos 9: closing <|im_end|>
```

Now the SFT mask. The rule from §3.1 is "1 only on assistant content, including the
trailing `<|im_end|>`." Apply it position-by-position:

```
pos:         0   1   2   3   4   5   6   7   8   9
input_id:   10  11  12  13  14  15  16  17  18  15
piece:       U   W   I   2   ?  /U   A   4   .  /A
loss_mask:   0   0   0   0   0   0   0   1   1   1
```

Legend: `U` = user header, `W/I/2/?` = user content, `/U` = user closing `<|im_end|>`,
`A` = assistant header, `4/.` = assistant content, `/A` = assistant closing `<|im_end|>`.

Two things to verify when staring at this:

- The assistant header at position 6 (`<|im_start|>assistant\n`) is `mask = 0`. Your
  inference code emits that token; the model never has to predict it.
- The assistant's closing `<|im_end|>` at position 9 is `mask = 1`. The model must
  learn when to stop, which means it must learn to emit that closing token.

Module 2 (`02-sft.md`) revisits the same example after the shift to the predict-next
frame, where the mask attaches to the *target* token rather than the *input* token.

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

The rule is: "the model is graded only on tokens that the assistant would have had to type."
The assistant would not type the user header, the user's message, or the assistant header if
your inference code supplies that header before generation. It *would* type the assistant
content and the closing marker that tells generation to stop. That boundary is the core SFT
data problem.

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

The last-token index is not an arbitrary pooling choice. In a causal transformer, the hidden
state at position `t` has seen only tokens up to `t`. The final non-pad token is the first
position that has seen the entire prompt and response. If you accidentally pool from the
first response token, the reward model is judging a response it has not read yet.

### 3.3 Prompt-only dataset (for PPO)

**Purpose.** Hand the policy a batch of prompts it hasn't seen yet, let it generate
completions, then score those completions with the RM and update the policy. For this
we need just prompts — no assistant answers attached.

Each example is a prompt ending with `<|im_start|>assistant\n` and nothing after. That
header is the signal "your turn to talk, model".

This is also why prompt-only examples are not just truncated SFT examples. If you leave part
of the gold assistant answer in the prompt, PPO is no longer learning to generate the answer;
it is continuing an answer that has already started. If you omit the assistant header, the
model may continue as the user or produce raw web text. The boundary must be exact.

We want the full context (prompt + generated response) to fit inside GPT-2's 1024-token
window. We split the budget: keep prompts to at most 512 tokens, and allow the model to
generate up to 256 new tokens.

**Left-pad** the prompts to the max length in the batch. Why left-pad instead of
right-pad? Because generation starts at the *last* real position of the prompt. If we
right-padded, the "last real position" would be at a different index for every row,
and we'd have to index into each row separately. Left-padding puts every row's "next
token to predict" at the same column, which is much simpler to handle in batched code.

Left-padding is slightly unnatural for GPT-style models because pretraining usually used
contiguous text without padding. But for batched generation it buys a very concrete
simplification: the final prompt token lines up across the batch. The attention mask still
prevents pad tokens from being treated as real context, and generation appends real tokens on
the right. The important thing is to be consistent and to test the first generated log-prob
against the correct prompt position.

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

Left truncation is a compromise. It may remove useful earlier context, but it preserves the
part of the example that produces the supervised signal. Right truncation is more dangerous:
it can chop off the assistant answer or the `<|im_end|>` marker, teaching the model from
incomplete responses and confusing stop behavior.

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

Good data notes include both numbers and interpretation. "p95 chosen length is 742 tokens"
is useful; "this means full-dialogue SFT will sometimes need truncation, and PPO prompts must
reserve response budget" is better. The goal is to connect the raw distribution to concrete
engineering choices in the dataloaders.
