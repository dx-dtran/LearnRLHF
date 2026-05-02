# 00 — Data: Anthropic HH-RLHF

## Purpose

This note documents the raw HH-RLHF dataset, the chat template used throughout the
course, and the three derived datasets produced by `data_hh.py` for SFT, the reward
model, and PPO. It is a data contract rather than a derivation. SFT depends on
assistant tokens being marked correctly, the reward model depends on chosen and
rejected responses being paired with the right prompt, and PPO depends on
prompt-only examples ending exactly where generation should begin.

A row in HH-RLHF contains two possible assistant endings, one preferred by a human.
The same source row produces three views: a single dialogue for SFT, a pair of
sequences for the reward model, and a prompt-only example for PPO rollouts.

---

## 1. What is HH-RLHF?

HH-RLHF stands for "Helpful and Harmless, with RLHF". Anthropic released it
alongside Bai et al. 2022 ("Training a Helpful and Harmless Assistant with
Reinforcement Learning from Human Feedback").

Each row in the dataset is a pair of conversations. Both conversations start the
same way (same user questions, same earlier assistant replies), but they end
differently. One ending was picked by a human labeler as "better", and the other
was not. The picked one is called **chosen** and the other is called **rejected**.

The criterion "better" is intentionally vague. It can mean more helpful, more
harmless, more truthful, less evasive, better formatted, or more satisfying to the
labeler. The reward model receives no decomposition of those reasons. It sees the
pairwise outcome: chosen beat rejected. For this reason the reward model is a
preference model rather than a supervised classifier with an absolute target score.

A raw row looks like:

```json
{
  "chosen":   "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: <chosen>",
  "rejected": "\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: <rejected>"
}
```

Both strings are the full dialogue, including all earlier turns. Usually only the
final assistant turn differs between chosen and rejected, but the conversations
may diverge earlier in principle. The safe mental model is two full trajectories
that share a prefix.

For the teaching implementation, a shared prompt and two candidate responses are
extracted, but the dataset itself does not promise a perfect
`(prompt, chosen_response, rejected_response)` schema. Parsing code should be
simple. When a row is malformed or surprising, print and inspect it rather than
forcing it through silently.

The dataset is split into two subsets, `helpful-base` and `harmless-base`. For
this course use the whole `Anthropic/hh-rlhf` mix without separating them.

When you run Problem 0.2, fill in the following statistics:

- Number of rows in train and test.
- How many turns the typical conversation has (most are 1–4 human turns).
- Token length percentiles (p50, p95, p99) for `chosen` and `rejected`, measured
  with the GPT-2 BPE tokenizer.
- Three random samples printed verbatim.

Looking at actual rows surfaces things that a percentile table hides: extra blank
lines, assistant turns that start with refusals, multi-turn contexts where the
final answer depends on earlier dialogue, and examples whose "rejected" answer is
not obviously terrible. Preference data is noisy, which is one reason reward
models reach pairwise accuracies well below 100%.

---

## 2. Chat template

GPT-2 was pretrained on raw web text and has no representation of speaker roles.
Role tags are introduced by wrapping every turn:

```
<|im_start|>user
<turn text><|im_end|>
<|im_start|>assistant
<turn text><|im_end|>
```

This format is called ChatML, popularized by OpenAI's chat models. It is used
everywhere in this course (SFT examples, RM preference pairs, PPO rollouts) so
the model sees the same structure during every phase of training.

The exact template is less important than the consistency of its use. A model
can learn many reasonable chat formats, but it struggles when SFT uses one
format, RM uses another, and PPO prompts use a third. The template defines the
contract between phases: the SFT model learns to answer after an assistant
header, the RM learns to score responses in that same frame, and PPO samples
from prompts that end at exactly that same assistant header.

### Special tokens

`<|im_start|>` and `<|im_end|>` are not real tokens in GPT-2's vocabulary. There
are two reasonable options:

1. Pick two unused slots in the vocab and treat them as new special tokens.
2. Type the literal text `<|im_start|>` and let the BPE tokenizer split it into
   ordinary subword tokens (`<`, `|`, `im`, `_start`, `|`, `>`, and so on).

The `train.jsonl` / `test.jsonl` files in this repo use option (2), and this
course follows that choice. It costs 6–8 tokens per tag rather than 1, but it
avoids resizing the embedding table or introducing custom tokenization. Option
(1) would yield shorter sequences and cleaner boundaries, but would also
require resizing embeddings, deciding how to initialize the new rows, and
ensuring tied output embeddings still behave correctly. Those mechanics
distract from the main line of the curriculum.

### Role labels

The roles are written as `user` and `assistant`. The raw HH data uses `Human:`
and `Assistant:`, so the formatter translates `Human:` → `user` and
`Assistant:` → `assistant` while wrapping everything in the ChatML tags.

### Worked example: raw HH row to chat-formatted string

A short raw HH row:

```json
{
  "chosen":   "\n\nHuman: What is 2+2?\n\nAssistant: 4.",
  "rejected": "\n\nHuman: What is 2+2?\n\nAssistant: I'm not sure."
}
```

After the formatter applies the ChatML template, the `chosen` side becomes:

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

Both share the prompt prefix up through the second `<|im_start|>assistant\n`. From
this point on, the SFT dataset uses the chosen string, the RM uses both, and the
PPO prompt-only dataset keeps just the prefix.

### Worked example: tokens and loss_mask alignment

The full BPE expansion of the chat template is verbose: each `<|im_start|>`
literal splits into about half a dozen subwords. To make the structure visible,
this note uses a tiny printed vocab with one token per logical piece. The actual
code uses the real GPT-2 BPE; the alignment rule is the same.

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

The SFT mask. The rule from §3.1 is "1 only on assistant content, including the
trailing `<|im_end|>`." Applied position-by-position:

```
pos:         0   1   2   3   4   5   6   7   8   9
input_id:   10  11  12  13  14  15  16  17  18  15
piece:       U   W   I   2   ?  /U   A   4   .  /A
loss_mask:   0   0   0   0   0   0   0   1   1   1
```

Legend: `U` = user header, `W/I/2/?` = user content, `/U` = user closing
`<|im_end|>`, `A` = assistant header, `4/.` = assistant content, `/A` = assistant
closing `<|im_end|>`.

Two properties of this mask:

- The assistant header at position 6 (`<|im_start|>assistant\n`) is `mask = 0`.
  The inference code emits that token, so the model never has to predict it.
- The assistant's closing `<|im_end|>` at position 9 is `mask = 1`. The model
  must learn when to stop, which means it must learn to emit that closing token.

Module 2 (`02-sft.md`) revisits the same example after the shift to the
predict-next frame, where the mask attaches to the *target* token rather than
the *input* token.

---

## 3. Three derived datasets

One source, three views. All three live in `data_hh.py`.

### 3.1 SFT dataset

**Purpose.** Teach the base model (a) to produce text in our chat format, and
(b) to imitate the `chosen` assistant responses.

Each example is one full formatted dialogue: the `chosen` string rewritten into
the ChatML template. The dataloader returns two tensors:

- `input_ids`: the whole dialogue tokenized into a 1D sequence.
- `loss_mask`: a 0/1 sequence of the same length. `loss_mask[t] = 1` only if
  the token at position `t` is inside an **assistant turn's content**. Content
  means the text between the `<|im_start|>assistant\n` header and the following
  `<|im_end|>`, including the `<|im_end|>` itself.

This mask is the most common source of bugs in an SFT implementation and is
covered in detail in `02-sft.md`. RM and PPO both assume this mask is correct.

The rule is: the model is graded only on tokens that the assistant would have
had to type. The assistant would not type the user header, the user's message,
or the assistant header (the inference code supplies that header before
generation). It would type the assistant content and the closing marker that
tells generation to stop.

### 3.2 Preference dataset (for the reward model)

**Purpose.** Train a reward model to assign a scalar quality score to a full
`(prompt, response)` conversation. The loss requires that the score for the
chosen response be higher than the score for the rejected response, on average.

Each example holds `(prompt, chosen_response, rejected_response)`, where
`prompt` is the shared prefix up through the last `<|im_start|>assistant\n`
header (just before the response that differs).

The implementation tokenizes `prompt + chosen_response` and
`prompt + rejected_response` as two separate sequences. Sharing work on the
prompt prefix is possible but introduces caching and bookkeeping that are not
worth the complexity in a teaching implementation.

For each of the two sequences the dataloader also returns an attention mask
and the index of the last real (non-pad) token. The RM reads off its scalar
score at that index.

The last-token index is not arbitrary. In a causal transformer, the hidden
state at position `t` has seen only tokens up to `t`. The final non-pad token
is the first position that has seen the entire prompt and response. Pooling
from any earlier position scores a response the model has not finished
reading.

### 3.3 Prompt-only dataset (for PPO)

**Purpose.** Hand the policy a batch of prompts it has not seen, let it
generate completions, then score those completions with the RM and update the
policy. This view contains prompts only, with no assistant answers attached.

Each example is a prompt ending with `<|im_start|>assistant\n` and nothing
after. That header is the signal that the model should generate next.

Prompt-only examples are not just truncated SFT examples. Leaving part of the
gold assistant answer in the prompt makes PPO continue an answer that has
already started, rather than learning to generate one from scratch. Omitting
the assistant header lets the model continue as the user or produce raw web
text. The boundary must match the SFT and RM templates exactly.

The full context (prompt + generated response) must fit inside GPT-2's
1024-token window. The default split is: prompts at most 512 tokens, response
generation at most 256 new tokens.

**Left-pad** the prompts to the max length in the batch. Generation starts at
the last real position of the prompt. Right-padding would put the last real
position at a different index for every row, requiring per-row indexing.
Left-padding aligns every row's "next token to predict" at the same column,
which simplifies batched code.

Left-padding is uncommon for GPT-style models because pretraining used
contiguous text without padding. The attention mask still prevents pad tokens
from being treated as real context, and generation appends real tokens on the
right. The first generated log-prob should be tested against the correct
prompt position to confirm alignment.

---

## 4. Token budget

GPT-2 small has a context window of 1024 tokens, so across all phases:

    prompt_tokens + response_tokens  <=  1024

The default split is:

    prompt_tokens     <=  512
    response_tokens   <=  256

This leaves headroom and matches the PPO config.

During SFT, some full dialogues exceed 1024 tokens. These are rare but real.
When a long example is encountered, either drop it or truncate from the
**left** (keeping the tail). The assistant's final response is the supervised
signal and must not be discarded.

Left truncation may remove useful earlier context. Right truncation can chop
off the assistant answer or the `<|im_end|>` marker, training the model from
incomplete responses and corrupting stop behavior.

---

## 5. What to commit to `notes/00-data.md`

Fill this file in while doing Problem 0.2:

- Number of train and test rows.
- A quick histogram of turn counts ("most examples have 1–2 human turns").
- p50 / p95 / p99 token lengths for chosen, rejected, and prompt-only.
- Three verbatim samples (truncate if very long).
- Any oddities (malformed rows, duplicate pairs, weird formatting).

Useful data notes include both numbers and interpretation. "p95 chosen length
is 742 tokens" is a fact; "this means full-dialogue SFT will sometimes need
truncation, and PPO prompts must reserve response budget" connects the
distribution to a concrete engineering choice in the dataloaders.
