# 06 — Evaluation

## Purpose

Methodology for Module 6 (Problems 6.1 through 6.3). There is almost no math left
at this stage — the theory is all behind you. This note is about:

1. How to fairly compare three models (base vs SFT vs RLHF).
2. How to compute a win-rate from blinded pairwise annotations.
3. What to write in your retrospective.

Fill this file in with your actual outputs as you run the problems.

### Quick metric formula (with tie handling)

Use the tie-aware win rate:

$$
\mathrm{win\_rate}_{\mathrm{RLHF}}
=
\frac{W_{\mathrm{RLHF}} + 0.5\,T}{N}
$$

where $W_{\mathrm{RLHF}}$ is RLHF wins, $T$ is ties, and $N$ is number of
prompts judged. Think of each tie as "half a win" so the metric stays fair.

---

## 1. Generation comparison (Problem 6.1)

Build a table with one row per held-out prompt (target: 20 prompts) and one
column per model (base, SFT, RLHF). Each cell holds the generated response,
trimmed to the first `<|im_end|>` or to a reasonable cutoff if the model never
stops.

### 1.1 Prompt selection

Pick 20 prompts from the HH test split that were **not** used in SFT or PPO
training. A reasonable mix:

- 5 helpful requests ("write me a Python function that ...").
- 5 conversational openers ("I've been feeling stressed about ...").
- 5 factual questions ("What's the capital of Kenya?").
- 5 edge cases where base GPT-2 is known to fail: long multi-step instructions,
  requests for specific formats (like JSON or tables), requests for lists.

### 1.2 Generation settings

**Keep them identical across all three models.** Temperature 1.0, top-p 0.9 (or
whatever you settle on). Use the same random seed per prompt across models. Any
difference you see should be attributable to the model, not to sampling
variance.

For a more deterministic comparison you can also generate with temperature 0
(greedy). More boring to read, but removes all sampling noise so the differences
are fully attributable to the model weights.

### 1.3 Format

A markdown table like:

```
| # | Prompt | Base | SFT | RLHF |
|---|--------|------|-----|------|
| 1 | ...    | ...  | ... | ...  |
```

Paste it under a "Results" section in this file.

---

## 2. Win-rate (Problem 6.2)

### 2.1 Setup

On the same 20 prompts, compare the SFT and RLHF responses pairwise. For each
prompt:

- Present both responses to yourself *without knowing which is which*. Write a
  simple randomizer that swaps the order with 50% probability and keeps the
  answer hidden.
- Rate one as "better" — more helpful, more accurate, more aligned with the
  instruction — or declare a tie.

### 2.2 Guarding against position bias

Humans (and LLM-as-judge evaluators) have a systematic preference for whichever
option they see first. Two mitigations:

- Randomize the order (you already did this above).
- Run the 20 comparisons once, then re-run them with order reversed, and average
  the results. If your ratings agree across the two passes, the randomizer was
  enough; if they differ, averaging corrects the bias.

### 2.3 Scoring

    win_rate_RLHF = (RLHF_wins + 0.5 * ties) / 20

Target: at least 0.55 — RLHF is meaningfully better than SFT on a majority of
prompts. If the win-rate is 0.5 or below, RLHF is no better than (or worse than)
SFT and the run failed. Likely reasons:

- Weak reward model.
- `beta` too large — the policy barely moved from SFT.
- Too few PPO iterations.
- Reward hacking that isn't visible on these particular prompts.

### 2.4 If RLHF didn't win

Don't paper over it. Write down honestly:

- What the RLHF responses look like that SFT's don't.
- What the RLHF responses are missing that you'd want.
- Which hyperparameter you'd change first, and why.

Then actually change it and try again, if you want to.

---

## 3. The retrospective (Problem 6.3)

One page, answering these six questions directly.

### 3.1 What was the hardest part?

Some candidates from past RLHF projects: aligning `logprobs_old` and
`logprobs_new` in the PPO loop; deciding between shared backbone and separate
value network; getting RM pairwise accuracy above the noise floor; managing
memory with four models loaded at once.

### 3.2 Where did you actually spend the most time?

Count your commits and your notes. The gap between "how hard I thought it would
be" and "how long it actually took" is diagnostic. Almost always, the answer
reveals where your mental model was the weakest.

### 3.3 Which gradient check saved you?

For every test in `tests/`, ask yourself: did it catch a real bug? Which bug?
Which tests were redundant? Use this to decide what to keep testing in your next
project.

### 3.4 What would you do differently next time?

Candidates: separate value network; adaptive `beta`; sample more prompts during
rollout; use `k_3` instead of `k_1` for the KL penalty; explicitly handle length
bias in the RM; initialize the value head from SFT instead of zero.

### 3.5 What did you get wrong and fix?

Trace back through your git history and identify bugs you caught during
development:

- Did you ever mask the wrong tokens in SFT? (Included user turns? Forgot to
  include `<|im_end|>`? Used an inverted mask?)
- Did you have an off-by-one or alignment bug between logprobs and tokens in the
  PPO rollout?
- Did you forget to clip the value loss, or compute advantages without masking?

These are the educational points of the whole exercise — the real bugs that
appear when you build RLHF from scratch, and how you learned to catch them.

### 3.6 What did you not understand before, and what do you understand now?

Pick one thing. Write two paragraphs. "I used to think X; now I think Y." If you
can't think of one, you didn't learn anything and you should redo the course.
But you *will* find one.

---

## 4. What to commit to `notes/06-eval.md`

- The 20-row generation table from 6.1.
- The 20-row win-rate tally from 6.2 with your scoring notes.
- The retrospective from 6.3.

This is the last file in the curriculum. When it's complete, the course is
complete.
